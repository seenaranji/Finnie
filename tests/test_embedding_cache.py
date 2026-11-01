"""CachedEmbedder: prefetch overlap, de-duplication of simultaneous asks, LRU memory, failures, and zero-cost hits. A fake inner embedder, no network."""
import threading
import time

import numpy as np
import pytest

from src.rag.embedder import EmbeddingResult
from src.rag.embedding_cache import CachedEmbedder


class Inner:
    def __init__(self, delay=0.0, fail_first=0):
        self.texts, self.delay, self.fail_first, self.lock = [], delay, fail_first, threading.Lock()

    def embed(self, texts):
        with self.lock:
            self.texts.append(list(texts))
            failing = len(self.texts) <= self.fail_first
        time.sleep(self.delay)
        if failing:
            raise RuntimeError("embedding service down")
        return EmbeddingResult(np.full((len(texts), 3), float(len(texts[0])), dtype=np.float32), "m", 10 * len(texts), 0.001, 0.5, 1)


def test_a_repeated_question_is_embedded_once_and_the_repeat_costs_nothing():
    inner = Inner()
    c = CachedEmbedder(inner)
    first, second = c.embed(["what is an ETF?"]), c.embed(["what is an ETF?"])
    assert len(inner.texts) == 1 and c.hits == 1
    assert np.array_equal(first.vectors, second.vectors) and first.cost_usd == 0.001 and (second.cost_usd, second.latency_s, second.input_tokens) == (0.0, 0.0, 0)


def test_prefetch_starts_the_call_in_the_background_and_embed_reuses_it():
    inner = Inner(delay=0.2)
    c = CachedEmbedder(inner)
    start = time.perf_counter()
    c.prefetch("question")
    assert time.perf_counter() - start < 0.1          # prefetch returns at once
    time.sleep(0.25)                                   # the router "runs" meanwhile
    begin = time.perf_counter()
    c.embed(["question"])
    assert time.perf_counter() - begin < 0.1 and len(inner.texts) == 1   # already finished: no waiting, no second call


def test_asking_while_the_prefetch_is_still_running_waits_for_it_instead_of_starting_another():
    inner = Inner(delay=0.3)
    c = CachedEmbedder(inner)
    c.prefetch("q")
    c.embed(["q"])
    assert len(inner.texts) == 1


def test_simultaneous_requests_for_one_text_share_a_single_call():
    inner = Inner(delay=0.2)
    c = CachedEmbedder(inner)
    threads = [threading.Thread(target=c.embed, args=(["same"],)) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(inner.texts) == 1


def test_the_least_recently_used_finished_entry_is_forgotten_beyond_the_limit():
    inner = Inner()
    c = CachedEmbedder(inner, max_entries=2)
    for text in ("a", "b", "c"):
        c.embed([text])
    c.embed(["c"]); c.embed(["b"])
    assert len(inner.texts) == 3     # b and c are remembered
    c.embed(["a"])
    assert len(inner.texts) == 4     # a was evicted


def test_with_no_memory_a_used_prefetch_is_dropped_and_asking_again_calls_the_embedder_again():
    inner = Inner()
    c = CachedEmbedder(inner, max_entries=0)
    c.prefetch("q"); c.embed(["q"])
    assert len(inner.texts) == 1
    c.embed(["q"])
    assert len(inner.texts) == 2


def test_with_no_memory_one_request_does_not_discard_anothers_pending_prefetch():
    inner = Inner(delay=0.2)
    c = CachedEmbedder(inner, max_entries=0)
    c.prefetch("other request")          # still running
    c.embed(["mine"])                    # finishes and is dropped
    c.embed(["other request"])           # must find its own prefetch, not start a second call
    assert sorted(t[0] for t in inner.texts) == ["mine", "other request"]


def test_a_failed_call_is_not_remembered_so_the_next_attempt_tries_again():
    inner = Inner(fail_first=1)
    c = CachedEmbedder(inner)
    with pytest.raises(RuntimeError):
        c.embed(["q"])
    time.sleep(0.05)
    assert c.embed(["q"]).vectors.shape == (1, 3) and len(inner.texts) == 2


def test_a_persistent_failure_still_reaches_the_caller_even_if_it_started_as_a_prefetch():
    c = CachedEmbedder(Inner(fail_first=99))
    c.prefetch("q")
    with pytest.raises(RuntimeError):
        c.embed(["q"])


def test_a_failed_prefetch_does_not_poison_the_real_request_which_tries_again():
    inner = Inner(fail_first=1)
    c = CachedEmbedder(inner)
    c.prefetch("q")
    time.sleep(0.1)                      # the background attempt has failed and been forgotten
    assert c.embed(["q"]).vectors.shape == (1, 3) and len(inner.texts) == 2


def test_batches_bypass_the_cache_and_blank_prefetches_are_ignored():
    inner = Inner()
    c = CachedEmbedder(inner)
    c.embed(["a", "b"]); c.embed(["a", "b"])
    assert len(inner.texts) == 2 and c._futures == {}
    c.prefetch("   ")
    assert c._futures == {}


def test_unconsumed_prefetches_do_not_pile_up_forever():
    c = CachedEmbedder(Inner(), max_entries=0)
    for i in range(200):
        c.prefetch(f"question {i}")      # routes that never search never read theirs
        time.sleep(0.001)
    time.sleep(0.2)
    c.prefetch("one more")
    assert len(c._futures) <= 40
