"""Retriever, HybridRetriever, dedupe, MemoryStore and the store factory, with fake embedder / store / keyword index (no network, no database)."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.config import load_settings
from src.rag.chunker import Chunk
from src.rag.factory import build_vector_store
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.keyword_index import KeywordHit
from src.rag.memory_store import MemoryStore
from src.rag.passages import Citation, Passage, dedupe_passages
from src.rag.retriever import Retriever
from src.rag.vector_store import SearchHit, VectorStore, category_filter


def settings(**rag):
    s = load_settings()
    for key, value in rag.items():
        setattr(s.rag, key, value)
    return s


def meta(doc, category="etfs"):
    return {"doc_id": doc, "title": doc.title(), "section": "S", "category": category, "source_name": "Src", "source_url": "https://x", "last_updated": "2026-10-30"}


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return SimpleNamespace(vectors=np.ones((len(texts), 3), dtype=np.float32))


class FakeStore(VectorStore):
    """Returns canned hits (best first) and records how it was asked."""
    def __init__(self, hits):
        self.hits, self.asked = hits, []

    def add(self, chunks, vectors): ...
    def reset(self): ...
    def count(self): return len(self.hits)

    def search(self, query_vector, top_k, where=None):
        self.asked.append((top_k, where))
        return self.hits[:top_k]


def words(chunk_id):
    """Six words that appear in no other chunk, so the dedupe step never mistakes two test chunks for copies."""
    return " ".join(f"{chunk_id.replace('#', 'n')}w{i}" for i in range(6))


def hit(chunk_id, similarity, text=None, category="etfs"):
    return SearchHit(chunk_id, text or words(chunk_id), meta(chunk_id.split("#")[0], category), similarity)


class FakeKeywords:
    def __init__(self, hits):
        self.hits, self.asked = hits, []

    def __len__(self): return len(self.hits)

    def search(self, query, top_k, where=None):
        self.asked.append((query, top_k, where))
        return self.hits[:top_k]


def kw(chunk_id, score):
    return KeywordHit(chunk_id, words(chunk_id), meta(chunk_id.split("#")[0]), score)


def dense(hits, **rag):
    s = settings(**rag)
    return Retriever(FakeStore(hits), FakeEmbedder(), settings=s)


# ---------- the Step 7 retriever ----------
def test_threshold_drops_weak_hits_and_the_survivors_carry_citations():
    r = dense([hit("a#0", 0.80), hit("b#0", 0.50), hit("c#0", 0.40)], similarity_threshold=0.465)
    result = r.retrieve("q")
    assert [p.chunk_id for p in result.passages] == ["a#0", "b#0"] and result.dropped_by_threshold == 1 and result.answerable
    assert result.passages[0].citation.label() == "A > S (Src, updated 2026-10-30)" and result.best_similarity == 0.80


def test_nothing_above_the_threshold_means_not_answerable_but_the_best_score_is_still_reported():
    result = dense([hit("a#0", 0.30)], similarity_threshold=0.465).retrieve("pizza")
    assert result.passages == [] and not result.answerable and result.best_similarity == 0.30


def test_an_empty_store_gives_no_best_similarity():
    result = dense([]).retrieve("q")
    assert not result.answerable and result.best_similarity is None and result.candidates == 0


def test_top_k_and_a_per_call_threshold_override_the_config():
    r = dense([hit(f"d{i}#0", 0.9 - i * 0.01) for i in range(8)], similarity_threshold=0.99, top_k=5)
    assert r.retrieve("q").passages == []
    assert len(r.retrieve("q", top_k=3, threshold=0.0).passages) == 3


def test_candidates_fetched_are_at_least_candidate_k_and_the_category_filter_reaches_the_store():
    r = dense([hit("a#0", 0.9)], candidate_k=20)
    r.retrieve("q", top_k=5, category=["etfs", "bonds"])
    assert r.store.asked == [(20, {"category": {"$in": ["etfs", "bonds"]}})]
    assert category_filter("etfs") == {"category": "etfs"} and category_filter(None) is None and category_filter([]) is None


def test_near_duplicate_chunks_are_dropped_and_counted():
    same = "an expense ratio is the yearly fee a fund charges to investors who hold it"
    result = dense([hit("a#0", 0.9, same), hit("b#0", 0.8, same + " today")]).retrieve("q", threshold=0.0)
    assert [p.chunk_id for p in result.passages] == ["a#0"] and result.dropped_as_duplicates == 1


def test_rerank_off_by_default_and_on_when_asked():
    class Reranker:
        last_cost_usd, last_latency_s = 0.002, 0.4
        def rerank(self, question, passages):
            return list(reversed(passages))
    r = Retriever(FakeStore([hit("a#0", 0.9), hit("b#0", 0.8)]), FakeEmbedder(), Reranker(), settings(rerank=False))
    assert [p.chunk_id for p in r.retrieve("q").passages] == ["a#0", "b#0"]
    result = r.retrieve("q", rerank=True)
    assert [p.chunk_id for p in result.passages] == ["b#0", "a#0"] and result.reranked and result.rerank_cost_usd == 0.002


def test_dedupe_rules_for_short_text_and_overlap():
    def p(text): return Passage("x", text, 0.5, Citation("d", "T", "S", "s", "u", "2026"))
    kept, dropped = dedupe_passages([p("tiny text"), p("tiny text"), p("other tiny")], overlap=0.8)
    assert len(kept) == 2 and dropped == 1   # under 5 words: only identical word sets count
    kept, dropped = dedupe_passages([p("one two three four five six"), p("one two three four five seven")], overlap=0.8)
    assert len(kept) == 1   # 5 of 6 words shared = 0.83
    kept, _ = dedupe_passages([p("one two three four five six"), p("one two three four nine ten")], overlap=0.8)
    assert len(kept) == 2


# ---------- hybrid ----------
def hybrid(dense_hits, keyword_hits, **rag):
    s = settings(**rag)
    return HybridRetriever(dense=Retriever(FakeStore(dense_hits), FakeEmbedder(), settings=s), keyword_index=FakeKeywords(keyword_hits), settings=s)


def test_dense_mode_is_exactly_the_step7_retriever_and_never_touches_the_keyword_index():
    h = hybrid([hit("a#0", 0.9)], [kw("z#0", 9.0)], retrieval_mode="dense")
    assert [p.chunk_id for p in h.retrieve("q").passages] == ["a#0"] and h.keyword_index.asked == []


def test_hybrid_ranks_a_chunk_found_by_both_lists_first():
    h = hybrid([hit("a#0", 0.9), hit("b#0", 0.8)], [kw("c#0", 5.0), kw("b#0", 4.0)], similarity_threshold=0.5, bm25_min_score=1.0, rrf_k=60)
    passages = h.retrieve("q", mode="hybrid").passages
    assert [p.chunk_id for p in passages][0] == "b#0"   # 1/62 + 1/62 beats 1/61 alone
    top = passages[0]
    assert top.dense_rank == 2 and top.bm25_rank == 2 and top.fused_score == pytest.approx(2 / 62)


def test_by_default_keyword_hits_cannot_open_the_gate():
    h = hybrid([hit("a#0", 0.30)], [kw("b#0", 9.0)], similarity_threshold=0.465, bm25_min_score=None)
    result = h.retrieve("q", mode="hybrid")
    assert result.passages == [] and not result.answerable and result.dropped_by_threshold == 2


def test_a_keyword_cutoff_lets_a_strong_keyword_hit_in_even_when_dense_is_weak():
    h = hybrid([hit("a#0", 0.30)], [kw("b#0", 9.0), kw("c#0", 1.0)], similarity_threshold=0.465, bm25_min_score=3.0)
    assert [p.chunk_id for p in h.retrieve("q", mode="hybrid").passages] == ["b#0"]
    forced = h.retrieve("q", mode="hybrid", bm25_min_score=0.0)   # a per-call override beats the config
    assert {p.chunk_id for p in forced.passages} == {"b#0", "c#0"}


def test_keyword_only_mode_never_asks_the_embedder():
    h = hybrid([hit("a#0", 0.9)], [kw("b#0", 4.0)], bm25_min_score=0.0)
    assert [p.chunk_id for p in h.retrieve("q", mode="bm25").passages] == ["b#0"] and h.dense.embedder.calls == []


def test_the_same_category_filter_goes_to_both_retrievers():
    h = hybrid([hit("a#0", 0.9)], [kw("a#0", 4.0)], bm25_min_score=0.0)
    h.retrieve("q", category="etfs", mode="hybrid")
    assert h.dense.store.asked[0][1] == {"category": "etfs"} and h.keyword_index.asked[0][2] == {"category": "etfs"}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        hybrid([], []).retrieve("q", mode="semantic")


# ---------- memory store + factory ----------
def chunk(chunk_id, text, category="etfs"):
    doc, _, index = chunk_id.partition("#")
    return Chunk(chunk_id, doc, text, {"category": category, "title": doc}, int(index), 0, len(text))


def test_memory_store_returns_best_first_with_cosine_similarity_and_filters():
    store = MemoryStore()
    store.add([chunk("a#0", "x"), chunk("b#0", "y", "bonds"), chunk("c#0", "z")], np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32))
    hits = store.search(np.array([1.0, 0.0]), 3)
    assert [h.chunk_id for h in hits] == ["a#0", "b#0", "c#0"] and hits[0].similarity == pytest.approx(1.0) and hits[1].similarity == pytest.approx(2 ** -0.5)
    assert [h.chunk_id for h in store.search(np.array([1.0, 0.0]), 3, {"category": "bonds"})] == ["b#0"]
    assert hits[0].metadata["doc_id"] == "a" and store.count() == 3
    store.reset()
    assert store.count() == 0


def test_memory_store_upserts_and_checks_lengths():
    store = MemoryStore()
    store.add([chunk("a#0", "old")], np.array([[1.0, 0.0]]))
    store.add([chunk("a#0", "new")], np.array([[0.0, 1.0]]))
    assert store.count() == 1 and store.search(np.array([0.0, 1.0]), 1)[0].text == "new"
    with pytest.raises(ValueError):
        store.add([chunk("b#0", "x")], np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_the_factory_refuses_an_unknown_store_name():
    with pytest.raises(ValueError):
        build_vector_store(load_settings(), "chroma")
