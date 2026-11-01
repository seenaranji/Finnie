"""Embedding wrapper with two independent speed-ups (Step 25). The question embedding call costs about 0.23 s of every retrieval answer.

  prefetch(text)   start embedding NOW in a background thread. The graph calls this for the user's message before the router runs, so the embedding
                   happens while the router is deciding (the two are independent: the router's answer does not change the text to embed). When the
                   Finance/Tax agent later asks for the same text, the embedding is already finished, or still running and shared (never done twice).
  cache            finished embeddings are remembered (least recently used, `max_entries` of them), so the same question asked again, for example a
                   starter button in the UI, costs no embedding call at all. max_entries=0 keeps nothing after a prefetched value is used.

Only single-text calls (a question) go through the cache; batches (indexing) pass straight to the real embedder.
"""
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

from src.rag.embedder import Embedder, EmbeddingResult


class CachedEmbedder(Embedder):
    def __init__(self, inner: Embedder, max_entries: int = 256, workers: int = 4) -> None:
        self.inner, self.max_entries = inner, max_entries
        self.model = getattr(inner, "model", None)
        self._futures: "OrderedDict[str, Future]" = OrderedDict()
        self._lock = threading.RLock()   # re-entrant: a done-callback may run on the thread that already holds it (a call that finishes before add_done_callback)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="embed-prefetch")
        self.calls = self.hits = 0   # calls made to the real embedder; lookups answered without one

    def _future_for(self, text: str) -> tuple[Future, bool]:
        """(future, was_already_there). Creating and registering happen under one lock, so two threads asking for the same text share one call."""
        with self._lock:
            future = self._futures.get(text)
            if future is not None:
                self._futures.move_to_end(text)
                return future, True
            future = self._pool.submit(self.inner.embed, [text])
            self.calls += 1
            self._futures[text] = future
            future.add_done_callback(lambda done, key=text: self._forget_failure(key, done))
            return future, False

    def _forget_failure(self, text: str, future: Future) -> None:
        if future.exception() is not None:   # never keep a failed call: the next request must try again
            with self._lock:
                if self._futures.get(text) is future:
                    del self._futures[text]

    def _trim(self, text: str) -> None:
        with self._lock:
            if self.max_entries == 0:
                self._futures.pop(text, None)   # consumed and nothing is kept (only OUR key: other requests' prefetches may still be running)
                return
            self._evict_finished(self.max_entries)

    def _evict_finished(self, limit: int) -> None:
        """Drop the oldest FINISHED entries above `limit`. A running one is never dropped: some request is about to read it."""
        for key in list(self._futures):
            if len(self._futures) <= limit:
                break
            if self._futures[key].done():
                del self._futures[key]

    def prefetch(self, text: str) -> None:
        """Fire and forget: start the embedding if nobody has asked for this text yet."""
        if text.strip():
            self._future_for(text)
            with self._lock:
                self._evict_finished(max(self.max_entries, 32))   # a prefetch for a route that never searches is never read: keep those from piling up

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if len(texts) != 1:
            return self.inner.embed(texts)
        future, existed = self._future_for(texts[0])
        result = future.result()
        if existed:
            self.hits += 1
            result = replace(result, cost_usd=0.0, latency_s=0.0, input_tokens=0)   # this lookup cost nothing: do not bill the same call twice
        self._trim(texts[0])
        return result
