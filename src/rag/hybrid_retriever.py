"""Hybrid retrieval: dense (meaning) + BM25 (exact words), merged with Reciprocal Rank Fusion.

Same interface as the Step 7 `Retriever` (same `retrieve(...)` arguments, same `RetrievalResult`),
so agents don't change. `rag.retrieval_mode` picks the behaviour:
  dense  -> exactly the Step 7 retriever
  bm25   -> keyword search only
  hybrid -> both lists, gated, then fused

THE GATE (why the Step 7 threshold cannot be applied after fusion):
RRF scores are tiny rank-based numbers (at most about 0.033), not similarities, so "score >= 0.465"
would refuse everything. Instead we gate BEFORE fusing, on each retriever's own score:
  - a dense candidate is eligible when its cosine similarity >= similarity_threshold (Step 7);
  - a BM25 candidate is eligible when its BM25 score >= bm25_min_score (None = BM25 cannot let anything in).
Only eligible candidates are fused. If none is eligible the result is empty: "the knowledge base
does not cover this", exactly like Step 7.
"""
from dataclasses import dataclass, replace

from src.core.config import Settings, load_settings
from src.rag.embedder import Embedder
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.keyword_index import KeywordHit, KeywordIndex
from src.rag.passages import Passage, dedupe_passages, passage_from_hit
from src.rag.reranker import Reranker
from src.rag.retriever import RetrievalResult, Retriever
from src.rag.vector_store import SearchHit, VectorStore

MODES = ("dense", "bm25", "hybrid")
_FROM_CONFIG = object()   # sentinel: "use rag.bm25_min_score" (None is a real value meaning "BM25 cannot open the gate")


class StaleKeywordIndexError(RuntimeError):
    """The BM25 index does not match the vector store (knowledge base changed since it was built)."""


@dataclass
class Candidates:
    """The raw, ungated, best-first lists from each retriever (used by the side-by-side demo)."""
    dense: list[Passage]
    keyword: list[KeywordHit]


class HybridRetriever:
    def __init__(self, dense: Retriever | None = None, keyword_index: KeywordIndex | None = None,
                 store: VectorStore | None = None, embedder: Embedder | None = None,
                 reranker: Reranker | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.dense = dense or Retriever(store, embedder, reranker, self.settings)
        self._keyword_index = keyword_index

    @property
    def keyword_index(self) -> KeywordIndex:
        if self._keyword_index is None:   # loaded only when a mode that needs it is used
            index = KeywordIndex.load(self.settings.rag.keyword_index_path)
            stored = self.dense.store.count()
            if len(index) != stored:
                raise StaleKeywordIndexError(
                    f"keyword index has {len(index)} chunks but the vector store has {stored}. "
                    f"Rebuild both: .venv/bin/python -m scripts.index_knowledge_base")
            self._keyword_index = index
        return self._keyword_index

    # ------------------------------------------------------------------ the two raw lists
    def candidates(self, question: str, category: str | None = None, n: int | None = None) -> Candidates:
        """Top n from each retriever with the SAME category filter, no gate. n defaults to candidates_per_retriever."""
        n = n or self.settings.rag.candidates_per_retriever
        where = {"category": category} if category else None
        query_vector = self.dense.embedder.embed([question]).vectors[0]
        dense = [passage_from_hit(hit) for hit in self.dense.store.search(query_vector, n, where)]
        return Candidates(dense, self.keyword_index.search(question, n, where))

    # ------------------------------------------------------------------ the Step 7 interface
    def retrieve(self, question: str, top_k: int | None = None, category: str | None = None,
                 threshold: float | None = None, rerank: bool | None = None,
                 mode: str | None = None, bm25_min_score=_FROM_CONFIG) -> RetrievalResult:
        """Arguments override config.yaml for this one call. `mode` and `bm25_min_score` are the only new (optional)
        arguments; pass bm25_min_score=0.0 to let every BM25 hit through (used to measure ranking without the gate)."""
        rag = self.settings.rag
        mode = mode or rag.retrieval_mode
        if mode not in MODES:
            raise ValueError(f"retrieval mode must be one of {MODES} (got '{mode}')")
        if mode == "dense":
            return self.dense.retrieve(question, top_k, category, threshold, rerank)

        top_k = top_k or rag.top_k
        threshold = rag.similarity_threshold if threshold is None else threshold
        use_rerank = rag.rerank if rerank is None else rerank

        found = self.candidates(question, category) if mode == "hybrid" else Candidates([], self.keyword_index.search(
            question, rag.candidates_per_retriever, {"category": category} if category else None))
        best_similarity = found.dense[0].similarity if found.dense else None

        # --- the gate, applied to each list on its own scale BEFORE fusing ---
        dense_ok = [p for p in found.dense if p.similarity >= threshold]
        min_bm25 = rag.bm25_min_score if bm25_min_score is _FROM_CONFIG else bm25_min_score
        keyword_ok = [h for h in found.keyword if min_bm25 is not None and h.score >= min_bm25]
        dropped = (len(found.dense) - len(dense_ok)) + (len(found.keyword) - len(keyword_ok))

        passages = self._fuse(dense_ok, keyword_ok, found.keyword, rag.rrf_k) if mode == "hybrid" \
            else self._keyword_only(keyword_ok)

        unique, duplicates = dedupe_passages(passages, rag.dedupe_overlap)
        reranked = use_rerank and len(unique) > 1
        cost = latency = 0.0
        if reranked:
            unique = self.dense.reranker.rerank(question, unique)
            cost, latency = self.dense.reranker.last_cost_usd, self.dense.reranker.last_latency_s

        final = unique[:top_k]
        return RetrievalResult(question, final, best_similarity, bool(final),
                               len(found.dense) + len(found.keyword), dropped, duplicates, reranked, cost, latency)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _keyword_only(keyword_ok: list[KeywordHit]) -> list[Passage]:
        passages = []
        for rank, hit in enumerate(keyword_ok, start=1):
            passage = passage_from_hit(SearchHit(hit.chunk_id, hit.text, hit.metadata, 0.0))
            passages.append(replace(passage, bm25_score=hit.score, bm25_rank=rank))
        return passages

    @staticmethod
    def _fuse(dense_ok: list[Passage], keyword_ok: list[KeywordHit], all_keyword: list[KeywordHit],
              rrf_k: int) -> list[Passage]:
        fused = reciprocal_rank_fusion([[p.chunk_id for p in dense_ok], [h.chunk_id for h in keyword_ok]], rrf_k)
        by_id = {p.chunk_id: p for p in dense_ok}
        dense_rank = {p.chunk_id: rank for rank, p in enumerate(dense_ok, start=1)}
        keyword_by_id = {h.chunk_id: h for h in keyword_ok}
        keyword_rank = {h.chunk_id: rank for rank, h in enumerate(keyword_ok, start=1)}
        raw_bm25 = {h.chunk_id: h.score for h in all_keyword}   # show the BM25 score even if the gate dropped it

        passages = []
        for chunk_id, fused_score in fused:
            if chunk_id in by_id:
                passage = by_id[chunk_id]
            else:   # found by BM25 only: dense search did not return it, so there is no cosine similarity (0.0)
                hit = keyword_by_id[chunk_id]
                passage = passage_from_hit(SearchHit(hit.chunk_id, hit.text, hit.metadata, 0.0))
            passages.append(replace(passage, bm25_score=raw_bm25.get(chunk_id), dense_rank=dense_rank.get(chunk_id),
                                    bm25_rank=keyword_rank.get(chunk_id), fused_score=fused_score))
        return passages
