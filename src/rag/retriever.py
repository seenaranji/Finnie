"""Retrieval: question in, relevant cited passages out (or nothing).

Pipeline:  embed question -> search candidate_k chunks (optional category filter)
           -> drop chunks below the similarity threshold -> drop near-duplicates
           -> (optional) rerank -> keep top_k.

An empty result means "the knowledge base does not cover this", and the caller should say so
instead of letting the LLM guess.
"""
from dataclasses import dataclass

from src.core.config import Settings, load_settings
from src.rag.embedder import Embedder, OpenAIEmbedder
from src.rag.passages import Passage, dedupe_passages, passage_from_hit
from src.rag.reranker import LLMReranker, Reranker
from src.rag.vector_store import ChromaStore, VectorStore, category_filter


@dataclass
class RetrievalResult:
    question: str
    passages: list[Passage]          # final passages, best first, each with its citation
    best_similarity: float | None    # best score BEFORE any filtering (None if nothing was found)
    answerable: bool                 # False when no passage survived
    candidates: int                  # chunks fetched from the store
    dropped_by_threshold: int
    dropped_as_duplicates: int
    reranked: bool
    rerank_cost_usd: float = 0.0
    rerank_latency_s: float = 0.0


class Retriever:
    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None,
                 reranker: Reranker | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.store = store or ChromaStore.from_settings(self.settings)
        self.embedder = embedder or OpenAIEmbedder(self.settings)
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:   # only built (and only needs the LLM) when reranking is actually used
            self._reranker = LLMReranker()
        return self._reranker

    def retrieve(self, question: str, top_k: int | None = None, category: str | list[str] | None = None,
                 threshold: float | None = None, rerank: bool | None = None) -> RetrievalResult:
        """Arguments override the config.yaml values for this one call (used by experiments)."""
        rag = self.settings.rag
        top_k = top_k or rag.top_k
        threshold = rag.similarity_threshold if threshold is None else threshold
        use_rerank = rag.rerank if rerank is None else rerank

        query_vector = self.embedder.embed([question]).vectors[0]
        where = category_filter(category)
        hits = self.store.search(query_vector, max(rag.candidate_k, top_k), where)
        passages = [passage_from_hit(hit) for hit in hits]
        best_similarity = passages[0].similarity if passages else None

        above_threshold = [passage for passage in passages if passage.similarity >= threshold]
        unique, duplicates = dedupe_passages(above_threshold, rag.dedupe_overlap)

        reranked = use_rerank and len(unique) > 1
        cost = latency = 0.0
        if reranked:
            unique = self.reranker.rerank(question, unique)
            cost, latency = self.reranker.last_cost_usd, self.reranker.last_latency_s

        final = unique[:top_k]
        return RetrievalResult(question, final, best_similarity, bool(final), len(passages),
                               len(passages) - len(above_threshold), duplicates, reranked, cost, latency)
