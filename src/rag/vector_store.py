"""Save chunk vectors and search them by similarity.

`VectorStore` is the interface the rest of Finnie uses. `PgVectorStore` (src/rag/pg_vector_store.py) is the implementation in production;
`MemoryStore` is for throwaway experiments. Another database could replace either without touching the retriever.

Safety rule: the store remembers which embedding model made its vectors. Opening it with a different model raises
EmbeddingModelMismatchError, because mixed models would not crash: they would silently return meaningless rankings.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.rag.chunker import Chunk


def category_filter(category: "str | list[str] | None") -> dict | None:
    """The metadata filter for one category, several categories, or none. Used by BOTH retrievers so they see the same chunks."""
    if not category:
        return None
    if isinstance(category, str):
        return {"category": category}
    return {"category": {"$in": list(category)}}


class EmbeddingModelMismatchError(RuntimeError):
    """The store was built with a different embedding model than the one in use."""


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict
    similarity: float   # cosine similarity, -1..1 (1 = same direction). NOT a probability.


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Save chunks with their vectors (same order). Re-adding an id replaces it."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int, where: dict | None = None) -> list[SearchHit]:
        """The top_k most similar chunks, best first. `where` filters on metadata."""

    @abstractmethod
    def count(self) -> int:
        """Number of chunks stored."""

    @abstractmethod
    def reset(self) -> None:
        """Delete every stored chunk (used before a full re-index)."""
