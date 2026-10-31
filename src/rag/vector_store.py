"""Save chunk vectors and search them by similarity.

`VectorStore` is the interface the rest of Finnie uses. `ChromaStore` is the Chroma
implementation, so another database could replace it without touching the retriever.

Safety rule: the collection remembers which embedding model made its vectors. Opening it
with a different model raises an error, because mixed models would not crash: they would
silently return meaningless rankings (Chroma only notices a different NUMBER of dimensions).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from src.core.config import Settings, load_settings
from src.core.logging import get_logger
from src.rag.chunker import Chunk

log = get_logger(__name__)


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


class ChromaStore(VectorStore):
    def __init__(self, embedding_model: str, path: str | Path = "chroma_db",
                 collection_name: str = "finnie_knowledge_base") -> None:
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._open_collection()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ChromaStore":
        settings = settings or load_settings()
        return cls(settings.rag.embedding_model, settings.rag.chroma_path, settings.rag.collection_name)

    def _open_collection(self):
        collection = self._client.get_or_create_collection(
            self.collection_name,
            embedding_function=None,   # we supply our own vectors; never let Chroma embed silently
            metadata={"embedding_model": self.embedding_model},
            configuration={"hnsw": {"space": "cosine"}},   # Chroma's default is squared-L2
        )
        stored_model = (collection.metadata or {}).get("embedding_model")
        if stored_model != self.embedding_model:
            raise EmbeddingModelMismatchError(
                f"collection '{self.collection_name}' holds vectors from '{stored_model}' but the "
                f"current embedding model is '{self.embedding_model}'. Re-index the knowledge base "
                f"(scripts.index_knowledge_base) instead of mixing models.")
        return collection

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[{**chunk.metadata, "doc_id": chunk.doc_id} for chunk in chunks],
        )

    def search(self, query_vector: np.ndarray, top_k: int, where: dict | None = None) -> list[SearchHit]:
        result = self._collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)
        return [SearchHit(chunk_id, text, metadata, 1.0 - distance)   # cosine distance = 1 - similarity
                for chunk_id, text, metadata, distance in zip(
                    result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._open_collection()
        log.info("reset collection '%s'", self.collection_name)
