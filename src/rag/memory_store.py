"""A throwaway in-memory VectorStore (exact cosine search with numpy). For experiments that need many small temporary indexes, such as the
chunk-size sweep; production uses PgVectorStore. Nothing is saved to disk."""
import numpy as np

from src.rag.chunker import Chunk
from src.rag.keyword_index import _matches
from src.rag.vector_store import SearchHit, VectorStore


class MemoryStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: dict[str, tuple[str, dict, np.ndarray]] = {}   # id -> (text, metadata, unit vector)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        for chunk, vector in zip(chunks, vectors):
            vector = np.asarray(vector, dtype=np.float32)
            self._chunks[chunk.chunk_id] = (chunk.text, {**chunk.metadata, "doc_id": chunk.doc_id}, vector / np.linalg.norm(vector))

    def search(self, query_vector: np.ndarray, top_k: int, where: dict | None = None) -> list[SearchHit]:
        query = np.asarray(query_vector, dtype=np.float32)
        query = query / np.linalg.norm(query)
        scored = [(float(vector @ query), chunk_id, text, meta) for chunk_id, (text, meta, vector) in self._chunks.items()
                  if not where or all(_matches(meta.get(key), value) for key, value in where.items())]
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [SearchHit(chunk_id, text, meta, similarity) for similarity, chunk_id, text, meta in scored[:top_k]]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks.clear()
