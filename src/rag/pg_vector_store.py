"""PostgreSQL + pgvector implementation of the VectorStore interface (Step 6, moved to Postgres in Step 23b).

Text, metadata, vector and keyword index live in ONE table (`chunks`), so a re-index is one transaction and nothing can drift apart.

Distance vs similarity: pgvector's `<=>` is cosine DISTANCE (0 = same direction, smaller = more similar), so similarity = 1 - distance.
Metadata filters are plain SQL `WHERE`. Every value goes through a placeholder (never an f-string); filter KEYS come from our own code and
are checked against a strict pattern anyway.
"""
import re
from datetime import date

import numpy as np
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from src.core.config import Settings, load_settings
from src.core.logging import get_logger
from src.data.db import get_pool
from src.rag.chunker import Chunk
from src.rag.vector_store import EmbeddingModelMismatchError, SearchHit, VectorStore

log = get_logger(__name__)

_COLUMNS = {"category"}   # filter keys that are real indexed columns; any other key is looked up inside the metadata JSON
_KEY = re.compile(r"^[a-z_][a-z0-9_]*$")


def where_sql(where: dict | None) -> tuple[str, list]:
    """Translate the store-neutral filter ({"category": "etfs"} or {"category": {"$in": [...]}}) into ('AND ...', params).
    Shared by the vector search and the keyword search, so both retrievers always see the same chunks."""
    if not where:
        return "", []
    clauses, params = [], []
    for key, wanted in where.items():
        if not _KEY.match(key):
            raise ValueError(f"unsafe filter key: {key!r}")
        column = key if key in _COLUMNS else f"metadata ->> '{key}'"   # key was validated above; the value is always a parameter
        if isinstance(wanted, dict) and "$in" in wanted:
            clauses.append(f"{column} = ANY(%s)")
            params.append(list(wanted["$in"]))
        else:
            clauses.append(f"{column} = %s")
            params.append(wanted)
    return " AND " + " AND ".join(clauses), params


class PgVectorStore(VectorStore):
    def __init__(self, embedding_model: str, pool: ConnectionPool) -> None:
        self.embedding_model = embedding_model
        self.pool = pool
        self._check_embedding_model()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "PgVectorStore":
        settings = settings or load_settings()
        return cls(settings.rag.embedding_model, get_pool(settings))

    # ------------------------------------------------------------------ the silent-mix-up guard
    def _check_embedding_model(self) -> None:
        with self.pool.connection() as connection:
            row = connection.execute("SELECT value FROM store_meta WHERE key = 'embedding_model'").fetchone()
            if row is None:   # first use of an empty store: remember the model
                connection.execute("INSERT INTO store_meta (key, value) VALUES ('embedding_model', %s) ON CONFLICT DO NOTHING",
                                   (self.embedding_model,))
            elif row[0] != self.embedding_model:
                raise EmbeddingModelMismatchError(
                    f"the database holds vectors from '{row[0]}' but the current embedding model is '{self.embedding_model}'. "
                    f"Re-index the knowledge base (scripts.index_knowledge_base) instead of mixing models.")

    # ------------------------------------------------------------------ VectorStore interface
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Upsert in ONE transaction: re-running never duplicates, and an article's chunks that no longer exist (the article got
        shorter) are removed, so no stale chunk survives an edit."""
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        articles: dict[str, dict] = {}
        for chunk in chunks:
            articles.setdefault(chunk.doc_id, chunk.metadata)
        with self.pool.connection() as connection, connection.transaction():
            for article_id, meta in articles.items():
                updated = meta.get("last_updated")
                connection.execute(
                    """INSERT INTO articles (id, title, category, difficulty, source_name, source_url, last_updated)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, category = EXCLUDED.category, difficulty = EXCLUDED.difficulty,
                           source_name = EXCLUDED.source_name, source_url = EXCLUDED.source_url, last_updated = EXCLUDED.last_updated""",
                    (article_id, meta.get("title", article_id), meta.get("category", ""), meta.get("difficulty"),
                     meta.get("source_name"), meta.get("source_url"), date.fromisoformat(updated) if updated else None))
            with connection.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO chunks (id, article_id, chunk_index, text, category, metadata, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO UPDATE SET article_id = EXCLUDED.article_id, chunk_index = EXCLUDED.chunk_index, text = EXCLUDED.text,
                           category = EXCLUDED.category, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                    [(chunk.chunk_id, chunk.doc_id, chunk.index, chunk.text, chunk.metadata.get("category", ""),
                      Jsonb({**chunk.metadata, "doc_id": chunk.doc_id}), np.asarray(vector, dtype=np.float32))
                     for chunk, vector in zip(chunks, vectors)])
            for article_id in articles:
                connection.execute("DELETE FROM chunks WHERE article_id = %s AND NOT (id = ANY(%s))",
                                   (article_id, [c.chunk_id for c in chunks if c.doc_id == article_id]))

    def search(self, query_vector: np.ndarray, top_k: int, where: dict | None = None) -> list[SearchHit]:
        clause, params = where_sql(where)
        sql = f"""SELECT id, text, metadata, 1 - (embedding <=> %s) AS similarity
                  FROM chunks WHERE true{clause}
                  ORDER BY embedding <=> %s LIMIT %s"""
        vector = np.asarray(query_vector, dtype=np.float32)
        with self.pool.connection() as connection:
            rows = connection.execute(sql, [vector, *params, vector, top_k]).fetchall()
        return [SearchHit(chunk_id, text, metadata, float(similarity)) for chunk_id, text, metadata, similarity in rows]

    def count(self) -> int:
        with self.pool.connection() as connection:
            return connection.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def reset(self) -> None:
        with self.pool.connection() as connection:
            connection.execute("TRUNCATE articles CASCADE")   # chunks go with their articles
        log.info("reset: all articles and chunks deleted")

    # ------------------------------------------------------------------ helpers for migration and checks
    def all_chunk_ids(self) -> list[str]:
        with self.pool.connection() as connection:
            return [row[0] for row in connection.execute("SELECT id FROM chunks ORDER BY id")]

    def content_checksum(self) -> str:
        """One hash over every chunk id and text, in id order: compare with the checksum of the source to prove nothing was lost or altered."""
        import hashlib
        digest = hashlib.sha256()
        with self.pool.connection() as connection:
            # sorted in Python, not SQL: the database orders text by locale rules ('a-b#1' vs 'a#1'), which would not match a hash made elsewhere
            for chunk_id, text in sorted(connection.execute("SELECT id, text FROM chunks").fetchall()):
                digest.update(chunk_id.encode() + b"\0" + text.encode() + b"\0")
        return digest.hexdigest()[:12]
