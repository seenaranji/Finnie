"""Pick the vector store and keyword index named by config (`rag.vector_store`), so the flag is the only thing that flips (Step 23b).

Every caller goes through here, so agents and the router never learn which store is in use.
"""
from src.core.config import Settings, load_settings
from src.rag.vector_store import VectorStore

STORES = ("pgvector",)   # the switch stays so another store (the optional FAISS comparison, Step 6b) can be added


def build_vector_store(settings: Settings | None = None, kind: str | None = None) -> VectorStore:
    settings = settings or load_settings()
    kind = kind or settings.rag.vector_store
    if kind == "pgvector":
        from src.rag.pg_vector_store import PgVectorStore
        return PgVectorStore.from_settings(settings)
    raise ValueError(f"rag.vector_store must be one of {STORES} (got '{kind}')")


def load_keyword_index(settings: Settings | None = None, kind: str | None = None):
    """The keyword search that goes with the store: PostgreSQL full-text search for pgvector."""
    settings = settings or load_settings()
    kind = kind or settings.rag.vector_store
    if kind == "pgvector":
        from src.rag.pg_keyword_index import PgKeywordIndex
        return PgKeywordIndex.from_settings(settings)
    raise ValueError(f"rag.vector_store must be one of {STORES} (got '{kind}')")
