"""Knowledge base tables: articles, chunks (vectors + full-text index in one table), store_meta (Steps 6 and 7c)

Revision ID: 0001
Revises:
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")   # an extension adds types and operators (vector, <=>) to the database
    op.execute("""
        CREATE TABLE articles (
            id            TEXT PRIMARY KEY,          -- the file name without extension, e.g. 'etf-basics'
            title         TEXT NOT NULL,
            category      TEXT NOT NULL,
            difficulty    TEXT,
            source_name   TEXT,
            source_url    TEXT,
            last_updated  DATE
        )""")
    # One table holds the text, the metadata, the vector and the keyword index: one transaction, one backup, nothing to keep in sync.
    # vector(1536) must match the embedding model (text-embedding-3-small makes 1536 numbers); a different size is rejected loudly.
    op.execute("""
        CREATE TABLE chunks (
            id           TEXT PRIMARY KEY,           -- 'etf-basics#2'
            article_id   TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
            chunk_index  INTEGER NOT NULL,
            text         TEXT NOT NULL,
            category     TEXT NOT NULL,
            metadata     JSONB NOT NULL,             -- exactly what the retriever turns into a citation
            embedding    vector(1536) NOT NULL,
            tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        )""")
    op.execute("CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")   # approximate nearest neighbour, cosine
    op.execute("CREATE INDEX chunks_category ON chunks (category)")
    op.execute("CREATE INDEX chunks_article_id ON chunks (article_id)")   # the foreign key's index: makes ON DELETE CASCADE and per-article cleanup fast
    op.execute("CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv)")   # full-text search (Step 7c)
    # Remembers which embedding model made the vectors. Mixing models does not crash, it silently ranks nonsense, so the store checks this.
    op.execute("CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


def downgrade() -> None:
    op.execute("DROP TABLE store_meta")
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE articles")
    op.execute("DROP EXTENSION IF EXISTS vector")
