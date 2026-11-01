"""Portfolios and holdings (Step 15b)

Revision ID: 0002
Revises: 0001
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # UUID public ids (an auto-increment id is guessable), TIMESTAMPTZ times, NUMERIC money and share counts (floats round: 0.1 + 0.2 != 0.3).
    op.execute("""
        CREATE TABLE portfolios (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name        TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )""")
    op.execute("""
        CREATE TABLE holdings (
            id            BIGSERIAL PRIMARY KEY,
            portfolio_id  UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
            ticker        TEXT NOT NULL,
            shares        NUMERIC NOT NULL CHECK (shares > 0),
            avg_cost      NUMERIC NOT NULL CHECK (avg_cost >= 0),
            UNIQUE (portfolio_id, ticker)
        )""")
    # NOTE: UNIQUE (portfolio_id, ticker) already creates an index that starts with portfolio_id, so lookups by portfolio are indexed.
    # The latency benchmark (scripts.step15b_benchmark) drops it on purpose, on a throwaway table, to show the difference.


def downgrade() -> None:
    op.execute("DROP TABLE holdings")
    op.execute("DROP TABLE portfolios")
