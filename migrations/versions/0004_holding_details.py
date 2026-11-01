"""What kind of holding it is, and when it was bought (typed-in portfolios)

Revision ID: 0004
Revises: 0003
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

TYPES = "('stock', 'etf', 'mutual_fund', 'bond', 'reit', 'cash', 'crypto', 'other')"   # keep in step with src/data/asset_types.py


def upgrade() -> None:
    # Existing rows (CSV-based and sample portfolios) become 'other': we never knew what kind they were, and guessing would be inventing data.
    op.execute(f"ALTER TABLE holdings ADD COLUMN asset_type TEXT NOT NULL DEFAULT 'other' CHECK (asset_type IN {TYPES})")
    op.execute("ALTER TABLE holdings ADD COLUMN purchase_date DATE")


def downgrade() -> None:
    op.execute("ALTER TABLE holdings DROP COLUMN purchase_date")
    op.execute("ALTER TABLE holdings DROP COLUMN asset_type")
