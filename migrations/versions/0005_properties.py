"""Real estate: properties owned inside a portfolio, valued by the owner (there is no market price for a house)

Revision ID: 0005
Revises: 0004
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

KINDS = "('house', 'condo', 'multi_family', 'land', 'commercial', 'other')"   # keep in step with src/data/asset_types.py


def upgrade() -> None:
    # Every amount is NUMERIC (exact). current_value is the OWNER's own estimate and is only as fresh as their last edit.
    # The mortgage is a single balance: Finnie does not track payments, interest rates or terms.
    op.execute(f"""
        CREATE TABLE properties (
            id                BIGSERIAL PRIMARY KEY,
            portfolio_id      UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
            name              TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 120),
            property_type     TEXT NOT NULL DEFAULT 'house' CHECK (property_type IN {KINDS}),
            purchase_price    NUMERIC NOT NULL CHECK (purchase_price >= 0),
            purchase_date     DATE,
            current_value     NUMERIC NOT NULL CHECK (current_value >= 0),
            mortgage_balance  NUMERIC NOT NULL DEFAULT 0 CHECK (mortgage_balance >= 0),
            monthly_rent      NUMERIC NOT NULL DEFAULT 0 CHECK (monthly_rent >= 0),
            monthly_expenses  NUMERIC NOT NULL DEFAULT 0 CHECK (monthly_expenses >= 0),
            UNIQUE (portfolio_id, name)
        )""")


def downgrade() -> None:
    op.execute("DROP TABLE properties")
