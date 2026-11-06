"""Mortgage terms on a property: the monthly payment and the interest rate, so a payoff time can be worked out on request

Revision ID: 0006
Revises: 0005
"""
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Both default to 0 = "not entered yet", so every existing property keeps working and simply has no payoff answer until the owner fills these in.
    op.execute("""
        ALTER TABLE properties
            ADD COLUMN monthly_mortgage_payment NUMERIC NOT NULL DEFAULT 0 CHECK (monthly_mortgage_payment >= 0),
            ADD COLUMN mortgage_rate_pct        NUMERIC NOT NULL DEFAULT 0 CHECK (mortgage_rate_pct >= 0 AND mortgage_rate_pct <= 100)""")


def downgrade() -> None:
    op.execute("ALTER TABLE properties DROP COLUMN monthly_mortgage_payment, DROP COLUMN mortgage_rate_pct")
