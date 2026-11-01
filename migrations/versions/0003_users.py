"""Users, and an owner for every saved portfolio (Step 27)

Revision ID: 0003
Revises: 0002
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The email is stored already lower-cased by the application and is UNIQUE, so "A@x.com" and "a@x.com" are one account.
    # Only the password HASH is stored (Argon2id, with its own salt inside the string); the password itself never touches the database.
    op.execute("""
        CREATE TABLE users (
            id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email          TEXT NOT NULL UNIQUE CHECK (email = lower(email) AND length(email) <= 254),
            password_hash  TEXT NOT NULL,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_login_at  TIMESTAMPTZ
        )""")
    # Nullable on purpose: portfolios saved before accounts existed (the three seeded samples) have no owner. Every query filters on the owner, so an
    # ownerless portfolio can be reached by nobody through the API. A deleted user takes their portfolios with them.
    op.execute("ALTER TABLE portfolios ADD COLUMN owner_id UUID REFERENCES users(id) ON DELETE CASCADE")
    op.execute("CREATE INDEX portfolios_owner_id ON portfolios (owner_id)")


def downgrade() -> None:
    op.execute("DROP INDEX portfolios_owner_id")
    op.execute("ALTER TABLE portfolios DROP COLUMN owner_id")
    op.execute("DROP TABLE users")
