"""Alembic environment: runs the versioned SQL files in migrations/versions against the database in DATABASE_URL.

We write the migrations as plain SQL (op.execute) on purpose: the SQL IS the lesson, and there is no model layer to keep in sync.
"""
import os

from alembic import context
from sqlalchemy import create_engine, pool

from src.core.config import Secrets


def database_url() -> str:
    url = os.environ.get("ALEMBIC_DATABASE_URL") or Secrets().require_database_url()   # the override lets the tests migrate a throwaway database
    # SQLAlchemy needs to be told which driver to use; ours is psycopg 3
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


def run_migrations_online() -> None:
    engine = create_engine(database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():   # PostgreSQL runs DDL inside transactions: a failed migration leaves nothing half-applied
            context.run_migrations()


run_migrations_online()
