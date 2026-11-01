"""Shared fixtures. Database tests use a THROWAWAY database (`finnie_test`, created and dropped here) on the same PostgreSQL server as the
app, migrated with the real Alembic files, so they never touch real data. If the server is not running they are skipped, not failed."""
import os
from pathlib import Path

import pytest

from src.core.config import PROJECT_ROOT, Secrets

TEST_DB = "finnie_test"


def _urls():
    url = Secrets().database_url
    if not url:
        return None, None
    base, _, _ = url.rpartition("/")
    return f"{base}/postgres", f"{base}/{TEST_DB}"


@pytest.fixture(scope="session")
def pg_pool():
    import psycopg
    from alembic import command
    from alembic.config import Config

    from src.data.db import make_pool
    admin_url, test_url = _urls()
    if admin_url is None:
        pytest.skip("DATABASE_URL is not set")
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=3) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
            admin.execute(f"CREATE DATABASE {TEST_DB}")
    except psycopg.OperationalError as error:
        pytest.skip(f"PostgreSQL is not reachable ({str(error).splitlines()[0]}): docker compose up -d")
    os.environ["ALEMBIC_DATABASE_URL"] = test_url
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    command.upgrade(config, "head")
    pool = make_pool(test_url, min_size=1, max_size=6)
    yield pool
    pool.close()
    os.environ.pop("ALEMBIC_DATABASE_URL", None)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")


@pytest.fixture
def pg(pg_pool):
    """The test database, emptied before each test."""
    with pg_pool.connection() as connection:
        connection.execute("TRUNCATE articles, portfolios, users, store_meta CASCADE")
    return pg_pool


@pytest.fixture
def two_users(pg):
    """Alice and Bob exist as real rows (portfolios.owner_id is a foreign key), with the same ids tests/auth_support.py puts in their tokens."""
    from tests.auth_support import ALICE, BOB
    with pg.connection() as connection:
        for user in (ALICE, BOB):
            connection.execute("INSERT INTO users (id, email, password_hash) VALUES (%s, %s, 'not-a-real-hash')", (user.id, user.email))
    return ALICE, BOB
