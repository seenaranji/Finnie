"""Alembic migrations: up, all the way down, and up again, on their own throwaway database."""
import os

import psycopg
import pytest
from alembic import command
from alembic.config import Config

from src.core.config import PROJECT_ROOT
from tests.conftest import _urls

NAME = "finnie_migtest"


def tables(url):
    with psycopg.connect(url) as connection:
        return {row[0] for row in connection.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}


def test_upgrade_downgrade_upgrade(monkeypatch):
    admin_url, test_url = _urls()
    if admin_url is None:
        pytest.skip("DATABASE_URL is not set")
    url = test_url.rsplit("/", 1)[0] + "/" + NAME
    try:
        admin = psycopg.connect(admin_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL is not reachable")
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {NAME} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {NAME}")
        try:
            monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)
            config = Config(str(PROJECT_ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
            command.upgrade(config, "head")
            assert {"articles", "chunks", "store_meta", "portfolios", "holdings"} <= tables(url)
            command.downgrade(config, "0001")   # one step: portfolio tables go, knowledge-base tables stay
            assert {"portfolios", "holdings"}.isdisjoint(tables(url)) and "chunks" in tables(url)
            command.downgrade(config, "base")
            assert tables(url) <= {"alembic_version"}
            command.upgrade(config, "head")
            assert "holdings" in tables(url)
        finally:
            admin.execute(f"DROP DATABASE IF EXISTS {NAME} WITH (FORCE)")
