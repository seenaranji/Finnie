"""Step 15b measurements on PostgreSQL: lookup latency vs table size with and without the index, write latency, concurrency, pool exhaustion.

Runs in a SCRATCH database (`finnie_bench`, created and dropped here, migrated with the real Alembic files), so your saved portfolios are untouched.

Run:  .venv/bin/python -m scripts.step15b_benchmark
Raw output: benchmarks/results/step15b_benchmark_<timestamp>.json
"""
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import psycopg
from alembic import command
from alembic.config import Config
from psycopg_pool import PoolTimeout

from src.core.config import PROJECT_ROOT, Secrets, load_settings
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.data.db import make_pool
from src.data.portfolio_repository import PortfolioRepository

setup_logging("WARNING")
settings = load_settings()
base = Secrets().require_database_url().rpartition("/")[0]
admin_url, bench_url = f"{base}/postgres", f"{base}/finnie_bench"
report = {}

with psycopg.connect(admin_url, autocommit=True) as admin:
    admin.execute("DROP DATABASE IF EXISTS finnie_bench WITH (FORCE)")
    admin.execute("CREATE DATABASE finnie_bench")
try:
    os.environ["ALEMBIC_DATABASE_URL"] = bench_url
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    command.upgrade(config, "head")
    pool = make_pool(bench_url, settings, max_size=10)
    repo = PortfolioRepository(pool)
    TICKERS = [f"T{i}" for i in range(600)]

    def grow_to(total: int) -> list:
        """Add portfolios (5 holdings each) in bulk until the table has `total`; returns every portfolio id."""
        with pool.connection() as connection, connection.transaction():
            have = connection.execute("SELECT count(*) FROM portfolios").fetchone()[0]
            with connection.cursor() as cursor:
                for n in range(have, total):
                    pid = connection.execute("INSERT INTO portfolios (name) VALUES (%s) RETURNING id", (f"bench {n}",)).fetchone()[0]
                    cursor.executemany("INSERT INTO holdings (portfolio_id, ticker, shares, avg_cost) VALUES (%s, %s, %s, %s)",
                                       [(pid, t, 10, 100) for t in random.sample(TICKERS, 5)])
            connection.execute("ANALYZE holdings")   # fresh statistics, so the planner chooses fairly
            return [r[0] for r in connection.execute("SELECT id FROM portfolios").fetchall()]

    def lookup_ms(ids, n=300):
        times = []
        for pid in random.choices(ids, k=n):
            start = time.perf_counter()
            repo.get(pid)
            times.append((time.perf_counter() - start) * 1000)
        return summarize(times)

    def plan(pid) -> str:
        with pool.connection() as connection:
            rows = connection.execute("EXPLAIN (ANALYZE) SELECT ticker, shares, avg_cost FROM holdings WHERE portfolio_id = %s ORDER BY id", (pid,)).fetchall()
        return "\n".join(r[0] for r in rows)

    # ---- 1. lookup latency: 10 / 1,000 / 10,000 portfolios, with and without the index on holdings.portfolio_id
    # The index is the one UNIQUE (portfolio_id, ticker) creates (its first column is portfolio_id). "Without" drops that constraint.
    random.seed(7)
    report["lookup"] = {}
    for total in (10, 1000, 10000):
        ids = grow_to(total)
        row = {"holdings_rows": total * 5}
        row["with_index"] = lookup_ms(ids)
        row["plan_with_index"] = plan(ids[0])
        with pool.connection() as connection:
            connection.execute("ALTER TABLE holdings DROP CONSTRAINT holdings_portfolio_id_ticker_key")
        row["without_index"] = lookup_ms(ids)
        row["plan_without_index"] = plan(ids[0])
        with pool.connection() as connection:
            connection.execute("ALTER TABLE holdings ADD CONSTRAINT holdings_portfolio_id_ticker_key UNIQUE (portfolio_id, ticker)")
            connection.execute("ANALYZE holdings")
        report["lookup"][total] = row
        print(f"get(id) with {total:>6} portfolios ({total * 5:>6} holdings rows): with index p50 {row['with_index']['p50']:.2f} ms p95 {row['with_index']['p95']:.2f} ms | "
              f"without p50 {row['without_index']['p50']:.2f} ms p95 {row['without_index']['p95']:.2f} ms")
    for total in (10, 10000):
        for label in ("plan_with_index", "plan_without_index"):
            print(f"\nEXPLAIN ANALYZE [{label}, {total} portfolios]\n{report['lookup'][total][label]}")

    # ---- 2. write latency: create() with 10 vs 500 holdings (each includes validation, one transaction, N inserts)
    report["create_ms"] = {}
    for size in (10, 500):
        holdings = [{"ticker": t, "shares": "10", "avg_cost": "100.25"} for t in TICKERS[:size]]
        times = []
        for _ in range(40):
            start = time.perf_counter()
            repo.create("write bench", holdings)
            times.append((time.perf_counter() - start) * 1000)
        report["create_ms"][size] = summarize(times)
        print(f"\ncreate() with {size:>3} holdings: p50 {report['create_ms'][size]['p50']:.1f} ms, p95 {report['create_ms'][size]['p95']:.1f} ms")

    # ---- 3. concurrency: 50 simultaneous requests (45 reads + 5 writes) through a pool of 10
    with pool.connection() as connection:
        ids = [r[0] for r in connection.execute("SELECT id FROM portfolios LIMIT 200").fetchall()]
    errors = []

    def request(n):
        start = time.perf_counter()
        try:
            if n % 10 == 0:
                repo.create(f"concurrent {n}", [{"ticker": "AAA", "shares": "1", "avg_cost": "1"}])
            else:
                assert len(repo.get(random.choice(ids)).holdings) == 5
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        return (time.perf_counter() - start) * 1000

    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=50) as executor:
        times = list(executor.map(request, range(50)))
    report["concurrency_50"] = {"errors": errors, "wall_ms": (time.perf_counter() - wall) * 1000, "request_ms": summarize(times), "pool_size": 10}
    print(f"\n50 simultaneous requests, pool of 10: {len(errors)} errors, wall {report['concurrency_50']['wall_ms']:.0f} ms, "
          f"request p50 {report['concurrency_50']['request_ms']['p50']:.0f} ms, max {report['concurrency_50']['request_ms']['max']:.0f} ms")

    # ---- 4. pool exhaustion: 2 connections, 8 requests that each hold one for 1 s, others wait at most 1.5 s
    small = make_pool(bench_url, settings, min_size=1, max_size=2, timeout=1.5)

    def hold(_):
        try:
            with small.connection() as connection:
                connection.execute("SELECT pg_sleep(1)")
            return "ok"
        except PoolTimeout:
            return "PoolTimeout"

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(hold, range(8)))
    report["pool_exhaustion"] = {"pool_size": 2, "wait_timeout_s": 1.5, "requests": 8, "outcomes": {k: outcomes.count(k) for k in set(outcomes)}, "wall_s": time.perf_counter() - start}
    print(f"pool of 2, 8 requests holding a connection 1 s each, wait limit 1.5 s: {report['pool_exhaustion']['outcomes']} in {report['pool_exhaustion']['wall_s']:.1f} s")
    small.close()
    pool.close()
finally:
    os.environ.pop("ALEMBIC_DATABASE_URL", None)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute("DROP DATABASE IF EXISTS finnie_bench WITH (FORCE)")

out = Path(f"benchmarks/results/step15b_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json")
out.write_text(json.dumps(report, indent=2, default=str))
print(f"\nsaved: {out}")
