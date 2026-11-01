"""PostgreSQL connection pool (one per process) shared by the vector store, the keyword search and the portfolio repository.

Why a pool: opening a connection costs a TCP handshake, authentication and a server process (milliseconds, and the server only allows
so many at once). A pool opens a few once and lends them out. `pool_max_size` caps how many requests can talk to the database at the
same time; when all are busy, a request waits up to `pool_timeout_seconds` and then fails with PoolTimeout instead of hanging forever.
"""
import threading

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from src.core.config import Settings, load_settings

_pools: dict[str, ConnectionPool] = {}
_lock = threading.Lock()


def _configure(connection) -> None:
    register_vector(connection)   # lets us pass numpy arrays as vector values and get them back
    # The filtered-search gotcha (Step 6): an HNSW scan finds its nearest candidates FIRST and applies WHERE afterwards, so a tight filter can return
    # fewer than k rows (measured: about 1 row instead of 5-7). Iterative scan keeps scanning until enough rows pass the filter; strict_order keeps the
    # results exactly ordered. Needs pgvector 0.8+ (the image we run has 0.8.7).
    connection.execute("SET hnsw.iterative_scan = strict_order")
    connection.commit()


def make_pool(url: str, settings: Settings | None = None, **overrides) -> ConnectionPool:
    """A new pool. Tests and benchmarks use this directly (own sizes, own database); the app uses get_pool()."""
    config = (settings or load_settings()).database
    options = dict(min_size=config.pool_min_size, max_size=config.pool_max_size, timeout=config.pool_timeout_seconds)
    options.update(overrides)
    pool = ConnectionPool(url, configure=_configure, open=False, **options)
    pool.open(wait=True, timeout=options["timeout"])
    return pool


def get_pool(settings: Settings | None = None) -> ConnectionPool:
    """The shared pool for DATABASE_URL, created on first use."""
    settings = settings or load_settings()
    url = settings.secrets.require_database_url()
    with _lock:
        if url not in _pools:
            _pools[url] = make_pool(url, settings)
        return _pools[url]


def close_pools() -> None:
    with _lock:
        for pool in _pools.values():
            pool.close()
        _pools.clear()
