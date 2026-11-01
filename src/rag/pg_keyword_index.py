"""Keyword search over the same `chunks` table, using PostgreSQL full-text search (Step 7c, moved to Postgres in Step 23b).

Same shape as the BM25 `KeywordIndex` (`search(query, top_k, where) -> [KeywordHit]`, `len()`, `fingerprint`), so the hybrid retriever does not
care which one it holds. There is nothing to build or rebuild: the `tsv` column is generated from `text` by the database, so it can never be stale.

Ranking is `ts_rank` with normalization flag 1 (divide by 1 + log(chunk length), the same idea as BM25's length adjustment). The spec suggests
`ts_rank_cd` (cover density); scripts/step23b_keyword_variants.py measured both plus 30 other setups against the golden dataset, and ts_rank + flag 1
had the best dev + expansion MRR. Either way it is similar in spirit to BM25 but NOT the same formula and NOT on the same scale: fine for RRF (ranks
only), but `rag.bm25_min_score` values tuned on bm25s (2.5, 3.3, 4.0) do not carry over; scores here are well below 1.

Matching: `websearch_to_tsquery` parses user text safely (quotes, "-word", "or") and stems with the `english` configuration, but joins plain words
with AND, so a long natural question would match nothing. With `database.keyword_match: any` we swap AND for OR, which is how BM25 behaves.
"""
import hashlib

from psycopg_pool import ConnectionPool

from src.core.config import Settings, load_settings
from src.data.db import get_pool
from src.rag.keyword_index import KeywordHit
from src.rag.pg_vector_store import where_sql

TS_CONFIG = "english"   # must equal the configuration of the generated tsv column (migration 0001)


class PgKeywordIndex:
    def __init__(self, pool: ConnectionPool, match: str = "any") -> None:
        if match not in ("any", "all"):
            raise ValueError(f"match must be 'any' or 'all' (got '{match}')")
        self.pool, self.match = pool, match

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "PgKeywordIndex":
        settings = settings or load_settings()
        return cls(get_pool(settings), settings.database.keyword_match)

    def __len__(self) -> int:
        with self.pool.connection() as connection:
            return connection.execute("SELECT count(*) FROM chunks").fetchone()[0]

    @property
    def fingerprint(self) -> str:
        with self.pool.connection() as connection:
            digest = hashlib.sha256()
            for chunk_id, text in sorted(connection.execute("SELECT id, text FROM chunks").fetchall()):   # Python order, same as KeywordIndex's fingerprint
                digest.update(chunk_id.encode() + b"\0" + text.encode() + b"\0")
        return digest.hexdigest()[:12]

    def search(self, query: str, top_k: int, where: dict | None = None) -> list[KeywordHit]:
        clause, params = where_sql(where)
        tsquery = "websearch_to_tsquery(%s, %s)"
        if self.match == "any":
            tsquery = f"replace({tsquery}::text, ' & ', ' | ')::tsquery"
        sql = f"""WITH q AS (SELECT {tsquery} AS query)
                  SELECT id, text, metadata, ts_rank(tsv, q.query, 1) AS score
                  FROM chunks, q WHERE tsv @@ q.query{clause}
                  ORDER BY score DESC, id LIMIT %s"""
        with self.pool.connection() as connection:
            rows = connection.execute(sql, [TS_CONFIG, query, *params, top_k]).fetchall()
        return [KeywordHit(chunk_id, text, metadata, float(score)) for chunk_id, text, metadata, score in rows]
