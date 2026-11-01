"""Step 6 on PostgreSQL + pgvector (run in Step 23b): index use, the filtered-search gotcha, approximate-index recall, latency, sizes.

Everything runs against the real `chunks` table. The one destructive-looking experiment (searching WITHOUT the HNSW index) is done inside a
transaction that is rolled back, so the index is never actually dropped.

Run:  .venv/bin/python -m scripts.step6_pgvector_check
Raw output: benchmarks/results/step6_pgvector_check_<timestamp>.json
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.core.config import load_settings
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.data.db import get_pool
from src.rag.embedder import OpenAIEmbedder

settings = load_settings()
setup_logging("WARNING")
pool = get_pool(settings)

queries = []
for name in ("retrieval.jsonl", "retrieval_holdout.jsonl", "retrieval_threshold.jsonl"):
    for line in Path("eval/datasets", name).read_text().splitlines():
        text = json.loads(line).get("query") or json.loads(line).get("question")
        if text and text not in queries and len(queries) < 50:
            queries.append(text)
vectors = [np.asarray(v, dtype=np.float32) for v in OpenAIEmbedder(settings).embed(queries).vectors]
SEARCH = "SELECT id, 1 - (embedding <=> %s) FROM chunks{where} ORDER BY embedding <=> %s LIMIT %s"
report = {}

with pool.connection() as connection:
    # ---- sizes
    report["sizes"] = {
        "chunks_rows": connection.execute("SELECT count(*) FROM chunks").fetchone()[0],
        "table_heap": connection.execute("SELECT pg_size_pretty(pg_relation_size('chunks'))").fetchone()[0],
        "toast_for_long_values": connection.execute("SELECT pg_size_pretty(pg_total_relation_size('chunks') - pg_relation_size('chunks') - pg_indexes_size('chunks'))").fetchone()[0],
        "indexes": {name: size for name, size in connection.execute(
            "SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)) FROM pg_stat_user_indexes WHERE relname = 'chunks' ORDER BY indexrelname")},
        "total": connection.execute("SELECT pg_size_pretty(pg_total_relation_size('chunks'))").fetchone()[0],
    }
    print("SIZES", json.dumps(report["sizes"], indent=1))

    # ---- what the planner chooses, and what EXPLAIN ANALYZE shows with / without the index
    def explain(label, setup=()):
        with connection.transaction(force_rollback=True):
            for statement in setup:
                connection.execute(statement)
            rows = connection.execute("EXPLAIN (ANALYZE, BUFFERS OFF, TIMING ON) " + SEARCH.format(where="") , (vectors[0], vectors[0], 5)).fetchall()
        text = re.sub(r"'\[[^\]]*\]'", "'[1536 numbers]'", "\n".join(r[0] for r in rows))   # the query vector literal is 15 KB of noise
        print(f"\nEXPLAIN ANALYZE [{label}]\n{text}")
        return text

    report["explain"] = {
        "planner_default": explain("planner default"),
        "hnsw_forced": explain("HNSW forced (enable_seqscan=off)", ["SET LOCAL enable_seqscan = off"]),
        "no_index_exact": explain("index dropped inside a rolled-back transaction", ["DROP INDEX chunks_embedding_hnsw"]),
    }

    # ---- latency p50/p95 (100 searches = 50 queries x 2): planner default vs HNSW forced vs exact scan
    def latency(setup=()):
        times = []
        with connection.transaction(force_rollback=True):
            for statement in setup:
                connection.execute(statement)
            for _ in range(2):
                for vector in vectors:
                    start = time.perf_counter()
                    connection.execute(SEARCH.format(where=""), (vector, vector, 10)).fetchall()
                    times.append((time.perf_counter() - start) * 1000)
        return summarize(times)

    report["latency_ms_top10"] = {"planner_default": latency(), "hnsw_forced": latency(["SET LOCAL enable_seqscan = off"]),
                                  "no_index_exact": latency(["DROP INDEX chunks_embedding_hnsw"])}
    print("\nLATENCY ms (top 10, 100 searches)", json.dumps({k: {"p50": round(v["p50"], 2), "p95": round(v["p95"], 2)} for k, v in report["latency_ms_top10"].items()}))

    # ---- recall of the approximate index against the exact scan, top 10, 50 queries
    def top_ids(vector, setup):
        with connection.transaction(force_rollback=True):
            for statement in setup:
                connection.execute(statement)
            return [r[0] for r in connection.execute(SEARCH.format(where=""), (vector, vector, 10)).fetchall()]

    exact = [top_ids(v, ["DROP INDEX chunks_embedding_hnsw"]) for v in vectors]
    for ef in (40, 10):   # 40 is pgvector's default hnsw.ef_search
        approx = [top_ids(v, ["SET LOCAL enable_seqscan = off", f"SET LOCAL hnsw.ef_search = {ef}"]) for v in vectors]
        recall = [len(set(a) & set(e)) / 10 for a, e in zip(approx, exact)]
        report[f"recall_at_10_ef_search_{ef}"] = {"mean": float(np.mean(recall)), "min": float(min(recall)), "queries_below_1": int(sum(r < 1 for r in recall))}
        print(f"RECALL@10 HNSW (ef_search={ef}) vs exact scan: mean {np.mean(recall):.3f}, min {min(recall):.2f}, queries below 1.0: {sum(r < 1 for r in recall)}/50")

    # ---- the filtered-search gotcha: the index finds its nearest candidates FIRST, the WHERE filter is applied AFTER
    categories = connection.execute("SELECT category, count(*) FROM chunks GROUP BY 1 ORDER BY 2").fetchall()
    report["category_sizes"] = {c: n for c, n in categories}
    print("\nCATEGORIES (chunks):", report["category_sizes"])
    rare = [c for c, n in categories[:4]]

    def filtered(category, setup):
        counts = []
        with connection.transaction(force_rollback=True):
            for statement in setup:
                connection.execute(statement)
            for vector in vectors:
                counts.append(len(connection.execute(SEARCH.format(where=" WHERE category = %s"), (vector, category, vector, 10)).fetchall()))
        return counts

    exact_available = {c: n for c, n in categories}
    gotcha = {}
    for category in rare:
        wanted = min(10, exact_available[category])
        row = {"chunks_in_category": exact_available[category], "rows_wanted": wanted}
        # The category btree index is dropped (inside the rolled-back transaction) so the planner has no choice but to use HNSW for the filtered search;
        # otherwise it picks the btree index and does an exact sort, which never shows the gotcha (the first version of this script did exactly that).
        no_btree = "DROP INDEX chunks_category"
        for label, setup in (("planner_default", []),
                             ("hnsw_iterative_off", [no_btree, "SET LOCAL enable_seqscan = off", "SET LOCAL hnsw.iterative_scan = off"]),   # what pgvector does without the setting
                             ("hnsw_iterative_relaxed", [no_btree, "SET LOCAL enable_seqscan = off", "SET LOCAL hnsw.iterative_scan = relaxed_order"]),
                             ("hnsw_iterative_strict", [no_btree, "SET LOCAL enable_seqscan = off"]),   # what the pool sets for every connection (src/data/db.py)
                             ("exact_scan", ["DROP INDEX chunks_embedding_hnsw"])):
            counts = filtered(category, setup)
            row[label] = {"mean_rows_returned": float(np.mean(counts)), "queries_returning_fewer_than_wanted": int(sum(c < wanted for c in counts))}
        gotcha[category] = row
        print(f"FILTER category={category!r} ({row['chunks_in_category']} chunks, want {wanted}): " +
              "; ".join(f"{k}: mean {row[k]['mean_rows_returned']:.1f} rows, short on {row[k]['queries_returning_fewer_than_wanted']}/50" for k in ("planner_default", "hnsw_iterative_off", "hnsw_iterative_relaxed", "hnsw_iterative_strict", "exact_scan")))
    report["filtered_search_gotcha"] = gotcha
    with connection.transaction(force_rollback=True):   # prove which index the "forced" variant really uses
        connection.execute("DROP INDEX chunks_category")
        connection.execute("SET LOCAL enable_seqscan = off")
        plan = connection.execute("EXPLAIN " + SEARCH.format(where=" WHERE category = %s"), (vectors[0], "stocks", vectors[0], 10)).fetchall()
    report["filtered_plan_without_btree"] = re.sub(r"'\[[^\]]*\]'", "'[1536 numbers]'", "\n".join(r[0] for r in plan))
    print("\nPLAN for the filtered search with the category index removed:\n" + report["filtered_plan_without_btree"])

out = Path(f"benchmarks/results/step6_pgvector_check_{datetime.now():%Y%m%d_%H%M%S}.json")
out.write_text(json.dumps(report, indent=2))
print(f"\nsaved: {out}")
