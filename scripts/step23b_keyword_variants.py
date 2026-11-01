"""Step 23b: which PostgreSQL full-text setup ranks best, compared with the BM25 index it replaces?

Keyword-only ranking (no embeddings, no cost) on the answerable questions of the golden dataset. Variants differ in: text-search configuration
('english' stems; 'simple' does not), rank function (ts_rank_cd cover density vs ts_rank), its normalization flag (0 none, 1 divide by 1+log(length),
2 divide by length, 32 scale to 0..1), and AND vs OR matching. Choose on dev + expansion; the test split only CONFIRMS the choice.

Run:  .venv/bin/python -m scripts.step23b_keyword_variants
Raw output: benchmarks/results/step23b_keyword_variants_<timestamp>.json
"""
import json
from datetime import datetime
from pathlib import Path

from src.core.config import load_settings
from src.core.logging import setup_logging
from src.data.db import get_pool
from src.rag.chunker import chunk_structured
from src.rag.keyword_index import KeywordIndex
from src.rag.loader import load_documents

settings = load_settings()
setup_logging("WARNING")
pool = get_pool(settings)
items = [json.loads(line) for line in Path("eval/datasets/retrieval.jsonl").read_text().splitlines() if line]
items = [i for i in items if i["answerable"]]
bm25 = KeywordIndex.build([c for d in load_documents() for c in chunk_structured(d, settings.rag.chunk_size)])   # the old keyword search, built in memory for comparison


def pg_search(query, config, rank_fn, flag, match, top=10):
    tsq = f"websearch_to_tsquery('{config}', %s)"
    if match == "any":
        tsq = f"replace({tsq}::text, ' & ', ' | ')::tsquery"
    # On-the-fly to_tsvector so any configuration can be tried (the stored tsv column is 'english'). Config names are our own constants, not user input.
    sql = f"""WITH q AS (SELECT {tsq} AS query)
              SELECT metadata ->> 'doc_id', {rank_fn}(to_tsvector('{config}', text), q.query, {flag}) AS score
              FROM chunks, q WHERE to_tsvector('{config}', text) @@ q.query ORDER BY score DESC, id LIMIT %s"""
    with pool.connection() as connection:
        return [row[0] for row in connection.execute(sql, (query, top)).fetchall()]


def metrics(ranker):
    out = {}
    for split in ("dev", "expansion", "test"):
        rows = [i for i in items if i["split"] == split]
        ranks = []
        for item in rows:
            docs = ranker(item["query"])
            ranks.append(next((r for r, d in enumerate(docs, 1) if d in item["expected_ids"]), None))
        out[split] = {"n": len(rows), "hit3": sum(1 for r in ranks if r and r <= 3) / len(rows), "mrr": sum(1 / r for r in ranks if r) / len(rows),
                      "no_hit_at_all": sum(1 for r in ranks if r is None)}
    return out


results = {"bm25s (old)": metrics(lambda q: [h.metadata["doc_id"] for h in bm25.search(q, 10)])}
for config in ("english", "simple"):
    for rank_fn in ("ts_rank_cd", "ts_rank"):
        for flag in (0, 1, 2, 32):
            for match in ("any", "all"):
                name = f"{config:7} {rank_fn:10} flag={flag:<2} {match}"
                results[name] = metrics(lambda q, c=config, r=rank_fn, f=flag, m=match: pg_search(q, c, r, f, m))

print(f"{'variant':36} | {'dev Hit@3':>9} {'MRR':>5} | {'exp Hit@3':>9} {'MRR':>5} | {'test Hit@3':>10} {'MRR':>5} | no-hit (dev/exp/test)")
for name, r in results.items():
    print(f"{name:36} | {r['dev']['hit3']:9.2f} {r['dev']['mrr']:5.2f} | {r['expansion']['hit3']:9.2f} {r['expansion']['mrr']:5.2f} | "
          f"{r['test']['hit3']:10.2f} {r['test']['mrr']:5.2f} | {r['dev']['no_hit_at_all']}/{r['expansion']['no_hit_at_all']}/{r['test']['no_hit_at_all']}")
out = Path(f"benchmarks/results/step23b_keyword_variants_{datetime.now():%Y%m%d_%H%M%S}.json")
out.write_text(json.dumps(results, indent=2))
print(f"saved: {out}")
