"""Step 18: the Tax Education agent. Educational question vs personal-advice question, plus a question about a yearly limit.

Run:  .venv/bin/python -m scripts.step18_tax_demo
"""
import json
from datetime import datetime
from pathlib import Path

from src.agents.tax_agent import TaxEducationAgent
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.schemas import AgentRequest
from src.rag.hybrid_retriever import HybridRetriever

setup_logging("WARNING")
retriever = HybridRetriever()
agent = TaxEducationAgent(LLMClient(), retriever)
CASES = ["What is the difference between a Roth IRA and a traditional IRA?",
         "How should I file my taxes this year?",
         "What is the 401(k) contribution limit for 2026?",
         "Should I convert my traditional IRA to a Roth IRA?",
         "How are cryptocurrency gains taxed?"]
rows = []
for question in CASES:
    response = agent.handle(AgentRequest(message=question))
    meta = response.metadata
    print("=" * 100, f"\nQUESTION: {question}\n" + "=" * 100)
    print(response.text)
    print(f"\n  retrieved: {meta['retrieved_ids']}   best_similarity={meta['best_similarity']}")
    print(f"  cited: {meta['cited_ids']}   invalid: {meta['invalid_citations']}   sources: {[s.title for s in response.sources]}")
    print(f"  personal-request detected: {meta['advice_requested']}   output flagged: {meta['output_flagged']}   model: {meta['model']}   cost: ${meta['cost_usd']:.5f}")
    print(f"  disclaimer: {response.disclaimer[:90]}...")
    rows.append({"question": question, "text": response.text, "metadata": meta})

print("\n" + "=" * 100, "\nCATEGORY FILTER CHECK (real Chroma + BM25): which categories come back for the same question?\n" + "=" * 100)
q = "What is a Roth IRA?"
for label, category in (("no filter", None), ("taxes only", "taxes"), ("taxes + retirement_accounts", ["taxes", "retirement_accounts"])):
    for mode in ("dense", "hybrid"):
        result = retriever.retrieve(q, top_k=5, category=category, threshold=0.0, mode=mode, bm25_min_score=0.0)
        cats = sorted({p.chunk_id.split("#")[0] for p in result.passages})
        print(f"  {label:30} {mode:7} -> articles: {cats}")

output_file = Path(f"benchmarks/results/step18_tax_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(rows, indent=2, default=str))
print(f"\nsaved: {output_file}")
