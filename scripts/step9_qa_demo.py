"""Step 9: Finance Q&A agent (RAG) vs the plain LLM, on four kinds of question.

  covered          -> answer should be grounded and cited
  partly covered   -> the article exists but lacks the specific fact
  outside (gate)   -> retrieval returns nothing, so the LLM is never called
  outside (no gate)-> gate switched off: retrieval hands over irrelevant text; does the PROMPT rule save us?

Run:  .venv/bin/python -m scripts.step9_qa_demo
"""
import json
from datetime import datetime
from pathlib import Path

from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.test_agent import TutorTestAgent
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.schemas import AgentRequest
from src.rag.hybrid_retriever import HybridRetriever

setup_logging("WARNING")
llm = LLMClient()
retriever = HybridRetriever()   # mode comes from config.yaml (dense); the agent does not care which retriever it gets

CASES = [
    ("covered", "What is the difference between an ETF and a mutual fund?", None),
    ("partly covered", "What is the 401(k) contribution limit this year?", None),
    ("outside (gate on)", "How does a 529 college savings plan work?", None),
    ("outside (gate OFF)", "How does a 529 college savings plan work?", 0.0),
]

rows = []
for label, question, threshold in CASES:
    rag_agent = FinanceQAAgent(llm, retriever, threshold=threshold)
    plain_agent = TutorTestAgent(llm)
    request = AgentRequest(message=question)
    rag = rag_agent.handle(request)
    plain = plain_agent.handle(request)

    print("=" * 100)
    print(f"[{label}] {question}")
    print("-" * 100)
    print(f"RAG ANSWER ({rag.metadata['model']}):\n{rag.text}\n")
    print(f"  retrieved: {rag.metadata['retrieved_ids']}  best_similarity={rag.metadata['best_similarity']}")
    print(f"  cited:     {rag.metadata['cited_ids']}  invalid citations: {rag.metadata['invalid_citations']}")
    print(f"  sources:   {[s.id for s in rag.sources]}")
    print(f"\nPLAIN LLM ANSWER ({plain.metadata['model']}):\n{plain.text}\n")
    rows.append({"case": label, "question": question,
                 "rag": {"text": rag.text, **{k: rag.metadata[k] for k in (
                     "model", "latency_s", "retrieval_latency_s", "input_tokens", "output_tokens", "cost_usd",
                     "retrieved_ids", "cited_ids", "invalid_citations", "best_similarity")}},
                 "plain": {"text": plain.text, **{k: plain.metadata[k] for k in (
                     "model", "latency_s", "input_tokens", "output_tokens", "cost_usd")}}})

print("=" * 100)
print("MEASUREMENTS (one run each; LLM latency is noisy, so read the pattern, not the decimals)")
print(f"{'case':20} | {'RAG in tok':>10} {'out tok':>8} {'LLM s':>6} {'retr s':>7} {'retr share':>10} {'cost $':>9} | "
      f"{'plain in':>8} {'out':>5} {'LLM s':>6} {'cost $':>9}")
for r in rows:
    g, p = r["rag"], r["plain"]
    total = g["latency_s"] + g["retrieval_latency_s"]
    print(f"{r['case']:20} | {g['input_tokens']:>10} {g['output_tokens']:>8} {g['latency_s']:>6.2f} {g['retrieval_latency_s']:>7.3f} "
          f"{g['retrieval_latency_s'] / total:>9.0%} {g['cost_usd']:>9.5f} | {p['input_tokens']:>8} {p['output_tokens']:>5} "
          f"{p['latency_s']:>6.2f} {p['cost_usd']:>9.5f}")

output_file = Path(f"benchmarks/results/step9_qa_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(rows, indent=2))
print(f"\nsaved: {output_file}")
