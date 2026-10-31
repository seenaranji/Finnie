"""Step 20: the small graph (router -> finance_qa | market): its diagram, five queries with state shown per node, and router cost/latency.

Run:  .venv/bin/python -m scripts.step20_graph_demo
"""
import json
from datetime import datetime
from pathlib import Path

from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.market_agent import MarketAnalysisAgent
from src.agents.market_tools import MarketTools
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.data.factory import build_market_data_service
from src.rag.hybrid_retriever import HybridRetriever
from src.workflow.graph import build_graph
from src.workflow.router import Router

setup_logging("ERROR")
llm = LLMClient()
router = Router(llm)
graph = build_graph(router, FinanceQAAgent(llm, HybridRetriever()), MarketAnalysisAgent(llm, MarketTools(build_market_data_service())))
report: dict = {}


def short(value, limit=110):
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 1] + "…"


print("=" * 100, "\nTHE GRAPH AS A MERMAID DIAGRAM (paste into any Mermaid viewer)\n" + "=" * 100)
print(graph.get_graph().draw_mermaid())

QUERIES = ["What is the difference between a Roth IRA and a traditional IRA?",
           "How is Apple doing today?",
           "What does an expense ratio mean for VOO?",
           "Is Microsoft above its 50-day moving average?",
           "Write me a poem about the ocean"]
runs = []
# One stream with BOTH modes, so each agent runs only once: chunks arrive as (mode, payload).
for number, query in enumerate(QUERIES, start=1):
    print("\n" + "=" * 100, f"\nQUERY {number}: {query}\n" + "=" * 100)
    initial = {"message": query, "session_id": "demo", "trace": []}
    print(f"initial state keys: {sorted(initial)}\n")
    final = None
    for mode, payload in graph.stream(initial, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node, update in payload.items():
                print(f"  NODE '{node}' returned an update with keys {sorted(update)}:")
                for key, value in update.items():
                    print(f"      {key:9} = {short(value)}")
        else:
            final = payload
            print(f"  state now has keys: {sorted(payload)}   (trace length {len(payload.get('trace', []))})\n")
    print(f"ROUTE CHOSEN: {final['route']}   router decision: {{'tokens': {final['router']['input_tokens']}+{final['router']['output_tokens']}, 'cost_usd': {final['router']['cost_usd']}, 'latency_s': {final['router']['latency_s']:.2f}}}")
    print(f"ANSWER ({final['response']['agent_name']}):\n{final['response']['text'][:500]}")
    runs.append({"query": query, "route": final["route"], "trace": final["trace"], "answer": final["response"]["text"]})
report["runs"] = runs

print("\n" + "=" * 100, "\nROUTER COST AND LATENCY (20 classifications, one run each; this is a sanity check, not the Step 21 evaluation)\n" + "=" * 100)
SANITY = [("What is an ETF?", "finance_qa"), ("How is VOO doing today?", "market"), ("How do bonds work?", "finance_qa"), ("What is Tesla's stock price?", "market"),
          ("Explain compound interest", "finance_qa"), ("Is NVDA above its 200-day moving average?", "market"), ("What is a Roth IRA?", "finance_qa"),
          ("How volatile has Apple been this year?", "market"), ("What does diversification mean?", "finance_qa"), ("Compare Microsoft and Apple today", "market"),
          ("How are dividends taxed?", "finance_qa"), ("What is SPY trading at?", "market"), ("What is an expense ratio?", "finance_qa"),
          ("What was the 52-week high for AMZN?", "market"), ("What is the difference between a stock and a bond?", "finance_qa"),
          ("How much is Google worth right now?", "market"), ("What is the 401(k) contribution limit?", "finance_qa"), ("Did the S&P 500 go up today?", "market"),
          ("What is inflation?", "finance_qa"), ("How did bonds do this week?", "market")]
decisions = [(query, label, router.classify(query)) for query, label in SANITY]
for query, label, d in decisions:
    print(f"  {'ok  ' if d.route == label else 'DIFF'} {query:52} -> {d.route:10} (my label: {label:10}) {d.latency_s:5.2f}s  {d.input_tokens}+{d.output_tokens} tokens  ${d.cost_usd:.6f}")
latencies = [d.latency_s for _, _, d in decisions]
print(f"\n  agrees with my label: {sum(d.route == label for _, label, d in decisions)}/{len(decisions)}   fell back to the default: {sum(d.fell_back for _, _, d in decisions)}")
print(f"  latency: {summarize(latencies)}")
print(f"  input tokens per call: {summarize([float(d.input_tokens) for _, _, d in decisions])['mean']:.0f}   output tokens per call: {summarize([float(d.output_tokens) for _, _, d in decisions])['mean']:.1f}")
print(f"  cost per classification: ${sum(d.cost_usd for _, _, d in decisions) / len(decisions):.6f}   per 1,000 messages: ${1000 * sum(d.cost_usd for _, _, d in decisions) / len(decisions):.3f}   model: {decisions[0][2].model}")
report["router_sanity"] = [{"query": q, "label": l, "route": d.route, "latency_s": d.latency_s, "cost_usd": d.cost_usd} for q, l, d in decisions]

output_file = Path(f"benchmarks/results/step20_graph_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(report, indent=2, default=str))
print(f"\nsaved: {output_file}")
