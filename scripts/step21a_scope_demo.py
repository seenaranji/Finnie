"""Step 21a: the out-of-scope gate in action, with COUNTERS proving what each question did and did not touch.

Part 1  the spec's seven questions through the real graph: route chosen + how many model / retrieval / market-API calls each caused
Part 2  the same 20 off-topic questions three ways:
          A  no gate: the router sends them on to the Finance Q&A agent (retrieval, and an answer if the similarity gate lets one through)
          B  the old behaviour (prompt v2, no out_of_scope label): they land on the `clarify` reply
          C  the scope gate (prompt v5): fixed reply from the router alone

Run:  .venv/bin/python -m scripts.step21a_scope_demo
"""
import json
import time
from datetime import datetime
from pathlib import Path

from src.core.config import load_settings
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.data.cache import TTLCache
from src.data.market_data import MarketDataService
from src.data.providers.base import MarketDataProvider
from src.data.providers.yfinance_provider import YFinanceProvider
from src.rag.hybrid_retriever import HybridRetriever
from src.workflow.factory import build_default_graph, config_for
from src.workflow.router import Router


class CountingLLM:
    def __init__(self, real):
        self._real, self.calls = real, []

    def _note(self, kwargs, result, kind):
        purpose = (kwargs.get("response_format") or {}).get("json_schema", {}).get("name") or ("tool-turn" if kind == "complete" else "answer")
        self.calls.append({"purpose": purpose, "tokens": result.input_tokens + result.output_tokens, "cost": result.cost_usd})
        return result

    def chat(self, *args, **kwargs):
        return self._note(kwargs, self._real.chat(*args, **kwargs), "chat")

    def complete(self, *args, **kwargs):
        return self._note(kwargs, self._real.complete(*args, **kwargs), "complete")

    def __getattr__(self, name):
        return getattr(self._real, name)


class CountingRetriever:
    def __init__(self, real):
        self._real, self.calls = real, 0

    def retrieve(self, *args, **kwargs):
        self.calls += 1
        return self._real.retrieve(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class CountingProvider(MarketDataProvider):
    name = "counting"

    def __init__(self, real):
        self.real, self.calls = real, 0

    def get_quote(self, symbol):
        self.calls += 1
        return self.real.get_quote(symbol)

    def get_history(self, symbol, period):
        self.calls += 1
        return self.real.get_history(symbol, period)


class SendEverythingToQA(Router):
    """Simulates a world WITHOUT the scope gate: the router call is paid for, then the message goes to the Q&A agent whatever it was."""
    def classify(self, message):
        decision = super().classify(message)
        decision.route = "finance_qa"
        return decision


setup_logging("ERROR")
settings, real = load_settings(), LLMClient()
llm, retriever, provider = CountingLLM(real), CountingRetriever(HybridRetriever(settings=settings)), CountingProvider(YFinanceProvider(10))
market = MarketDataService(provider, TTLCache(), settings)
report = {}


def reset():
    llm.calls.clear(); retriever.calls = 0; provider.calls = 0


def used():
    answering = [c for c in llm.calls if c["purpose"] not in ("route", "standalone")]
    return {"router_calls": sum(c["purpose"] == "route" for c in llm.calls), "answering_model_calls": len(answering), "retrieval_calls": retriever.calls,
            "market_api_calls": provider.calls, "tokens": sum(c["tokens"] for c in llm.calls), "cost": sum(c["cost"] for c in llm.calls)}


print("=" * 100 + "\nPART 1: the spec's questions through the real graph (router prompt v5), with counters\n" + "=" * 100)
graph = build_default_graph(llm=llm, retriever=retriever, market_service=market)
CASES = [("clearly off-topic", "what is the price of a pizza?"), ("clearly off-topic", "who won the game last night?"), ("clearly off-topic", "write me a poem"),
         ("borderline, in scope", "is inflation making groceries expensive?"), ("borderline, in scope", "how is Domino's stock doing?"),
         ("borderline, in scope", "should I budget for eating out?"), ("plain finance", "what is an ETF?")]
log_path = Path(settings.workflow.scope_log_path)
log_before = log_path.read_text().count("\n") if log_path.exists() else 0
print(f"{'kind':22}{'question':42}{'route':13}{'conf':8}{'router':>7}{'answer-LLM':>11}{'retrieval':>10}{'market API':>11}")
part1 = []
for number, (kind, question) in enumerate(CASES):
    reset()
    state = graph.invoke({"message": question, "session_id": f"scope-demo-{number}"}, config_for(f"scope-demo-{number}"))
    u = used()
    part1.append({"kind": kind, "question": question, "route": state["route"], **u, "answer": state["response"]["text"]})
    print(f"{kind:22}{question:42}{state['route']:13}{state['router'].get('confidence', ''):8}{u['router_calls']:>7}{u['answering_model_calls']:>11}{u['retrieval_calls']:>10}{u['market_api_calls']:>11}")
print("\nWhat the user saw for the three off-topic questions:")
for row in part1[:3]:
    print(f"  {row['question']!r} -> {row['answer']!r}")
off = [r for r in part1 if r["kind"] == "clearly off-topic"]
print(f"\nCHECK: off-topic questions that touched retrieval / the market API / an answering model: "
      f"{sum(r['retrieval_calls'] > 0 for r in off)} / {sum(r['market_api_calls'] > 0 for r in off)} / {sum(r['answering_model_calls'] > 0 for r in off)}   (must all be 0)")
log_after = log_path.read_text().splitlines()[log_before:] if log_path.exists() else []
print(f"\nDECISION LOG ({log_path}): {len(log_after)} new entries")
for line in log_after:
    entry = json.loads(line); print(f"  {entry['at']}  confidence={entry['confidence']}  query={entry['query']!r}")
report["part1"] = part1

print("\n" + "=" * 100 + "\nPART 2: the same 20 off-topic questions, three ways (end-to-end through the graph)\n" + "=" * 100)
queries = [json.loads(line)["query"] for line in Path("eval/datasets/scope.jsonl").read_text().splitlines() if line and json.loads(line)["kind"] == "off_topic" and json.loads(line)["split"] in ("dev", "test")]
configs = {"A  no gate (router -> Finance Q&A)": build_default_graph(llm=llm, retriever=retriever, market_service=market, router=SendEverythingToQA(llm)),
           "B  old: prompt v2 -> clarify": build_default_graph(llm=llm, retriever=retriever, market_service=market, router_prompt="v2"),
           "C  scope gate: prompt v5": graph}
rows = {}
for name, g in configs.items():
    seconds, agg, routes, replies = [], {"router_calls": 0, "answering_model_calls": 0, "retrieval_calls": 0, "market_api_calls": 0, "tokens": 0, "cost": 0.0}, [], []
    answered_by_llm = 0
    for i, q in enumerate(queries):
        reset()
        start = time.perf_counter()
        state = g.invoke({"message": q, "session_id": f"p2-{name[0]}-{i}"}, config_for(f"p2-{name[0]}-{i}"))
        seconds.append(time.perf_counter() - start)
        u = used()
        for k in agg: agg[k] += u[k]
        answered_by_llm += u["answering_model_calls"] > 0
        routes.append(state["route"]); replies.append(state["response"]["text"])
    rows[name] = {**agg, "seconds": seconds, "routes": routes, "answered_by_llm": answered_by_llm, "distinct_replies": len(set(replies))}
print(f"{'configuration':40}{'p50 s':>7}{'p95 s':>7}{'tokens/q':>9}{'cost/q':>11}{'retrieval':>10}{'LLM answers':>12}  routes")
for name, r in rows.items():
    s = summarize(r["seconds"])
    from collections import Counter
    print(f"{name:40}{s['p50']:>7.2f}{s['p95']:>7.2f}{r['tokens'] / len(queries):>9.0f}${r['cost'] / len(queries):>10.6f}{r['retrieval_calls']:>10}{r['answered_by_llm']:>12}  {dict(Counter(r['routes']))}")
a, c = rows["A  no gate (router -> Finance Q&A)"], rows["C  scope gate: prompt v5"]
print(f"\nGate vs no gate on {len(queries)} off-topic questions: retrieval calls {a['retrieval_calls']} -> {c['retrieval_calls']}; questions that reached an answering model {a['answered_by_llm']} -> {c['answered_by_llm']}; "
      f"median latency {summarize(a['seconds'])['p50']:.2f}s -> {summarize(c['seconds'])['p50']:.2f}s; cost per question ${a['cost'] / len(queries):.6f} -> ${c['cost'] / len(queries):.6f}")
report["part2"] = {k: {kk: vv for kk, vv in v.items()} for k, v in rows.items()}
out = Path(f"benchmarks/results/step21a_scope_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
out.write_text(json.dumps(report, indent=2, default=str))
print(f"\nsaved: {out}")
