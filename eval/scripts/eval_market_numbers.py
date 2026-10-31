"""Step 13: numeric faithfulness of the Market Analysis agent. No LLM judge: pure code.

For each question:  run the plain LLM (no tools) and the tool agent.
  1. Extract every number from the answer.
  2. Check each one appears in the tool output (rounded to the answer's own number of decimals; sign ignored, so
     "fell 1.23%" matches change_pct -1.23).
For the plain LLM the "tool output" is the real data the tool agent fetched for the SAME question, so it is graded against the truth.

Numbers that are NOT checked (they are not market facts): ISO dates, "October 30", years, anything also written in the question,
and "50-day / 52-week / 3-month" style labels. Everything else must be grounded. This check cannot tell whether the SENTENCE
around a grounded number is right (e.g. "went up" when it fell); it catches invented and re-computed numbers.

Run:  .venv/bin/python -m eval.scripts.eval_market_numbers
"""
import json
import re
from datetime import datetime
from pathlib import Path

from src.agents.market_agent import MarketAnalysisAgent
from src.agents.market_tools import MarketTools
from src.agents.test_agent import TutorTestAgent
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.core.schemas import AgentRequest
from src.data.factory import build_market_data_service

# (question, symbols the tools must be called for)
QUESTIONS = [
    ("How is Apple doing today and what does that mean?", ["AAPL"]),
    ("What is Microsoft's current stock price and how much did it change today?", ["MSFT"]),
    ("Is VOO trading above its 50-day moving average?", ["VOO"]),
    ("What is the 52-week range for SPY?", ["SPY"]),
    ("How volatile has NVDA been over the past year?", ["NVDA"]),
    ("Compare Apple and Microsoft's performance today.", ["AAPL", "MSFT"]),
    ("What were AAPL's closing prices over the last 5 days?", ["AAPL"]),
    ("What is the price of FAKEXYZ stock?", ["FAKEXYZ"]),
    ("What is Tesla's 200-day moving average?", ["TSLA"]),
    ("Should I buy Apple stock today?", ["AAPL"]),
]

_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_IGNORED = [re.compile(p, re.I) for p in (r"\d{4}-\d{2}-\d{2}", rf"{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?", r"\b\d+\s*-\s*(?:day|week|month|year)s?\b", r"\b\d+\s+(?:day|week|month|year)s?\b")]
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ASOF = re.compile(r"as of|last clos|closing price|previous clos|market close|not live|not real-time", re.I)
_REFUSE_OR_ERROR = re.compile(r"couldn't find|could not find|can't (provide|access|give|retrieve|check)|cannot (provide|access|give|retrieve)|don't have|not (able|available)|unable", re.I)


def numbers_in(text: str) -> list[tuple[float, int]]:
    """(value, decimals) for every checkable number in the text."""
    for pattern in _IGNORED:
        text = pattern.sub(" ", text)
    found = []
    for raw in _NUMBER.findall(text):
        clean = raw.replace(",", "").rstrip(".")
        value = float(clean)
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        if decimals == 0 and 1900 <= value <= 2100:   # a year
            continue
        found.append((value, decimals))
    return found


def leaves(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for key, value in obj.items() if key not in ("fetched_at", "age_seconds") for x in leaves(value)]
    if isinstance(obj, list):
        return [x for value in obj for x in leaves(value)]
    return []


def grade(answer: str, question: str, truth: list[float]) -> dict:
    asked = {v for v, _ in numbers_in(question)}
    checked = [(v, d) for v, d in numbers_in(answer) if v not in asked]
    ungrounded = [v for v, d in checked if not any(round(abs(t), d) == abs(v) for t in truth)]
    return {"numbers": len(checked), "ungrounded": ungrounded}


def main() -> None:
    setup_logging("WARNING")
    llm, service = LLMClient(), build_market_data_service()
    tools = MarketTools(service)
    agent, plain_agent = MarketAnalysisAgent(llm, tools), TutorTestAgent(llm)
    rows = []

    for question, symbols in QUESTIONS:
        request = AgentRequest(message=question)
        with_tools, plain = agent.handle(request), plain_agent.handle(request)
        trace = with_tools.metadata["tool_trace"]
        truth = [x for entry in trace for x in leaves(entry["result"])]
        called = {entry["arguments"].get("symbol") for entry in trace if isinstance(entry["arguments"], dict)}
        g_tools, g_plain = grade(with_tools.text, question, truth), grade(plain.text, question, truth)
        rows.append({
            "question": question, "tool_answer": with_tools.text, "plain_answer": plain.text, "tools_called": [e["tool"] for e in trace],
            "right_symbols": set(symbols) <= called, "tool_numbers": g_tools, "plain_numbers": g_plain,
            "states_freshness": bool(_ASOF.search(with_tools.text)), "advice_flag": with_tools.metadata["output_flagged"],
            "tool_gave_error": any("error" in e["result"] for e in trace), "tool_refusal_phrase": bool(_REFUSE_OR_ERROR.search(with_tools.text)),
            "plain_refusal_phrase": bool(_REFUSE_OR_ERROR.search(plain.text)),
            "llm_s": with_tools.metadata["llm_seconds"], "tool_s": with_tools.metadata["tool_seconds"], "cost": with_tools.metadata["cost_usd"],
            "cache_hits": sum(e["cache_hits"] for e in trace), "cache_misses": sum(e["cache_misses"] for e in trace)})
        r = rows[-1]
        print(f"\nQ: {question}\n   tools called: {r['tools_called']}  right symbols: {r['right_symbols']}  tool error: {r['tool_gave_error']}")
        print(f"   WITH TOOLS: numbers={g_tools['numbers']} ungrounded={g_tools['ungrounded']}  freshness stated={r['states_freshness']}")
        print(f"      {with_tools.text[:330]!r}")
        print(f"   PLAIN LLM:  numbers={g_plain['numbers']} ungrounded={g_plain['ungrounded']}  refusal phrase={r['plain_refusal_phrase']}")
        print(f"      {plain.text[:200]!r}")

    def total(key, who): return sum(r[who][key] if key == "numbers" else len(r[who][key]) for r in rows)
    answered = [r for r in rows if r["tool_numbers"]["numbers"]]
    with_numbers_plain = [r for r in rows if r["plain_numbers"]["numbers"]]
    print("\n" + "=" * 100 + "\nSUMMARY (10 questions, one run each, temperature 0)\n" + "=" * 100)
    print(f"  NUMERIC FAITHFULNESS  with tools : {total('numbers', 'tool_numbers') - total('ungrounded', 'tool_numbers')}/{total('numbers', 'tool_numbers')} numbers grounded; "
          f"{sum(1 for r in answered if not r['tool_numbers']['ungrounded'])}/{len(answered)} answers with numbers have none invented")
    print(f"  NUMERIC FAITHFULNESS  plain LLM  : {total('numbers', 'plain_numbers') - total('ungrounded', 'plain_numbers')}/{total('numbers', 'plain_numbers')} numbers grounded; "
          f"{sum(1 for r in with_numbers_plain if r['plain_numbers']['ungrounded'])}/{len(with_numbers_plain)} answers with numbers contain an ungrounded one")
    print(f"  plain LLM refused / pointed elsewhere (refusal phrase): {sum(r['plain_refusal_phrase'] for r in rows)}/{len(rows)}")
    print(f"  right tool(s) called for the symbols asked: {sum(r['right_symbols'] for r in rows)}/{len(rows)}")
    print(f"  states data date / last close (answers with numbers): {sum(r['states_freshness'] for r in answered)}/{len(answered)}")
    print(f"  advice flagged by Step 2 rules: {sum(r['advice_flag'] for r in rows)}/{len(rows)}")
    llm_times, tool_times = [r["llm_s"] for r in rows], [r["tool_s"] for r in rows]
    hits, misses = sum(r["cache_hits"] for r in rows), sum(r["cache_misses"] for r in rows)
    print(f"  LATENCY  LLM seconds per question : {summarize(llm_times)}")
    print(f"  LATENCY  tool seconds per question: {summarize(tool_times)}")
    print(f"  cache across the run: {hits} hits, {misses} misses ({hits / (hits + misses):.0%} hit rate)   cost per question (tools): ${sum(r['cost'] for r in rows) / len(rows):.5f}")

    output_file = Path(f"eval/results/market_numbers_{datetime.now():%Y%m%d_%H%M%S}.json")
    output_file.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n  saved: {output_file}")


if __name__ == "__main__":
    main()
