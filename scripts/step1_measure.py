"""Step 1 measurement: 20 identical calls, then latency percentiles, tokens, cost.

Run:  .venv/bin/python -m scripts.step1_measure
"""
import json
from datetime import datetime
from pathlib import Path

from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import Recorder

setup_logging("WARNING")
N = 20
Q = "What is compound interest?"
TUTOR = "You are a friendly tutor. The learner is a beginner. Answer in under 120 words."

client = LLMClient()
rec = Recorder()
calls = []
print(f"{'#':>2} {'in':>4} {'out':>4} {'secs':>5} {'$':>9} tries")
for i in range(1, N + 1):
    r = client.chat(Q, system=TUTOR, temperature=0)
    rec.add("latency_s", r.latency_s)
    rec.add("input_tokens", r.input_tokens)
    rec.add("output_tokens", r.output_tokens)
    rec.add("cost_usd", r.cost_usd)
    calls.append({"n": i, "text": r.text, "in": r.input_tokens, "out": r.output_tokens,
                  "latency_s": r.latency_s, "cost_usd": r.cost_usd, "attempts": r.attempts})
    print(f"{i:>2} {r.input_tokens:>4} {r.output_tokens:>4} {r.latency_s:>5.2f} "
          f"{r.cost_usd:>9.5f} {r.attempts:>5}")

print("\n===== SUMMARY =====")
summary = rec.summary()
for name, s in summary.items():
    print(name, {k: round(v, 5) for k, v in s.items()})
print(f"cost per 1,000 queries: ${summary['cost_usd']['mean'] * 1000:.2f}")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = Path(f"benchmarks/results/step1_measure_{stamp}.json")
out.write_text(json.dumps({"summary": summary, "calls": calls}, indent=2))
print(f"saved: {out}")

print("\n===== FULL RAW OUTPUT =====")
for c in calls:
    print(f"\n--- call {c['n']} ---\n{c['text']}")
