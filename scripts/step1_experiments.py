"""Step 1 experiments: how system prompt and temperature change the answer.

Run:  .venv/bin/python -m scripts.step1_experiments
"""
import json
from pathlib import Path

from src.core.llm import LLMClient
from src.core.logging import setup_logging

setup_logging("WARNING")  # keep output readable; the table below has the numbers
Q = "What is compound interest?"
TUTOR = "You are a friendly tutor. The learner is a beginner. Answer in under 120 words."

client = LLMClient()
runs = [
    ("a: no system prompt", dict(), 2),
    ("b: tutor system prompt", dict(system=TUTOR), 2),
    ("c: tutor, temperature=0", dict(system=TUTOR, temperature=0), 2),
    ("c: tutor, temperature=1", dict(system=TUTOR, temperature=1), 2),
]

results = {}
print(f"{'setting':28} {'run':>3} {'in':>4} {'out':>4} {'secs':>5} {'$':>9}")
for name, kw, n in runs:
    results[name] = []
    for i in range(n):
        r = client.chat(Q, **kw)
        results[name].append(r.text)
        print(f"{name:28} {i+1:>3} {r.input_tokens:>4} {r.output_tokens:>4} "
              f"{r.latency_s:>5.2f} {r.cost_usd:>9.5f}")

print()
for name in ("c: tutor, temperature=0", "c: tutor, temperature=1"):
    a, b = results[name]
    print(f"{name}: two runs identical? {a == b}")

out = Path("benchmarks/results/step1_experiments.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nfull texts saved to {out}")
