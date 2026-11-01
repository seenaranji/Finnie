"""Step 25 experiment: does the length of the router's system prompt change its latency? Same message, same model, same output schema; prompts of different
length (the real v6 prompt, its first half, its first quarter, and a one-line prompt) called in interleaved order so drift in the API's speed hits all equally.
This measures LATENCY ONLY: a shorter prompt would route worse, so any real change would need the router evaluation (make eval-router) first.

  python -m benchmarks.exp_router_prompt [--n 20]        costs about $0.01
"""
import argparse
import random

from benchmarks.common import save, summarize
from src.core.llm import LLMClient
from src.workflow.router import LABELS, PROMPTS, response_format_for

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=20)
n = parser.parse_args().n
full = PROMPTS["v6"]
variants = {"full v6": full, "first half": full[: len(full) // 2], "first quarter": full[: len(full) // 4], "one line": "Classify the message into one route and give your confidence."}
llm, fmt = LLMClient(), response_format_for(LABELS["v6"])
messages = ["What is an ETF?", "How is Apple doing today?", "Is my portfolio diversified?", "I want to save $20000 in 4 years", "What is the latest news about Tesla?"]
rows = {name: [] for name in variants}
tokens = {}
schedule = [name for _ in range(n) for name in variants]
random.seed(3)
random.shuffle(schedule)
for i, name in enumerate(schedule):
    result = llm.chat(f"MESSAGE: {messages[i % len(messages)]}", system=variants[name], model_key="router_model", temperature=0, max_output_tokens=30, response_format=fmt)
    rows[name].append(result.latency_s)
    tokens[name] = result.input_tokens
print(f"{'prompt':14} {'tokens in':>9} {'n':>3} {'p50 s':>7} {'p95 s':>7} {'mean s':>7}")
for name, values in rows.items():
    s = summarize(values)
    print(f"{name:14} {tokens[name]:9d} {s['n']:3d} {s['p50']:7.3f} {s['p95']:7.3f} {s['mean']:7.3f}")
print("raw:", save("step25_exp_router_prompt", {"tokens": tokens, "latency_s": rows}))
