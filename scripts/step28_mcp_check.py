"""Step 28 live check: start the MCP server as a real subprocess (exactly what Claude Desktop does), call every tool over stdio, and compare latency with calling the same functions directly.

Run: .venv/bin/python -m scripts.step28_mcp_check [--runs 10]
Costs: search_knowledge makes one embedding call per run (about $0.00001); analyze_portfolio makes ONE model call (a fraction of a cent). Needs PostgreSQL up for search_knowledge.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from mcp import Client, StdioServerParameters

from src.mcp_server.tools import FinnieTools

CSV = "ticker,shares,avg_cost\nAAPL,10,150\nVTI,5,200\n"
CASES = {
    "get_quote": {"ticker": "AAPL"},
    "plan_goal": {"goal": 100000, "years": 10, "current_savings": 5000},
    "search_knowledge": {"query": "what is an expense ratio"},
}


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


async def main(runs: int) -> int:
    direct = FinnieTools()
    params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).resolve().parent / "run_mcp_server.py")], cwd="/")   # cwd="/" on purpose: a client does not start in the project folder
    rows, failures = [], 0
    async with Client(params) as client:
        print("tools:", [t.name for t in (await client.list_tools()).tools])
        for name, args in CASES.items():
            getattr(direct, name)(**args)   # warm-up for both paths (first call loads indexes / clients)
            await client.call_tool(name, args)
            mcp_t, dir_t, ok = [], [], 0
            for _ in range(runs):
                t = time.perf_counter(); r = await client.call_tool(name, args); mcp_t.append(time.perf_counter() - t)
                ok += not r.is_error
                t = time.perf_counter(); getattr(direct, name)(**args); dir_t.append(time.perf_counter() - t)
            failures += runs - ok
            rows.append((name, ok, runs, statistics.median(mcp_t), pct(mcp_t, .95), statistics.median(dir_t), pct(dir_t, .95)))
        r = await client.call_tool("analyze_portfolio", {"holdings_csv": CSV})
        print("\nanalyze_portfolio (1 run, uses a model):", "ERROR" if r.is_error else "ok", "|", r.content[0].text[:200].replace("\n", " "))
        failures += r.is_error
        bad = await client.call_tool("get_quote", {"ticker": "../x"})
        print("bad ticker is a tool error, not a crash:", bad.is_error, "|", bad.content[0].text)
    print(f"\n{'tool':18}{'success':>9}{'MCP p50':>10}{'MCP p95':>10}{'direct p50':>12}{'direct p95':>12}   (seconds)")
    for n, ok, runs_, m50, m95, d50, d95 in rows:
        print(f"{n:18}{ok:>5}/{runs_:<3}{m50:>10.4f}{m95:>10.4f}{d50:>12.4f}{d95:>12.4f}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    sys.exit(asyncio.run(main(parser.parse_args().runs)))
