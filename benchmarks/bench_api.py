"""Step 22 API measurements against a REAL uvicorn server over real HTTP.

  python -m benchmarks.bench_api

Per endpoint: cold (first) call, then N timed calls -> p50 / p95 / max, error rate and response size. Quotes are measured twice: distinct tickers
(every call misses the cache) and one repeated ticker (every call hits it). /chat uses a fresh session per call, one question per route.
Then a concurrency check: simultaneous calls must return CORRECT answers, not merely fast ones. Raw numbers go to benchmarks/results/.
Costs real money for /chat and /portfolio/analyze (about $0.1 in total).
"""
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx

from benchmarks.common import bearer_headers

ROOT = Path(__file__).resolve().parents[1]
PORT, BASE = 8765, "http://127.0.0.1:8765"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT", "KO", "PEP", "DIS", "NFLX", "INTC", "AMD", "BA", "XOM", "CVX", "PG",
           "VOO", "SPY", "QQQ", "VTI", "BND", "VXUS", "IWM", "GLD", "T", "IBM"]
KNOWLEDGE_QUERIES = ["what is an expense ratio", "how does compound interest work", "difference between stocks and bonds", "what is diversification",
                     "roth ira versus traditional ira", "what is a bond yield", "how are dividends taxed", "what is an index fund", "what is inflation",
                     "what does risk tolerance mean"] * 2
CHAT = [("finance_qa", "What is an ETF?"), ("finance_qa", "How does diversification reduce risk?"), ("market", "How is Microsoft doing today?"),
        ("market", "What is the price of VOO?"), ("goal", "I want to save $20000 in 4 years, I have $2000, medium risk"), ("tax", "What is a wash sale?"),
        ("news", "What is the latest news about Tesla?"), ("out_of_scope", "What is the price of a pizza?"),
        ("multi", "How is Apple doing, what is the latest news on it, and what is an expense ratio?"), ("finance_qa", "What is a mutual fund?")]
CSV = (ROOT / "src/data/sample_portfolios/diversified.csv").read_text()


def pct(values, p):
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))], 4)


def measure(client, name, calls):
    """calls: list of zero-argument callables returning an httpx.Response. The first is reported as 'cold'."""
    times, statuses, sizes = [], [], []
    for call in calls:
        start = time.perf_counter()
        response = call()
        times.append(time.perf_counter() - start)
        statuses.append(response.status_code)
        sizes.append(len(response.content))
    timed = times[1:] or times
    errors = sum(1 for s in statuses if s >= 400)
    row = {"endpoint": name, "n": len(times), "cold_s": round(times[0], 4), "p50_s": pct(timed, 50), "p95_s": pct(timed, 95), "max_s": round(max(timed), 4),
           "errors": errors, "error_rate": round(errors / len(times), 3), "status_codes": sorted(set(statuses)), "bytes_median": int(statistics.median(sizes)), "bytes_max": max(sizes)}
    print(f"  {name:34} n={row['n']:<3} cold {row['cold_s']:>7.3f}s  p50 {row['p50_s']:>7.3f}s  p95 {row['p95_s']:>7.3f}s  max {row['max_s']:>7.3f}s  "
          f"errors {errors}/{len(times)}  codes {row['status_codes']}  size p50 {row['bytes_median']:,}B max {row['bytes_max']:,}B")
    return row


def main() -> None:
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.web_app.api:app", "--port", str(PORT), "--log-level", "warning"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if httpx.get(f"{BASE}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.25)
        client = httpx.Client(base_url=BASE, timeout=60, headers=bearer_headers(BASE))
        rows, started = [], datetime.now()
        print("\nSTAGE 1: per-endpoint latency (real HTTP, one request at a time)")
        rows.append(measure(client, "GET /health", [lambda: client.get("/health")] * 30))
        rows.append(measure(client, "GET /market/quote (cache MISS)", [lambda t=t: client.get(f"/market/quote/{t}") for t in TICKERS]))
        rows.append(measure(client, "GET /market/quote (cache HIT)", [lambda: client.get("/market/quote/AAPL")] * 30))
        rows.append(measure(client, "GET /market/history 1y (MISS)", [lambda t=t: client.get(f"/market/history/{t}?period=1y") for t in TICKERS]))
        rows.append(measure(client, "GET /market/overview", [lambda: client.get("/market/overview")] * 20))
        rows.append(measure(client, "POST /goals/plan", [lambda i=i: client.post("/goals/plan", json={"goal": 10000 + i * 500, "years": 5, "current_savings": 1000}) for i in range(30)]))
        rows.append(measure(client, "GET /knowledge/search", [lambda q=q: client.get("/knowledge/search", params={"q": q}) for q in KNOWLEDGE_QUERIES]))
        rows.append(measure(client, "POST /portfolio/analyze", [lambda: client.post("/portfolio/analyze", json={"csv_text": CSV})] * 8))
        chat_rows = []
        for route, message in CHAT:
            row = measure(client, f"POST /chat [{route}]", [lambda m=message: client.post("/chat", json={"message": m})] * 1)
            chat_rows.append(row)
        chat_times = [r["cold_s"] for r in chat_rows]
        print(f"  /chat overall (10 questions, one per case): p50 {pct(chat_times, 50):.3f}s  p95 {pct(chat_times, 95):.3f}s  max {max(chat_times):.3f}s")
        rows += chat_rows

        print("\nSTAGE 2: error paths (should be fast and use the standard error shape)")
        bad = [("POST /goals/plan (422)", lambda: client.post("/goals/plan", json={"goal": -1})), ("GET /market/quote/ZZZZ (404)", lambda: client.get("/market/quote/ZZZZ")),
               ("POST /chat empty (422)", lambda: client.post("/chat", json={"message": ""})), ("GET /nope (404)", lambda: client.get("/nope"))]
        for name, call in bad:
            start = time.perf_counter(); response = call(); ms = (time.perf_counter() - start) * 1000
            keys = sorted(response.json())
            print(f"  {name:34} status {response.status_code}  {ms:7.1f} ms  body keys {keys}")
            rows.append({"endpoint": name, "status": response.status_code, "ms": round(ms, 1), "body_keys": keys})

        print("\nSTAGE 3: concurrency (correctness first, then speed)")
        concurrency = {}
        # (a) 4 simultaneous portfolio analyses of DIFFERENT portfolios: each answer must describe its own portfolio.
        files = ["diversified.csv", "bond_heavy.csv", "concentrated_single_stock.csv", "diversified.csv"]
        csvs = [(ROOT / "src/data/sample_portfolios" / f).read_text() for f in files]
        start = time.perf_counter()
        with ThreadPoolExecutor(4) as pool:
            replies = list(pool.map(lambda c: client.post("/portfolio/analyze", json={"csv_text": c}).json(), csvs))
        wall = time.perf_counter() - start
        counts = [r["metrics"]["holding_count"] for r in replies]
        expected = [len([l for l in c.strip().splitlines()[1:] if l.strip()]) for c in csvs]
        mix_up = [r["metrics"]["holdings"][0]["ticker"] for r in replies]
        print(f"  4 simultaneous portfolio analyses: holding counts {counts} expected {expected}  match={counts == expected}  wall {wall:.2f}s  first tickers {mix_up}")
        concurrency["portfolio_4_parallel"] = {"counts": counts, "expected": expected, "match": counts == expected, "wall_s": round(wall, 2)}
        # (b) 8 simultaneous chats, different sessions.
        questions = [m for _, m in CHAT[:8]]
        start = time.perf_counter()
        with ThreadPoolExecutor(8) as pool:
            out = list(pool.map(lambda m: client.post("/chat", json={"message": m}), questions))
        wall = time.perf_counter() - start
        ok = [r.status_code == 200 for r in out]
        routes = [r.json().get("route") for r in out]
        print(f"  8 simultaneous chats (different sessions): all 200={all(ok)}  wall {wall:.2f}s  routes {routes}")
        concurrency["chat_8_parallel"] = {"all_ok": all(ok), "wall_s": round(wall, 2), "routes": routes}
        # (c) 4 simultaneous turns of ONE session: the lock should serialise them; turns must be 1,2,3,4 with none lost.
        sid = "race-test-1"
        with ThreadPoolExecutor(4) as pool:
            out = list(pool.map(lambda m: client.post("/chat", json={"message": m, "session_id": sid}).json(), ["What is an ETF?", "What is a bond?", "What is a stock?", "What is inflation?"]))
        turns = sorted(r["turn"] for r in out)
        print(f"  4 simultaneous turns of ONE session: turns seen {turns}  (expect [1, 2, 3, 4])")
        concurrency["one_session_4_parallel"] = {"turns": turns, "ok": turns == [1, 2, 3, 4]}

        total_cost = None
        result = {"started": started.isoformat(timespec="seconds"), "endpoints": rows, "concurrency": concurrency}
        out_path = ROOT / "benchmarks" / "results" / f"step22_api_{started:%Y%m%d_%H%M%S}.json"
        out_path.write_text(json.dumps(result, indent=1))
        print(f"\nraw results: {out_path.relative_to(ROOT)}")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
