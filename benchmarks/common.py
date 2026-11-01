"""Shared pieces of the Step 25 benchmarks: the six query types, a throwaway uvicorn server, percentiles, result files."""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
PORT = 8770
BASE = f"http://127.0.0.1:{PORT}"
SAMPLES = ROOT / "src" / "data" / "sample_portfolios"
CSVS = [(SAMPLES / name).read_text() for name in ("diversified.csv", "bond_heavy.csv", "concentrated_single_stock.csv")]

QA = ["What is an ETF?", "How does diversification reduce risk?", "What is an expense ratio?", "What is the difference between stocks and bonds?",
      "What is compound interest?", "What is a mutual fund?", "How does inflation affect my savings?", "What is a bond yield?", "What is an index fund?",
      "Why is owning only one stock risky?"]
TAX = ["What is the 401(k) contribution limit?", "How are dividends taxed?", "What is a capital gain?", "How are long-term capital gains taxed?",
       "What is a Roth IRA?", "What is the IRA contribution limit?", "What is a tax-advantaged account?", "How is interest income taxed?",
       "What is cost basis?", "What is a required minimum distribution?"]
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VOO", "SPY", "QQQ", "VTI", "TSLA"]
MARKET = [f"How is {t} doing today?" if i % 2 else f"What is the price of {t}?" for i, t in enumerate(TICKERS)]
NEWS = [f"What is the latest news about {t}?" for t in ("Tesla", "Apple", "Microsoft", "Nvidia", "Amazon", "Google", "Meta", "Netflix", "Disney", "Boeing")]
GOAL = [f"I want to save ${goal} in {years} years, I have ${have}, medium risk" for goal, years, have in
        ((20000, 4, 2000), (50000, 5, 2000), (10000, 3, 500), (100000, 10, 5000), (30000, 6, 1000), (15000, 2, 3000), (60000, 8, 4000), (25000, 5, 0),
         (80000, 12, 2500), (40000, 7, 1500))]
PORTFOLIO = [("Explain my portfolio in simple terms.", CSVS[i % 3]) for i in range(10)]

# route -> list of (message, portfolio_csv or None); the same ten requests repeat three times for 30 warm runs
# 30 portfolios of 5 different tickers each, none repeated across the 30 (a pool of 150 used exactly once): the market-data cache is cold for every one
_POOL = """AAPL MSFT GOOGL AMZN NVDA META TSLA JPM V WMT KO PEP DIS NFLX INTC AMD BA XOM CVX PG JNJ UNH HD MA BAC PFE CSCO ORCL ADBE CRM T VZ MRK ABBV TMO COST MCD
NKE LLY AVGO TXN QCOM HON IBM CAT GE MMM LOW SBUX AMGN GS MS BLK SCHW AXP USB PNC C WFC CVS CI MDT ISRG GILD BMY LMT RTX UPS FDX DE MO PM CL KMB GIS K HSY SYY
TGT DG ROST TJX EL BKNG MAR HLT YUM CMG ORLY AZO KR DAL UAL LUV AAL CCL RCL NEE DUK SO D AEP EXC SRE PCG SPY QQQ VTI VOO VXUS BND AGG TLT IWM EFA EEM GLD SLV
VNQ XLF XLK XLE XLV XLY XLP XLI XLU ARKK DIA SCHD VIG VUG VTV IJH IJR LQD HYG TIP""".split()
PORTFOLIO_NEW = [("Explain my portfolio in simple terms.", "ticker,shares,avg_cost\n" + "\n".join(f"{t},10,100" for t in _POOL[i * 5:i * 5 + 5])) for i in range(30)]

ROUTES = {"qa": [(q, None) for q in QA], "market": [(q, None) for q in MARKET], "portfolio": PORTFOLIO, "goal": [(q, None) for q in GOAL],
          "news": [(q, None) for q in NEWS], "tax": [(q, None) for q in TAX],
          "portfolio_new": PORTFOLIO_NEW}   # only used when asked for by name: it is the cold-data case


def percentile(values, p):
    return float(np.percentile(values, p)) if len(values) else float("nan")


def summarize(values):
    return {"n": len(values), "mean": float(np.mean(values)), "p50": percentile(values, 50), "p95": percentile(values, 95), "min": float(min(values)), "max": float(max(values))} if len(values) else {"n": 0}


class Server:
    """A real uvicorn process on its own port. Extra environment (e.g. an optimization switch) can be passed in."""
    def __init__(self, env=None, workers=1):
        import os
        self.process = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.web_app.api:app", "--port", str(PORT), "--log-level", "warning", "--workers", str(workers)],
                                        cwd=ROOT, env={**os.environ, **(env or {})}, stdout=subprocess.DEVNULL,
                                        stderr=open(os.environ["BENCH_SERVER_LOG"], "ab") if os.environ.get("BENCH_SERVER_LOG") else subprocess.DEVNULL)   # set BENCH_SERVER_LOG=file to keep the server's log
        for _ in range(120):
            try:
                if httpx.get(f"{BASE}/health", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.25)
        self.process.terminate()
        raise RuntimeError("server did not start")

    def stop(self):
        self.process.terminate()
        self.process.wait(timeout=15)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def chat(client, message, csv=None):
    body = {"message": message, **({"portfolio_csv": csv} if csv else {})}
    start = time.perf_counter()
    response = client.post("/chat", json=body)
    seconds = time.perf_counter() - start
    data = response.json() if response.status_code == 200 else {}
    return {"seconds": seconds, "status": response.status_code, "route": data.get("route"), "answered_by": data.get("answered_by"), "cost_usd": data.get("cost_usd", 0.0),
            "degraded": data.get("degraded"), "server_latency_s": data.get("latency_s")}


def save(name, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path
