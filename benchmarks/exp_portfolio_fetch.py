"""Step 25 experiment: how long does the Portfolio agent take to fetch prices and do its math when the market-data cache is COLD, one fetch thread
(the old sequential behaviour) vs several?

Fairness rules learned the hard way: (1) every run draws DIFFERENT random tickers from a pool of 150 (the first version reused 5 tickers and the data
provider answered faster after the first pass, which flattered the sequential code); (2) the two variants alternate run by run, so a slow minute at the
provider hits both; (3) every run builds a fresh market-data service, so our own cache is empty. The model is faked (instant), so only fetch + math is timed.
Real yfinance calls: about 2 per holding per run.

  python -m benchmarks.exp_portfolio_fetch [--runs 10] [--threads 8]
"""
import argparse
import random
import time

import src.agents.portfolio_agent as portfolio_agent
from benchmarks.common import save, summarize
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.core.config import load_settings
from src.core.llm import LLMResult
from src.data.factory import build_market_data_service

POOL = """AAPL MSFT GOOGL AMZN NVDA META TSLA JPM V WMT KO PEP DIS NFLX INTC AMD BA XOM CVX PG JNJ UNH HD MA BAC PFE CSCO ORCL ADBE CRM T VZ
MRK ABBV TMO COST MCD NKE LLY AVGO TXN QCOM HON IBM CAT GE MMM LOW SBUX AMGN GS MS BLK SCHW AXP USB PNC C WFC CVS CI MDT ISRG GILD BMY LMT RTX
UPS FDX DE MO PM CL KMB GIS K HSY SYY TGT DG ROST TJX EL BKNG MAR HLT YUM CMG ORLY AZO KR DAL UAL LUV AAL CCL RCL NEE DUK SO D AEP EXC SRE PCG
SPY QQQ VTI VOO VXUS BND AGG TLT IWM EFA EEM GLD SLV VNQ XLF XLK XLE XLV XLY XLP XLI XLU ARKK DIA SCHD VIG VUG VTV IJH IJR LQD HYG TIP""".split()


class FakeLLM:
    def chat(self, prompt, system=None, **kwargs):
        return LLMResult("ok", "fake", 1, 1, 0.0, 0.0, 1)


def csv_for(tickers):
    return "ticker,shares,avg_cost\n" + "\n".join(f"{t},10,100" for t in tickers)


parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--threads", type=int, default=8)
args = parser.parse_args()
settings = load_settings()
random.seed(5)
build_market_data_service(settings).get_quote("AAPL")   # import and connection warm-up, not timed
report = {}
print(f"{'portfolio':14} {'fetch threads':>13} {'n':>3} {'p50 s':>7} {'p95 s':>7} {'mean s':>7} {'max s':>7}")
for size in (5, 12):
    times = {1: [], args.threads: []}
    for _ in range(args.runs):
        for threads in random.sample(list(times), len(times)):   # random order inside each pair
            portfolio_agent.MAX_FETCH_THREADS = threads
            agent = PortfolioAnalysisAgent(FakeLLM(), build_market_data_service(settings), {})
            start = time.perf_counter()
            response = agent.run(csv_for(random.sample(POOL, size)))
            times[threads].append(time.perf_counter() - start)
            if response.metadata.get("metrics") is None:
                times[threads].pop()   # a failed fetch would make the timing meaningless
    for threads, values in times.items():
        report[f"{size} holdings, {threads} thread(s)"] = summarize(values)
        s = report[f"{size} holdings, {threads} thread(s)"]
        print(f"{size:>2} holdings     {threads:13d} {s['n']:3d} {s['p50']:7.2f} {s['p95']:7.2f} {s['mean']:7.2f} {s['max']:7.2f}")
print("raw:", save("step25_exp_portfolio_fetch", report))
