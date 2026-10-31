"""Step 14: analyze the three sample portfolios with live prices, check one number by hand, and time the math at 5 / 50 / 500 holdings.

Run:  .venv/bin/python -m scripts.step14_portfolio_demo
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.agents import portfolio_calc as pc
from src.core.config import PROJECT_ROOT
from src.core.logging import setup_logging
from src.data.errors import MarketDataError
from src.data.factory import build_market_data_service

setup_logging("WARNING")
PORTFOLIO_DIR = PROJECT_ROOT / "src" / "data" / "sample_portfolios"
metadata = json.loads((PROJECT_ROOT / "src" / "data" / "ticker_metadata.json").read_text())
service = build_market_data_service()
report: dict = {}


def market_inputs(holdings):
    prices, closes = {}, {}
    for h in holdings:
        try:
            prices[h.ticker] = service.get_quote(h.ticker).price
            closes[h.ticker] = [(p.date, p.close) for p in service.get_history(h.ticker, "1y").points]
        except MarketDataError as error:
            print(f"  (could not fetch {h.ticker}: {error.user_message})")
    return prices, closes


def show(name, result):
    print("=" * 100, f"\n{name}\n" + "=" * 100)
    print(f"  {'ticker':7}{'shares':>8}{'avg cost':>10}{'price':>10}{'value':>12}{'weight':>9}{'gain':>9}")
    for p in result["positions"]:
        gain = "n/a" if p.gain_pct is None else f"{p.gain_pct:+.1f}%"
        print(f"  {p.ticker:7}{p.shares:>8g}{p.avg_cost:>10.2f}{p.price:>10.2f}{p.value:>12,.2f}{p.weight:>9.1%}{gain:>9}")
    print(f"  total value ${result['total_value']:,.2f}   total cost ${result['total_cost']:,.2f}")
    print(f"  by asset class: { {k: f'{v:.1%}' for k, v in result['by_asset_class'].items()} }")
    print(f"  by sector     : { {k: f'{v:.1%}' for k, v in result['by_sector'].items()} }")
    ticker, weight = result["largest_position"]
    print(f"  largest position: {ticker} {weight:.1%}   HHI {result['hhi']:.3f}   effective holdings {result['effective_holdings']:.2f}")
    if result["volatility_pct"] is not None:
        print(f"  annualized volatility {result['volatility_pct']:.2f}%  (from {result['returns_days']} daily returns)   weighted avg correlation {result['avg_correlation']:.2f}")
        print(f"  per-asset volatility %: {result['asset_volatility_pct']}")
        print(f"  pairs with correlation >= 0.8: {[(a, b, round(c, 2)) for a, b, c in result['correlated_pairs']] or 'none'}")
    d = result["diversification"]
    print(f"  DIVERSIFICATION SCORE {d['score']}/100   (concentration {d['concentration']}, asset-class spread {d['asset_class_spread']}, low correlation {d['low_correlation']})")
    for note in result["notes"]:
        print(f"  note: {note}")


results = {}
for file in ("diversified", "concentrated_single_stock", "bond_heavy"):
    holdings = pc.parse_portfolio_csv((PORTFOLIO_DIR / f"{file}.csv").read_text())
    prices, closes = market_inputs(holdings)
    results[file] = pc.analyze(holdings, prices, closes, metadata)
    show(file, results[file])

print("\n" + "=" * 100, "\nCOMPARISON\n" + "=" * 100)
print(f"  {'portfolio':28}{'largest':>9}{'HHI':>7}{'eff. N':>8}{'vol %':>8}{'avg corr':>10}{'score':>8}")
for name, r in results.items():
    print(f"  {name:28}{r['largest_position'][1]:>9.1%}{r['hhi']:>7.3f}{r['effective_holdings']:>8.2f}{r['volatility_pct']:>8.2f}{r['avg_correlation']:>10.2f}{r['diversification']['score']:>8.1f}")

print("\n" + "=" * 100, "\nHAND CHECK: concentrated_single_stock, HHI and score recomputed from the printed weights\n" + "=" * 100)
r = results["concentrated_single_stock"]
weights = [p.weight for p in r["positions"]]
terms = " + ".join(f"{w:.4f}^2" for w in weights)
print(f"  HHI = {terms} = {' + '.join(f'{w * w:.4f}' for w in weights)} = {sum(w * w for w in weights):.4f}   (program says {r['hhi']:.4f})")
print(f"  effective holdings = 1 / HHI = {1 / sum(w * w for w in weights):.3f}   (program says {r['effective_holdings']:.3f})")
cls = list(r["by_asset_class"].values())
conc, spread = min(1, 1 / sum(w * w for w in weights) / 10), min(1, 1 / sum(w * w for w in cls) / 4)
low = min(1, max(0, 1 - r["avg_correlation"]))
print(f"  score = 100 * (0.4*{conc:.3f} + 0.3*{spread:.3f} + 0.3*{low:.3f}) = {100 * (0.4 * conc + 0.3 * spread + 0.3 * low):.1f}   (program says {r['diversification']['score']})")
print("  (to check the volatility, paste the weights and the tickers' daily closes into a spreadsheet: =SQRT(MMULT(MMULT(TRANSPOSE(w),COVARIANCE.S matrix),w))*SQRT(252))")

print("\n" + "=" * 100, "\nCOMPUTE TIME (synthetic data, 252 daily returns, math only, no network; median of 5 runs)\n" + "=" * 100)
rng = np.random.default_rng(1)
timings = {}
for n in (5, 50, 500):
    dates = [f"d{i:04d}" for i in range(253)]
    closes = {f"T{i}": list(zip(dates, (100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 253))).tolist())) for i in range(n)}
    holdings = [pc.Holding(t, 10, 100.0) for t in closes]
    prices = {t: series[-1][1] for t, series in closes.items()}
    runs = []
    for _ in range(5):
        start = time.perf_counter()
        pc.analyze(holdings, prices, closes, {})
        runs.append(time.perf_counter() - start)
    timings[n] = float(np.median(runs))
    print(f"  {n:4d} holdings: {timings[n] * 1000:9.2f} ms")
report["timings_s"] = timings
report["portfolios"] = {name: {k: v for k, v in r.items() if k not in ("positions", "correlation", "correlation_tickers")} |
                        {"positions": [vars(p) for p in r["positions"]]} for name, r in results.items()}
output_file = Path(f"benchmarks/results/step14_portfolio_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(report, indent=2, default=str))
print(f"\nsaved: {output_file}")
