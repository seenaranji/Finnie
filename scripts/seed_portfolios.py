"""Step 15b: save the three sample CSVs (src/data/sample_portfolios/) as portfolios in PostgreSQL. Safe to re-run: a name that already exists is skipped.

Run:  .venv/bin/python -m scripts.seed_portfolios
"""
from pathlib import Path

from src.agents.portfolio_calc import parse_portfolio_csv
from src.core.config import PROJECT_ROOT
from src.data.portfolio_repository import PortfolioRepository

repository = PortfolioRepository.from_settings()
existing = {summary.name for summary in repository.list()}
for path in sorted((PROJECT_ROOT / "src" / "data" / "sample_portfolios").glob("*.csv")):
    name = f"sample: {path.stem}"
    if name in existing:
        print(f"skipped (already saved): {name}")
        continue
    holdings = [{"ticker": h.ticker, "shares": str(h.shares), "avg_cost": str(h.avg_cost)} for h in parse_portfolio_csv(path.read_text())]
    print(f"saved {name}: {len(holdings)} holdings, id {repository.create(name, holdings)}")
print("\n*** SEEDING FINISHED ***")
