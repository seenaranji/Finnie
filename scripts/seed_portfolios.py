"""Step 15b: save the three sample CSVs (src/data/sample_portfolios/) as portfolios in PostgreSQL FOR ONE USER. Safe to re-run: a name that already exists
for that user is skipped. Since Step 27 every portfolio has an owner, so say whose they are (the account must already exist: register in the app first).

Run:  .venv/bin/python -m scripts.seed_portfolios --email you@example.com
"""
import argparse

from src.agents.portfolio_calc import parse_portfolio_csv
from src.auth.repository import UserRepository
from src.auth.security import normalize_email
from src.core.config import PROJECT_ROOT
from src.data.portfolio_repository import PortfolioRepository

parser = argparse.ArgumentParser()
parser.add_argument("--email", required=True, help="the registered user who will own the samples")
user = UserRepository.from_settings().get_by_email(normalize_email(parser.parse_args().email))
if user is None:
    raise SystemExit("No account with that email. Register in the app first, then run this again.")
repository = PortfolioRepository.from_settings()
existing = {summary.name for summary in repository.list(owner_id=user.id)}
for path in sorted((PROJECT_ROOT / "src" / "data" / "sample_portfolios").glob("*.csv")):
    name = f"sample: {path.stem}"
    if name in existing:
        print(f"skipped (already saved): {name}")
        continue
    holdings = [{"ticker": h.ticker, "shares": str(h.shares), "avg_cost": str(h.avg_cost)} for h in parse_portfolio_csv(path.read_text())]
    print(f"saved {name}: {len(holdings)} holdings, id {repository.create(name, holdings, owner_id=user.id)}")
print("\n*** SEEDING FINISHED ***")
