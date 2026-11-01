"""The kinds of holding a user can pick when typing a portfolio in. One list, shared by the database check (migration 0004), the API validation and the UI."""
from typing import Literal

AssetType = Literal["stock", "etf", "mutual_fund", "bond", "reit", "cash", "crypto", "other"]   # a test checks this matches ASSET_TYPES and the database CHECK

ASSET_TYPES = {
    "stock": "Stock",
    "etf": "ETF",
    "mutual_fund": "Mutual fund",
    "bond": "Bond",
    "reit": "REIT (real estate)",
    "cash": "Cash",
    "crypto": "Crypto",
    "other": "Other",
}
PLURAL = {"stock": "Stocks", "etf": "ETFs", "mutual_fund": "Mutual funds", "bond": "Bonds", "reit": "REITs", "cash": "Cash", "crypto": "Crypto", "other": "Other"}
