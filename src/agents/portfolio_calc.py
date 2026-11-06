"""Portfolio math (Step 14). No LLM, no network: data in, numbers out, so every function is testable by hand.

Conventions
  weights      fractions that sum to 1 (market value / total market value)
  HHI          sum of squared weights, 0 to 1 (some books use 0 to 10,000: multiply by 10,000)
  volatility   annualized standard deviation of DAILY simple returns, as a percent: sqrt(w' S w) * sqrt(252)
  correlation  Pearson correlation of daily returns, -1 to +1
Past volatility and correlation describe the period measured. They are not forecasts, and correlations rise in crashes.

Diversification score (0-100) is OUR heuristic, not an industry standard. Weights 0.4 / 0.3 / 0.3 are a judgement call:
  score = 100 * (0.4 * concentration + 0.3 * asset_class_spread + 0.3 * low_correlation)
  concentration      = min(1, (1 / HHI of holdings) / 10)            10 equally sized holdings = full marks
  asset_class_spread = min(1, (1 / HHI of asset classes) / 4)        4 equally sized asset classes = full marks
  low_correlation    = 1 - weighted average pairwise correlation, clipped to [0, 1]   (a single holding counts as correlation 1)
"""
import csv
import io
import math
import re
from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252
MIN_RETURN_DAYS = 30   # fewer overlapping daily returns than this and volatility / correlation are not reported
_TICKER_OK = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


class PortfolioInputError(ValueError):
    """Bad portfolio input. `problems` lists EVERY issue found, so the user can fix them all in one go."""
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Holding:
    ticker: str
    shares: float
    avg_cost: float


@dataclass(frozen=True)
class Position:
    ticker: str
    shares: float
    avg_cost: float
    price: float
    value: float
    weight: float
    cost_basis: float
    gain_pct: float | None


# ---------- parsing ----------
def parse_portfolio_csv(text: str) -> list[Holding]:
    """Columns: ticker, shares, avg_cost. Validates every row; duplicate tickers are merged (shares add, cost is share-weighted)."""
    reader = csv.DictReader(io.StringIO(text.strip()))
    columns = [c.strip().lower() for c in (reader.fieldnames or [])]
    missing = [c for c in ("ticker", "shares", "avg_cost") if c not in columns]
    if missing:
        raise PortfolioInputError([f"missing column(s): {', '.join(missing)} (expected: ticker, shares, avg_cost)"])
    problems, merged = [], {}
    for line_number, raw in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        ticker = row["ticker"].upper()
        if not _TICKER_OK.match(ticker):
            problems.append(f"line {line_number}: '{row['ticker']}' is not a valid ticker")
            continue
        try:
            shares, cost = float(row["shares"]), float(row["avg_cost"])
        except ValueError:
            problems.append(f"line {line_number} ({ticker}): shares and avg_cost must be numbers")
            continue
        if not (math.isfinite(shares) and shares > 0):
            problems.append(f"line {line_number} ({ticker}): shares must be greater than 0")
        elif not (math.isfinite(cost) and cost >= 0):
            problems.append(f"line {line_number} ({ticker}): avg_cost cannot be negative")
        else:
            old_shares, old_cost = merged.get(ticker, (0.0, 0.0))
            total = old_shares + shares
            merged[ticker] = (total, (old_shares * old_cost + shares * cost) / total)
    if not merged and not problems:
        problems.append("the file has no holdings")
    if problems:
        raise PortfolioInputError(problems)
    return [Holding(t, s, c) for t, (s, c) in merged.items()]


# ---------- values, weights, breakdowns ----------
def build_positions(holdings: list[Holding], prices: dict[str, float]) -> tuple[list[Position], list[str]]:
    """Price each holding. A holding with no price is returned in `unpriced`, never valued at a guess."""
    priced = [(h, prices[h.ticker]) for h in holdings if prices.get(h.ticker)]
    unpriced = [h.ticker for h in holdings if not prices.get(h.ticker)]
    total = sum(h.shares * p for h, p in priced)
    positions = [Position(h.ticker, h.shares, h.avg_cost, p, h.shares * p, h.shares * p / total, h.shares * h.avg_cost,
                          (p / h.avg_cost - 1) * 100 if h.avg_cost else None) for h, p in priced]
    return sorted(positions, key=lambda x: -x.value), unpriced


def breakdown(positions: list[Position], metadata: dict, field: str) -> dict[str, float]:
    """Weight per category (asset_class or sector). Tickers missing from the metadata go under 'Unknown'."""
    totals: dict[str, float] = {}
    for p in positions:
        category = metadata.get(p.ticker, {}).get(field, "Unknown")
        totals[category] = totals.get(category, 0.0) + p.weight
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def hhi(weights) -> float:
    return float(sum(w * w for w in weights))


def effective_holdings(weights) -> float:
    """1 / HHI: how many EQUALLY sized holdings would be as concentrated as this portfolio."""
    h = hhi(weights)
    return 1 / h if h else 0.0


# ---------- returns, volatility, correlation ----------
def returns_matrix(closes: dict[str, list[tuple[str, float]]]) -> tuple[list[str], np.ndarray] | None:
    """Daily simple returns on the dates ALL tickers share. Rows are days, columns follow the returned ticker order.
    None when fewer than MIN_RETURN_DAYS overlapping returns exist."""
    tickers = list(closes)
    if not tickers:
        return None
    common = sorted(set.intersection(*(set(d for d, _ in series) for series in closes.values())))
    if len(common) < MIN_RETURN_DAYS + 1:
        return None
    lookup = {t: dict(closes[t]) for t in tickers}   # built once per ticker, not once per date
    prices = np.column_stack([[lookup[t][d] for d in common] for t in tickers]).astype(float)
    return tickers, prices[1:] / prices[:-1] - 1


def portfolio_volatility_pct(weights: np.ndarray, returns: np.ndarray) -> float:
    covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
    return float(math.sqrt(max(weights @ covariance @ weights, 0.0)) * math.sqrt(TRADING_DAYS) * 100)


def asset_volatility_pct(returns: np.ndarray) -> np.ndarray:
    return np.std(returns, axis=0, ddof=1) * math.sqrt(TRADING_DAYS) * 100


def correlation_matrix(returns: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.corrcoef(returns, rowvar=False))


def weighted_avg_correlation(weights: np.ndarray, corr: np.ndarray) -> float:
    """Average of the pairwise correlations, each pair weighted by w_i * w_j. One holding: defined as 1 (nothing to diversify with)."""
    if len(weights) < 2:
        return 1.0
    pair_weights = np.outer(weights, weights)
    np.fill_diagonal(pair_weights, 0)
    return float((pair_weights * corr).sum() / pair_weights.sum())


def correlated_pairs(tickers: list[str], corr: np.ndarray, threshold: float = 0.8) -> list[tuple[str, str, float]]:
    pairs = [(tickers[i], tickers[j], float(corr[i, j])) for i in range(len(tickers)) for j in range(i + 1, len(tickers)) if corr[i, j] >= threshold]
    return sorted(pairs, key=lambda x: -x[2])


# ---------- the documented score ----------
def diversification_score(holding_weights, class_weights, avg_correlation: float) -> dict:
    concentration = min(1.0, effective_holdings(holding_weights) / 10)
    spread = min(1.0, effective_holdings(class_weights) / 4)
    low_corr = min(1.0, max(0.0, 1 - avg_correlation))
    return {"score": round(100 * (0.4 * concentration + 0.3 * spread + 0.3 * low_corr), 1),
            "concentration": round(concentration, 3), "asset_class_spread": round(spread, 3), "low_correlation": round(low_corr, 3)}


# ---------- what-if: move part of the portfolio from one sector into another kind of holding ----------
# The options a growth question is answered with. Labels match asset_class in the ticker metadata (and ASSET_TYPES for typed-in holdings); the trade-off text is fixed
# here, in code, so the model paraphrases it instead of inventing claims. Asset classes only: never a specific fund or stock.
WHATIF_OPTIONS = {
    "Bond": {"aliases": ("Bond",), "trade_off": "usually lower expected growth and smaller swings; prices can fall when interest rates rise"},
    "International stock": {"aliases": ("International stock",), "trade_off": "keeps stock-market growth potential and spreads across other regions; still has stock-market risk plus currency swings"},
    "Real estate": {"aliases": ("Real estate", "REIT (real estate)"), "trade_off": "can add rental-style income and some inflation protection; can fall when interest rates rise and is not as steady as it looks"},
    "Cash": {"aliases": ("Cash",), "trade_off": "very stable and easy to use, but little growth and it can lose buying power to inflation"},
}


# Sectors that are already spread out or are not an equity sector: nothing to "move out of", so they are never the source of a what-if.
SPREAD_SECTORS = {"Broad market", "Aggregate bonds", "Long-term Treasuries", "Inflation-protected bonds", "Real estate", "Unknown", "Unspecified"}
NOT_A_SOURCE_CLASSES = {"Bond", "Cash"}   # already the steady side: a what-if never moves money out of these
MIN_SOURCE_WEIGHT = 0.20   # a sector below this share is not concentrated enough to suggest moving money out of


def concentrated_sector(result: dict, metadata: dict) -> str | None:
    """The sector holding the biggest share of the portfolio, counting only holdings that are not bonds or cash and not in a spread-out sector.
    None when no such sector reaches MIN_SOURCE_WEIGHT."""
    weights: dict[str, float] = {}
    for p in result["positions"]:
        info = metadata.get(p.ticker, {})
        sector = info.get("sector", "Unknown")
        if info.get("asset_class") not in NOT_A_SOURCE_CLASSES and sector not in SPREAD_SECTORS:
            weights[sector] = weights.get(sector, 0.0) + p.weight
    weights = {sector: w for sector, w in weights.items() if w >= MIN_SOURCE_WEIGHT}
    return max(weights, key=weights.get) if weights else None


def rebalance_scenarios(result: dict, metadata: dict, shift: float = 0.10, from_sector: str | None = None, options: dict = WHATIF_OPTIONS) -> list[dict]:
    """Illustrative what-ifs, pure arithmetic: for each option, take `shift` of the WHOLE portfolio (10 percentage points, not 10% of the sector) out of the holdings
    in `from_sector` (default: the portfolio's most concentrated sector), pro rata, and put it in one new holding of that option's asset class. [] when there is no
    such sector. Volatility is NOT re-estimated (the new holding has no price history here) and the average correlation is carried over unchanged."""
    from_sector = from_sector or concentrated_sector(result, metadata)
    if from_sector is None:
        return []
    positions = result["positions"]
    in_sector = [p for p in positions if metadata.get(p.ticker, {}).get("sector", "Unknown") == from_sector]
    sector_weight = sum(p.weight for p in in_sector)
    if not sector_weight:
        return []
    moved = min(shift, sector_weight)
    keep = (sector_weight - moved) / sector_weight
    weights = [p.weight * (keep if p in in_sector else 1.0) for p in positions] + [moved]
    avg_corr = result["avg_correlation"] if result["avg_correlation"] is not None else 1.0
    scenarios = []
    for label, option in options.items():
        canonical = lambda cls: label if cls in option["aliases"] else cls   # a typed-in 'REIT (real estate)' and a listed 'Real estate' are one class
        class_weights: dict[str, float] = {}
        for p, w in zip(positions, weights):
            cls = canonical(metadata.get(p.ticker, {}).get("asset_class", "Unknown"))
            class_weights[cls] = class_weights.get(cls, 0.0) + w
        class_weights[label] = class_weights.get(label, 0.0) + moved
        before = sum(v for cls, v in result["by_asset_class"].items() if cls in option["aliases"])
        scenarios.append({"option": label, "from_sector": from_sector, "trade_off": option["trade_off"], "moved": moved, "sector_before": sector_weight, "sector_after": sector_weight - moved,
                          "class_before": before, "class_after": class_weights[label],
                          "hhi_before": result["hhi"], "hhi_after": hhi(weights),
                          "effective_before": result["effective_holdings"], "effective_after": effective_holdings(weights),
                          "score_before": result["diversification"]["score"],
                          "score_after": diversification_score(weights, list(class_weights.values()), avg_corr)["score"]})
    return scenarios


# ---------- everything together (pure: prices and closes are passed in) ----------
def analyze(holdings: list[Holding], prices: dict[str, float], closes: dict[str, list[tuple[str, float]]], metadata: dict) -> dict:
    notes = []
    positions, unpriced = build_positions(holdings, prices)
    if unpriced:
        notes.append(f"no price for {unpriced}: left out of every figure below")
    if not positions:
        raise PortfolioInputError(["none of the holdings could be priced"])
    weights = [p.weight for p in positions]
    class_weights = breakdown(positions, metadata, "asset_class")
    result = {"total_value": sum(p.value for p in positions), "total_cost": sum(p.cost_basis for p in positions), "positions": positions,
              "unpriced": unpriced, "by_asset_class": class_weights, "by_sector": breakdown(positions, metadata, "sector"),
              "largest_position": (positions[0].ticker, positions[0].weight), "hhi": hhi(weights),
              "effective_holdings": effective_holdings(weights), "volatility_pct": None, "avg_correlation": None,
              "correlated_pairs": [], "returns_days": 0, "asset_volatility_pct": {}, "notes": notes}
    if result["by_asset_class"].get("Unknown"):
        notes.append(f"{result['by_asset_class']['Unknown']:.0%} of the portfolio is in tickers with no classification")

    usable = [p for p in positions if p.ticker in closes]
    if len(usable) < len(positions):
        notes.append(f"no price history for {[p.ticker for p in positions if p.ticker not in closes]}: volatility and correlation cover the rest")
    matrix = returns_matrix({p.ticker: closes[p.ticker] for p in usable}) if usable else None
    if matrix is None:
        notes.append(f"fewer than {MIN_RETURN_DAYS} overlapping daily returns: volatility and correlation not reported")
    else:
        tickers, returns = matrix
        by_ticker = {p.ticker: p.weight for p in usable}
        w = np.array([by_ticker[t] for t in tickers])
        w = w / w.sum()   # re-normalize across the holdings that have history
        corr = correlation_matrix(returns)
        result.update({"volatility_pct": portfolio_volatility_pct(w, returns), "avg_correlation": weighted_avg_correlation(w, corr),
                       "correlated_pairs": correlated_pairs(tickers, corr), "returns_days": len(returns),
                       "asset_volatility_pct": dict(zip(tickers, asset_volatility_pct(returns).round(2).tolist())),
                       "correlation_tickers": tickers, "correlation": corr})
    avg_corr = result["avg_correlation"] if result["avg_correlation"] is not None else 1.0
    result["diversification"] = diversification_score(weights, list(class_weights.values()), avg_corr)
    if result["avg_correlation"] is None:
        notes.append("correlation unknown, so the score assumes correlation 1 (the pessimistic case)")
    return result
