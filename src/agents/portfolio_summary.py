"""A saved portfolio grouped by the kind of holding the USER said it is (stock, ETF, bond, ...). No model, no network: holdings and prices in, numbers out.

  cost      shares x average cost                       what was paid
  value     shares x current price                      only for holdings that have a price
  gain      value / cost of the PRICED holdings - 1     compares like with like: an unpriced holding is left out of both sides
  weight    value / total value of the priced holdings  so weights of the priced holdings add to 100%

Money arithmetic uses Decimal and is rounded to cents only when the numbers leave this function.
"""
from decimal import Decimal

from src.data.asset_types import ASSET_TYPES, PLURAL
from src.data.portfolio_repository import StoredHolding

CENT = Decimal("0.01")
HUNDRED = Decimal(100)


def _money(x: Decimal) -> float:
    return float(x.quantize(CENT))


def _gain_pct(value: Decimal, cost: Decimal) -> float | None:
    return None if cost <= 0 else float(((value / cost - 1) * HUNDRED).quantize(CENT))


def group_by_type(holdings: list[StoredHolding], prices: dict[str, Decimal | float | None]) -> dict:
    """`prices` maps ticker -> price; a missing key or None means the price could not be fetched (the holding is shown, with its cost, but has no value)."""
    rows = []
    for h in holdings:
        price = prices.get(h.ticker)
        price = None if price is None else Decimal(str(price))
        cost = h.shares * h.avg_cost
        value = None if price is None else h.shares * price
        rows.append({"holding": h, "cost": cost, "value": value, "price": price})
    total_value = sum((r["value"] for r in rows if r["value"] is not None), Decimal(0))
    priced_cost = sum((r["cost"] for r in rows if r["value"] is not None), Decimal(0))

    groups = []
    for asset_type in ASSET_TYPES:
        members = [r for r in rows if r["holding"].asset_type == asset_type]
        if not members:
            continue
        value = sum((r["value"] for r in members if r["value"] is not None), Decimal(0))
        cost = sum((r["cost"] for r in members), Decimal(0))
        priced_cost_here = sum((r["cost"] for r in members if r["value"] is not None), Decimal(0))
        members.sort(key=lambda r: (r["value"] is None, -(r["value"] or Decimal(0)), r["holding"].ticker))
        groups.append({
            "asset_type": asset_type, "label": PLURAL[asset_type], "count": len(members), "unpriced_count": sum(r["value"] is None for r in members),
            "cost": _money(cost), "value": _money(value), "gain_pct": _gain_pct(value, priced_cost_here),
            "weight_pct": float((value / total_value * HUNDRED).quantize(CENT)) if total_value > 0 and len(members) > sum(r["value"] is None for r in members) else None,
            "holdings": [{
                "ticker": r["holding"].ticker, "asset_type": asset_type, "shares": float(r["holding"].shares), "avg_cost": float(r["holding"].avg_cost),
                "purchase_date": r["holding"].purchase_date, "cost": _money(r["cost"]), "price": None if r["price"] is None else float(r["price"]),
                "value": None if r["value"] is None else _money(r["value"]), "gain_pct": None if r["value"] is None else _gain_pct(r["value"], r["cost"]),
                "weight_pct": None if r["value"] is None or total_value <= 0 else float((r["value"] / total_value * HUNDRED).quantize(CENT))} for r in members]})
    groups.sort(key=lambda g: (-g["value"], g["asset_type"]))   # biggest group first; ties in a fixed order
    return {
        "total_cost": _money(sum((r["cost"] for r in rows), Decimal(0))), "total_value": _money(total_value), "gain_pct": _gain_pct(total_value, priced_cost),
        "holding_count": len(rows), "unpriced": sorted(r["holding"].ticker for r in rows if r["value"] is None), "groups": groups}
