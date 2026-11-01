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


def summarize_portfolio(holdings: list[StoredHolding], properties: list, prices: dict) -> dict:
    """Market holdings (grouped by kind) AND properties in one view. Real estate becomes one more group, valued at the owner's own estimate. Totals, gain and
    every share-of-total are recomputed over everything together. The mortgage is NOT subtracted from the group's value (the group shows what the property is
    worth); it appears separately as `mortgage_owed`, and `net_value` is total value minus it."""
    from src.agents.real_estate import real_estate_totals
    base = group_by_type(holdings, prices)
    real_estate = real_estate_totals(properties)
    if real_estate is None:
        return {**base, "property_count": 0, "mortgage_owed": 0.0, "net_value": base["total_value"], "real_estate": None}

    market_value, re_value = Decimal(str(base["total_value"])), Decimal(str(real_estate["value"]))
    total_value = market_value + re_value
    priced_cost = sum((Decimal(str(h["cost"])) for g in base["groups"] for h in g["holdings"] if h["value"] is not None), Decimal(0)) + Decimal(str(real_estate["purchase_price"]))

    def share(value) -> float | None:
        return None if value is None or total_value <= 0 else float((Decimal(str(value)) / total_value * HUNDRED).quantize(CENT))

    groups = []
    for group in base["groups"]:
        groups.append({**group, "weight_pct": share(group["value"]) if group["count"] > group["unpriced_count"] else None,
                       "holdings": [{**h, "weight_pct": share(h["value"])} for h in group["holdings"]]})
    groups.append({"asset_type": "real_estate", "label": "Real estate", "count": real_estate["count"], "unpriced_count": 0, "cost": real_estate["purchase_price"],
                   "value": real_estate["value"], "gain_pct": real_estate["gain_pct"], "weight_pct": share(real_estate["value"]), "holdings": []})
    groups.sort(key=lambda g: (-g["value"], g["asset_type"]))
    mortgage = Decimal(str(real_estate["mortgage"]))
    return {"total_cost": _money(Decimal(str(base["total_cost"])) + Decimal(str(real_estate["purchase_price"]))), "total_value": _money(total_value),
            "gain_pct": _gain_pct(total_value, priced_cost), "holding_count": base["holding_count"] + real_estate["count"], "property_count": real_estate["count"],
            "unpriced": base["unpriced"], "groups": groups, "mortgage_owed": _money(mortgage), "net_value": _money(total_value - mortgage), "real_estate": real_estate}
