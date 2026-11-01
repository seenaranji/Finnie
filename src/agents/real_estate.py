"""Real estate arithmetic. No model, no network: what the owner typed in, numbers out. Decimal throughout; rounded to cents / hundredths only on the way out.

  equity               value - mortgage                       what you would own outright (negative if the mortgage is bigger than the value)
  gain                 value - purchase price                 on paper, before selling costs, taxes and any improvements you paid for
  annual rent          monthly rent x 12
  annual expenses      monthly expenses x 12                  tax, insurance, upkeep, fees: everything EXCEPT the mortgage payment
  net operating income annual rent - annual expenses          before the mortgage payment (Finnie does not know your payment)
  gross yield          annual rent / value                    on what the property is worth NOW
  net yield            net operating income / value           the same measure investors call the "cap rate"
  loan-to-value        mortgage / value

Value, mortgage, rent and expenses are the owner's own numbers and only as current as the owner's last edit. None of this is a valuation or advice.
"""
from decimal import Decimal

from src.data.portfolio_repository import PropertyIn

CENT, HUNDRED, TWELVE = Decimal("0.01"), Decimal(100), Decimal(12)


def _money(x: Decimal) -> float:
    return float(x.quantize(CENT))


def _ratio_pct(top: Decimal, bottom: Decimal) -> float | None:
    return None if bottom <= 0 else float((top / bottom * HUNDRED).quantize(CENT))


def _figures(purchase: Decimal, value: Decimal, mortgage: Decimal, rent: Decimal, expenses: Decimal) -> dict:
    annual_rent, annual_expenses = rent * TWELVE, expenses * TWELVE
    noi = annual_rent - annual_expenses
    return {"purchase_price": _money(purchase), "value": _money(value), "mortgage": _money(mortgage), "equity": _money(value - mortgage),
            "gain": _money(value - purchase), "gain_pct": _ratio_pct(value - purchase, purchase) if purchase > 0 else None,
            "annual_rent": _money(annual_rent), "annual_expenses": _money(annual_expenses), "net_operating_income": _money(noi),
            "gross_yield_pct": _ratio_pct(annual_rent, value), "net_yield_pct": _ratio_pct(noi, value), "loan_to_value_pct": _ratio_pct(mortgage, value)}


def property_metrics(p: PropertyIn) -> dict:
    return {"name": p.name, "property_type": p.property_type, "purchase_date": p.purchase_date,
            **_figures(p.purchase_price, p.current_value, p.mortgage_balance, p.monthly_rent, p.monthly_expenses)}


def real_estate_totals(properties: list[PropertyIn]) -> dict | None:
    """All the properties together (yields are on the TOTAL value, not an average of averages). None when there are no properties."""
    if not properties:
        return None
    total = lambda attr: sum((getattr(p, attr) for p in properties), Decimal(0))
    return {"count": len(properties), **_figures(total("purchase_price"), total("current_value"), total("mortgage_balance"), total("monthly_rent"), total("monthly_expenses")),
            "properties": [property_metrics(p) for p in properties]}
