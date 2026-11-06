"""Mortgage payoff arithmetic (an amortization calculator). No model, no network: what the owner typed in, numbers out. Decimal throughout.

Each month:  interest = balance x (annual rate / 12), rounded to the cent;  balance = balance + interest - payment.  The last payment is whatever is left.
  months to pay off     how many payments until the balance reaches 0 (the last one is usually smaller)
  total interest        the interest paid over those months
  with extra            the same, paying `extra` more every month on top of the regular payment, from now on
  saved                 months and interest saved by paying the extra

Assumes a fixed rate, a payment made at the end of each month, and that the payment shown is principal + interest only (no escrow for tax or insurance).
If the payment does not even cover the monthly interest, the balance never falls: the answer says so instead of a number. Not advice; your lender's statement wins.
"""
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from src.data.portfolio_repository import PropertyIn

CENT, TWELVE, HUNDRED = Decimal("0.01"), Decimal(12), Decimal(100)
MAX_MONTHS = 100 * 12   # a schedule longer than 100 years is reported as "never"


def _money(x: Decimal) -> float:
    return float(x.quantize(CENT, ROUND_HALF_UP))


def amortize(balance: Decimal, annual_rate_pct: Decimal, payment: Decimal) -> dict | None:
    """Pay the loan off month by month. Returns {"months", "total_interest", "total_paid", "last_payment"}, or None when it never pays off within 100 years."""
    if balance <= 0:
        return {"months": 0, "total_interest": Decimal(0), "total_paid": Decimal(0), "last_payment": Decimal(0)}
    if payment <= 0:
        return None
    rate = annual_rate_pct / HUNDRED / TWELVE
    owed, interest_total, months, last = balance, Decimal(0), 0, Decimal(0)
    while owed > 0:
        if months >= MAX_MONTHS:
            return None
        interest = (owed * rate).quantize(CENT, ROUND_HALF_UP)
        if payment <= interest:
            return None   # the balance would never go down
        last = min(payment, owed + interest)
        owed = owed + interest - last
        interest_total += interest
        months += 1
    return {"months": months, "total_interest": interest_total, "total_paid": balance + interest_total, "last_payment": last}


def _plan(result: dict | None, balance: Decimal, today: date) -> dict | None:
    if result is None:
        return None
    months = result["months"]
    index = today.year * 12 + today.month - 1 + months
    return {"months": months, "years": float((Decimal(months) / TWELVE).quantize(CENT)), "payoff_date": date(index // 12, index % 12 + 1, 1).isoformat() if months else None,
            "total_interest": _money(result["total_interest"]), "total_paid": _money(result["total_paid"]), "last_payment": _money(result["last_payment"])}


def mortgage_payoff(p: PropertyIn, extra: Decimal = Decimal(0), today: date | None = None) -> dict:
    """Payoff answer for one property. `status` says whether there is something to compute and, if not, what is missing."""
    today = today or date.today()
    base = {"name": p.name, "balance": _money(p.mortgage_balance), "monthly_payment": _money(p.monthly_mortgage_payment), "rate_pct": float(p.mortgage_rate_pct), "extra": _money(extra)}
    if p.mortgage_balance <= 0:
        return {**base, "status": "no_mortgage"}
    if p.monthly_mortgage_payment <= 0:
        return {**base, "status": "missing_payment"}
    regular = amortize(p.mortgage_balance, p.mortgage_rate_pct, p.monthly_mortgage_payment)
    if regular is None:
        return {**base, "status": "never_paid_off", "monthly_interest": _money((p.mortgage_balance * p.mortgage_rate_pct / HUNDRED / TWELVE).quantize(CENT, ROUND_HALF_UP))}
    answer = {**base, "status": "ok", "regular": _plan(regular, p.mortgage_balance, today), "with_extra": None, "saved": None}
    if extra > 0:
        faster = amortize(p.mortgage_balance, p.mortgage_rate_pct, p.monthly_mortgage_payment + extra)   # never None: a larger payment than one that already pays off
        answer["with_extra"] = _plan(faster, p.mortgage_balance, today)
        saved_months = regular["months"] - faster["months"]
        answer["saved"] = {"months": saved_months, "years": float((Decimal(saved_months) / TWELVE).quantize(CENT)), "interest": _money(regular["total_interest"] - faster["total_interest"])}
    return answer
