"""Answers "how long until my mortgage is paid off?" in chat. No model: the question is recognized by rules, the numbers come from src/agents/mortgage.py, and the
wording is fixed text, so every figure in the reply is one the calculator produced. Only runs when the person asks; nothing is shown otherwise."""
import difflib
import re
from decimal import Decimal

from src.agents.mortgage import mortgage_payoff
from src.core.guardrails import DISCLAIMER_STRONG
from src.core.schemas import AgentResponse
from src.workflow.nodes import fixed_response

MORTGAGE = re.compile(r"\b(mortgages?|home loans?|amorti[sz]\w*)\b", re.I)
PAYOFF = re.compile(r"(pay(?:ing)?\s+(?:it\s+|this\s+|that\s+|my\s+\w+\s+)?off|payoff|pay-off|close|closing|clear|how long|how many (?:years|months)|how soon|years? (?:left|to)|"
                    r"extra|additional|amorti[sz]\w*|sooner|earlier|faster|finish)", re.I)
PERSONAL = re.compile(r"\b(my|our|mine|i|i'd|i'm|we)\b|amorti", re.I)
NUMBER = r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?"
EXTRA_PATTERNS = (re.compile(r"(?:extra|additional|more|add(?:ing)?|plus)\s*(?:of\s*)?" + NUMBER, re.I),
                  re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?\s*(?:extra|more|additional|a month|per month|/\s*month|monthly|on top)", re.I),
                  re.compile(r"pay(?:ing)?\s*\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", re.I),
                  re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", re.I))
WANTS_EXTRA = re.compile(r"\b(extra|additional|more|sooner|earlier|faster|on top)\b", re.I)


STREET = r"(?:road|rd|street|st|avenue|ave|av|lane|ln|drive|dr|court|ct|way|boulevard|blvd|place|pl|circle|cir|terrace|ter|highway|hwy)"
ADDRESS = re.compile(r"\b\d+\s+(?:[a-z']+\s+){0,3}" + STREET + r"\b", re.I)
# What a short follow-up to a mortgage answer sounds like: it names a property or an address, or asks about paying off / paying extra. "What about my tax?" is NOT one.
FOLLOW_UP = re.compile(r"\b(real estate|propert(?:y|ies)|house|houses|condo|apartment|rental|address|the other one|that one|this one|both of them|all of them|extra|additional|sooner|earlier|faster|"
                       r"pay(?:ing)?\s+off|payoff|how long|how many (?:years|months)|" + STREET + r")\b", re.I)
ABBREVIATIONS = {"rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "ln": "lane", "dr": "drive", "ct": "court", "pl": "place", "blvd": "boulevard", "cir": "circle", "ter": "terrace", "hwy": "highway"}
GENERIC = {"house", "home", "property", "rental", "condo", "apartment", "the", "my", "at", "on", "of"}   # words that do not tell one property from another


def is_mortgage_payoff_question(message: str) -> bool:
    return bool(MORTGAGE.search(message) and PAYOFF.search(message) and PERSONAL.search(message))


def is_mortgage_follow_up(message: str) -> bool:
    """A short question that continues a mortgage conversation without saying 'mortgage' ("what about my 123 Opossum Road property?", "and if I pay $500 extra?")."""
    return bool(FOLLOW_UP.search(message) or ADDRESS.search(message) or extra_amount(message))


def _words(text: str) -> list[str]:
    return [ABBREVIATIONS.get(w, w) for w in re.findall(r"[a-z0-9]+", text.lower())]


def _alike(a: str, b: str) -> bool:
    return a == b or (a.isalpha() and b.isalpha() and len(a) >= 4 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8)   # 'opposum' still finds 'Opossum'


def mentions(name: str, message: str) -> bool:
    """Does the message talk about this property? Tolerant of 'Rd' / 'road', of capitals and of a small typo; house numbers must match exactly."""
    wanted = [w for w in _words(name) if w not in GENERIC]
    said = _words(message)
    if not wanted:
        return False
    found = [w for w in wanted if any(_alike(w, other) for other in said)]
    return len(found) / len(wanted) >= 0.6 and all(w in found for w in wanted if w.isdigit())


def extra_amount(message: str) -> Decimal | None:
    """The extra monthly payment the person mentioned ("$500 extra", "an additional 1,000 a month", "pay $2k more"), or None."""
    for pattern in EXTRA_PATTERNS:
        found = pattern.search(message)
        if found:
            amount = Decimal(found.group(1).replace(",", "")) * (1000 if found.group(2) else 1)
            if amount > 0:
                return amount
    return None


def _money(x: float) -> str:
    return f"${x:,.0f}" if float(x).is_integer() else f"${x:,.2f}"


def _span(plan: dict) -> str:
    years, months = divmod(plan["months"], 12)
    length = " ".join(part for part in (f"{years} year{'s' if years != 1 else ''}" if years else "", f"{months} month{'s' if months != 1 else ''}" if months else "") if part)
    return f"{length}, paid off around {plan['payoff_date'][:7]}, with {_money(plan['total_interest'])} total interest"


def describe(answer: dict) -> str:
    head = f"**{answer['name']}**: {_money(answer['balance'])} owed, {_money(answer['monthly_payment'])} a month at {answer['rate_pct']:g}%."
    status = answer["status"]
    if status == "missing_payment":
        return f"**{answer['name']}**: you haven't entered a monthly payment or rate yet. Add them under Portfolio → My saved portfolios → Mortgage payoff calculator, then ask again."
    if status == "never_paid_off":
        return f"{head}\n- At this payment the balance never goes down: the first month's interest alone is about {_money(answer['monthly_interest'])}. Please check the payment and rate you entered."
    lines = [head, f"- Paying the regular amount: {_span(answer['regular'])}."]
    if answer["with_extra"]:
        saved = answer["saved"]
        lines += [f"- Paying {_money(answer['extra'])} more every month: {_span(answer['with_extra'])}.",
                  f"- That is {_span_short(saved['months'])} sooner and {_money(saved['interest'])} less interest."]
    return "\n".join(lines)


def _span_short(months: int) -> str:
    years, rest = divmod(months, 12)
    return " ".join(part for part in (f"{years} year{'s' if years != 1 else ''}" if years else "", f"{rest} month{'s' if rest != 1 else ''}" if rest else "") if part) or "0 months"


class MortgagePayoffChat:
    name = "mortgage_payoff"

    def __init__(self, repository_getter) -> None:
        self.repository_getter = repository_getter   # called only when a question arrives, so the app starts even if the database is down

    def run(self, message: str, portfolio_id, owner_id, earlier: str | None = None) -> AgentResponse:
        """Answers for EVERY property with a mortgage in ALL of the person's saved portfolios (or just the ones the message names). `earlier` is the person's previous
        mortgage question in this conversation: an extra monthly amount mentioned there still applies to a follow-up that does not repeat it."""
        if not owner_id:
            return fixed_response("To work out your mortgage payoff I need to know who you are. Please log in and ask again.", self.name, "needs the owner", DISCLAIMER_STRONG)
        repository = self.repository_getter()
        portfolios = [repository.get(summary.id, owner_id=owner_id) for summary in repository.list(owner_id=owner_id)]
        every = [(portfolio.name, p) for portfolio in portfolios for p in portfolio.properties]
        owing = [(name, p) for name, p in every if p.mortgage_balance > 0]
        if not owing:
            where = "your saved portfolios have" if len(portfolios) != 1 else f"'{portfolios[0].name}' has"
            return fixed_response(f"None of {where} a property with a mortgage balance, so there is nothing to pay off." if len(portfolios) != 1
                                  else f"'{portfolios[0].name}' has no property with a mortgage balance, so there is nothing to pay off.", self.name, "no mortgage", DISCLAIMER_STRONG)
        named = [(name, p) for name, p in every if mentions(p.name, message)]
        if named and not any(p.mortgage_balance > 0 for _, p in named):
            return fixed_response("; ".join(f"**{p.name}** (in {name}) has no mortgage balance saved, so there is nothing to pay off." for name, p in named), self.name, "no mortgage", DISCLAIMER_STRONG)
        chosen = [(name, p) for name, p in named if p.mortgage_balance > 0] or owing
        extra = extra_amount(message) or (extra_amount(earlier) if earlier else None) or Decimal(0)
        several = len({name for name, _ in owing}) > 1 or len(owing) > 1
        parts = []
        if not named and ADDRESS.search(message):
            parts.append("I couldn't find a property matching that in your saved portfolios, so here are all of your mortgages:")
        elif not named and len(owing) > 1:
            parts.append(f"You have {len(owing)} properties with a mortgage across your saved portfolios:")
        for name, p in chosen:
            answer = mortgage_payoff(p, extra)
            if several:
                answer["name"] = f"{p.name} (in {name})"
            parts.append(describe(answer))
        text = "\n\n".join(parts)
        if extra == 0 and WANTS_EXTRA.search(message):
            text += "\n\nTell me how much extra you would pay each month (for example, \"what if I pay $500 extra?\") and I'll show how much sooner it ends."
        text += "\n\nThis assumes a fixed rate and a payment that is principal and interest only. Your lender's statement is the final word."
        return fixed_response(text, self.name, "calculated, no model", DISCLAIMER_STRONG)
