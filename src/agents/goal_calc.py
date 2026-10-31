"""Goal planning math (Step 16). No LLM, no network. Pure functions, testable against hand calculations.

Model (every simplification is deliberate and listed here):
  - Savings are invested once a month, at the END of each month (an "ordinary annuity"). Contributions stay flat (they do not grow with inflation).
  - The annual return is an EFFECTIVE annual rate: the monthly rate is i = (1 + r)^(1/12) - 1, so twelve months of growth equal exactly r.
    (A calculator that uses r/12 gives a slightly different answer; the difference is shown in scripts/step16_goal_demo.py.)
  - The return is constant. Real markets are not. This is why plan_goal returns three scenarios, never one number.
  - No taxes, fees or account limits.

Formulas (derived in the Step 16 lesson):
  future value  FV = P (1+i)^N + C ((1+i)^N - 1) / i            P = current savings, C = monthly contribution, N = months
  required      C  = (G - P (1+i)^N) * i / ((1+i)^N - 1)         G = goal
  at 0% return  FV = P + C N,   C = (G - P) / N                 (the formulas above divide by i = 0)
  inflation     inflation-adjusted target = G (1 + pi)^years     keeps the target's purchasing power
"""
from dataclasses import dataclass

from src.core.config import GoalPlanningConfig

MAX_YEARS = 60
MAX_AMOUNT = 1e9


class GoalInputError(ValueError):
    """Invalid goal inputs. The message is safe to show the user."""


# ---------- building blocks ----------
def monthly_rate(annual_return_pct: float) -> float:
    if annual_return_pct <= -100:
        raise GoalInputError("a yearly return of -100% or less is not possible")
    return (1 + annual_return_pct / 100) ** (1 / 12) - 1


def future_value(principal: float, monthly_contribution: float, annual_return_pct: float, years: float) -> float:
    n, i = round(years * 12), monthly_rate(annual_return_pct)
    if i == 0:
        return principal + monthly_contribution * n
    growth = (1 + i) ** n
    return principal * growth + monthly_contribution * (growth - 1) / i


def required_monthly(goal: float, principal: float, annual_return_pct: float, years: float) -> float:
    """Monthly amount that makes the balance reach `goal` after `years`. 0 if the current savings alone already get there."""
    n, i = round(years * 12), monthly_rate(annual_return_pct)
    if future_value(principal, 0, annual_return_pct, years) >= goal:
        return 0.0
    if n == 0:
        raise GoalInputError(f"there is no time left to save: ${goal - principal:,.2f} is still missing")
    if i == 0:
        return (goal - principal) / n
    growth = (1 + i) ** n
    return (goal - principal * growth) * i / (growth - 1)


def inflation_adjusted_target(goal: float, inflation_pct: float, years: float) -> float:
    """What the goal must be in future dollars to buy what `goal` buys today."""
    return goal * (1 + inflation_pct / 100) ** years


def real_return_pct(nominal_pct: float, inflation_pct: float) -> float:
    """Return after inflation (Fisher): (1 + nominal) / (1 + inflation) - 1."""
    return ((1 + nominal_pct / 100) / (1 + inflation_pct / 100) - 1) * 100


def balance_path(principal: float, monthly_contribution: float, annual_return_pct: float, years: int) -> list[float]:
    """Balance at the end of each year, starting with year 0. Month-by-month simulation (independent of the closed formula)."""
    i, balance, path = monthly_rate(annual_return_pct), principal, [principal]
    for month in range(1, years * 12 + 1):
        balance = balance * (1 + i) + monthly_contribution
        if month % 12 == 0:
            path.append(balance)
    return path


def _validate(goal: float, years: float, current_savings: float) -> None:
    if not 0 < goal <= MAX_AMOUNT:
        raise GoalInputError("the goal amount must be greater than 0 (and below one billion dollars)")
    if current_savings < 0 or current_savings > MAX_AMOUNT:
        raise GoalInputError("current savings cannot be negative")
    if years < 0 or years > MAX_YEARS:
        raise GoalInputError(f"the time horizon must be between 0 and {MAX_YEARS} years")


# ---------- the plan: three scenarios ----------
@dataclass(frozen=True)
class ScenarioPlan:
    scenario: str
    annual_return_pct: float
    monthly_needed: float                    # to reach the goal as stated (in future dollars)
    monthly_needed_inflation_adjusted: float # to reach the goal with today's purchasing power
    current_savings_grow_to: float           # what today's savings become with no new contributions
    total_contributed: float                 # monthly_needed * months
    growth_from_returns: float               # goal - current savings - contributions: the part markets are assumed to supply
    already_reached: bool


@dataclass(frozen=True)
class GoalPlan:
    goal: float
    years: float
    current_savings: float
    risk_profile: str
    inflation_pct: float
    inflation_adjusted_goal: float
    scenarios: list[ScenarioPlan]
    assumptions: str


def plan_goal(goal: float, years: float, current_savings: float, risk_profile: str, config: GoalPlanningConfig) -> GoalPlan:
    _validate(goal, years, current_savings)
    profile = config.risk_profiles.get(risk_profile.lower().strip())
    if profile is None:
        raise GoalInputError(f"risk profile must be one of: {', '.join(config.risk_profiles)}")
    adjusted_goal = inflation_adjusted_target(goal, config.inflation_pct, years)
    scenarios = []
    for name in ("conservative", "expected", "optimistic"):
        rate = getattr(profile, name)
        grows_to = future_value(current_savings, 0, rate, years)
        needed = required_monthly(goal, current_savings, rate, years)
        contributed = needed * round(years * 12)
        scenarios.append(ScenarioPlan(name, rate, needed, required_monthly(adjusted_goal, current_savings, rate, years), grows_to,
                                      contributed, goal - current_savings - contributed, grows_to >= goal))
    assumptions = (f"ASSUMPTIONS, not forecasts: yearly returns of {profile.conservative}% / {profile.expected}% / {profile.optimistic}% "
                   f"(conservative / expected / optimistic) for a {risk_profile} risk profile, {config.inflation_pct}% yearly inflation, "
                   f"flat monthly savings at the end of each month, no taxes or fees. Real returns vary and can be negative.")
    return GoalPlan(goal, years, current_savings, risk_profile.lower().strip(), config.inflation_pct, adjusted_goal, scenarios, assumptions)


def assess(plan: GoalPlan, large_monthly_usd: float) -> list[str]:
    """Plain-English flags about the plan itself, from the math only (we do not know the user's income or situation)."""
    flags = []
    expected = plan.scenarios[1]
    if all(s.already_reached for s in plan.scenarios):
        flags.append("Current savings alone are projected to reach this goal in all three scenarios, so no extra monthly saving is needed.")
    elif expected.monthly_needed > large_monthly_usd:
        flags.append(f"The expected scenario needs ${expected.monthly_needed:,.0f} a month, which is a very large commitment. "
                     f"A longer time horizon, a smaller goal or a bigger starting amount would lower it.")
    if plan.years < 1:
        flags.append("With less than a year to go, investment returns are very uncertain, so the real result could be far from these scenarios.")
    return flags


def sensitivity_table(goal: float, current_savings: float, years_list: list[float], returns_pct: list[float]) -> dict[float, dict[float, float]]:
    """Monthly amount needed for each (years, annual return) pair."""
    return {y: {r: required_monthly(goal, current_savings, r, y) for r in returns_pct} for y in years_list}
