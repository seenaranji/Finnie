"""Step 16: goal planning math. The plan, a hand check, 5 vs 10 vs 20 years, sensitivity tables, edge cases.

Run:  .venv/bin/python -m scripts.step16_goal_demo
"""
import json
import time
from datetime import datetime
from pathlib import Path

from src.agents import goal_calc as gc
from src.core.config import load_settings

config = load_settings().goal_planning
report: dict = {}

print("=" * 100, "\nA. THE PLAN: $50,000 in 5 years, $2,000 saved already, medium risk\n" + "=" * 100)
plan = gc.plan_goal(50_000, 5, 2_000, "medium", config)
print(f"  {plan.assumptions}")
print(f"  inflation-adjusted goal: $50,000 today-dollars needs ${plan.inflation_adjusted_goal:,.2f} in 5 years\n")
print(f"  {'scenario':13}{'return':>8}{'save / month':>14}{'(inflation-adj.)':>18}{'contributed':>13}{'from returns':>14}{'savings grow to':>17}")
for s in plan.scenarios:
    print(f"  {s.scenario:13}{s.annual_return_pct:>7.1f}%{s.monthly_needed:>14,.2f}{s.monthly_needed_inflation_adjusted:>18,.2f}{s.total_contributed:>13,.0f}{s.growth_from_returns:>14,.0f}{s.current_savings_grow_to:>17,.0f}")
report["plan"] = {"goal": plan.goal, "scenarios": [vars(s) for s in plan.scenarios]}

print("\n" + "=" * 100, "\nB. HAND CHECK of the expected scenario (5% a year), step by step\n" + "=" * 100)
r, P, G, years = 0.05, 2000.0, 50000.0, 5
i, N = (1 + r) ** (1 / 12) - 1, 60
growth = (1 + i) ** N
print(f"  monthly rate  i = 1.05^(1/12) - 1            = {i:.8f}")
print(f"  growth factor (1+i)^60 = 1.05^5              = {growth:.7f}")
print(f"  current savings grow to  2000 x {growth:.7f} = {P * growth:,.2f}")
print(f"  still missing  50000 - {P * growth:,.2f}          = {G - P * growth:,.2f}")
print(f"  annuity factor ((1+i)^60 - 1) / i            = {(growth - 1) / i:.4f}")
print(f"  monthly saving = {G - P * growth:,.2f} / {(growth - 1) / i:.4f}        = {(G - P * growth) / ((growth - 1) / i):,.4f}   (program: {plan.scenarios[1].monthly_needed:,.4f})")
path = gc.balance_path(P, plan.scenarios[1].monthly_needed, 5.0, 5)
print(f"  month-by-month simulation, year-end balances: {[round(b) for b in path]}   (ends at the goal: {path[-1]:,.2f})")
nominal_i = r / 12
nominal_growth = (1 + nominal_i) ** N
nominal_c = (G - P * nominal_growth) / ((nominal_growth - 1) / nominal_i)
print(f"\n  CONVENTION MATTERS: a calculator that uses 5%/12 per month instead of an effective 5% a year gets ${nominal_c:,.2f} a month,")
print(f"  ${plan.scenarios[1].monthly_needed - nominal_c:+.2f} different from ours, because 5%/12 compounds to {((1 + nominal_i) ** 12 - 1) * 100:.3f}% a year, not 5%.")

print("\n" + "=" * 100, "\nC. 5 vs 10 vs 20 YEARS (goal $50,000, $2,000 saved, medium risk)\n" + "=" * 100)
print(f"  {'years':>6}{'conservative':>14}{'expected':>12}{'optimistic':>12}   {'expected: contributed':>22}{'from returns':>14}")
horizons = {}
for y in (5, 10, 20):
    p = gc.plan_goal(50_000, y, 2_000, "medium", config)
    horizons[y] = [s.monthly_needed for s in p.scenarios]
    e = p.scenarios[1]
    print(f"  {y:>6}{p.scenarios[0].monthly_needed:>14,.2f}{e.monthly_needed:>12,.2f}{p.scenarios[2].monthly_needed:>12,.2f}   {e.total_contributed:>22,.0f}{e.growth_from_returns:>14,.0f}")
report["horizons"] = horizons

print("\n" + "=" * 100, "\nD. SENSITIVITY: monthly saving needed for $50,000 with $2,000 saved\n" + "=" * 100)
returns = [0, 2, 3, 5, 8, 11]
table = gc.sensitivity_table(50_000, 2_000, [3, 5, 10, 15, 20, 30], returns)
corner = "years / return"
print(f"  {corner:>16}" + "".join(f"{r:>9}%" for r in returns))
for y, row in table.items():
    print(f"  {y:>16}" + "".join(f"{row[r]:>10,.0f}" for r in returns))
print("\n  Inflation effect (5% return, 5 years, goal $50,000 in today's purchasing power):")
print(f"  {'inflation':>10}{'target in future $':>20}{'save / month':>14}{'extra vs stated goal':>22}")
base = gc.required_monthly(50_000, 2_000, 5.0, 5)
for inflation in (0, 2, 3, 5, 8):
    target = gc.inflation_adjusted_target(50_000, inflation, 5)
    c = gc.required_monthly(target, 2_000, 5.0, 5)
    print(f"  {inflation:>9}%{target:>20,.0f}{c:>14,.2f}{c - base:>+22,.2f}")
print(f"  real return at 5% nominal and 3% inflation: {gc.real_return_pct(5.0, 3.0):.3f}%")
report["sensitivity"] = {str(y): {str(r): v for r, v in row.items()} for y, row in table.items()}

print("\n" + "=" * 100, "\nE. EDGE CASES\n" + "=" * 100)
cases = [("0% return, 5 years", lambda: gc.required_monthly(50_000, 2_000, 0, 5)),
         ("goal already reached ($60k saved)", lambda: gc.required_monthly(50_000, 60_000, 5, 5)),
         ("0 years, goal already reached", lambda: gc.required_monthly(50_000, 50_000, 5, 0)),
         ("0 years, goal NOT reached", lambda: gc.required_monthly(50_000, 2_000, 5, 0)),
         ("negative goal", lambda: gc.plan_goal(-5, 5, 0, "medium", config)),
         ("negative savings", lambda: gc.plan_goal(1000, 5, -1, "medium", config)),
         ("negative years", lambda: gc.plan_goal(1000, -2, 0, "medium", config)),
         ("100 years", lambda: gc.plan_goal(1000, 100, 0, "medium", config)),
         ("risk profile 'yolo'", lambda: gc.plan_goal(1000, 5, 0, "yolo", config)),
         ("negative return (-3%), 5 years", lambda: gc.required_monthly(50_000, 2_000, -3, 5))]
for label, call in cases:
    try:
        result = call()
        shown = f"${result:,.2f} a month" if isinstance(result, float) else "ok"
    except gc.GoalInputError as error:
        shown = f"GoalInputError: {error}"
    print(f"  {label:36} -> {shown}")

start = time.perf_counter()
for _ in range(1000):
    gc.plan_goal(50_000, 20, 2_000, "medium", config)
print(f"\n  compute time: {(time.perf_counter() - start):.3f} ms per full 3-scenario plan (mean of 1000)")

output_file = Path(f"benchmarks/results/step16_goal_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(report, indent=2, default=str))
print(f"\nsaved: {output_file}")
