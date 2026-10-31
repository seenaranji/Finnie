import random

import pytest
from pydantic import ValidationError

from src.agents import goal_calc as gc
from src.core.config import GoalPlanningConfig, ScenarioReturns, load_settings

CONFIG = GoalPlanningConfig(inflation_pct=3.0, risk_profiles={
    "medium": ScenarioReturns(conservative=2.0, expected=5.0, optimistic=8.0)})


# ---------- known answers ----------
def test_a_lump_sum_compounds_by_the_effective_annual_rate():
    assert gc.future_value(1000, 0, 10, 1) == pytest.approx(1100.0)     # 1000 * 1.10
    assert gc.future_value(1000, 0, 10, 2) == pytest.approx(1210.0)     # 1000 * 1.10^2 (compounding on the interest)


def test_twelve_monthly_steps_equal_exactly_one_year_of_the_annual_rate():
    assert (1 + gc.monthly_rate(7.0)) ** 12 == pytest.approx(1.07)


def test_textbook_annuity_factor():
    # 1% a month for 12 months: the future value factor of $1 saved monthly is 12.6825 (a standard table value)
    annual = ((1.01) ** 12 - 1) * 100
    assert gc.future_value(0, 100, annual, 1) == pytest.approx(1268.25, abs=0.01)


def test_zero_return_is_plain_arithmetic():
    assert gc.future_value(1000, 100, 0, 5) == 1000 + 100 * 60
    assert gc.required_monthly(7000, 1000, 0, 5) == pytest.approx(100.0)   # (7000 - 1000) / 60


def test_hand_calculated_plan_5_years_medium_expected():
    # P=2000, goal=50000, 5% a year, 5 years. growth = 1.05^5 = 1.2762816; i = 1.05^(1/12) - 1 = 0.00407412
    # C = (50000 - 2000 * 1.2762816) * i / (growth - 1) = 47447.437 * 0.00407412 / 0.2762816 = 699.67
    # (checked to 12 digits with 40-digit Decimal arithmetic AND a bisection over a month-by-month simulation: 699.6729325562)
    assert gc.required_monthly(50000, 2000, 5.0, 5) == pytest.approx(699.6729, abs=0.001)


def test_inflation_and_real_return():
    assert gc.inflation_adjusted_target(1000, 3.0, 2) == pytest.approx(1060.9)
    assert gc.real_return_pct(5.0, 3.0) == pytest.approx(1.9417, abs=1e-3)
    assert gc.real_return_pct(3.0, 3.0) == pytest.approx(0.0, abs=1e-12)


# ---------- properties ----------
def test_saving_the_required_amount_reaches_the_goal_for_many_random_inputs():
    rng = random.Random(0)
    for _ in range(200):
        goal, principal = rng.uniform(1_000, 500_000), rng.uniform(0, 100_000)
        rate, years = rng.uniform(-5, 15), rng.randint(1, 40)
        c = gc.required_monthly(goal, principal, rate, years)
        assert gc.future_value(principal, c, rate, years) == pytest.approx(goal, rel=1e-9) if c > 0 else gc.future_value(principal, 0, rate, years) >= goal


def test_closed_formula_matches_a_month_by_month_simulation():
    path = gc.balance_path(2000, 699.6729, 5.0, 5)
    assert path[0] == 2000 and len(path) == 6
    assert path[-1] == pytest.approx(gc.future_value(2000, 699.6729, 5.0, 5), rel=1e-12)
    assert path[-1] == pytest.approx(50000, abs=0.01)   # saving the required amount lands on the goal


def test_more_return_or_more_time_means_less_to_save_each_month():
    assert gc.required_monthly(50000, 2000, 3, 10) > gc.required_monthly(50000, 2000, 6, 10)
    assert gc.required_monthly(50000, 2000, 5, 5) > gc.required_monthly(50000, 2000, 5, 10) > gc.required_monthly(50000, 2000, 5, 20)


# ---------- edge cases ----------
def test_goal_already_reached_needs_nothing():
    assert gc.required_monthly(50000, 60000, 5, 5) == 0.0
    assert gc.required_monthly(50000, 50000, 5, 0) == 0.0   # reached, and no time left: still fine


def test_savings_alone_will_reach_the_goal_flag():
    plan = gc.plan_goal(10_000, 20, 9_000, "medium", CONFIG)   # 9000 at 2% for 20 years is already ~13,371
    assert all(s.monthly_needed == 0.0 and s.already_reached for s in plan.scenarios)


def test_no_time_left_and_goal_not_reached_is_a_clear_error():
    with pytest.raises(gc.GoalInputError, match="no time left"):
        gc.required_monthly(50000, 2000, 5, 0)


@pytest.mark.parametrize("goal,years,savings", [(0, 5, 0), (-5, 5, 0), (1000, -1, 0), (1000, 5, -1), (1000, 99, 0), (2e9, 5, 0)])
def test_invalid_inputs_are_rejected_with_a_message(goal, years, savings):
    with pytest.raises(gc.GoalInputError):
        gc.plan_goal(goal, years, savings, "medium", CONFIG)


def test_unknown_risk_profile_lists_the_valid_ones():
    with pytest.raises(gc.GoalInputError, match="medium"):
        gc.plan_goal(1000, 5, 0, "yolo", CONFIG)


def test_a_return_of_minus_100_or_worse_is_rejected():
    with pytest.raises(gc.GoalInputError):
        gc.monthly_rate(-100)


# ---------- the plan ----------
def test_plan_has_three_ordered_scenarios_and_the_assumptions_are_stated():
    plan = gc.plan_goal(50000, 5, 2000, "Medium ", CONFIG)
    needed = [s.monthly_needed for s in plan.scenarios]
    assert [s.scenario for s in plan.scenarios] == ["conservative", "expected", "optimistic"]
    assert needed[0] > needed[1] > needed[2]                                   # worse markets mean more saving
    assert plan.scenarios[1].monthly_needed == pytest.approx(699.6729, abs=0.001)
    assert plan.inflation_adjusted_goal == pytest.approx(50000 * 1.03 ** 5)
    assert all(s.monthly_needed_inflation_adjusted > s.monthly_needed for s in plan.scenarios)
    assert "not forecasts" in plan.assumptions
    s = plan.scenarios[1]
    assert s.total_contributed + s.growth_from_returns + plan.current_savings == pytest.approx(plan.goal)


# ---------- config ----------
def test_scenario_returns_must_be_ordered_and_sane():
    with pytest.raises(ValidationError):
        ScenarioReturns(conservative=5, expected=3, optimistic=8)
    with pytest.raises(ValidationError):
        ScenarioReturns(conservative=1, expected=3, optimistic=80)


def test_the_real_config_loads_all_three_profiles():
    assert set(load_settings().goal_planning.risk_profiles) == {"low", "medium", "high"}
