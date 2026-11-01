"""Real estate arithmetic and the combined portfolio view. Every number was worked out by hand first (see the comments)."""
from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from src.agents.portfolio_summary import summarize_portfolio
from src.agents.real_estate import property_metrics, real_estate_totals
from src.data.asset_types import PROPERTY_TYPES, PropertyType
from src.data.portfolio_repository import PortfolioIn, PropertyIn, StoredHolding

from typing import get_args


def prop(name="Maple St", kind="house", price=300000, value=400000, mortgage=200000, rent=2500, expenses=800, **extra):
    return PropertyIn(name=name, property_type=kind, purchase_price=D(price), current_value=D(value), mortgage_balance=D(mortgage), monthly_rent=D(rent), monthly_expenses=D(expenses), **extra)


MAPLE = prop()                                                                  # equity 200,000; gain 100,000 (33.33%); rent 30,000 - costs 9,600 = NOI 20,400
OAK = prop("Oak condo", "condo", price=150000, value=140000, mortgage=90000, rent=1000, expenses=450)   # equity 50,000; gain -10,000 (-6.67%); NOI 12,000 - 5,400 = 6,600


def test_one_property_by_hand():
    m = property_metrics(MAPLE)
    assert (m["equity"], m["gain"], m["gain_pct"]) == (200000.0, 100000.0, 33.33)
    assert (m["annual_rent"], m["annual_expenses"], m["net_operating_income"]) == (30000.0, 9600.0, 20400.0)
    assert (m["gross_yield_pct"], m["net_yield_pct"], m["loan_to_value_pct"]) == (7.5, 5.1, 50.0)          # 30,000/400,000; 20,400/400,000; 200,000/400,000


def test_a_property_that_lost_value_shows_a_negative_gain():
    m = property_metrics(OAK)
    assert (m["equity"], m["gain"], m["gain_pct"], m["gross_yield_pct"], m["net_yield_pct"], m["loan_to_value_pct"]) == (50000.0, -10000.0, -6.67, 8.57, 4.71, 64.29)


def test_totals_use_the_total_value_not_an_average_of_yields():
    t = real_estate_totals([MAPLE, OAK])
    assert (t["count"], t["purchase_price"], t["value"], t["mortgage"], t["equity"], t["gain"], t["gain_pct"]) == (2, 450000.0, 540000.0, 290000.0, 250000.0, 90000.0, 20.0)
    assert (t["annual_rent"], t["annual_expenses"], t["net_operating_income"]) == (42000.0, 15000.0, 27000.0)
    assert (t["gross_yield_pct"], t["net_yield_pct"], t["loan_to_value_pct"]) == (7.78, 5.0, 53.7)         # 42,000/540,000; 27,000/540,000; 290,000/540,000
    assert (7.5 + 8.57) / 2 != t["gross_yield_pct"]                                                       # a plain average of the two yields (8.04) would be wrong


def test_no_properties_means_no_real_estate_section():
    assert real_estate_totals([]) is None


def test_a_mortgage_bigger_than_the_value_gives_negative_equity_and_a_loan_to_value_over_100():
    m = property_metrics(prop(value=100000, mortgage=130000, price=120000))
    assert (m["equity"], m["loan_to_value_pct"], m["gain"]) == (-30000.0, 130.0, -20000.0)


def test_zero_value_or_zero_price_gives_no_percentages_instead_of_dividing_by_zero():
    m = property_metrics(prop(price=0, value=0, mortgage=0, rent=500, expenses=100))
    assert m["gain_pct"] is None and m["gross_yield_pct"] is None and m["net_yield_pct"] is None and m["loan_to_value_pct"] is None and m["net_operating_income"] == 4800.0


def test_a_property_with_no_rent_has_zero_yield_and_negative_net_income_from_its_costs():
    m = property_metrics(prop(rent=0, expenses=300))
    assert (m["gross_yield_pct"], m["net_operating_income"], m["net_yield_pct"]) == (0.0, -3600.0, -0.9)    # a home you live in still costs money


def test_money_is_exact():
    m = property_metrics(prop(price="0.1", value="0.3", mortgage=0, rent=0, expenses=0))
    assert m["gain"] == 0.2                                                                                 # floats would give 0.19999999999999998


# ---- validation
@pytest.mark.parametrize("field", ["purchase_price", "current_value", "mortgage_balance", "monthly_rent", "monthly_expenses"])
def test_negative_money_is_refused(field):
    with pytest.raises(ValidationError):
        PropertyIn(**{"name": "x", "purchase_price": 1, "current_value": 1, field: -1})


def test_a_property_needs_a_name_a_price_and_a_value_and_the_future_is_refused():
    with pytest.raises(ValidationError):
        PropertyIn(name="  ", purchase_price=1, current_value=1)
    with pytest.raises(ValidationError):
        PropertyIn(name="x", current_value=1)
    with pytest.raises(ValidationError):
        prop(purchase_date="2999-01-01")
    with pytest.raises(ValidationError):
        prop(kind="castle")


def test_a_portfolio_may_hold_only_properties_but_not_nothing():
    assert PortfolioIn(name="rentals", properties=[MAPLE.model_dump()]).holdings == []
    with pytest.raises(ValidationError) as caught:
        PortfolioIn(name="empty")
    assert "at least one holding or one property" in str(caught.value)


def test_two_properties_with_one_name_are_refused_in_any_letter_case():
    with pytest.raises(ValidationError) as caught:
        PortfolioIn(name="x", properties=[prop("Maple St").model_dump(), prop("MAPLE ST").model_dump()])
    assert "same name" in str(caught.value)


def test_the_property_kinds_agree_everywhere():
    assert set(get_args(PropertyType)) == set(PROPERTY_TYPES)


# ---- the combined view
def test_real_estate_is_one_more_group_and_every_share_is_over_the_whole_portfolio():
    holdings = [StoredHolding(ticker="VTI", shares=D(10), avg_cost=D(200), asset_type="etf")]                # cost 2,000; value 10 x 250 = 2,500
    r = summarize_portfolio(holdings, [MAPLE, OAK], {"VTI": 250})
    assert (r["total_cost"], r["total_value"], r["holding_count"], r["property_count"]) == (452000.0, 542500.0, 3, 2)
    assert (r["mortgage_owed"], r["net_value"]) == (290000.0, 252500.0)                                      # 542,500 - 290,000
    assert r["gain_pct"] == 20.02                                                                           # 542,500 / 452,000 - 1
    groups = {g["asset_type"]: g for g in r["groups"]}
    assert [g["asset_type"] for g in r["groups"]] == ["real_estate", "etf"]                                  # biggest first
    assert (groups["real_estate"]["label"], groups["real_estate"]["value"], groups["real_estate"]["weight_pct"], groups["real_estate"]["gain_pct"]) == ("Real estate", 540000.0, 99.54, 20.0)
    assert (groups["etf"]["weight_pct"], groups["etf"]["holdings"][0]["weight_pct"]) == (0.46, 0.46)         # 2,500 / 542,500
    assert r["real_estate"]["equity"] == 250000.0 and len(r["real_estate"]["properties"]) == 2


def test_a_portfolio_of_only_properties_works_and_without_properties_nothing_changes():
    r = summarize_portfolio([], [MAPLE], {})
    assert (r["total_value"], r["holding_count"], r["unpriced"], [g["asset_type"] for g in r["groups"]]) == (400000.0, 1, [], ["real_estate"])
    plain = summarize_portfolio([StoredHolding(ticker="VTI", shares=D(1), avg_cost=D(1), asset_type="etf")], [], {"VTI": 2})
    assert plain["real_estate"] is None and plain["mortgage_owed"] == 0.0 and plain["net_value"] == plain["total_value"] == 2.0


def test_an_unpriced_holding_does_not_distort_the_gain_when_real_estate_is_included():
    holdings = [StoredHolding(ticker="VTI", shares=D(10), avg_cost=D(200), asset_type="etf"), StoredHolding(ticker="XYZ", shares=D(5), avg_cost=D(100), asset_type="stock")]
    r = summarize_portfolio(holdings, [MAPLE], {"VTI": 250, "XYZ": None})
    assert r["unpriced"] == ["XYZ"] and r["total_cost"] == 302500.0 and r["total_value"] == 402500.0   # XYZ's 500 of cost counts in total cost only
    assert r["gain_pct"] == 33.28                                                                       # 402,500 / (2,000 + 300,000) - 1 = 1.33278
