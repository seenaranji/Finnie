"""Grouping a portfolio by the kind of holding. Every number below was worked out by hand first."""
from datetime import date
from decimal import Decimal as D
from typing import get_args

import pytest

from src.agents.portfolio_summary import group_by_type
from src.data.asset_types import ASSET_TYPES, PLURAL, AssetType
from src.data.portfolio_repository import StoredHolding as H


def holdings():
    return [H(ticker="VTI", shares=D(10), avg_cost=D(200), asset_type="etf"),       # cost 2000
            H(ticker="AAPL", shares=D(5), avg_cost=D(100), asset_type="stock"),     # cost 500
            H(ticker="MSFT", shares=D(2), avg_cost=D(300), asset_type="stock"),     # cost 600
            H(ticker="BND", shares=D(20), avg_cost=D(70), asset_type="bond")]       # cost 1400


PRICES = {"VTI": 250, "AAPL": 150, "MSFT": 400, "BND": 72}   # values 2500, 750, 800, 1440


def by_type(result):
    return {g["asset_type"]: g for g in result["groups"]}


def test_group_totals_gain_and_weight():
    r = group_by_type(holdings(), PRICES)
    assert (r["total_cost"], r["total_value"], r["holding_count"], r["unpriced"]) == (4500.0, 5490.0, 4, [])
    assert r["gain_pct"] == 22.0                                        # 5490 / 4500 - 1
    g = by_type(r)
    assert (g["stock"]["cost"], g["stock"]["value"], g["stock"]["count"]) == (1100.0, 1550.0, 2)
    assert g["stock"]["gain_pct"] == 40.91                              # 1550 / 1100 - 1
    assert (g["etf"]["value"], g["etf"]["weight_pct"], g["bond"]["weight_pct"], g["stock"]["weight_pct"]) == (2500.0, 45.54, 26.23, 28.23)   # 2500, 1440, 1550 of 5490
    assert g["bond"]["gain_pct"] == 2.86                                # 1440 / 1400 - 1


def test_weights_of_the_groups_add_up_to_100():
    r = group_by_type(holdings(), PRICES)
    assert sum(g["weight_pct"] for g in r["groups"]) == pytest.approx(100, abs=0.02)
    assert sum(h["weight_pct"] for g in r["groups"] for h in g["holdings"]) == pytest.approx(100, abs=0.05)


def test_groups_are_ordered_biggest_first_and_empty_kinds_are_left_out():
    r = group_by_type(holdings(), PRICES)
    assert [g["asset_type"] for g in r["groups"]] == ["etf", "stock", "bond"]
    assert {"reit", "cash", "crypto", "mutual_fund", "other"}.isdisjoint(g["asset_type"] for g in r["groups"])


def test_holdings_inside_a_group_are_ordered_by_value_with_labels():
    stocks = by_type(group_by_type(holdings(), PRICES))["stock"]
    assert [h["ticker"] for h in stocks["holdings"]] == ["MSFT", "AAPL"] and stocks["label"] == "Stocks"   # 800 before 750
    assert stocks["holdings"][0]["gain_pct"] == 33.33 and stocks["holdings"][0]["price"] == 400.0


def test_an_unpriced_holding_keeps_its_cost_but_has_no_value_and_does_not_distort_the_gain():
    r = group_by_type(holdings(), {"VTI": 250, "AAPL": 150, "MSFT": 400, "BND": None})
    assert r["unpriced"] == ["BND"] and r["total_cost"] == 4500.0 and r["total_value"] == 4050.0
    assert r["gain_pct"] == 30.65                                       # 4050 / 3100 - 1: the unpriced bond's 1400 of cost is left out of BOTH sides
    bond = by_type(r)["bond"]
    assert (bond["value"], bond["gain_pct"], bond["weight_pct"], bond["unpriced_count"], bond["cost"]) == (0.0, None, None, 1, 1400.0)
    assert bond["holdings"][0]["value"] is None and bond["holdings"][0]["weight_pct"] is None


def test_a_price_missing_from_the_dictionary_counts_as_unpriced():
    r = group_by_type(holdings(), {"VTI": 250})
    assert sorted(r["unpriced"]) == ["AAPL", "BND", "MSFT"] and r["total_value"] == 2500.0 and r["gain_pct"] == 25.0


def test_nothing_priced_at_all_gives_no_percentages_instead_of_dividing_by_zero():
    r = group_by_type(holdings(), {})
    assert r["total_value"] == 0.0 and r["gain_pct"] is None and all(g["weight_pct"] is None for g in r["groups"])


def test_money_is_exact_decimal_arithmetic():
    r = group_by_type([H(ticker="X", shares=D("0.1"), avg_cost=D("0.2"), asset_type="stock")], {"X": "0.3"})
    assert r["total_cost"] == 0.02 and r["total_value"] == 0.03          # floats would give 0.020000000000000004


def test_cash_priced_at_one_is_just_a_holding_with_gain_zero():
    r = group_by_type([H(ticker="CASH", shares=D(1500), avg_cost=D(1), asset_type="cash")], {"CASH": 1})
    assert by_type(r)["cash"]["value"] == 1500.0 and by_type(r)["cash"]["gain_pct"] == 0.0 and by_type(r)["cash"]["label"] == "Cash"


def test_purchase_dates_pass_through():
    r = group_by_type([H(ticker="X", shares=D(1), avg_cost=D(1), asset_type="stock", purchase_date=date(2024, 5, 1))], {"X": 2})
    assert r["groups"][0]["holdings"][0]["purchase_date"] == date(2024, 5, 1)


def test_an_empty_portfolio_is_not_a_crash():
    r = group_by_type([], {})
    assert r["groups"] == [] and r["holding_count"] == 0 and r["total_value"] == 0.0


def test_the_type_lists_agree_everywhere():
    assert set(get_args(AssetType)) == set(ASSET_TYPES) == set(PLURAL)
