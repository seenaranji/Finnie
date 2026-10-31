import math

import numpy as np
import pytest

from src.agents import portfolio_calc as pc

META = {"A": {"asset_class": "US stock", "sector": "Tech"}, "B": {"asset_class": "Bond", "sector": "Bonds"}}


# ---------- parsing ----------
def test_valid_csv_is_parsed_and_duplicates_are_merged_with_a_share_weighted_cost():
    holdings = pc.parse_portfolio_csv("ticker,shares,avg_cost\nvti, 10, 100\nBND,5,70\nVTI,30,200\n")
    assert [(h.ticker, h.shares, h.avg_cost) for h in holdings] == [("VTI", 40.0, 175.0), ("BND", 5.0, 70.0)]   # (10*100 + 30*200) / 40


def test_every_problem_is_reported_at_once():
    text = "ticker,shares,avg_cost\n,5,10\nAAPL,-3,10\nMSFT,abc,10\nVTI,5,-1\nBND,2,70\n"
    with pytest.raises(pc.PortfolioInputError) as error:
        pc.parse_portfolio_csv(text)
    assert len(error.value.problems) == 4 and not any("BND" in p for p in error.value.problems)


@pytest.mark.parametrize("text", ["", "ticker,shares\nAAPL,5", "ticker,shares,avg_cost\n"])
def test_missing_columns_and_empty_files_are_rejected(text):
    with pytest.raises(pc.PortfolioInputError):
        pc.parse_portfolio_csv(text)


# ---------- weights, HHI, breakdowns (hand-calculated) ----------
def test_weights_hhi_and_effective_holdings():
    holdings = [pc.Holding("A", 10, 50.0), pc.Holding("B", 10, 50.0)]
    positions, unpriced = pc.build_positions(holdings, {"A": 75.0, "B": 25.0})   # values 750 / 250
    assert [round(p.weight, 6) for p in positions] == [0.75, 0.25] and unpriced == []
    assert pc.hhi([p.weight for p in positions]) == pytest.approx(0.625)         # 0.75^2 + 0.25^2
    assert pc.effective_holdings([0.75, 0.25]) == pytest.approx(1.6)
    assert positions[0].gain_pct == pytest.approx(50.0)                          # bought at 50, now 75


def test_hhi_extremes():
    assert pc.hhi([1.0]) == 1.0
    assert pc.hhi([0.25] * 4) == pytest.approx(0.25)
    assert pc.hhi([0.1] * 10) == pytest.approx(0.1)


def test_a_holding_without_a_price_is_left_out_not_guessed():
    positions, unpriced = pc.build_positions([pc.Holding("A", 1, 1), pc.Holding("B", 1, 1)], {"A": 10.0})
    assert unpriced == ["B"] and len(positions) == 1 and positions[0].weight == 1.0


def test_breakdown_sums_weights_per_category_and_reports_unknown():
    positions, _ = pc.build_positions([pc.Holding("A", 1, 1), pc.Holding("B", 1, 1), pc.Holding("Z", 1, 1)], {"A": 50.0, "B": 30.0, "Z": 20.0})
    assert pc.breakdown(positions, META, "asset_class") == pytest.approx({"US stock": 0.5, "Bond": 0.3, "Unknown": 0.2})


# ---------- volatility and correlation (hand-calculated) ----------
RNG = np.random.default_rng(0)
BASE = RNG.normal(0, 0.01, 200)


def test_perfectly_correlated_assets_do_not_diversify():
    returns = np.column_stack([BASE, 2 * BASE])   # second asset moves exactly twice as much
    vols = pc.asset_volatility_pct(returns)
    assert vols[1] == pytest.approx(2 * vols[0])
    assert pc.portfolio_volatility_pct(np.array([0.5, 0.5]), returns) == pytest.approx(0.5 * vols[0] + 0.5 * vols[1])   # = weighted average, no benefit
    assert pc.correlation_matrix(returns)[0, 1] == pytest.approx(1.0)


def test_perfectly_negative_correlation_cancels_out():
    returns = np.column_stack([BASE, -BASE])
    assert pc.portfolio_volatility_pct(np.array([0.5, 0.5]), returns) == pytest.approx(0.0, abs=1e-9)
    assert pc.correlation_matrix(returns)[0, 1] == pytest.approx(-1.0)


def test_volatility_of_a_single_asset_is_std_times_sqrt_252():
    returns = BASE.reshape(-1, 1)
    assert pc.portfolio_volatility_pct(np.array([1.0]), returns) == pytest.approx(float(np.std(BASE, ddof=1)) * math.sqrt(252) * 100)


def test_returns_use_only_shared_dates_and_need_enough_of_them():
    dates = [f"2026-01-{d:02d}" for d in range(1, 29)] + [f"2026-02-{d:02d}" for d in range(1, 11)]   # 38 dates
    a = [(d, 100.0 + i) for i, d in enumerate(dates)]
    b = [(d, 50.0) for d in dates[5:]]                                                              # starts later
    tickers, returns = pc.returns_matrix({"A": a, "B": b})
    assert returns.shape == (len(dates) - 5 - 1, 2) and returns[0, 1] == 0.0
    assert pc.returns_matrix({"A": a[:10], "B": b[:10]}) is None   # too few overlapping days


def test_weighted_average_correlation():
    corr = np.array([[1, 0.9, 0.0], [0.9, 1, 0.0], [0.0, 0.0, 1]])
    assert pc.weighted_avg_correlation(np.array([0.5, 0.5, 0.0]), corr) == pytest.approx(0.9)
    assert pc.weighted_avg_correlation(np.array([1.0]), np.array([[1.0]])) == 1.0   # a single holding: nothing to diversify with
    assert pc.correlated_pairs(["A", "B", "C"], corr) == [("A", "B", pytest.approx(0.9))]


# ---------- the documented score ----------
def test_score_floor_for_one_holding_is_11_5_not_0_and_the_ideal_case_is_100():
    # one holding: concentration 1/10, spread 1/4, correlation term 0 -> 100 * (0.4*0.1 + 0.3*0.25) = 11.5
    assert pc.diversification_score([1.0], [1.0], 1.0)["score"] == pytest.approx(11.5)
    ideal = pc.diversification_score([0.1] * 10, [0.25] * 4, 0.0)
    assert ideal["score"] == 100.0


def test_concentrated_scores_lower_than_spread():
    spread = pc.diversification_score([0.2] * 5, [0.4, 0.3, 0.2, 0.1], 0.3)["score"]
    concentrated = pc.diversification_score([0.9, 0.05, 0.05], [0.95, 0.05], 0.8)["score"]
    assert concentrated < spread


# ---------- analyze() end to end on synthetic data ----------
def test_analyze_reports_notes_when_history_is_missing():
    holdings = [pc.Holding("A", 10, 50.0), pc.Holding("B", 10, 50.0), pc.Holding("C", 10, 50.0)]
    result = pc.analyze(holdings, {"A": 100.0, "B": 100.0}, {}, META)   # C has no price; nobody has history
    assert result["unpriced"] == ["C"] and result["volatility_pct"] is None
    assert any("no price for" in n for n in result["notes"]) and any("correlation unknown" in n for n in result["notes"])
    assert result["diversification"]["low_correlation"] == 0.0   # pessimistic assumption, stated
