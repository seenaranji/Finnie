import pytest

from src.data import stats


def test_change_pct():
    assert stats.change_pct(110, 100) == 10.0
    assert stats.change_pct(95, 100) == -5.0
    assert stats.change_pct(5, None) is None and stats.change_pct(5, 0) is None


def test_moving_average_uses_only_the_last_window_closes():
    assert stats.moving_average([1, 2, 3, 4, 5], 3) == 4.0   # (3 + 4 + 5) / 3
    assert stats.moving_average([1, 2], 3) is None            # not enough data: None, not a guess


def test_price_vs_average():
    assert stats.price_vs_average_pct(105, 100) == 5.0 and stats.price_vs_average_pct(105, None) is None


def test_range_52w_flags_a_short_history():
    closes = [float(x) for x in range(100, 160)]   # 60 bars
    result = stats.range_52w(closes)
    assert (result["low"], result["high"], result["bars"], result["covers_full_year"]) == (100.0, 159.0, 60, False)
    assert stats.range_52w([1.0] * 10) is None
    assert stats.range_52w([float(x) for x in range(300)])["covers_full_year"] is True


def test_volatility_is_zero_for_a_flat_price_and_needs_enough_data():
    assert stats.annualized_volatility_pct([100.0] * 30) == 0.0
    assert stats.annualized_volatility_pct([100.0] * 20) is None


def test_volatility_matches_a_hand_calculation():
    closes = [100.0, 101.0] * 15 + [100.0]   # 31 closes, alternating +1% / -0.99% moves
    value = stats.annualized_volatility_pct(closes)
    assert 15 < value < 17   # daily std about 1% -> about 1% * sqrt(252) = 15.9%
