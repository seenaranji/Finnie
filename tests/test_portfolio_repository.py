"""PortfolioRepository against a real (throwaway) PostgreSQL database: round trip, rollback, cascade, injection, validation, concurrency."""
from decimal import Decimal
from threading import Thread
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.data.portfolio_repository import HoldingIn, PortfolioIn, PortfolioNotFound, PortfolioRepository

HOLDINGS = [{"ticker": "VTI", "shares": "60", "avg_cost": "220.10"}, {"ticker": "BND", "shares": "100.5", "avg_cost": "74"}]


@pytest.fixture
def repo(pg):
    return PortfolioRepository(pg)


def count(pg, table):
    with pg.connection() as connection:
        return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_round_trip_returns_identical_holdings(repo):
    portfolio_id = repo.create("Mine", HOLDINGS)
    stored = repo.get(portfolio_id)
    assert stored.name == "Mine" and stored.id == portfolio_id
    assert [(h.ticker, h.shares, h.avg_cost) for h in stored.holdings] == [("VTI", Decimal("60"), Decimal("220.10")), ("BND", Decimal("100.5"), Decimal("74"))]


def test_decimals_are_exact_not_floating_point(repo):
    portfolio_id = repo.create("Exact", [{"ticker": "VTI", "shares": "0.1", "avg_cost": "0.2"}])
    holding = repo.get(portfolio_id).holdings[0]
    assert holding.shares + holding.avg_cost == Decimal("0.3")   # with floats, 0.1 + 0.2 is 0.30000000000000004


def test_missing_id_is_a_clean_not_found(repo):
    for call in (lambda: repo.get(uuid4()), lambda: repo.delete(uuid4()), lambda: repo.update(uuid4(), HOLDINGS, "x")):
        with pytest.raises(PortfolioNotFound):
            call()


def test_bad_input_is_rejected_before_the_database_is_touched(repo, pg):
    bad = [[{"ticker": "VTI", "shares": "-5", "avg_cost": "10"}], [{"ticker": "", "shares": "5", "avg_cost": "10"}],
           [{"ticker": "VTI", "shares": "0", "avg_cost": "10"}], [{"ticker": "VTI", "shares": "5", "avg_cost": "-1"}],
           [{"ticker": "VTI", "shares": "NaN", "avg_cost": "1"}], [{"ticker": "NOT A TICKER!", "shares": "1", "avg_cost": "1"}], []]
    for holdings in bad:
        with pytest.raises(ValidationError):
            repo.create("Bad", holdings)
    with pytest.raises(ValidationError):
        repo.create("   ", HOLDINGS)
    assert count(pg, "portfolios") == 0 and count(pg, "holdings") == 0


def test_tickers_are_uppercased_and_duplicates_merged_with_weighted_cost():
    portfolio = PortfolioIn(name="x", holdings=[{"ticker": " vti ", "shares": "10", "avg_cost": "100"}, {"ticker": "VTI", "shares": "30", "avg_cost": "200"}])
    assert [(h.ticker, h.shares, h.avg_cost) for h in portfolio.holdings] == [("VTI", Decimal("40"), Decimal("175"))]


def test_cap_on_holdings():
    many = [{"ticker": f"T{i}", "shares": "1", "avg_cost": "1"} for i in range(501)]
    with pytest.raises(ValidationError):
        PortfolioIn(name="big", holdings=many)


def test_a_save_that_fails_halfway_leaves_nothing(repo, pg):
    broken = HoldingIn.model_construct(ticker="BAD", shares=Decimal("-1"), avg_cost=Decimal("1"))   # skips validation, so the DATABASE's own CHECK must catch it
    good = HoldingIn(ticker="VTI", shares=Decimal("1"), avg_cost=Decimal("1"))
    with pytest.raises(Exception):
        repo.create("Half", [good, broken])   # the portfolio row and the first holding are inserted BEFORE the failure
    assert count(pg, "portfolios") == 0 and count(pg, "holdings") == 0


def test_delete_cascades_to_holdings(repo, pg):
    keep, drop = repo.create("keep", HOLDINGS), repo.create("drop", HOLDINGS)
    assert count(pg, "holdings") == 4
    repo.delete(drop)
    assert count(pg, "holdings") == 2 and repo.get(keep).holdings


def test_sql_injection_text_is_stored_as_plain_text(repo, pg):
    name = "'; DROP TABLE portfolios; --"
    portfolio_id = repo.create(name, HOLDINGS)
    assert repo.get(portfolio_id).name == name
    assert count(pg, "portfolios") == 1   # the table still exists


def test_update_replaces_holdings_and_bumps_updated_at(repo):
    portfolio_id = repo.create("Old", HOLDINGS)
    before = repo.get(portfolio_id)
    repo.update(portfolio_id, [{"ticker": "VXUS", "shares": "5", "avg_cost": "58"}], "New")
    after = repo.get(portfolio_id)
    assert after.name == "New" and [h.ticker for h in after.holdings] == ["VXUS"]
    assert after.created_at == before.created_at and after.updated_at > before.updated_at


def test_update_without_a_name_keeps_the_name(repo):
    portfolio_id = repo.create("Keep me", HOLDINGS)
    repo.update(portfolio_id, [{"ticker": "VXUS", "shares": "5", "avg_cost": "58"}])
    assert repo.get(portfolio_id).name == "Keep me"


def test_a_failed_update_keeps_the_old_holdings(repo):
    portfolio_id = repo.create("Safe", HOLDINGS)
    broken = HoldingIn.model_construct(ticker="BAD", shares=Decimal("-1"), avg_cost=Decimal("1"))
    with pytest.raises(Exception):
        repo.update(portfolio_id, [broken], "Changed")
    after = repo.get(portfolio_id)
    assert after.name == "Safe" and [h.ticker for h in after.holdings] == ["VTI", "BND"]


def test_list_shows_counts_newest_first(repo):
    first = repo.create("first", HOLDINGS)
    second = repo.create("second", HOLDINGS[:1])
    listed = repo.list()
    assert [(p.id, p.holding_count) for p in listed] == [(second, 1), (first, 2)]


def test_many_threads_share_a_small_pool_without_errors(repo):
    errors = []

    def work(n):
        try:
            portfolio_id = repo.create(f"p{n}", HOLDINGS)
            assert len(repo.get(portfolio_id).holdings) == 2
        except Exception as error:   # collected so the main thread can fail the test
            errors.append(error)

    threads = [Thread(target=work, args=(n,)) for n in range(30)]   # 30 callers, pool of 6
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and len(repo.list()) == 30
