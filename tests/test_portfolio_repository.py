"""PortfolioRepository against a real (throwaway) PostgreSQL database: round trip, rollback, cascade, injection, validation, concurrency."""
from decimal import Decimal
from threading import Thread
from uuid import uuid4

import pytest
from pydantic import ValidationError

from tests.auth_support import ALICE, BOB
from src.data.portfolio_repository import HoldingIn, PortfolioIn, PortfolioNotFound, PortfolioRepository

HOLDINGS = [{"ticker": "VTI", "shares": "60", "avg_cost": "220.10"}, {"ticker": "BND", "shares": "100.5", "avg_cost": "74"}]


@pytest.fixture
def repo(pg, two_users):
    return PortfolioRepository(pg)


ME = ALICE.id


def count(pg, table):
    with pg.connection() as connection:
        return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_round_trip_returns_identical_holdings(repo):
    portfolio_id = repo.create("Mine", HOLDINGS, owner_id=ME)
    stored = repo.get(portfolio_id, owner_id=ME)
    assert stored.name == "Mine" and stored.id == portfolio_id
    assert [(h.ticker, h.shares, h.avg_cost) for h in stored.holdings] == [("VTI", Decimal("60"), Decimal("220.10")), ("BND", Decimal("100.5"), Decimal("74"))]


def test_decimals_are_exact_not_floating_point(repo):
    portfolio_id = repo.create("Exact", [{"ticker": "VTI", "shares": "0.1", "avg_cost": "0.2"}], owner_id=ME)
    holding = repo.get(portfolio_id, owner_id=ME).holdings[0]
    assert holding.shares + holding.avg_cost == Decimal("0.3")   # with floats, 0.1 + 0.2 is 0.30000000000000004


def test_missing_id_is_a_clean_not_found(repo):
    for call in (lambda: repo.get(uuid4(), owner_id=ME), lambda: repo.delete(uuid4(), owner_id=ME), lambda: repo.update(uuid4(), HOLDINGS, "x", owner_id=ME)):
        with pytest.raises(PortfolioNotFound):
            call()


def test_bad_input_is_rejected_before_the_database_is_touched(repo, pg):
    bad = [[{"ticker": "VTI", "shares": "-5", "avg_cost": "10"}], [{"ticker": "", "shares": "5", "avg_cost": "10"}],
           [{"ticker": "VTI", "shares": "0", "avg_cost": "10"}], [{"ticker": "VTI", "shares": "5", "avg_cost": "-1"}],
           [{"ticker": "VTI", "shares": "NaN", "avg_cost": "1"}], [{"ticker": "NOT A TICKER!", "shares": "1", "avg_cost": "1"}], []]
    for holdings in bad:
        with pytest.raises(ValidationError):
            repo.create("Bad", holdings, owner_id=ME)
    with pytest.raises(ValidationError):
        repo.create("   ", HOLDINGS, owner_id=ME)
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
        repo.create("Half", [good, broken], owner_id=ME)   # the portfolio row and the first holding are inserted BEFORE the failure
    assert count(pg, "portfolios") == 0 and count(pg, "holdings") == 0


def test_delete_cascades_to_holdings(repo, pg):
    keep, drop = repo.create("keep", HOLDINGS, owner_id=ME), repo.create("drop", HOLDINGS, owner_id=ME)
    assert count(pg, "holdings") == 4
    repo.delete(drop, owner_id=ME)
    assert count(pg, "holdings") == 2 and repo.get(keep, owner_id=ME).holdings


def test_sql_injection_text_is_stored_as_plain_text(repo, pg):
    name = "'; DROP TABLE portfolios; --"
    portfolio_id = repo.create(name, HOLDINGS, owner_id=ME)
    assert repo.get(portfolio_id, owner_id=ME).name == name
    assert count(pg, "portfolios") == 1   # the table still exists


def test_update_replaces_holdings_and_bumps_updated_at(repo):
    portfolio_id = repo.create("Old", HOLDINGS, owner_id=ME)
    before = repo.get(portfolio_id, owner_id=ME)
    repo.update(portfolio_id, [{"ticker": "VXUS", "shares": "5", "avg_cost": "58"}], "New", owner_id=ME)
    after = repo.get(portfolio_id, owner_id=ME)
    assert after.name == "New" and [h.ticker for h in after.holdings] == ["VXUS"]
    assert after.created_at == before.created_at and after.updated_at > before.updated_at


def test_update_without_a_name_keeps_the_name(repo):
    portfolio_id = repo.create("Keep me", HOLDINGS, owner_id=ME)
    repo.update(portfolio_id, [{"ticker": "VXUS", "shares": "5", "avg_cost": "58"}], owner_id=ME)
    assert repo.get(portfolio_id, owner_id=ME).name == "Keep me"


def test_a_failed_update_keeps_the_old_holdings(repo):
    portfolio_id = repo.create("Safe", HOLDINGS, owner_id=ME)
    broken = HoldingIn.model_construct(ticker="BAD", shares=Decimal("-1"), avg_cost=Decimal("1"))
    with pytest.raises(Exception):
        repo.update(portfolio_id, [broken], "Changed", owner_id=ME)
    after = repo.get(portfolio_id, owner_id=ME)
    assert after.name == "Safe" and [h.ticker for h in after.holdings] == ["VTI", "BND"]


def test_list_shows_counts_newest_first(repo):
    first = repo.create("first", HOLDINGS, owner_id=ME)
    second = repo.create("second", HOLDINGS[:1], owner_id=ME)
    listed = repo.list(owner_id=ME)
    assert [(p.id, p.holding_count) for p in listed] == [(second, 1), (first, 2)]


def test_many_threads_share_a_small_pool_without_errors(repo):
    errors = []

    def work(n):
        try:
            portfolio_id = repo.create(f"p{n}", HOLDINGS, owner_id=ME)
            assert len(repo.get(portfolio_id, owner_id=ME).holdings) == 2
        except Exception as error:   # collected so the main thread can fail the test
            errors.append(error)

    threads = [Thread(target=work, args=(n,)) for n in range(30)]   # 30 callers, pool of 6
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and len(repo.list(owner_id=ME)) == 30


# ---------- Step 27: ownership ----------
def test_another_users_portfolio_looks_exactly_like_one_that_does_not_exist(repo):
    mine = repo.create("mine", HOLDINGS, owner_id=ALICE.id)
    for call in (lambda: repo.get(mine, owner_id=BOB.id), lambda: repo.delete(mine, owner_id=BOB.id), lambda: repo.update(mine, HOLDINGS, "hijacked", owner_id=BOB.id),
                 lambda: repo.update(mine, HOLDINGS, owner_id=BOB.id)):
        with pytest.raises(PortfolioNotFound):
            call()
    stored = repo.get(mine, owner_id=ALICE.id)
    assert stored.name == "mine" and len(stored.holdings) == 2   # untouched by Bob's attempts


def test_list_shows_only_my_own_portfolios(repo):
    repo.create("alice 1", HOLDINGS, owner_id=ALICE.id)
    repo.create("alice 2", HOLDINGS, owner_id=ALICE.id)
    repo.create("bob 1", HOLDINGS, owner_id=BOB.id)
    assert sorted(p.name for p in repo.list(owner_id=ALICE.id)) == ["alice 1", "alice 2"]
    assert [p.name for p in repo.list(owner_id=BOB.id)] == ["bob 1"]


def test_a_portfolio_with_no_owner_is_reachable_by_nobody(repo, pg):
    with pg.connection() as connection:
        orphan = connection.execute("INSERT INTO portfolios (name) VALUES ('old sample') RETURNING id").fetchone()[0]
    assert repo.list(owner_id=ALICE.id) == []
    with pytest.raises(PortfolioNotFound):
        repo.get(orphan, owner_id=ALICE.id)


def test_deleting_a_user_deletes_their_portfolios(repo, pg):
    repo.create("goes away", HOLDINGS, owner_id=BOB.id)
    with pg.connection() as connection:
        connection.execute("DELETE FROM users WHERE id = %s", (BOB.id,))
    assert count(pg, "portfolios") == 0 and count(pg, "holdings") == 0


def test_an_owner_is_required_and_cannot_be_forgotten(repo):
    with pytest.raises(TypeError):
        repo.get(uuid4())   # keyword-only and required: forgetting it is an error, not a silent "everyone's"
