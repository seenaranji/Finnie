"""Saved portfolios in PostgreSQL (Step 15b, built directly on Postgres in Step 23b).

The rest of the app calls `PortfolioRepository` and never writes SQL (same idea as `VectorStore` and `MarketDataProvider`).

What is stored: the HOLDINGS the user typed in (ticker, shares, average cost). What is not stored: metrics, values, weights. Prices change, so
those are recomputed every time a portfolio is viewed.

Money and share counts are `Decimal` in Python and `NUMERIC` in PostgreSQL, so what you save is exactly what you get back. Every statement uses
placeholders (`%s`) with the values passed separately, so user text can never become SQL. Each write is one transaction: a portfolio and all of
its holdings are saved together or not at all.

Ownership (Step 27): every portfolio belongs to a user, and EVERY method takes `owner_id` as a required keyword argument and puts it in the SQL
(`WHERE ... AND owner_id = %s`), so one user can never read, change or delete another's portfolio, and a caller cannot forget to say whose it is.
A portfolio that is not yours looks exactly like one that does not exist (PortfolioNotFound), so ids cannot be probed for existence.
"""
import re
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator
from psycopg_pool import ConnectionPool

from src.core.config import Settings, load_settings
from src.data.asset_types import AssetType
from src.data.db import get_pool

MAX_HOLDINGS = load_settings().database.max_holdings_per_portfolio
_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")   # the same rule portfolio_calc applies to a CSV


class PortfolioNotFound(LookupError):
    def __init__(self, portfolio_id) -> None:
        super().__init__(f"no portfolio with id {portfolio_id}")
        self.portfolio_id = portfolio_id


class HoldingIn(BaseModel):
    ticker: str
    shares: Decimal = Field(allow_inf_nan=False)
    avg_cost: Decimal = Field(allow_inf_nan=False)
    asset_type: AssetType = "other"          # what kind of holding it is; the user's own word, never guessed from the ticker
    purchase_date: date | None = None        # optional: when it was bought

    @field_validator("purchase_date")
    @classmethod
    def _not_in_the_future(cls, value: date | None) -> date | None:
        if value is not None and value > date.today():
            raise ValueError("purchase date cannot be in the future")
        return value

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, value: str) -> str:
        value = value.strip().upper()
        if not _TICKER.match(value):
            raise ValueError(f"'{value}' is not a valid ticker (letters, digits, '.' or '-', up to 10 characters)")
        return value

    @field_validator("shares")
    @classmethod
    def _shares(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("shares must be greater than 0")
        return value

    @field_validator("avg_cost")
    @classmethod
    def _cost(cls, value: Decimal) -> Decimal:
        if value < 0:
            raise ValueError("avg_cost cannot be negative")
        return value


class PortfolioIn(BaseModel):
    """A portfolio ready to save. Duplicate tickers are merged (shares add, cost is share-weighted), like the CSV parser does."""
    name: str = Field(min_length=1, max_length=100)
    holdings: list[HoldingIn] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name cannot be blank")
        return value

    @model_validator(mode="after")
    def _merge_and_cap(self):
        merged: dict[str, HoldingIn] = {}
        for holding in self.holdings:
            if holding.ticker in merged:
                old = merged[holding.ticker]
                if old.asset_type != holding.asset_type:
                    raise ValueError(f"{holding.ticker} is listed as both '{old.asset_type}' and '{holding.asset_type}': pick one type")
                shares = old.shares + holding.shares
                dates = [d for d in (old.purchase_date, holding.purchase_date) if d]   # two purchases of one ticker: keep the earlier date
                merged[holding.ticker] = HoldingIn(ticker=holding.ticker, shares=shares, asset_type=old.asset_type, purchase_date=min(dates) if dates else None,
                                                   avg_cost=(old.shares * old.avg_cost + holding.shares * holding.avg_cost) / shares)
            else:
                merged[holding.ticker] = holding
        if len(merged) > MAX_HOLDINGS:
            raise ValueError(f"a portfolio can hold at most {MAX_HOLDINGS} different tickers (got {len(merged)})")
        self.holdings = list(merged.values())
        return self


class StoredHolding(BaseModel):
    ticker: str
    shares: Decimal
    avg_cost: Decimal
    asset_type: AssetType = "other"
    purchase_date: date | None = None


class StoredPortfolio(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime
    holdings: list[StoredHolding]


class PortfolioSummary(BaseModel):
    id: UUID
    name: str
    holding_count: int
    created_at: datetime
    updated_at: datetime


class PortfolioRepository:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "PortfolioRepository":
        return cls(get_pool(settings))

    @staticmethod
    def _insert_holdings(connection, portfolio_id: UUID, holdings: list[HoldingIn]) -> None:
        with connection.cursor() as cursor:
            cursor.executemany("INSERT INTO holdings (portfolio_id, ticker, shares, avg_cost, asset_type, purchase_date) VALUES (%s, %s, %s, %s, %s, %s)",
                               [(portfolio_id, h.ticker, h.shares, h.avg_cost, h.asset_type, h.purchase_date) for h in holdings])

    def create(self, name: str, holdings: list[HoldingIn | dict], *, owner_id: UUID) -> UUID:
        """Validate, then save the portfolio and ALL its holdings in one transaction. Returns the new id.
        Bad input raises pydantic.ValidationError before the database is touched."""
        portfolio = PortfolioIn(name=name, holdings=holdings)
        with self.pool.connection() as connection, connection.transaction():
            portfolio_id = connection.execute("INSERT INTO portfolios (name, owner_id) VALUES (%s, %s) RETURNING id", (portfolio.name, owner_id)).fetchone()[0]
            self._insert_holdings(connection, portfolio_id, portfolio.holdings)
        return portfolio_id

    def get(self, portfolio_id: UUID, *, owner_id: UUID) -> StoredPortfolio:
        with self.pool.connection() as connection:
            row = connection.execute("SELECT id, name, created_at, updated_at FROM portfolios WHERE id = %s AND owner_id = %s", (portfolio_id, owner_id)).fetchone()
            if row is None:
                raise PortfolioNotFound(portfolio_id)
            holdings = connection.execute("SELECT ticker, shares, avg_cost, asset_type, purchase_date FROM holdings WHERE portfolio_id = %s ORDER BY id", (portfolio_id,)).fetchall()
        return StoredPortfolio(id=row[0], name=row[1], created_at=row[2], updated_at=row[3],
                               holdings=[StoredHolding(ticker=t, shares=s, avg_cost=c, asset_type=a, purchase_date=d) for t, s, c, a, d in holdings])

    def update(self, portfolio_id: UUID, holdings: list[HoldingIn | dict], name: str | None = None, *, owner_id: UUID) -> None:
        """Replace the holdings (and optionally the name) in one transaction: either the whole new set is saved or the old one stays."""
        current = self.get(portfolio_id, owner_id=owner_id) if name is None else None
        portfolio = PortfolioIn(name=name if name is not None else current.name, holdings=holdings)
        with self.pool.connection() as connection, connection.transaction():
            updated = connection.execute("UPDATE portfolios SET name = %s, updated_at = now() WHERE id = %s AND owner_id = %s RETURNING id",
                                         (portfolio.name, portfolio_id, owner_id)).fetchone()
            if updated is None:
                raise PortfolioNotFound(portfolio_id)
            connection.execute("DELETE FROM holdings WHERE portfolio_id = %s", (portfolio_id,))
            self._insert_holdings(connection, portfolio_id, portfolio.holdings)

    def delete(self, portfolio_id: UUID, *, owner_id: UUID) -> None:
        """Delete the portfolio; its holdings go with it (ON DELETE CASCADE)."""
        with self.pool.connection() as connection:
            deleted = connection.execute("DELETE FROM portfolios WHERE id = %s AND owner_id = %s RETURNING id", (portfolio_id, owner_id)).fetchone()
        if deleted is None:
            raise PortfolioNotFound(portfolio_id)

    # defined last on purpose: inside this class body the name `list` means this method, so no annotation below it may use list[...]
    def list(self, *, owner_id: UUID) -> list[PortfolioSummary]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """SELECT p.id, p.name, count(h.id), p.created_at, p.updated_at
                   FROM portfolios p LEFT JOIN holdings h ON h.portfolio_id = p.id
                   WHERE p.owner_id = %s
                   GROUP BY p.id ORDER BY p.created_at DESC, p.id""", (owner_id,)).fetchall()
        return [PortfolioSummary(id=i, name=n, holding_count=c, created_at=ca, updated_at=ua) for i, n, c, ca, ua in rows]
