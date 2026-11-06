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
from src.data.asset_types import AssetType, PropertyType
from src.data.db import get_pool

MAX_HOLDINGS = load_settings().database.max_holdings_per_portfolio
MAX_PROPERTIES = 50
_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")   # the same rule portfolio_calc applies to a CSV


class PropertyNotFound(LookupError):
    def __init__(self, name) -> None:
        super().__init__(f"no property named {name!r} in this portfolio")
        self.name = name


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


class PropertyIn(BaseModel):
    """A property the user owns. All money is exact decimal and entered by the user; Finnie looks nothing up. The mortgage is a balance plus, optionally, the monthly payment and rate (used only when the owner asks how long it takes to pay off)."""
    name: str = Field(min_length=1, max_length=120, description="a name or address, however the owner thinks of it")
    property_type: PropertyType = "house"
    purchase_price: Decimal = Field(ge=0, allow_inf_nan=False)
    purchase_date: date | None = None
    current_value: Decimal = Field(ge=0, allow_inf_nan=False, description="the owner's own estimate of what it is worth today")
    mortgage_balance: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False)
    monthly_rent: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False)
    monthly_expenses: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False, description="tax, insurance, upkeep, fees: everything except the mortgage payment")
    monthly_mortgage_payment: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False, description="principal + interest paid each month (0 = not entered)")
    mortgage_rate_pct: Decimal = Field(default=Decimal(0), ge=0, le=100, allow_inf_nan=False, description="yearly interest rate in percent, e.g. 6.5 (0 = not entered)")

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name cannot be blank")
        return value

    @field_validator("purchase_date")
    @classmethod
    def _not_in_the_future(cls, value: date | None) -> date | None:
        if value is not None and value > date.today():
            raise ValueError("purchase date cannot be in the future")
        return value


class PropertyTerms(BaseModel):
    """The two mortgage numbers that can be changed on a saved property."""
    payment: Decimal = Field(ge=0, allow_inf_nan=False)
    rate_pct: Decimal = Field(ge=0, le=100, allow_inf_nan=False)


class PortfolioIn(BaseModel):
    """A portfolio ready to save: market holdings, properties, or both (at least one). Duplicate tickers are merged (shares add, cost is share-weighted),
    like the CSV parser does; two properties with the same name are refused (they would be indistinguishable on screen)."""
    name: str = Field(min_length=1, max_length=100)
    holdings: list[HoldingIn] = Field(default_factory=list)
    properties: list[PropertyIn] = Field(default_factory=list)

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
        names = [p.name.lower() for p in self.properties]
        if len(set(names)) != len(names):
            raise ValueError("two properties have the same name: give each one its own name or address")
        if len(self.properties) > MAX_PROPERTIES:
            raise ValueError(f"a portfolio can hold at most {MAX_PROPERTIES} properties (got {len(self.properties)})")
        if not self.holdings and not self.properties:
            raise ValueError("add at least one holding or one property")
        return self


class StoredHolding(BaseModel):
    ticker: str
    shares: Decimal
    avg_cost: Decimal
    asset_type: AssetType = "other"
    purchase_date: date | None = None


class StoredProperty(PropertyIn):
    """A saved property: the same fields as the input (so it can be fed straight back in to an update)."""


class StoredPortfolio(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime
    holdings: list[StoredHolding]
    properties: list[StoredProperty] = Field(default_factory=list)


class PortfolioSummary(BaseModel):
    id: UUID
    name: str
    holding_count: int
    property_count: int = 0
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

    @staticmethod
    def _insert_properties(connection, portfolio_id: UUID, properties: list[PropertyIn]) -> None:
        with connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO properties (portfolio_id, name, property_type, purchase_price, purchase_date, current_value, mortgage_balance, monthly_rent, monthly_expenses,
                                       monthly_mortgage_payment, mortgage_rate_pct)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [(portfolio_id, p.name, p.property_type, p.purchase_price, p.purchase_date, p.current_value, p.mortgage_balance, p.monthly_rent, p.monthly_expenses,
                  p.monthly_mortgage_payment, p.mortgage_rate_pct) for p in properties])

    def create(self, name: str, holdings: list[HoldingIn | dict] = (), *, owner_id: UUID, properties: list[PropertyIn | dict] = ()) -> UUID:
        """Validate, then save the portfolio with ALL its holdings and properties in one transaction. Returns the new id.
        Bad input raises pydantic.ValidationError before the database is touched."""
        portfolio = PortfolioIn(name=name, holdings=list(holdings), properties=list(properties))
        with self.pool.connection() as connection, connection.transaction():
            portfolio_id = connection.execute("INSERT INTO portfolios (name, owner_id) VALUES (%s, %s) RETURNING id", (portfolio.name, owner_id)).fetchone()[0]
            self._insert_holdings(connection, portfolio_id, portfolio.holdings)
            self._insert_properties(connection, portfolio_id, portfolio.properties)
        return portfolio_id

    def get(self, portfolio_id: UUID, *, owner_id: UUID) -> StoredPortfolio:
        with self.pool.connection() as connection:
            row = connection.execute("SELECT id, name, created_at, updated_at FROM portfolios WHERE id = %s AND owner_id = %s", (portfolio_id, owner_id)).fetchone()
            if row is None:
                raise PortfolioNotFound(portfolio_id)
            holdings = connection.execute("SELECT ticker, shares, avg_cost, asset_type, purchase_date FROM holdings WHERE portfolio_id = %s ORDER BY id", (portfolio_id,)).fetchall()
            properties = connection.execute(
                """SELECT name, property_type, purchase_price, purchase_date, current_value, mortgage_balance, monthly_rent, monthly_expenses,
                          monthly_mortgage_payment, mortgage_rate_pct
                   FROM properties WHERE portfolio_id = %s ORDER BY id""", (portfolio_id,)).fetchall()
        return StoredPortfolio(id=row[0], name=row[1], created_at=row[2], updated_at=row[3],
                               holdings=[StoredHolding(ticker=t, shares=s, avg_cost=c, asset_type=a, purchase_date=d) for t, s, c, a, d in holdings],
                               properties=[StoredProperty(name=n, property_type=k, purchase_price=pp, purchase_date=pd_, current_value=cv, mortgage_balance=mb, monthly_rent=mr, monthly_expenses=me,
                                                          monthly_mortgage_payment=mp, mortgage_rate_pct=mrate)
                                           for n, k, pp, pd_, cv, mb, mr, me, mp, mrate in properties])

    def update(self, portfolio_id: UUID, holdings: list[HoldingIn | dict], name: str | None = None, *, owner_id: UUID, properties: list[PropertyIn | dict] | None = None) -> None:
        """Replace the holdings (and optionally the name and the properties) in one transaction: either the whole new set is saved or the old one stays.
        `properties=None` leaves the saved properties alone; pass a list (even an empty one) to replace them."""
        current = self.get(portfolio_id, owner_id=owner_id)   # also the ownership check, up front
        keep_properties = properties is None
        portfolio = PortfolioIn(name=name if name is not None else current.name, holdings=list(holdings),
                                properties=[p.model_dump() for p in current.properties] if keep_properties else list(properties))
        with self.pool.connection() as connection, connection.transaction():
            updated = connection.execute("UPDATE portfolios SET name = %s, updated_at = now() WHERE id = %s AND owner_id = %s RETURNING id",
                                         (portfolio.name, portfolio_id, owner_id)).fetchone()
            if updated is None:
                raise PortfolioNotFound(portfolio_id)
            connection.execute("DELETE FROM holdings WHERE portfolio_id = %s", (portfolio_id,))
            self._insert_holdings(connection, portfolio_id, portfolio.holdings)
            if not keep_properties:
                connection.execute("DELETE FROM properties WHERE portfolio_id = %s", (portfolio_id,))
                self._insert_properties(connection, portfolio_id, portfolio.properties)

    def set_mortgage_terms(self, portfolio_id: UUID, property_name: str, payment: Decimal, rate_pct: Decimal, *, owner_id: UUID) -> None:
        """Fill in (or change) the monthly payment and yearly rate of ONE saved property, leaving everything else alone. This is how existing portfolios get these values."""
        PropertyTerms(payment=payment, rate_pct=rate_pct)   # same limits as when the property is first saved
        with self.pool.connection() as connection:
            if connection.execute("SELECT 1 FROM portfolios WHERE id = %s AND owner_id = %s", (portfolio_id, owner_id)).fetchone() is None:
                raise PortfolioNotFound(portfolio_id)
            updated = connection.execute("UPDATE properties SET monthly_mortgage_payment = %s, mortgage_rate_pct = %s WHERE portfolio_id = %s AND name = %s RETURNING id",
                                         (payment, rate_pct, portfolio_id, property_name)).fetchone()
            if updated is None:
                raise PropertyNotFound(property_name)

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
                """SELECT p.id, p.name, (SELECT count(*) FROM holdings h WHERE h.portfolio_id = p.id), (SELECT count(*) FROM properties q WHERE q.portfolio_id = p.id),
                          p.created_at, p.updated_at
                   FROM portfolios p WHERE p.owner_id = %s
                   ORDER BY p.created_at DESC, p.id""", (owner_id,)).fetchall()
        return [PortfolioSummary(id=i, name=n, holding_count=c, property_count=pc, created_at=ca, updated_at=ua) for i, n, c, pc, ca, ua in rows]
