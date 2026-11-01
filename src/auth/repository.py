"""Users in PostgreSQL. Parameterized SQL only. The repository stores and fetches; the rules (what a valid email or password is) live in security.py."""
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import psycopg.errors
from psycopg_pool import ConnectionPool

from src.core.config import Settings
from src.data.db import get_pool


class EmailTaken(Exception):
    pass


@dataclass(frozen=True)
class UserRecord:
    id: UUID
    email: str
    password_hash: str
    created_at: datetime


class UserRepository:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "UserRepository":
        return cls(get_pool(settings))

    def create(self, email: str, password_hash: str) -> UserRecord:
        try:
            with self.pool.connection() as connection:
                row = connection.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id, email, password_hash, created_at",
                                         (email, password_hash)).fetchone()
        except psycopg.errors.UniqueViolation:
            raise EmailTaken(email) from None   # the UNIQUE constraint is the real guard: two simultaneous sign-ups cannot both win
        return UserRecord(*row)

    def get_by_email(self, email: str) -> UserRecord | None:
        with self.pool.connection() as connection:
            row = connection.execute("SELECT id, email, password_hash, created_at FROM users WHERE email = %s", (email,)).fetchone()
        return UserRecord(*row) if row else None

    def record_login(self, user_id: UUID, new_hash: str | None = None) -> None:
        """Stamp the login time, and swap in a fresh hash if the hashing settings have been strengthened since this one was made."""
        with self.pool.connection() as connection:
            if new_hash:
                connection.execute("UPDATE users SET last_login_at = now(), password_hash = %s WHERE id = %s", (new_hash, user_id))
            else:
                connection.execute("UPDATE users SET last_login_at = now() WHERE id = %s", (user_id,))
