"""The security building blocks, with no web framework and no database in them, so every rule can be tested alone.

  passwords   Argon2id (a deliberately slow, memory-hungry hash with its own random salt). We store only the hash, never the password.
  tokens      a JWT signed with HS256: {sub: user id, email, iat, exp, iss}. Anyone holding a valid, unexpired token IS that user, so tokens
              live in the browser session only, are short-lived, and the signing secret comes from the environment.
  throttling  a sliding-window counter so guessing passwords or mass-creating accounts from one address gets slow and then refused.
"""
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from src.core.config import AuthConfig

ISSUER = "finnie"
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Not a real defence against weak passwords (length is), just the first things anyone tries.
_COMMON = frozenset("password password1 password123 1234567890 qwertyuiop 12345678910 iloveyou123 letmein123 welcome123 admin12345".split())


class AuthInputError(ValueError):
    """Registration or login input that breaks a rule. `problems` lists every broken rule, so the user can fix them in one go."""
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class InvalidToken(Exception):
    """The token is missing, malformed, tampered with, signed with another secret, or expired. The reason is for logs; the user is only told 'log in again'."""


class TooManyAttempts(Exception):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(f"too many attempts; retry in {retry_after_seconds} s")
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class AuthUser:
    id: UUID
    email: str


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not email or len(email) > 254 or not _EMAIL.match(email):
        raise AuthInputError(["email: enter a valid email address such as name@example.com"])
    return email


def check_password(password: str, config: AuthConfig, email: str = "") -> None:
    problems = []
    if len(password) < config.min_password_length:
        problems.append(f"password: use at least {config.min_password_length} characters")
    if len(password) > config.max_password_length:
        problems.append(f"password: use at most {config.max_password_length} characters")
    if password.strip().lower() in _COMMON or (email and password.strip().lower() == email):
        problems.append("password: that one is too easy to guess; choose something less common and not your email")
    if problems:
        raise AuthInputError(problems)


class Passwords:
    def __init__(self) -> None:
        self._hasher = PasswordHasher()   # argon2-cffi's defaults follow the RFC 9106 recommendation (Argon2id)
        # Verified against when the email is unknown, so "no such user" costs the same time as "wrong password" and timing does not reveal which emails exist.
        self._dummy = self._hasher.hash("finnie-dummy-password-for-timing")

    def hash(self, password: str) -> str:
        return self._hasher.hash(password)

    def verify(self, stored_hash: str | None, password: str) -> bool:
        try:
            return self._hasher.verify(stored_hash or self._dummy, password) and stored_hash is not None
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False

    def needs_rehash(self, stored_hash: str) -> bool:
        return self._hasher.check_needs_rehash(stored_hash)


def create_token(user: AuthUser, secret: str, minutes: int, now: float | None = None) -> str:
    issued = int(now if now is not None else time.time())
    return jwt.encode({"sub": str(user.id), "email": user.email, "iat": issued, "exp": issued + minutes * 60, "iss": ISSUER}, secret, algorithm="HS256")


def decode_token(token: str, secret: str, now: float | None = None) -> AuthUser:
    try:
        claims = jwt.decode(token, secret, algorithms=["HS256"], issuer=ISSUER, options={"require": ["exp", "iat", "sub", "iss"]},
                            leeway=0) if now is None else _decode_at(token, secret, now)
        return AuthUser(UUID(claims["sub"]), claims["email"])
    except (jwt.PyJWTError, ValueError, KeyError, TypeError) as error:
        raise InvalidToken(type(error).__name__) from None


def _decode_at(token: str, secret: str, now: float) -> dict:
    """decode_token with a pretend clock, for tests of expiry (PyJWT reads the real clock, so we check exp ourselves after verifying the signature)."""
    claims = jwt.decode(token, secret, algorithms=["HS256"], issuer=ISSUER, options={"require": ["exp", "iat", "sub", "iss"], "verify_exp": False})
    if claims["exp"] <= now:
        raise jwt.ExpiredSignatureError("expired")
    return claims


class RateLimiter:
    """At most `limit` events per `window_seconds` for each key (in memory, so it is per server process and forgotten on restart: fine for one worker,
    and a known limit for several; see PROGRESS.md). Old entries are dropped as it goes, and the table is capped so a flood of different keys cannot grow it forever."""
    MAX_KEYS = 10_000

    def __init__(self, limit: int, window_seconds: float, clock=time.monotonic) -> None:
        self.limit, self.window, self._clock = limit, window_seconds, clock
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str) -> deque:
        events = self._events.setdefault(key, deque())
        cutoff = self._clock() - self.window
        while events and events[0] <= cutoff:
            events.popleft()
        if len(self._events) > self.MAX_KEYS:
            for stale in [k for k, v in self._events.items() if not v or v[-1] <= cutoff][: self.MAX_KEYS // 2]:
                del self._events[stale]
        return events

    def retry_after(self, key: str) -> int:
        """0 if another event is allowed now, else the whole seconds until the oldest one ages out."""
        with self._lock:
            events = self._recent(key)
            return 0 if len(events) < self.limit else max(1, int(events[0] + self.window - self._clock()) + 1)

    def add(self, key: str) -> None:
        with self._lock:
            self._recent(key).append(self._clock())

    def clear(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)
