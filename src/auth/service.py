"""Register and log in. The rules, the throttling and the choice of error live here; the API only translates them into HTTP."""
import logging

from src.auth.repository import EmailTaken, UserRepository
from src.auth.security import AuthInputError, AuthUser, InvalidToken, Passwords, RateLimiter, TooManyAttempts, check_password, create_token, decode_token, normalize_email
from src.core.config import Settings

log = logging.getLogger("finnie.auth")


class BadCredentials(Exception):
    """Wrong email or wrong password. Deliberately one error for both, so the reply never says which emails have accounts."""


class AuthService:
    def __init__(self, users: UserRepository | None, settings: Settings) -> None:
        self._users, self.settings, self.config = users, settings, settings.auth
        self.passwords = Passwords()
        self.failed_logins = RateLimiter(self.config.max_failed_logins, self.config.lockout_minutes * 60)
        self.registrations = RateLimiter(self.config.max_registrations_per_hour, 3600)

    @property
    def users(self) -> UserRepository:
        """Created on first use, so checking a token (which needs only the secret) works without the database."""
        if self._users is None:
            self._users = UserRepository.from_settings(self.settings)
        return self._users

    @property
    def secret(self) -> str:
        return self.settings.secrets.require_jwt_secret()

    def token_for(self, user: AuthUser) -> str:
        return create_token(user, self.secret, self.config.token_minutes)

    def verify_token(self, token: str) -> AuthUser:
        return decode_token(token, self.secret)

    def register(self, email: str, password: str, client_ip: str) -> AuthUser:
        wait = self.registrations.retry_after(client_ip)
        if wait:
            raise TooManyAttempts(wait)
        problems = []
        try:
            email = normalize_email(email)
        except AuthInputError as error:
            problems += error.problems
        try:
            check_password(password, self.config, email)
        except AuthInputError as error:
            problems += error.problems
        if problems:
            raise AuthInputError(problems)
        self.registrations.add(client_ip)   # counted only for well-formed attempts that reach the database
        try:
            record = self.users.create(email, self.passwords.hash(password))
        except EmailTaken:
            raise AuthInputError(["email: an account with this email already exists; log in instead"]) from None
        log.info("registered user %s", record.id)   # the id, never the email or anything secret
        return AuthUser(record.id, record.email)

    def login(self, email: str, password: str, client_ip: str) -> AuthUser:
        try:
            email = normalize_email(email)
        except AuthInputError:
            raise BadCredentials() from None
        key = f"{email}|{client_ip}"
        wait = self.failed_logins.retry_after(key)
        if wait:
            raise TooManyAttempts(wait)
        record = self.users.get_by_email(email)
        ok = self.passwords.verify(record.password_hash if record else None, password)   # runs a real hash even for an unknown email
        if not ok:
            self.failed_logins.add(key)
            log.info("failed login")
            raise BadCredentials()
        self.failed_logins.clear(key)
        self.users.record_login(record.id, self.passwords.hash(password) if self.passwords.needs_rehash(record.password_hash) else None)
        return AuthUser(record.id, record.email)
