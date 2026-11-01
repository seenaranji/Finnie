"""Test helpers for authenticated API calls. Tokens are stateless (signature + expiry), so a test can mint one without a database or a registration call."""
from uuid import UUID, uuid4

from src.auth.security import AuthUser, create_token
from src.core.config import load_settings

TEST_SECRET = "test-secret-" + "x" * 40   # never used outside tests
ALICE = AuthUser(UUID("11111111-1111-4111-8111-111111111111"), "alice@example.com")
BOB = AuthUser(UUID("22222222-2222-4222-8222-222222222222"), "bob@example.com")


def settings_with_secret(secret: str = TEST_SECRET):
    settings = load_settings()
    settings.secrets.jwt_secret = secret   # whatever is in .env, tests use their own
    return settings


def headers_for(user: AuthUser = ALICE, secret: str = TEST_SECRET, minutes: int = 60) -> dict:
    return {"Authorization": f"Bearer {create_token(user, secret, minutes)}"}


def new_user() -> AuthUser:
    return AuthUser(uuid4(), f"user-{uuid4().hex[:8]}@example.com")
