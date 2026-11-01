"""Registration and login against a real (throwaway) PostgreSQL: what is stored, what is refused, and what the throttles do."""
import pytest

from src.auth.repository import EmailTaken, UserRepository
from src.auth.security import AuthInputError, TooManyAttempts
from src.auth.service import AuthService, BadCredentials
from tests.auth_support import settings_with_secret

PASSWORD = "correct horse battery"
IP = "203.0.113.5"


@pytest.fixture
def auth(pg):
    return AuthService(UserRepository(pg), settings_with_secret())


def stored(pg, email):
    with pg.connection() as connection:
        return connection.execute("SELECT id, email, password_hash, last_login_at FROM users WHERE email = %s", (email,)).fetchone()


def test_registering_stores_a_hash_never_the_password(auth, pg):
    user = auth.register("Alice@Example.com", PASSWORD, IP)
    row = stored(pg, "alice@example.com")
    assert row[0] == user.id and row[2].startswith("$argon2id$") and PASSWORD not in row[2]
    with pg.connection() as connection:
        everything = " ".join(str(v) for r in connection.execute("SELECT * FROM users").fetchall() for v in r)
    assert PASSWORD not in everything


def test_the_same_email_in_any_case_cannot_register_twice(auth):
    auth.register("alice@example.com", PASSWORD, IP)
    with pytest.raises(AuthInputError) as caught:
        auth.register("ALICE@example.com ", PASSWORD, IP)
    assert "already exists" in caught.value.problems[0]


def test_two_simultaneous_sign_ups_for_one_email_cannot_both_win(pg):
    import threading
    results = []
    service = AuthService(UserRepository(pg), settings_with_secret())

    def go():
        try:
            service.register("race@example.com", PASSWORD, IP)
            results.append("ok")
        except AuthInputError:
            results.append("taken")
    threads = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == ["ok", "taken", "taken", "taken"]   # the database's UNIQUE constraint decides, not our check-then-insert


def test_every_registration_problem_is_reported_at_once(auth):
    with pytest.raises(AuthInputError) as caught:
        auth.register("not-an-email", "short", IP)
    assert len(caught.value.problems) == 2


def test_login_succeeds_with_any_case_of_the_email_and_stamps_the_time(auth, pg):
    registered = auth.register("alice@example.com", PASSWORD, IP)
    assert stored(pg, "alice@example.com")[3] is None
    user = auth.login(" ALICE@example.com", PASSWORD, IP)
    assert user.id == registered.id and stored(pg, "alice@example.com")[3] is not None


def test_wrong_password_and_unknown_email_give_the_same_error(auth):
    auth.register("alice@example.com", PASSWORD, IP)
    for email, password in (("alice@example.com", "wrong password here"), ("nobody@example.com", PASSWORD), ("garbage", PASSWORD)):
        with pytest.raises(BadCredentials):
            auth.login(email, password, IP)


def test_repeated_failures_lock_that_email_from_that_address_only(auth):
    auth.register("alice@example.com", PASSWORD, IP)
    limit = auth.config.max_failed_logins
    for _ in range(limit):
        with pytest.raises(BadCredentials):
            auth.login("alice@example.com", "wrong password here", IP)
    with pytest.raises(TooManyAttempts) as caught:
        auth.login("alice@example.com", PASSWORD, IP)           # even the RIGHT password is refused during the lockout
    assert caught.value.retry_after_seconds > 0
    assert auth.login("alice@example.com", PASSWORD, "198.51.100.9").email == "alice@example.com"   # Alice herself, from elsewhere, is not locked out by an attacker's address


def test_a_successful_login_clears_the_failure_count(auth):
    auth.register("alice@example.com", PASSWORD, IP)
    for _ in range(auth.config.max_failed_logins - 1):
        with pytest.raises(BadCredentials):
            auth.login("alice@example.com", "wrong password here", IP)
    auth.login("alice@example.com", PASSWORD, IP)
    for _ in range(auth.config.max_failed_logins - 1):          # a fresh allowance
        with pytest.raises(BadCredentials):
            auth.login("alice@example.com", "wrong password here", IP)


def test_registrations_are_limited_per_address(auth):
    for i in range(auth.config.max_registrations_per_hour):
        auth.register(f"user{i}@example.com", PASSWORD, IP)
    with pytest.raises(TooManyAttempts):
        auth.register("one-too-many@example.com", PASSWORD, IP)
    auth.register("other-address@example.com", PASSWORD, "198.51.100.9")


def test_an_old_hash_is_upgraded_on_the_next_login(auth, pg):
    from argon2 import PasswordHasher
    weak = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash(PASSWORD)
    UserRepository(pg).create("old@example.com", weak)
    auth.login("old@example.com", PASSWORD, IP)
    assert stored(pg, "old@example.com")[2] != weak and auth.login("old@example.com", PASSWORD, IP)


def test_the_repository_refuses_a_duplicate_directly(pg):
    repo = UserRepository(pg)
    repo.create("dup@example.com", "hash")
    with pytest.raises(EmailTaken):
        repo.create("dup@example.com", "hash")


def test_the_database_itself_refuses_a_mixed_case_email(pg):
    import psycopg
    with pytest.raises(psycopg.errors.CheckViolation), pg.connection() as connection:
        connection.execute("INSERT INTO users (email, password_hash) VALUES ('Mixed@Example.com', 'x')")
