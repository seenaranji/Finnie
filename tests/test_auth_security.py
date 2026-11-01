"""Passwords, tokens and the rate limiter, with no web framework and no database."""
import base64
import json
import time

import jwt
import pytest

from src.auth.security import (AuthInputError, AuthUser, InvalidToken, Passwords, RateLimiter, check_password, create_token, decode_token, normalize_email)
from src.core.config import AuthConfig
from tests.auth_support import ALICE, TEST_SECRET

CONFIG = AuthConfig()


# ---------- email and password rules ----------
@pytest.mark.parametrize("raw,expected", [("  Alice@Example.COM ", "alice@example.com"), ("a.b+tag@sub.example.org", "a.b+tag@sub.example.org")])
def test_emails_are_trimmed_and_lowercased(raw, expected):
    assert normalize_email(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", "no-at-sign", "two@@example.com", "a@b", "spaces in@example.com", "x@y.z " + "a" * 260, None])
def test_bad_emails_are_rejected(bad):
    with pytest.raises(AuthInputError) as caught:
        normalize_email(bad)
    assert "email" in caught.value.problems[0]


def test_password_length_rules_use_the_configured_limits():
    check_password("a" * CONFIG.min_password_length, CONFIG)
    for bad in ("a" * (CONFIG.min_password_length - 1), "a" * (CONFIG.max_password_length + 1), ""):
        with pytest.raises(AuthInputError):
            check_password(bad, CONFIG)


def test_common_passwords_and_the_email_itself_are_refused_and_every_problem_is_listed():
    with pytest.raises(AuthInputError) as caught:
        check_password("password123", CONFIG)
    assert "too easy to guess" in caught.value.problems[0]
    with pytest.raises(AuthInputError):
        check_password("alice@example.com", CONFIG, email="alice@example.com")
    with pytest.raises(AuthInputError) as caught:
        check_password("short", CONFIG)
    assert len(caught.value.problems) == 1


# ---------- hashing ----------
def test_a_password_hashes_to_a_salted_argon2id_string_that_verifies_only_for_the_right_password():
    p = Passwords()
    one, two = p.hash("correct horse battery"), p.hash("correct horse battery")
    assert one.startswith("$argon2id$") and one != two                       # a fresh random salt each time
    assert "correct horse" not in one
    assert p.verify(one, "correct horse battery") and not p.verify(one, "correct horse batterY")


def test_verifying_an_unknown_user_is_false_but_still_does_the_work():
    p = Passwords()
    start = time.perf_counter()
    assert p.verify(None, "anything at all") is False
    unknown = time.perf_counter() - start
    start = time.perf_counter()
    p.verify(p.hash("x" * 12), "y" * 12)
    known = time.perf_counter() - start
    assert unknown > known / 5   # a real hash was computed (without this, "no such email" would answer instantly and reveal which emails exist)


def test_garbage_in_the_stored_hash_is_a_failed_login_not_a_crash():
    assert Passwords().verify("not-a-hash", "whatever") is False


def test_old_hash_settings_are_noticed():
    from argon2 import PasswordHasher
    weak = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash("x" * 12)
    assert Passwords().needs_rehash(weak) and not Passwords().needs_rehash(Passwords().hash("x" * 12))


# ---------- tokens ----------
def test_a_token_round_trips_to_the_same_user():
    token = create_token(ALICE, TEST_SECRET, 60)
    assert decode_token(token, TEST_SECRET) == ALICE


def test_an_expired_token_is_refused():
    token = create_token(ALICE, TEST_SECRET, 60, now=time.time() - 3 * 3600)
    with pytest.raises(InvalidToken):
        decode_token(token, TEST_SECRET)


def test_the_pretend_clock_agrees_about_the_exact_expiry_second():
    token = create_token(ALICE, TEST_SECRET, 1, now=1_000_000)
    assert decode_token(token, TEST_SECRET, now=1_000_059) == ALICE
    with pytest.raises(InvalidToken):
        decode_token(token, TEST_SECRET, now=1_000_060)


def test_a_token_signed_with_another_secret_is_refused():
    with pytest.raises(InvalidToken):
        decode_token(create_token(ALICE, "another-secret-" + "y" * 40, 60), TEST_SECRET)


def test_a_tampered_payload_is_refused():
    header, payload, signature = create_token(ALICE, TEST_SECRET, 60).split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=="))
    claims["sub"] = "99999999-9999-4999-8999-999999999999"   # pretend to be someone else, keep the old signature
    forged = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    with pytest.raises(InvalidToken):
        decode_token(f"{header}.{forged}.{signature}", TEST_SECRET)


def test_the_alg_none_attack_is_refused():
    unsigned = jwt.encode({"sub": str(ALICE.id), "email": ALICE.email, "iat": int(time.time()), "exp": int(time.time()) + 600, "iss": "finnie"}, key=None, algorithm="none")
    with pytest.raises(InvalidToken):
        decode_token(unsigned, TEST_SECRET)


@pytest.mark.parametrize("missing", ["exp", "sub", "iss", "iat"])
def test_a_token_missing_a_required_claim_is_refused(missing):
    claims = {"sub": str(ALICE.id), "email": ALICE.email, "iat": int(time.time()), "exp": int(time.time()) + 600, "iss": "finnie"}
    del claims[missing]
    with pytest.raises(InvalidToken):
        decode_token(jwt.encode(claims, TEST_SECRET, algorithm="HS256"), TEST_SECRET)


def test_a_token_from_another_issuer_or_with_a_bad_subject_is_refused():
    base = {"email": "x@example.com", "iat": int(time.time()), "exp": int(time.time()) + 600}
    for claims in ({**base, "sub": str(ALICE.id), "iss": "someone-else"}, {**base, "sub": "not-a-uuid", "iss": "finnie"}):
        with pytest.raises(InvalidToken):
            decode_token(jwt.encode(claims, TEST_SECRET, algorithm="HS256"), TEST_SECRET)


@pytest.mark.parametrize("junk", ["", "abc", "a.b.c", "Bearer x", None])
def test_junk_is_refused_not_a_crash(junk):
    with pytest.raises(InvalidToken):
        decode_token(junk, TEST_SECRET)


# ---------- rate limiter ----------
class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_the_limiter_allows_the_limit_then_says_how_long_to_wait_and_then_forgets():
    clock = Clock()
    limiter = RateLimiter(3, 60, clock)
    for _ in range(3):
        assert limiter.retry_after("k") == 0
        limiter.add("k")
    clock.now = 10
    assert 49 <= limiter.retry_after("k") <= 51
    clock.now = 61
    assert limiter.retry_after("k") == 0


def test_keys_are_independent_and_clear_resets_one():
    limiter = RateLimiter(1, 60, Clock())
    limiter.add("a")
    assert limiter.retry_after("a") > 0 and limiter.retry_after("b") == 0
    limiter.clear("a")
    assert limiter.retry_after("a") == 0


def test_the_table_cannot_grow_without_bound():
    clock = Clock()
    limiter = RateLimiter(1, 1, clock)
    for i in range(RateLimiter.MAX_KEYS + 500):
        limiter.add(f"k{i}")
        clock.now += 0.01
        limiter.retry_after(f"k{i}")
    assert len(limiter._events) <= RateLimiter.MAX_KEYS + 1
