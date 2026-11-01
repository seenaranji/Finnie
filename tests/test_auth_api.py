"""Authentication through the HTTP API: register, log in, the locked front door, token abuse, and conversations that belong to one user."""
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from src.auth.repository import UserRepository
from src.auth.security import AuthUser, create_token
from src.auth.service import AuthService
from src.core.config import MissingSecretError
from src.data.cache import TTLCache
from src.data.portfolio_repository import PortfolioRepository
from src.data.market_data import MarketDataService
from src.web_app.api import create_app
from src.web_app.services import Services
from tests.auth_support import ALICE, BOB, TEST_SECRET, headers_for, settings_with_secret
from tests.test_api import FakeRetriever, FlakyProvider, ReplyLLM

PASSWORD = "correct horse battery"
OPEN_ROUTES = {("GET", "/health"), ("POST", "/auth/register"), ("POST", "/auth/login")}   # the only doors without a lock; the docs are added by FastAPI itself


def make_app(pg=None, graph=None):
    settings = settings_with_secret()
    market = MarketDataService(FlakyProvider(), TTLCache(), settings, sleep=lambda s: None)
    auth = AuthService(UserRepository(pg) if pg is not None else None, settings)
    services = Services(settings, market=market, retriever=FakeRetriever(), llm=ReplyLLM(), graph=graph, auth=auth,
                        portfolios=PortfolioRepository(pg) if pg is not None else None)   # never fall back to the real database inside a test
    return create_app(services)


@pytest.fixture
def api(pg):
    return TestClient(make_app(pg), raise_server_exceptions=False)


def register(api, email="alice@example.com", password=PASSWORD):
    return api.post("/auth/register", json={"email": email, "password": password})


# ---------- the front door ----------
def operations(app):
    """Every (METHOD, path, operation) the app serves, from its OpenAPI schema. (app.routes does not list routes added through an included router,
    so a walk over it would silently skip most of the app: this test's first version did exactly that and passed while checking nothing.)"""
    return [(method.upper(), path, op) for path, item in app.openapi()["paths"].items() for method, op in item.items()]


def test_every_route_except_the_open_ones_refuses_a_request_with_no_token():
    app = make_app()
    client = TestClient(app, raise_server_exceptions=False)
    checked = []
    for method, path, _ in operations(app):
        if (method, path) in OPEN_ROUTES:
            continue
        url = path.replace("{ticker}", "AAPL").replace("{portfolio_id}", "00000000-0000-4000-8000-000000000000").replace("{doc_id}", "x")
        response = client.request(method, url, json={} if method in ("POST", "PUT") else None)
        assert response.status_code == 401 and response.json()["code"] == "not_authenticated", f"{method} {path} is open to anyone"
        checked.append((method, path))
    assert len(checked) >= 18   # guards the guard: if the walk finds almost nothing, it would pass for the wrong reason


def test_the_open_routes_are_exactly_the_documented_ones():
    # an operation is locked when its schema declares the bearer requirement; adding a route without the lock makes this fail
    unlocked = {(method, path) for method, path, op in operations(make_app()) if "security" not in op}
    assert unlocked == OPEN_ROUTES


def test_health_and_the_docs_are_public():
    client = TestClient(make_app())
    assert client.get("/health").status_code == 200 and client.get("/openapi.json").status_code == 200 and client.get("/docs").status_code == 200


def test_the_401_has_the_standard_error_shape_and_the_bearer_challenge():
    r = TestClient(make_app()).get("/market/overview")
    assert r.status_code == 401 and set(r.json()) == {"code", "message", "request_id", "details"} and r.headers["www-authenticate"] == "Bearer" and r.headers["x-request-id"]


@pytest.mark.parametrize("header", ["Bearer ", "Bearer not.a.token", "Basic YWxpY2U6cGFzcw==", "Token abc", "bearer"])
def test_malformed_credentials_are_401(header):
    assert TestClient(make_app()).get("/auth/me", headers={"Authorization": header}).status_code == 401


def test_an_expired_token_is_401():
    expired = create_token(ALICE, TEST_SECRET, 60, now=time.time() - 2 * 3600)
    assert TestClient(make_app()).get("/auth/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401


def test_a_token_signed_with_the_wrong_secret_or_without_a_signature_is_401():
    client = TestClient(make_app())
    other = create_token(ALICE, "some-other-secret-" + "z" * 40, 60)
    unsigned = jwt.encode({"sub": str(ALICE.id), "email": ALICE.email, "iat": int(time.time()), "exp": int(time.time()) + 600, "iss": "finnie"}, key=None, algorithm="none")
    for token in (other, unsigned):
        assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_a_valid_token_gets_in_and_me_returns_the_user():
    r = TestClient(make_app()).get("/auth/me", headers=headers_for(BOB))
    assert r.status_code == 200 and r.json() == {"id": str(BOB.id), "email": BOB.email}


def test_options_preflight_is_answered_and_allows_the_authorization_header():
    r = TestClient(make_app()).options("/chat", headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
    assert r.status_code == 200 and "authorization" in r.headers["access-control-allow-headers"].lower()


def test_the_server_refuses_to_start_without_a_signing_secret():
    settings = settings_with_secret(secret="short")
    app = create_app(Services(settings, auth=AuthService(None, settings)))
    with pytest.raises(MissingSecretError):
        with TestClient(app):
            pass


# ---------- register and log in ----------
def test_registering_returns_a_working_token_for_the_new_user(api):
    r = register(api)
    body = r.json()
    assert r.status_code == 201 and body["token_type"] == "bearer" and body["expires_in"] == 3600 and body["user"]["email"] == "alice@example.com"
    assert api.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"}).json()["email"] == "alice@example.com"
    assert "password" not in r.text


def test_logging_in_returns_a_token_for_the_same_user(api):
    registered = register(api).json()
    r = api.post("/auth/login", json={"email": "ALICE@example.com", "password": PASSWORD})
    assert r.status_code == 200 and r.json()["user"]["id"] == registered["user"]["id"]
    assert api.get("/market/overview", headers={"Authorization": f"Bearer {r.json()['access_token']}"}).status_code in (200, 502)   # past the door (the fake market decides the rest)


def test_wrong_password_and_unknown_email_are_the_same_401(api):
    register(api)
    wrong = api.post("/auth/login", json={"email": "alice@example.com", "password": "not the password"})
    unknown = api.post("/auth/login", json={"email": "nobody@example.com", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["code"] == unknown.json()["code"] == "invalid_credentials" and wrong.json()["message"] == unknown.json()["message"]
    assert wrong.headers["www-authenticate"] == "Bearer"


def test_weak_input_is_422_with_every_problem_listed(api):
    r = register(api, email="not-an-email", password="short")
    assert r.status_code == 422 and r.json()["code"] == "invalid_auth_input" and len(r.json()["details"]) == 2
    assert register(api, email="a@example.com", password="x" * 300).status_code == 422   # over the model's size cap


def test_a_second_registration_with_the_same_email_is_refused(api):
    register(api)
    r = register(api)
    assert r.status_code == 422 and "already exists" in r.json()["details"][0]["problem"]


def test_too_many_wrong_passwords_is_429_with_retry_after(api):
    register(api)
    for _ in range(5):
        assert api.post("/auth/login", json={"email": "alice@example.com", "password": "wrong password!"}).status_code == 401
    r = api.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert r.status_code == 429 and r.json()["code"] == "too_many_attempts" and int(r.headers["retry-after"]) > 0


def test_the_response_and_the_log_never_contain_the_password(api, caplog):
    import logging
    with caplog.at_level(logging.DEBUG):
        register(api, password="a-very-unusual-password-9")
        api.post("/auth/login", json={"email": "alice@example.com", "password": "another-unusual-password-9"})
    assert "unusual-password" not in caplog.text


def test_a_user_who_registered_can_save_a_portfolio_straight_away(api):
    token = register(api).json()["access_token"]
    holdings = [{"ticker": "AAA", "shares": "1", "avg_cost": "1"}]
    saved = api.post("/portfolios", json={"name": "mine", "holdings": holdings}, headers={"Authorization": f"Bearer {token}"})
    assert saved.status_code == 201
    other = register(api, email="bob@example.com").json()["access_token"]
    assert api.get(f"/portfolios/{saved.json()['id']}", headers={"Authorization": f"Bearer {other}"}).status_code == 404


# ---------- conversations belong to one user ----------
class RecordingGraph:
    """Stands in for the workflow graph: remembers the session id and thread id it was given, answers with a fixed reply."""
    def __init__(self):
        self.seen = []

    def invoke(self, state, config):
        self.seen.append((state["session_id"], config["configurable"]["thread_id"]))
        reply = {"text": "hi", "agent_name": "finance_qa", "sources": [], "disclaimer": "", "metadata": {}}
        return {"response": reply, "route": "finance_qa", "turn": 1, "trace": []}


def test_two_users_sending_the_same_session_id_get_separate_conversations():
    graph = RecordingGraph()
    client = TestClient(make_app(graph=graph))
    a = client.post("/chat", json={"message": "hello", "session_id": "shared-id"}, headers=headers_for(ALICE))
    b = client.post("/chat", json={"message": "hello", "session_id": "shared-id"}, headers=headers_for(BOB))
    assert a.json()["session_id"] == b.json()["session_id"] == "shared-id"            # the client sees the plain id
    (a_state, a_thread), (b_state, b_thread) = graph.seen
    assert a_thread == f"{ALICE.id}:shared-id" and b_thread == f"{BOB.id}:shared-id" and a_thread != b_thread   # but memory is keyed by owner + id


def test_a_session_started_by_the_server_is_also_tied_to_the_user():
    graph = RecordingGraph()
    r = TestClient(make_app(graph=graph)).post("/chat", json={"message": "hello"}, headers=headers_for(ALICE))
    assert graph.seen[0][1] == f"{ALICE.id}:{r.json()['session_id']}"
