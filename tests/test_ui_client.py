import httpx
import pytest

from src.web_app.client import ApiError, FinnieClient


def client_with(handler):
    return FinnieClient("http://api.test", transport=httpx.MockTransport(handler))


def test_a_good_reply_is_returned_as_json():
    assert client_with(lambda r: httpx.Response(200, json={"status": "ok"})).health() == {"status": "ok"}


def test_an_api_error_body_becomes_an_api_error_with_the_request_id():
    body = {"code": "ticker_not_found", "message": "I couldn't find a ticker called ZZZZ.", "request_id": "abc", "details": []}
    with pytest.raises(ApiError) as caught:
        client_with(lambda r: httpx.Response(404, json=body)).quote("ZZZZ")
    error = caught.value
    assert (error.status, error.code, error.request_id, error.retryable) == (404, "ticker_not_found", "abc", False)


def test_a_server_down_connection_error_is_a_friendly_retryable_error():
    def refuse(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(ApiError) as caught:
        client_with(refuse).health()
    assert "can't reach" in caught.value.message and caught.value.retryable and "refused" not in caught.value.message


def test_a_timeout_is_a_friendly_retryable_error():
    def slow(request):
        raise httpx.ReadTimeout("slow")
    with pytest.raises(ApiError) as caught:
        client_with(slow).chat("hi")
    assert "too long" in caught.value.message and caught.value.retryable


def test_a_garbled_error_page_does_not_leak_into_the_message():
    with pytest.raises(ApiError) as caught:
        client_with(lambda r: httpx.Response(502, text="<html>Bad gateway nginx 1.2</html>")).health()
    assert "nginx" not in caught.value.message and caught.value.status == 502


def test_caret_symbols_are_encoded_and_a_session_id_is_only_sent_when_known():
    seen = []

    def spy(request):
        seen.append(request)
        return httpx.Response(200, json={})
    c = client_with(spy)
    c.quote("^GSPC")
    c.chat("hi")
    c.chat("hi", session_id="s1", portfolio_csv="a,b")
    assert "%5EGSPC" in str(seen[0].url) and b"session_id" not in seen[1].content and b'"session_id":"s1"' in seen[2].content.replace(b" ", b"")


def test_news_calls_the_market_and_ticker_endpoints():
    seen = []
    client = client_with(lambda r: (seen.append(r.url.path), httpx.Response(200, json={"items": []}))[1])
    client.market_news()
    client.ticker_news(" AAPL ")
    assert seen == ["/news/market", "/news/ticker/AAPL"]


def test_the_token_is_sent_on_every_call_once_set_and_not_before():
    seen = []
    client = client_with(lambda r: (seen.append(r.headers.get("authorization")), httpx.Response(200, json={}))[1])
    client.health()
    client.token = "abc.def.ghi"
    client.overview()
    client.chat("hi")
    assert seen == [None, "Bearer abc.def.ghi", "Bearer abc.def.ghi"]


def test_login_and_register_post_the_credentials_as_json():
    sent = []
    client = client_with(lambda r: (sent.append((r.method, r.url.path, r.content)), httpx.Response(200, json={"access_token": "t"}))[1])
    client.login("a@example.com", "pw")
    client.register("b@example.com", "pw2")
    assert [(m, p) for m, p, _ in sent] == [("POST", "/auth/login"), ("POST", "/auth/register")]
    assert b'"email":"a@example.com"' in sent[0][2].replace(b" ", b"")


def test_a_401_becomes_an_api_error_with_status_401_so_the_ui_can_log_out():
    body = {"code": "not_authenticated", "message": "Please log in to use Finnie.", "request_id": "r1", "details": []}
    with pytest.raises(ApiError) as caught:
        client_with(lambda r: httpx.Response(401, json=body)).overview()
    assert (caught.value.status, caught.value.code, caught.value.retryable) == (401, "not_authenticated", False)
