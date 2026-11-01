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
