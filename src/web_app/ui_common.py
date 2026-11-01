"""Small helpers every tab shares."""
import time

import streamlit as st

from src.web_app.client import ApiError


def md(text) -> str:
    """Text that came from a model, a feed or a database, made safe to give to st.markdown / st.caption.
    Streamlit reads a pair of dollar signs as a formula, so "$80,000 ... $2.2 million" turned into code-looking symbols and "$" amounts in answers broke
    (found in the news list and the chat starters). An escaped dollar sign shows as a plain one. (This does NOT neutralise other markdown such as links or emphasis; text that must be inert
    should go through st.text instead.)"""
    return str(text).replace("\\", "\\\\").replace("$", "\\$")


DISCLAIMER = "Finnie is an educational tool, not financial advice. Nothing here is a recommendation to buy or sell anything."


def show_error(error: ApiError, what: str = "that") -> None:
    """One way to show every failure: what happened in plain words, whether trying again can help, and the request id to quote."""
    if error.status == 401 and st.session_state.get("auth"):
        # The server no longer accepts our token (it expired, or the server's secret changed). Back to the login screen with a plain explanation.
        from src.web_app.auth_screen import sign_out
        sign_out(st.session_state.client, "Your session has ended. Please log in again.")
        st.rerun()
    st.error(error.message)
    hint = "This may be temporary. Try again in a moment." if error.retryable else "Please change the input and try again."
    st.caption(f"{hint}" + (f"  Request id: `{error.request_id}`" if error.request_id else ""))
    for detail in error.details:
        st.caption(f"- {detail.get('field', '')} {detail.get('problem', '')}".strip())


RETRY_AFTER_SECONDS = 15


def guarded(key, call, force: bool = False):
    """Run `call`, but do not re-send a request that just failed. Streamlit re-runs the script on every click and never caches an exception, so without this a
    failing request (an unknown ticker that takes a second to refuse, a server that is down) is sent again on EVERY rerun. `force=True` (the user pressed the
    button again) always tries. Raises the remembered ApiError while the failure is fresh."""
    failures = st.session_state.setdefault("_failures", {})
    if force:
        failures.pop(key, None)
    seen = failures.get(key)
    if seen and time.time() - seen[0] < RETRY_AFTER_SECONDS:
        raise seen[1]
    try:
        result = call()
    except ApiError as error:
        failures[key] = (time.time(), error)
        raise
    failures.pop(key, None)
    return result
