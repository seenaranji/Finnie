"""Small helpers every tab shares."""
import time

import streamlit as st

from src.web_app.client import ApiError

DISCLAIMER = "Finnie is an educational tool, not financial advice. Nothing here is a recommendation to buy or sell anything."


def show_error(error: ApiError, what: str = "that") -> None:
    """One way to show every failure: what happened in plain words, whether trying again can help, and the request id to quote."""
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
