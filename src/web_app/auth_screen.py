"""The login / create-account screen, shown instead of the app until someone is signed in.

The signed-in state is `st.session_state.auth = {"token": ..., "email": ...}`. Streamlit keeps session_state per browser tab, so one person's token is never
visible to another. It is lost when the page is reloaded or the tab is closed (the user simply logs in again): safer than a cookie we would have to protect.
"""
import streamlit as st

from src.web_app.client import ApiError, FinnieClient
from src.web_app.ui_common import show_error

KEEP = ("client",)   # the only session key that survives login and logout: the API client object itself


def signed_in() -> dict | None:
    return st.session_state.get("auth")


def sign_in(client: FinnieClient, reply: dict) -> None:
    """Start fresh for the new user: nothing from an earlier user (chat history, loaded portfolio, remembered failures) may carry over."""
    for key in [k for k in st.session_state if k not in KEEP]:
        del st.session_state[key]
    st.session_state.auth = {"token": reply["access_token"], "email": reply["user"]["email"]}
    client.token = reply["access_token"]


def sign_out(client: FinnieClient, notice: str | None = None) -> None:
    for key in [k for k in st.session_state if k not in KEEP]:
        del st.session_state[key]
    client.token = None
    if notice:
        st.session_state.login_notice = notice


def render(client: FinnieClient) -> None:
    st.title("Finnie")
    st.caption("A finance tutor for beginners. Please log in or create an account to continue.")
    if st.session_state.get("login_notice"):
        st.warning(st.session_state.pop("login_notice"))
    log_in, create = st.tabs(["Log in", "Create account"])
    with log_in:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            if not email.strip() or not password:
                st.error("Enter your email and password.")
            else:
                try:
                    sign_in(client, client.login(email.strip(), password))
                    st.rerun()
                except ApiError as error:
                    if error.status == 429:
                        st.error(error.message)
                    else:
                        show_error(error)
    with create:
        with st.form("register_form"):
            new_email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Password (at least 10 characters)", type="password", key="register_password")
            confirm = st.text_input("Repeat the password", type="password", key="register_confirm")
            created = st.form_submit_button("Create account", type="primary")
        if created:
            if new_password != confirm:
                st.error("The two passwords are not the same.")
            else:
                try:
                    sign_in(client, client.register(new_email.strip(), new_password))
                    st.rerun()
                except ApiError as error:
                    show_error(error)
