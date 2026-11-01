"""The login / create-account screen, shown instead of the app until someone is signed in.

The signed-in state is `st.session_state.auth = {"token": ..., "email": ...}`. Streamlit keeps session_state per browser tab, so one person's token is never
visible to another. It is lost when the page is reloaded or the tab is closed (the user simply logs in again): safer than a cookie we would have to protect.
"""
import streamlit as st

from src.web_app import style
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


HERO = """
<div class="hero">
  <h1>Learn money, one plain-English answer at a time.</h1>
  <p class="lead">Finnie explains investing, checks live prices, tracks your own portfolio and plans your savings goals. It teaches; it never tells you what to buy.</p>
  <ul>
    <li><span class="ic">💬</span><div><b>Ask anything</b><br>ETFs, bonds, taxes, jargon, with the source of every answer.</div></li>
    <li><span class="ic">📈</span><div><b>See the market</b><br>Index levels, prices and the latest headlines.</div></li>
    <li><span class="ic">🥧</span><div><b>Know what you own</b><br>Save your holdings and see them grouped by kind.</div></li>
    <li><span class="ic">🎯</span><div><b>Plan a goal</b><br>What to save each month, with honest what-if ranges.</div></li>
  </ul>
</div>
"""   # fixed text written by us (it goes into HTML): nothing a user or a model supplied may ever be placed in here


def render(client: FinnieClient) -> None:
    hero, card = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with hero:
        style.brand_header()
        st.markdown(HERO, unsafe_allow_html=True)
    with card:
        st.subheader("Welcome")
        st.caption("Log in, or create a free account to get started.")
        if st.session_state.get("login_notice"):
            st.warning(st.session_state.pop("login_notice"))
        log_in, create = st.tabs(["Log in", "Create account"])
        with log_in:
            with st.form("login_form", border=True):
                email = st.text_input("Email", key="login_email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Log in", type="primary", width="stretch")
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
            with st.form("register_form", border=True):
                new_email = st.text_input("Email", key="register_email", placeholder="you@example.com")
                new_password = st.text_input("Password (at least 10 characters)", type="password", key="register_password")
                confirm = st.text_input("Repeat the password", type="password", key="register_confirm")
                created = st.form_submit_button("Create account", type="primary", width="stretch")
            if created:
                if new_password != confirm:
                    st.error("The two passwords are not the same.")
                else:
                    try:
                        sign_in(client, client.register(new_email.strip(), new_password))
                        st.rerun()
                    except ApiError as error:
                        show_error(error)
    style.disclaimer()
