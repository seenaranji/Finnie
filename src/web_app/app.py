"""Finnie's web UI.  Start it with:  streamlit run src/web_app/app.py   (the API must be running: uvicorn src.web_app.api:app)

Streamlit re-runs this whole file from the top on every click or keystroke. What must survive a rerun lives in st.session_state; what is expensive
and repeatable is wrapped in st.cache_data (see the tabs).
"""
import sys
from pathlib import Path

import streamlit as st

# Streamlit puts this script's folder on sys.path, not the project root, so `src.*` imports need the root added.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.web_app import auth_screen, style
from src.web_app.client import DEFAULT_URL, ApiError, FinnieClient
from src.web_app.tabs import chat, goals, knowledge, markets, news, portfolio
from src.web_app.ui_common import md

st.set_page_config(page_title="Finnie · your finance tutor", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
style.inject()

TABS = [":material/forum: Chat", ":material/show_chart: Markets", ":material/newspaper: News", ":material/pie_chart: Portfolio", ":material/flag: Goals", ":material/menu_book: Knowledge"]


def get_client() -> FinnieClient:
    if "client" not in st.session_state:   # a test can place a stand-in here before the script runs
        st.session_state.client = FinnieClient(DEFAULT_URL)
    return st.session_state.client


@st.cache_data(ttl=10, show_spinner=False)
def server_status(_client: FinnieClient, url: str) -> dict:
    """Cached for 10 s so the many reruns of a chat turn make one health call. (The underscore tells Streamlit not to try hashing the client; `url` is the cache key.)
    A failure raises, and Streamlit never caches an exception, so a server that is down is re-checked on the next rerun."""
    return _client.health()


def header(client: FinnieClient) -> None:
    """Brand on the left; on the right the connection state and an account menu (replaces the old sidebar, which took a third of the screen)."""
    brand, status, account = st.columns([5, 2, 2.4], vertical_alignment="center")
    with brand:
        style.brand_header()
    problem = None
    with status:
        try:
            server_status(client, client.base_url)
            style.status_pill(True)
        except ApiError as error:
            problem = error
            style.status_pill(False)
    with account:
        with st.popover(f":material/account_circle: {md(st.session_state.auth['email'])}", width="stretch"):
            st.caption("Signed in as")
            st.text(st.session_state.auth["email"])   # st.text: shown exactly as typed, never read as markdown or turned into a link
            if st.button("Log out", width="stretch", icon=":material/logout:"):
                auth_screen.sign_out(client)
                st.rerun()
    if problem:
        st.error(problem.message)
    style.disclaimer()


client = get_client()
if not auth_screen.signed_in():   # nobody gets past this line without an account: no tabs, no API calls
    client.token = None
    auth_screen.render(client)
    st.stop()
client.token = st.session_state.auth["token"]
header(client)
chat_tab, markets_tab, news_tab, portfolio_tab, goals_tab, knowledge_tab = st.tabs(TABS)
with chat_tab:
    chat.render(client)
with markets_tab:
    markets.render(client)
with news_tab:
    news.render(client)
with portfolio_tab:
    portfolio.render(client)
with goals_tab:
    goals.render(client)
with knowledge_tab:
    knowledge.render(client)
