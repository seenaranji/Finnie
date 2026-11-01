"""Finnie's web UI.  Start it with:  streamlit run src/web_app/app.py   (the API must be running: uvicorn src.web_app.api:app)

Streamlit re-runs this whole file from the top on every click or keystroke. What must survive a rerun lives in st.session_state; what is expensive
and repeatable is wrapped in st.cache_data (see the tabs).
"""
import sys
from pathlib import Path

import streamlit as st

# Streamlit puts this script's folder on sys.path, not the project root, so `src.*` imports need the root added.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.web_app.client import DEFAULT_URL, ApiError, FinnieClient
from src.web_app.tabs import chat, goals, knowledge, markets, news, portfolio
from src.web_app.ui_common import DISCLAIMER

st.set_page_config(page_title="Finnie", page_icon="💬", layout="wide")


def get_client() -> FinnieClient:
    if "client" not in st.session_state:   # a test can place a stand-in here before the script runs
        st.session_state.client = FinnieClient(DEFAULT_URL)
    return st.session_state.client


@st.cache_data(ttl=10, show_spinner=False)
def server_status(_client: FinnieClient, url: str) -> dict:
    """Cached for 10 s so the many reruns of a chat turn make one health call. (The underscore tells Streamlit not to try hashing the client; `url` is the cache key.)
    A failure raises, and Streamlit never caches an exception, so a server that is down is re-checked on the next rerun."""
    return _client.health()


def sidebar(client: FinnieClient) -> None:
    with st.sidebar:
        st.title("Finnie")
        st.caption("A finance tutor for beginners")
        try:
            server_status(client, client.base_url)
            st.success("Server connected", icon="✅")
        except ApiError as error:
            st.error(error.message)
        if st.button("New conversation", width="stretch"):
            chat.new_conversation()
            st.rerun()
        if st.session_state.get("session_id"):
            st.caption(f"Session `{st.session_state.session_id}`")
        st.caption("Portfolio loaded for chat: " + ("yes" if st.session_state.get("portfolio_csv") else "no"))


client = get_client()
sidebar(client)
st.info(DISCLAIMER, icon="ℹ️")
chat_tab, markets_tab, news_tab, portfolio_tab, goals_tab, knowledge_tab = st.tabs(["Chat", "Markets", "News", "Portfolio", "Goals", "Knowledge"])
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
