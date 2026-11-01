"""Tab: News. Headlines from the same feeds the news agent reads. The Chat tab can summarize them ("what's the news on AAPL?"); this tab just lists them.
Streamlit concepts: st.cache_data with a short ttl, a form for the optional ticker."""
import streamlit as st

from src.web_app.client import ApiError, FinnieClient
from src.web_app.ui_common import guarded, show_error


@st.cache_data(ttl=300, show_spinner=False)
def cached_news(_client: FinnieClient, url: str, ticker: str) -> dict:
    return _client.ticker_news(ticker) if ticker else _client.market_news()


def render(client: FinnieClient) -> None:
    with st.form("news_form"):
        ticker = st.text_input("Ticker (leave empty for general market news)", placeholder="e.g. AAPL", max_chars=10)
        submitted = st.form_submit_button("Show news")
    if submitted:
        st.session_state.news_ticker = ticker.strip().upper()
    ticker = st.session_state.get("news_ticker", "")
    try:
        with st.spinner("Loading headlines…"):
            batch = guarded(("news", ticker), lambda: cached_news(client, client.base_url, ticker), force=submitted)
    except ApiError as error:
        show_error(error)
        return
    st.subheader(f"News: {ticker}" if ticker else "Market news")
    st.caption(f"Fetched {batch['fetched_at'][:16].replace('T', ' ')} UTC. Headlines come from third-party feeds; Finnie does not vouch for them.")
    if batch["feed_errors"]:
        st.warning("Some news sources did not respond, so this list may be incomplete.")
    if batch["withheld"]:
        st.caption(f"{batch['withheld']} item(s) were left out because their text looked unsafe.")
    if not batch["items"]:
        st.info("No recent headlines found.")
    for item in batch["items"]:
        with st.container(border=True):
            st.markdown(f"**[{item['title']}]({item['link']})**")
            if item["summary"]:
                st.write(item["summary"])
            st.caption(f"{item['source']} · {item['published'][:10] if item['published'] else 'date unknown'}")
