"""Tab 2: Markets. Streamlit concepts: st.cache_data with a TTL (a rerun must not refetch prices), st.form (typing does not rerun), st.metric, Plotly."""
import plotly.graph_objects as go
import streamlit as st

from src.web_app.client import ApiError, FinnieClient
from src.web_app.ui_common import guarded, show_error

PERIODS = {"5 days": "5d", "1 month": "1mo", "3 months": "3mo", "6 months": "6mo", "1 year": "1y", "Year to date": "ytd"}


# The leading underscore keeps Streamlit from hashing the client; `url` (and the ticker/period) are the cache key. Exceptions are never cached.
@st.cache_data(ttl=60, show_spinner=False)
def cached_overview(_client: FinnieClient, url: str) -> dict:
    return _client.overview()


@st.cache_data(ttl=60, show_spinner=False)
def cached_quote(_client: FinnieClient, url: str, ticker: str) -> dict:
    return _client.quote(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_history(_client: FinnieClient, url: str, ticker: str, period: str) -> dict:
    return _client.history(ticker, period)


def history_figure(history: dict) -> go.Figure:
    dates = [p["date"] for p in history["points"]]
    closes = [p["close"] for p in history["points"]]
    figure = go.Figure(go.Scatter(x=dates, y=closes, mode="lines", line={"width": 2}, name=history["symbol"], hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>"))
    figure.update_layout(title=f"{history['symbol']} daily closing price ({history['period']})", yaxis_title="Close (USD)", xaxis_title=None,
                         margin={"l": 10, "r": 10, "t": 50, "b": 10}, height=380)
    figure.update_yaxes(tickprefix="$", tickformat=",.0f")
    return figure


def freshness(item: dict) -> None:
    """Say how old a number is. Stale data is shown, but never silently."""
    if item["is_stale"]:
        st.warning(f"The data provider is not responding, so this is an older saved value (about {item['age_seconds'] / 60:.0f} min old). It may be out of date.")
    as_of = item.get("as_of")
    st.caption(f"Closing price as of {as_of}" if as_of else "Latest available price" + f"  ·  source: {item['source']}  ·  fetched {item['fetched_at'][11:19]} UTC")


def overview_section(client: FinnieClient) -> None:
    st.subheader("Market overview")
    refresh = st.button("Refresh", key="refresh_overview")
    if refresh:
        cached_overview.clear()
    try:
        overview = guarded("overview", lambda: cached_overview(client, client.base_url), force=refresh)
    except ApiError as error:
        show_error(error)
        return
    columns = st.columns(max(len(overview["indices"]), 1))
    for column, index in zip(columns, overview["indices"]):
        quote = index["quote"]
        column.metric(index["name"], f"{quote['price']:,.2f}", None if quote["change_pct"] is None else f"{quote['change_pct']:+.2f}%")
    if overview["indices"]:
        freshness(overview["indices"][0]["quote"])
    if overview["unavailable"]:
        st.warning("Could not load: " + ", ".join(overview["unavailable"]))


def lookup_section(client: FinnieClient) -> None:
    st.subheader("Look up a stock or fund")
    with st.form("lookup"):
        left, right = st.columns([2, 1])
        ticker = left.text_input("Ticker symbol", placeholder="AAPL, VOO, BRK-B …", max_chars=10)
        label = right.selectbox("Period", list(PERIODS), index=1)
        submitted = st.form_submit_button("Look up")
    if submitted:
        st.session_state.market_lookup = (ticker.strip().upper(), PERIODS[label])
        st.session_state.market_lookup_forced = True
    if "market_lookup" not in st.session_state:
        return
    symbol, period = st.session_state.market_lookup
    if not symbol:
        st.warning("Please type a ticker symbol, such as AAPL.")
        return
    force = st.session_state.pop("market_lookup_forced", False)   # true only on the run where the user pressed Look up
    try:
        with st.spinner(f"Loading {symbol}…"):
            quote = guarded(("quote", symbol), lambda: cached_quote(client, client.base_url, symbol), force=force)
            history = guarded(("history", symbol, period), lambda: cached_history(client, client.base_url, symbol, period), force=force)
    except ApiError as error:
        if error.status == 422:
            st.error(f"'{symbol}' doesn't look like a ticker symbol. Use letters and numbers, for example AAPL or BRK-B.")
        else:
            show_error(error)
        return
    unit = "" if quote["symbol"].startswith("^") else "$"   # an index level is points, not dollars
    st.metric(quote["symbol"], f"{unit}{quote['price']:,.2f}", None if quote["change_pct"] is None else f"{quote['change_pct']:+.2f}% vs previous close")
    freshness(quote)
    if history["points"]:
        st.plotly_chart(history_figure(history), width="stretch")
        if history["period"] != period:
            st.caption(f"Only {history['period']} of data was available for this period.")
    else:
        st.info("No price history is available for this symbol.")


def render(client: FinnieClient) -> None:
    overview_section(client)
    st.divider()
    lookup_section(client)
