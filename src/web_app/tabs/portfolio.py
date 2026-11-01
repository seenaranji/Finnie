"""Tab 3: Portfolio. Streamlit concepts: st.file_uploader, st.form, a cached POST (an analysis costs a model call, so the same input must not run twice)."""
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.web_app.client import ApiError, FinnieClient
from src.web_app.ui_common import guarded, show_error

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_portfolios"
SAMPLES = {"Diversified": "diversified.csv", "Bond heavy": "bond_heavy.csv", "One big stock": "concentrated_single_stock.csv"}
MAX_CHARS = 20000   # the API's limit; checked here only to give a friendlier message than a 422
EXAMPLE = "ticker,shares,avg_cost\nVTI,60,220.00\nBND,100,74.00"


@st.cache_data(ttl=300, show_spinner=False)
def cached_analysis(_client: FinnieClient, url: str, csv_text: str, question: str) -> dict:
    return _client.analyze_portfolio(csv_text, question or None)


def allocation_figure(asset_class_pct: dict) -> go.Figure:
    figure = go.Figure(go.Pie(labels=list(asset_class_pct), values=list(asset_class_pct.values()), hole=0.4, sort=False, textinfo="label+percent"))
    figure.update_layout(title="Allocation by asset class", showlegend=False, margin={"l": 10, "r": 10, "t": 50, "b": 10}, height=360)
    return figure


def concentration_figure(holdings: list[dict]) -> go.Figure:
    ordered = sorted(holdings, key=lambda h: h["weight_pct"])
    figure = go.Figure(go.Bar(x=[h["weight_pct"] for h in ordered], y=[h["ticker"] for h in ordered], orientation="h",
                              text=[f"{h['weight_pct']:.1f}%" for h in ordered], textposition="outside"))
    figure.update_layout(title="Share of the portfolio by holding", xaxis_title="% of portfolio", xaxis_range=[0, 105], margin={"l": 10, "r": 10, "t": 50, "b": 10},
                         height=max(240, 60 + 40 * len(ordered)))
    return figure


def correlation_figure(correlation: dict) -> go.Figure:
    tickers, matrix = correlation["tickers"], correlation["matrix"]
    figure = go.Figure(go.Heatmap(z=matrix, x=tickers, y=tickers, zmin=-1, zmax=1, colorscale="RdBu_r", text=matrix, texttemplate="%{text:.2f}",
                                  colorbar={"title": "correlation"}))
    figure.update_layout(title="How similarly the holdings moved (daily returns, past year)", yaxis_autorange="reversed", margin={"l": 10, "r": 10, "t": 50, "b": 10},
                         height=max(320, 90 + 55 * len(tickers)))
    return figure


def pick_source() -> tuple[str, str | None]:
    """Chosen OUTSIDE the form: a widget inside st.form does not rerun the script until the form is submitted, so a choice that changes which inputs
    appear (upload vs paste) must live out here."""
    source = st.radio("Where are your holdings?", ["Sample portfolio", "Upload a CSV", "Paste CSV text"], horizontal=True)
    sample = st.selectbox("Sample", list(SAMPLES)) if source == "Sample portfolio" else None
    return source, sample


def read_input(source: str, sample: str | None) -> tuple[str | None, str | None]:
    """Inside the form. Returns (csv text, problem). Exactly one is None."""
    if source == "Sample portfolio":
        return (SAMPLES_DIR / SAMPLES[sample]).read_text(), None
    if source == "Upload a CSV":
        uploaded = st.file_uploader("CSV with the columns ticker, shares, avg_cost", type=["csv", "txt"])
        if uploaded is None:
            return None, "Choose a file first."
        try:
            return uploaded.getvalue().decode("utf-8-sig"), None
        except UnicodeDecodeError:
            return None, "That file isn't plain text. Save it as a CSV (UTF-8) and try again."
    text = st.text_area("Paste your holdings", value="", placeholder=EXAMPLE, height=140)
    return (text, None) if text.strip() else (None, "Paste some holdings first.")


def show_result(result: dict) -> None:
    metrics = result["metrics"]
    st.markdown(result["text"])
    st.caption(result["disclaimer"])
    columns = st.columns(4)
    columns[0].metric("Total value", f"${metrics['total_value_usd']:,}")
    columns[1].metric("Total gain", "n/a" if metrics["total_gain_pct"] is None else f"{metrics['total_gain_pct']:+.1f}%")
    columns[2].metric("Holdings", metrics["holding_count"])
    score = metrics["diversification_score_out_of_100"]["score"]
    columns[3].metric("Diversification score", f"{score:.0f} / 100", help=metrics["diversification_score_out_of_100"]["built_from"])
    left, right = st.columns(2)
    left.plotly_chart(allocation_figure(metrics["asset_class_pct"]), width="stretch")
    right.plotly_chart(concentration_figure(metrics["holdings"]), width="stretch")
    if result.get("correlation"):
        st.plotly_chart(correlation_figure(result["correlation"]), width="stretch")
    else:
        st.info("There isn't enough overlapping price history to show how the holdings move together.")
    for note in metrics.get("notes", []):
        st.info(note)
    if result["sources"]:
        st.caption("Data: " + "; ".join(s["title"] for s in result["sources"]))


def render(client: FinnieClient) -> None:
    st.markdown("See how a set of holdings is spread out. Nothing is stored: your file is sent for the analysis and kept only in this browser session.")
    source, sample = pick_source()
    with st.form("portfolio"):
        csv_text, problem = read_input(source, sample)
        question = st.text_input("Anything specific to ask? (optional)", max_chars=500)
        submitted = st.form_submit_button("Analyze")
    if submitted:
        st.session_state.portfolio_request = None if problem else (csv_text, question.strip())
        st.session_state.portfolio_forced = True
        st.session_state.portfolio_problem = problem
    if st.session_state.get("portfolio_problem"):
        st.warning(st.session_state.portfolio_problem)
        return
    request = st.session_state.get("portfolio_request")
    if not request:
        return
    csv_text, question = request
    if len(csv_text) > MAX_CHARS:
        st.error(f"That file is too large ({len(csv_text):,} characters; the limit is {MAX_CHARS:,}).")
        return
    try:
        with st.spinner("Analyzing your portfolio…"):
            result = guarded(("portfolio", csv_text, question), lambda: cached_analysis(client, client.base_url, csv_text, question),
                             force=st.session_state.pop("portfolio_forced", False))
    except ApiError as error:
        show_error(error)
        if error.code == "invalid_portfolio":
            st.code(EXAMPLE, language="text")
        return
    st.session_state.portfolio_csv = csv_text   # the Chat tab now sends it with every message
    st.success("Portfolio analyzed. Chat questions about 'my portfolio' will now use it.")
    show_result(result)
    if st.button("Forget this portfolio"):
        st.session_state.portfolio_csv = None
        st.session_state.portfolio_request = None
        st.rerun()
