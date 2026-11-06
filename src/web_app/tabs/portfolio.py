"""Tab 3: Portfolio, in three modes (a radio decides which one runs, so the others make no API calls):
  Enter my holdings           type symbol, kind, shares, cost and date into a table, name it, save it (PostgreSQL gives it a unique id)
  My saved portfolios         pick or load by id; shown GROUPED by kind with cost, value, gain and share of the total; one editable dashboard of everything, explain, use in chat, delete
  Analyze a file or sample    the original: nothing is stored, a CSV is sent for one analysis
Streamlit concepts: st.data_editor (a spreadsheet-like table), widget keys that change to reset a form, st.cache_data keyed by user, st.file_uploader, a cached POST.
"""
import re
from datetime import date
from pathlib import Path
from uuid import UUID

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.asset_types import ASSET_TYPES, PROPERTY_TYPES
from src.web_app import charts
from src.web_app.client import ApiError, FinnieClient
from src.web_app.ui_common import guarded, md, show_error

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_portfolios"
SAMPLES = {"Diversified": "diversified.csv", "Bond heavy": "bond_heavy.csv", "One big stock": "concentrated_single_stock.csv"}
MAX_CHARS = 20000   # the API's limit; checked here only to give a friendlier message than a 422
EXAMPLE = "ticker,shares,avg_cost\nVTI,60,220.00\nBND,100,74.00"
MODES = ["Enter my holdings", "My saved portfolios", "Analyze a file or sample"]
LABEL_TO_KEY = {label: key for key, label in ASSET_TYPES.items()}
COLUMNS = {"symbol": "Symbol", "type": "Type", "shares": "Shares", "cost": "Avg cost per share ($)", "date": "Purchase date"}
PROPERTY_LABEL_TO_KEY = {label: key for key, label in PROPERTY_TYPES.items()}
PCOLS = {"name": "Property (name or address)", "type": "Kind", "price": "Purchase price ($)", "date": "Purchase date", "value": "Current value ($)",
         "mortgage": "Mortgage owed ($)", "payment": "Monthly mortgage payment ($)", "rate": "Mortgage rate (%)", "rent": "Monthly rent ($)", "costs": "Monthly expenses ($)"}
PROPERTY_FIELD_NAMES = {"name": "name", "property_type": "kind", "purchase_price": "purchase price", "purchase_date": "purchase date", "current_value": "current value",
                        "mortgage_balance": "mortgage", "monthly_mortgage_payment": "monthly mortgage payment", "mortgage_rate_pct": "mortgage rate", "monthly_rent": "monthly rent", "monthly_expenses": "monthly expenses"}
FIELD_NAMES = {"ticker": "symbol", "asset_type": "type", "shares": "shares", "avg_cost": "average cost", "purchase_date": "purchase date"}


@st.cache_data(ttl=300, show_spinner=False)
def cached_analysis(_client: FinnieClient, url: str, csv_text: str, question: str) -> dict:
    return _client.analyze_portfolio(csv_text, question or None)


def allocation_figure(asset_class_pct: dict) -> go.Figure:
    return charts.donut(list(asset_class_pct), list(asset_class_pct.values()), "Allocation by asset class", height=360)


def concentration_figure(holdings: list[dict]) -> go.Figure:
    ordered = sorted(holdings, key=lambda h: h["weight_pct"])
    figure = go.Figure(go.Bar(x=[h["weight_pct"] for h in ordered], y=[h["ticker"] for h in ordered], orientation="h", marker={"color": charts.PALETTE[0]},
                              text=[f"{h['weight_pct']:.1f}%" for h in ordered], textposition="outside", cliponaxis=False))
    figure.update_layout(xaxis_title="% of portfolio", xaxis_range=[0, 108])
    return charts.style(figure, "Share of the portfolio by holding", height=max(260, 80 + 40 * len(ordered)))


def correlation_figure(correlation: dict) -> go.Figure:
    tickers, matrix = correlation["tickers"], correlation["matrix"]
    figure = go.Figure(go.Heatmap(z=matrix, x=tickers, y=tickers, zmin=-1, zmax=1, colorscale=charts.SEQUENTIAL_RED_BLUE, text=matrix, texttemplate="%{text:.2f}",
                                  colorbar={"title": "correlation"}))
    figure.update_layout(yaxis_autorange="reversed")
    return charts.style(figure, "How similarly the holdings moved (daily returns, past year)", height=max(320, 90 + 55 * len(tickers)))


def whatif_figure(scenarios: list[dict]) -> go.Figure:
    """Diversification score before vs after each option: the one number all the options change in the same way."""
    options = [x["option"] for x in scenarios]
    figure = go.Figure([go.Bar(name="Now", x=options, y=[x["diversification_score_out_of_100"]["before"] for x in scenarios], marker={"color": charts.MUTED}),
                        go.Bar(name="After the example change", x=options, y=[x["diversification_score_out_of_100"]["after"] for x in scenarios], marker={"color": charts.PALETTE[0]})])
    figure.update_layout(barmode="group", yaxis_title="Diversification score (out of 100)", yaxis_range=[0, 100])
    return charts.style(figure, "Diversification score: now vs after each example change", height=340, legend_below=True)


def whatif_table(scenarios: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Option": x["option"], "Share of that kind (now → after)": f"{x['option_asset_class_pct']['before']:.1f}% → {x['option_asset_class_pct']['after']:.1f}%",
                          f"{x['from_sector']} share (now → after)": f"{x['sector_pct']['before']:.1f}% → {x['sector_pct']['after']:.1f}%",
                          "Concentration (HHI)": f"{x['hhi_concentration_0_to_1']['before']:.3f} → {x['hhi_concentration_0_to_1']['after']:.3f}",
                          "Score": f"{x['diversification_score_out_of_100']['before']:.0f} → {x['diversification_score_out_of_100']['after']:.0f}",
                          "Trade-off": x["trade_off"]} for x in scenarios])


def show_whatifs(metrics: dict) -> None:
    """Only present when the question was about growing the portfolio and one sector holds a large share. Numbers come from the server, never from the model."""
    scenarios = metrics.get("what_if_scenarios")
    if not scenarios:
        return
    first = scenarios[0]
    st.subheader("Ideas for growing this portfolio: example what-ifs")
    st.warning(f"These are illustrations, not advice. Each one moves {first['moved_pct_of_portfolio']:.0f}% of the whole portfolio out of {first['from_sector']} into one other kind of holding. "
               "They ignore taxes, trading costs and your goals, and none of them forecasts growth: every option can lose value, and a steadier one usually means lower expected "
               "growth. Talk to a licensed financial professional before changing your portfolio.", icon=":material/warning:")
    st.plotly_chart(whatif_figure(scenarios), width="stretch")
    st.dataframe(whatif_table(scenarios), hide_index=True, width="stretch")
    st.caption("Volatility is not re-estimated for these examples; the score changes come from concentration and spread only.")


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
    st.markdown(md(result["text"]))
    st.caption(md(result["disclaimer"]))
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
    show_whatifs(metrics)
    for note in metrics.get("notes", []):
        st.info(md(note))
    if result["sources"]:
        st.caption("Data: " + "; ".join(s["title"] for s in result["sources"]))


# ---------------------------------------------------------------- entering a portfolio
def blank_table() -> pd.DataFrame:
    return pd.DataFrame({COLUMNS["symbol"]: [""], COLUMNS["type"]: [ASSET_TYPES["stock"]], COLUMNS["shares"]: [None], COLUMNS["cost"]: [None],
                         COLUMNS["date"]: pd.Series([pd.NaT], dtype="datetime64[ns]")})


def rows_to_holdings(table: pd.DataFrame) -> tuple[list[dict], list[int], list[str]]:
    """The table the user filled in -> (holdings to send, the visible row number of each, plain-language problems). Completely empty rows are ignored.
    Numbers travel as text so that 220.10 stays exactly 220.10 (the database stores exact decimals, not floating point)."""
    holdings, row_numbers, problems = [], [], []
    for number, row in enumerate(table.to_dict("records"), start=1):
        raw_symbol = row[COLUMNS["symbol"]]
        symbol = "" if pd.isna(raw_symbol) else str(raw_symbol).strip()   # a row added with the table's + has NO symbol (NaN, which is truthy: str(NaN) would be the text "nan")
        shares, cost, bought = row[COLUMNS["shares"]], row[COLUMNS["cost"]], row[COLUMNS["date"]]
        if not symbol and pd.isna(shares) and pd.isna(cost) and pd.isna(bought):
            continue
        before = len(problems)
        if not symbol:
            problems.append(f"Row {number}: enter the symbol (for example VTI or AAPL).")
        if pd.isna(shares) or shares <= 0:
            problems.append(f"Row {number}: shares must be a number greater than 0.")
        if pd.isna(cost) or cost < 0:
            problems.append(f"Row {number}: enter what you paid per share (0 or more).")
        if not pd.isna(bought) and bought.date() > date.today():
            problems.append(f"Row {number}: the purchase date cannot be in the future.")
        if len(problems) > before:
            continue
        holding = {"ticker": symbol, "asset_type": LABEL_TO_KEY.get(row[COLUMNS["type"]], "other"), "shares": repr(float(shares)), "avg_cost": repr(float(cost))}
        if not pd.isna(bought):
            holding["purchase_date"] = bought.date().isoformat()
        holdings.append(holding)
        row_numbers.append(number)
    return holdings, row_numbers, problems


def blank_property_table() -> pd.DataFrame:
    return pd.DataFrame({PCOLS["name"]: pd.Series([], dtype="object"), PCOLS["type"]: pd.Series([], dtype="object"), PCOLS["price"]: pd.Series([], dtype="float64"),
                         PCOLS["date"]: pd.Series([], dtype="datetime64[ns]"), PCOLS["value"]: pd.Series([], dtype="float64"), PCOLS["mortgage"]: pd.Series([], dtype="float64"),
                         PCOLS["payment"]: pd.Series([], dtype="float64"), PCOLS["rate"]: pd.Series([], dtype="float64"), PCOLS["rent"]: pd.Series([], dtype="float64"), PCOLS["costs"]: pd.Series([], dtype="float64")})


def rows_to_properties(table: pd.DataFrame) -> tuple[list[dict], list[int], list[str]]:
    """The real estate table -> (properties to send, the visible row number of each, plain-language problems). Empty rows are ignored. A property needs a name, a
    purchase price and a current value; mortgage, rent and expenses default to 0. Numbers travel as text so nothing is rounded on the way."""
    properties, row_numbers, problems = [], [], []
    for number, row in enumerate(table.to_dict("records"), start=1):
        raw_name = row[PCOLS["name"]]
        name = "" if pd.isna(raw_name) else str(raw_name).strip()
        money = {key: row.get(PCOLS[key], float("nan")) for key in ("price", "value", "mortgage", "payment", "rate", "rent", "costs")}   # a table without the mortgage-terms columns counts them as empty
        bought = row[PCOLS["date"]]
        if not name and all(pd.isna(v) for v in money.values()) and pd.isna(bought):
            continue
        before = len(problems)
        if not name:
            problems.append(f"Property row {number}: enter a name or address.")
        if pd.isna(money["price"]) or money["price"] < 0:
            problems.append(f"Property row {number}: enter what you paid (0 or more).")
        if pd.isna(money["value"]) or money["value"] < 0:
            problems.append(f"Property row {number}: enter what you think it is worth today (0 or more).")
        for key, label in (("mortgage", "mortgage"), ("payment", "monthly mortgage payment"), ("rate", "mortgage rate"), ("rent", "monthly rent"), ("costs", "monthly expenses")):
            if not pd.isna(money[key]) and money[key] < 0:
                problems.append(f"Property row {number}: {label} cannot be negative.")
        if not pd.isna(money["rate"]) and money["rate"] > 100:
            problems.append(f"Property row {number}: the mortgage rate is a yearly percent, like 6.5 (it cannot be over 100).")
        if not pd.isna(bought) and bought.date() > date.today():
            problems.append(f"Property row {number}: the purchase date cannot be in the future.")
        if len(problems) > before:
            continue
        kind = row[PCOLS["type"]]
        prop = {"name": name, "property_type": PROPERTY_LABEL_TO_KEY.get(kind, "house" if pd.isna(kind) else "other"), "purchase_price": repr(float(money["price"])),
                "current_value": repr(float(money["value"])), "mortgage_balance": repr(float(0 if pd.isna(money["mortgage"]) else money["mortgage"])),
                "monthly_mortgage_payment": repr(float(0 if pd.isna(money["payment"]) else money["payment"])),
                "mortgage_rate_pct": repr(float(0 if pd.isna(money["rate"]) else money["rate"])),
                "monthly_rent": repr(float(0 if pd.isna(money["rent"]) else money["rent"])), "monthly_expenses": repr(float(0 if pd.isna(money["costs"]) else money["costs"]))}
        if not pd.isna(bought):
            prop["purchase_date"] = bought.date().isoformat()
        properties.append(prop)
        row_numbers.append(number)
    return properties, row_numbers, problems


def server_problems_by_row(error: ApiError, row_numbers: list[int], property_rows: list[int] | None = None) -> list[str]:
    """The server names a problem 'holdings.2.shares' or 'properties.0.monthly_rent'; the person sees their own tables, so say 'Row 4, shares' or 'Property row 1,
    monthly rent' (the server counts only the rows we sent)."""
    lines = []
    for detail in error.details:
        found = re.match(r"(holdings|properties)\.(\d+)\.(\w+)", detail.get("field", ""))
        text = re.sub(r"^Value error, ", "", detail.get("problem", ""))
        rows, names, prefix = (row_numbers, FIELD_NAMES, "Row") if found and found.group(1) == "holdings" else (property_rows or [], PROPERTY_FIELD_NAMES, "Property row")
        if found and int(found.group(2)) < len(rows):
            lines.append(f"{prefix} {rows[int(found.group(2))]}, {names.get(found.group(3), found.group(3))}: {text}")
        else:
            lines.append(text)
    return lines


def holdings_editor(data: pd.DataFrame, key: str, extra: dict | None = None, disabled: tuple = (), locked: bool = False) -> pd.DataFrame:
    """`locked` = view only: no cell can be changed and rows cannot be added or removed."""
    return st.data_editor(
        data, key=key, num_rows="fixed" if locked else "dynamic", hide_index=True, width="stretch",
        column_config={
            COLUMNS["symbol"]: st.column_config.TextColumn("Symbol", help="The ticker, such as VTI, AAPL or BND. For cash use CASH.", max_chars=10, required=False),
            COLUMNS["type"]: st.column_config.SelectboxColumn("Type", help="What kind of holding it is", options=list(ASSET_TYPES.values()), required=True),
            COLUMNS["shares"]: st.column_config.NumberColumn("Shares", help="How many shares or units you own (for cash: the dollar amount)", min_value=0.0, format="%g"),
            COLUMNS["cost"]: st.column_config.NumberColumn("Avg cost per share ($)", help="What you paid per share on average (for cash: 1)", min_value=0.0, format="%.2f"),
            COLUMNS["date"]: st.column_config.DateColumn("Purchase date", help="Optional", max_value=date.today()), **(extra or {})}, disabled=True if locked else disabled)


def properties_editor(data: pd.DataFrame, key: str, extra: dict | None = None, disabled: tuple = (), locked: bool = False) -> pd.DataFrame:
    """`locked` = view only: no cell can be changed and rows cannot be added or removed."""
    return st.data_editor(
        data, key=key, num_rows="fixed" if locked else "dynamic", hide_index=True, width="stretch",
        column_config={
            PCOLS["name"]: st.column_config.TextColumn("Property (name or address)", max_chars=120, help="For example: Maple St rental"),
            PCOLS["type"]: st.column_config.SelectboxColumn("Kind", options=list(PROPERTY_TYPES.values()), help="Defaults to House if left empty"),
            PCOLS["price"]: st.column_config.NumberColumn("Purchase price ($)", min_value=0.0, format="dollar", help="What you paid"),
            PCOLS["date"]: st.column_config.DateColumn("Purchase date", max_value=date.today(), help="Optional"),
            PCOLS["value"]: st.column_config.NumberColumn("Current value ($)", min_value=0.0, format="dollar", help="Your own estimate of what it is worth today"),
            PCOLS["mortgage"]: st.column_config.NumberColumn("Mortgage owed ($)", min_value=0.0, format="dollar", help="The balance still owed; leave empty for none"),
            PCOLS["payment"]: st.column_config.NumberColumn("Monthly mortgage payment ($)", min_value=0.0, format="dollar", help="What you pay the lender each month (principal + interest). Only used if you ask how long the mortgage will take to pay off"),
            PCOLS["rate"]: st.column_config.NumberColumn("Mortgage rate (%)", min_value=0.0, max_value=100.0, format="%.3f", help="Yearly interest rate, for example 6.5"),
            PCOLS["rent"]: st.column_config.NumberColumn("Monthly rent ($)", min_value=0.0, format="dollar", help="Rent you receive each month; leave empty for none"),
            PCOLS["costs"]: st.column_config.NumberColumn("Monthly expenses ($)", min_value=0.0, format="dollar", help="Tax, insurance, upkeep, fees: not the mortgage payment"), **(extra or {})}, disabled=True if locked else disabled)


def render_enter(client: FinnieClient) -> None:
    st.markdown("Type in what you own, one holding per row, and save it. Each saved portfolio gets its own **unique id** and belongs only to your account.")
    version = st.session_state.setdefault("pf_version", 0)   # saving bumps this, which gives the widgets new keys and so clears them
    name = st.text_input("Portfolio name", max_chars=100, key=f"pf_name_{version}", placeholder="for example: Retirement account")
    table = holdings_editor(blank_table(), f"pf_table_{version}")
    st.caption("Add rows with the + at the bottom of the table. Empty rows are ignored. The same symbol on two rows is combined into one holding.")
    st.markdown("#### Real estate (optional)")
    st.caption("Properties you own. Finnie looks nothing up: the current value, mortgage, rent and expenses are your own numbers, and only as fresh as your last edit. "
               "Expenses are everything except the mortgage payment (tax, insurance, upkeep, fees). To add a property, click the empty row under the headings and start typing.")
    properties_table = properties_editor(blank_property_table(), f"pf_props_{version}")
    if not st.button("Save portfolio", type="primary", icon=":material/save:", key=f"pf_save_{version}"):
        return
    holdings, row_numbers, problems = rows_to_holdings(table)
    properties, property_rows, property_problems = rows_to_properties(properties_table)
    problems += property_problems
    if not name.strip():
        problems.insert(0, "Give the portfolio a name.")
    if not holdings and not properties and not problems:
        problems.append("Add at least one holding or property.")
    if problems:
        for problem in problems:
            st.warning(problem)
        return
    try:
        saved = client.save_portfolio(name.strip(), holdings, properties)
    except ApiError as error:
        if error.code in ("invalid_request", "invalid_portfolio"):
            st.error("The server could not accept some rows:")
            for line in server_problems_by_row(error, row_numbers, property_rows):
                st.caption(f"- {line}")
        else:
            show_error(error)
        return
    st.session_state.pf_version = version + 1
    st.session_state.selected_portfolio_id = saved["id"]
    st.session_state.pf_saved_notice = (name.strip(), saved["id"])
    st.session_state.pf_goto = MODES[1]   # take them to the grouped view of what they just saved (applied before the mode radio exists next run)
    st.rerun()


# ---------------------------------------------------------------- viewing saved portfolios
def money(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def pct(x: float | None, signed: bool = False) -> str:
    return "n/a" if x is None else (f"{x:+.1f}%" if signed else f"{x:.1f}%")


@st.cache_data(ttl=60, show_spinner=False)
def cached_summary(_client: FinnieClient, url: str, user: str, portfolio_id: str, version: int) -> dict:
    """`user` is part of the cache key on purpose: st.cache_data is shared by every visitor of the server, and a portfolio summary must never be served to someone else."""
    return _client.portfolio_summary(portfolio_id)


@st.cache_data(ttl=300, show_spinner=False)
def cached_saved_analysis(_client: FinnieClient, url: str, user: str, portfolio_id: str, question: str) -> dict:
    return _client.analyze_saved_portfolio(portfolio_id, question or None)


def show_payoff(answer: dict) -> None:
    """One property's payoff answer. Only ever called after the owner asks for it."""
    st.markdown(f"**{md(answer['name'])}**: {money(answer['balance'])} owed, {money(answer['monthly_payment'])} a month at {answer['rate_pct']:g}%")
    status = answer["status"]
    if status == "missing_payment":
        st.info("Enter the monthly payment above (and the rate) and save, then ask again.")
        return
    if status == "never_paid_off":
        st.warning(f"At this payment the balance never goes down: the first month's interest alone is about {money(answer['monthly_interest'])}. Check the payment and rate you entered.")
        return
    span = lambda plan: f"{plan['years']:g} years ({plan['months']} months), paid off around {plan['payoff_date']}, total interest {money(plan['total_interest'])}"
    st.write(f"- Paying the regular amount: {span(answer['regular'])}.")
    if answer["with_extra"]:
        saved = answer["saved"]
        st.write(f"- Paying {money(answer['extra'])} more every month: {span(answer['with_extra'])}.")
        st.write(f"- That is **{saved['years']:g} years ({saved['months']} months) sooner** and **{money(saved['interest'])} less interest**.")


def render_mortgage_payoff(client: FinnieClient, saved: list[dict]) -> None:
    """Every mortgage in EVERY saved portfolio in one place. Hidden until asked: nothing is calculated or shown until the owner presses the button."""
    user, version = st.session_state.auth["email"], st.session_state.get("pf_data_version", 0)
    owing = []   # (portfolio id, portfolio name, property)
    for entry in saved:
        try:
            stored = guarded(("stored", user, entry["id"], version), lambda i=entry["id"]: cached_stored(client, client.base_url, user, i, version))
        except ApiError:
            continue   # the dashboard above already said this portfolio could not be loaded
        owing += [(entry["id"], entry["name"], p) for p in stored["properties"] if float(p["mortgage_balance"]) > 0]
    if not owing:
        return
    note = st.session_state.pop("pf_terms_notice", None)
    if note:
        st.success(md(f"Saved the payment and rate for {note}."))
    if st.button("Mortgage payoff calculator", key="pf_payoff_open", icon=":material/calculate:"):
        st.session_state.pf_payoff_shown = not st.session_state.get("pf_payoff_shown", False)
        st.rerun()
    if not st.session_state.get("pf_payoff_shown"):
        return
    several = len({pid for pid, _, _ in owing}) > 1
    st.caption(f"{len(owing)} mortgage(s) across your saved portfolios. Add each one's monthly payment (principal + interest) and yearly rate once; they are saved with the property. "
               "Assumes a fixed rate. Not advice: your lender's statement wins.")
    for index, (pid, portfolio, prop) in enumerate(owing):
        columns = st.columns([3, 2, 2, 1])
        columns[0].write(f"**{md(prop['name'])}**" + (f" ({md(portfolio)})" if several else "") + f"  \n{money(float(prop['mortgage_balance']))} owed")
        payment = columns[1].number_input("Monthly payment ($)", min_value=0.0, value=float(prop["monthly_mortgage_payment"]), step=50.0, key=f"pf_pay_{pid}_{index}")
        rate = columns[2].number_input("Rate (% a year)", min_value=0.0, max_value=100.0, value=float(prop["mortgage_rate_pct"]), step=0.125, format="%.3f", key=f"pf_rate_{pid}_{index}")
        if columns[3].button("Save", key=f"pf_terms_{pid}_{index}"):
            try:
                client.set_mortgage_terms(pid, prop["name"], payment, rate)
            except ApiError as error:
                show_error(error)
            else:
                st.session_state.pf_data_version = version + 1   # the saved numbers changed, so the cached copies are stale
                st.session_state.pf_terms_notice = prop["name"]
                st.rerun()
    extra = st.number_input("Extra I could pay each month ($, optional)", min_value=0.0, value=0.0, step=50.0, key="pf_extra")
    if st.button("Work out the payoff", key="pf_payoff_go", type="primary", icon=":material/play_arrow:"):
        for pid in dict.fromkeys(pid for pid, _, _ in owing):
            name = next(n for i, n, _ in owing if i == pid)
            try:
                result = client.mortgage_payoff(pid, extra)
            except ApiError as error:
                show_error(error)
                continue
            for answer in result["properties"]:
                show_payoff({**answer, "name": f"{answer['name']} ({name})"} if several else answer)


def chat_csv(summary: dict) -> str:
    """The holdings as the CSV the chat expects. Cash has no market price, so it is left out of what chat analyzes."""
    lines = [f"{h['ticker']},{h['shares']:g},{h['avg_cost']}" for g in summary["groups"] if g["asset_type"] != "cash" for h in g["holdings"]]
    return "ticker,shares,avg_cost\n" + "\n".join(lines)


PF_CHAT_HINT = "e.g. How would my portfolio be affected if the market is up tomorrow? What if I buy NVDA?"


def render_portfolio_chat(client: FinnieClient, csv_text: str, scope: str) -> None:
    """A chat box under the portfolio. Questions go to the same /chat endpoint with this portfolio attached, in their own conversation
    (kept per `scope`, so switching portfolios starts a fresh thread and the Chat tab's conversation is untouched)."""
    from src.web_app.tabs.chat import AVATARS, render_message, saved_portfolio_kwargs   # imported here: chat.py is a sibling tab, only this function needs it
    st.divider()
    st.subheader("Ask about this portfolio")
    state = st.session_state.setdefault("pf_chats", {}).setdefault(scope, {"messages": [], "session_id": None})
    st.caption("Ask how your holdings could be affected by the market, or what adding a stock might do. Answers explain; they are not personal advice.")
    for message in state["messages"]:
        render_message(message)
    prompt = st.chat_input(PF_CHAT_HINT, max_chars=2000, key="pf_chat_input")
    if state["messages"] and st.button("Clear this conversation", key="pf_chat_clear", icon=":material/add:"):
        state["messages"], state["session_id"] = [], None
        st.rerun()
    if prompt is None:
        return
    if not prompt.strip():
        st.warning("Please type a question first.")
        return
    state["messages"].append({"role": "user", "content": prompt})
    render_message(state["messages"][-1])
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Thinking…"):
                reply = client.chat(prompt, state["session_id"], csv_text, **saved_portfolio_kwargs(scope[6:] if scope.startswith("saved:") else None))
        except ApiError as error:
            state["messages"].append({"role": "assistant", "error": error})
            st.rerun()
    state["session_id"] = reply["session_id"]
    state["messages"].append({"role": "assistant", "content": reply["text"], "meta": reply})
    st.rerun()


def choose_portfolio(client: FinnieClient) -> str | None:
    try:
        saved = client.list_portfolios()
    except ApiError as error:
        show_error(error)
        return None
    with st.expander("Have a portfolio id? Load it here", expanded=bool(st.session_state.get("pf_load_id"))):
        typed = st.text_input("Portfolio id", key="pf_load_id", placeholder="paste a portfolio id", help="Only your own portfolios can be loaded.")
    if typed.strip():
        try:
            return str(UUID(typed.strip()))
        except ValueError:
            st.warning("That doesn't look like a portfolio id (36 characters, like 3f2b8c1e-5d4a-4c7e-9b1a-0a1b2c3d4e5f).")
            return None
    if not saved:
        st.info("You haven't saved a portfolio yet. Use 'Enter my holdings' to add one.")
        return None
    ids = [p["id"] for p in saved]
    wanted = st.session_state.get("selected_portfolio_id")
    labels = {p["id"]: f"{p['name']} — {p['holding_count']} holding(s)" + (f" + {p['property_count']} propert{'y' if p['property_count'] == 1 else 'ies'}" if p.get("property_count") else "")
              + f" — {p['id'][:8]}…" for p in saved}
    return st.selectbox("Your portfolios", ids, index=ids.index(wanted) if wanted in ids else 0, format_func=lambda i: md(labels[i]), key="pf_pick")


@st.cache_data(ttl=60, show_spinner=False)
def cached_stored(_client: FinnieClient, url: str, user: str, portfolio_id: str, version: int) -> dict:
    """The saved rows as typed (exact numbers, dates, mortgage terms): what the editable tables start from. `user` is in the key for the same reason as cached_summary."""
    return _client.get_portfolio(portfolio_id)


STOCK_COLUMNS = ["Portfolio", COLUMNS["symbol"], COLUMNS["type"], COLUMNS["shares"], COLUMNS["cost"], COLUMNS["date"], "Price", "Value", "Gain"]
STOCK_READONLY = ("Price", "Value", "Gain")
ESTATE_COLUMNS = ["Portfolio", *PCOLS.values(), "Equity", "Net yield"]
ESTATE_READONLY = ("Equity", "Net yield")


def portfolio_labels(saved: list[dict]) -> dict[str, str]:
    """id -> the name shown in the Portfolio column. Two portfolios with the same name get the start of their id added so each row still says which one it belongs to."""
    names = [p["name"] for p in saved]
    return {p["id"]: p["name"] if names.count(p["name"]) == 1 else f"{p['name']} ({p['id'][:8]})" for p in saved}


def _typed(rows: list[dict], columns: list[str], numbers: list[str], dates: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    for name in numbers:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
    for name in dates:
        frame[name] = pd.to_datetime(frame[name])
    return frame


def stocks_frame(stored: dict[str, dict], summaries: dict[str, dict], labels: dict[str, str]) -> pd.DataFrame:
    """Every market holding of every portfolio as ONE table. Symbol, type, shares, cost and date can be edited; price, value and gain are Finnie's numbers (read-only)."""
    rows = []
    for pid, portfolio in stored.items():
        priced = {h["ticker"]: h for g in summaries[pid]["groups"] for h in g["holdings"]}
        for h in portfolio["holdings"]:
            now = priced.get(h["ticker"], {})
            rows.append({"Portfolio": labels[pid], COLUMNS["symbol"]: h["ticker"], COLUMNS["type"]: ASSET_TYPES.get(h["asset_type"], ASSET_TYPES["other"]), COLUMNS["shares"]: h["shares"],
                         COLUMNS["cost"]: h["avg_cost"], COLUMNS["date"]: h.get("purchase_date"), "Price": now.get("price"), "Value": now.get("value"), "Gain": now.get("gain_pct")})
    return _typed(rows, STOCK_COLUMNS, [COLUMNS["shares"], COLUMNS["cost"], "Price", "Value", "Gain"], [COLUMNS["date"]])


def estate_frame(stored: dict[str, dict], summaries: dict[str, dict], labels: dict[str, str]) -> pd.DataFrame:
    """Every property of every portfolio as ONE table. All the numbers you typed can be edited; equity and net yield are worked out by Finnie (read-only)."""
    rows = []
    for pid, portfolio in stored.items():
        worked = {p["name"]: p for p in (summaries[pid].get("real_estate") or {}).get("properties", [])}
        for p in portfolio["properties"]:
            now = worked.get(p["name"], {})
            rows.append({"Portfolio": labels[pid], PCOLS["name"]: p["name"], PCOLS["type"]: PROPERTY_TYPES.get(p["property_type"], PROPERTY_TYPES["other"]), PCOLS["price"]: p["purchase_price"],
                         PCOLS["date"]: p.get("purchase_date"), PCOLS["value"]: p["current_value"], PCOLS["mortgage"]: p["mortgage_balance"], PCOLS["payment"]: p["monthly_mortgage_payment"],
                         PCOLS["rate"]: p["mortgage_rate_pct"], PCOLS["rent"]: p["monthly_rent"], PCOLS["costs"]: p["monthly_expenses"], "Equity": now.get("equity"), "Net yield": now.get("net_yield_pct")})
    return _typed(rows, ESTATE_COLUMNS, [c for c in ESTATE_COLUMNS if c not in ("Portfolio", PCOLS["name"], PCOLS["type"], PCOLS["date"])], [PCOLS["date"]])


def rows_by_portfolio(stocks: pd.DataFrame, estate: pd.DataFrame, ids: dict[str, str]) -> tuple[dict[str, dict], list[str]]:
    """The two tables -> {portfolio id: its holdings and properties} (+ the visible row numbers, to name problems), and plain-language problems.
    `ids` maps the Portfolio column's text to the id. A row that names no portfolio is a problem, not silently dropped."""
    per = {pid: {"holdings": [], "hrows": [], "properties": [], "prows": []} for pid in ids.values()}
    holdings, hrows, problems = rows_to_holdings(stocks)
    properties, prows, property_problems = rows_to_properties(estate)
    for item, number, table, kind, bucket, rows in [(h, n, stocks, "Row", "holdings", "hrows") for h, n in zip(holdings, hrows)] + [(p, n, estate, "Property row", "properties", "prows") for p, n in zip(properties, prows)]:
        owner = ids.get(table.iloc[number - 1]["Portfolio"])
        if owner is None:
            problems.append(f"{kind} {number}: choose which portfolio it belongs to.")
        else:
            per[owner][bucket].append(item)
            per[owner][rows].append(number)
    return per, problems + property_problems


def _comparable(entry: dict) -> tuple:
    return (sorted(entry["holdings"], key=lambda h: h["ticker"]), sorted(entry["properties"], key=lambda p: p["name"].lower()))


def render_dashboard(client: FinnieClient, saved: list[dict]) -> None:
    """Everything the user owns on one screen: totals, ONE editable stocks table and ONE editable real estate table covering all portfolios, and the buttons under them."""
    user, version = st.session_state.auth["email"], st.session_state.get("pf_data_version", 0)
    stored, summaries = {}, {}
    with st.spinner("Loading your dashboard…"):
        for entry in saved:
            pid = entry["id"]
            try:
                summaries[pid] = guarded(("summary", user, pid, version), lambda i=pid: cached_summary(client, client.base_url, user, i, version))
                stored[pid] = guarded(("stored", user, pid, version), lambda i=pid: cached_stored(client, client.base_url, user, i, version))
            except ApiError as error:
                summaries.pop(pid, None)
                st.warning(md(f"Could not load '{entry['name']}': {error}"))
    if not stored:
        return
    labels = portfolio_labels([p for p in saved if p["id"] in stored])
    ids = {label: pid for pid, label in labels.items()}
    value = sum(s["total_value"] for s in summaries.values())
    mortgage = sum(s.get("mortgage_owed", 0.0) for s in summaries.values())
    paid_for_gain = sum(s["total_value"] / (1 + s["gain_pct"] / 100) for s in summaries.values() if s["gain_pct"] is not None and s["total_value"] > 0)
    st.subheader("My dashboard")
    top = st.columns(5)
    top[0].metric("Portfolios", len(stored))
    top[1].metric("Total value", money(value))
    top[2].metric("Gain", pct((value / paid_for_gain - 1) * 100 if paid_for_gain > 0 else None, signed=True), help="Today's value against what you paid, for everything that has a value")
    top[3].metric("Mortgage owed", money(mortgage))
    top[4].metric("Net worth", money(value - mortgage), help="Everything you own minus mortgages")
    stocks, estate = stocks_frame(stored, summaries, labels), estate_frame(stored, summaries, labels)
    key = f"{st.session_state.get('pf_edit_version', 0)}_{version}"   # saving or cancelling changes the key, which throws away the half-made edits
    editing = st.session_state.get("pf_editing_table")                  # "stocks", "estate" or None: only one table is open for editing at a time
    where = {"Portfolio": st.column_config.SelectboxColumn("Portfolio", options=list(labels.values()), required=True, help="Which portfolio this row belongs to")}
    tabs = st.tabs([f":material/show_chart: Stocks & funds ({len(stocks)})", f":material/home: Real estate ({len(estate)})"])
    stocks_now, estate_now, pressed = stocks, estate, None

    def buttons(which: str, noun: str) -> None:
        """Under a table: Edit while looking, Save changes + Cancel while editing."""
        nonlocal pressed
        if editing != which:
            if st.button(f"Edit {noun}", key=f"pf_edit_{which}", type="primary", icon=":material/edit:", help="Change the values in this table"):
                st.session_state.pf_editing_table = which
                st.rerun()
            return
        save, cancel, _ = st.columns([2, 2, 3])
        if cancel.button("Cancel", key=f"pf_cancel_{which}", icon=":material/close:", width="stretch"):
            st.session_state.pf_editing_table = None
            st.session_state.pf_edit_version = st.session_state.get("pf_edit_version", 0) + 1
            st.rerun()
        if save.button("Save changes", key=f"pf_save_{which}", type="primary", icon=":material/save:", width="stretch"):
            pressed = which

    stock_view = {**where, "Price": st.column_config.NumberColumn(format="dollar"), "Value": st.column_config.NumberColumn(format="dollar"), "Gain": st.column_config.NumberColumn(format="%+.1f%%")}
    estate_view = {**where, "Equity": st.column_config.NumberColumn(format="dollar"), "Net yield": st.column_config.NumberColumn(format="%.1f%%")}
    with tabs[0]:
        if editing == "stocks":
            st.caption("Editing. Click a cell to change it, use the + at the bottom to add a row, or select a row and press Delete. Price, value and gain are filled in by Finnie.")
            stocks_now = holdings_editor(stocks, f"dash_stocks_edit_{key}", extra=stock_view, disabled=STOCK_READONLY)
        else:
            holdings_editor(stocks, f"dash_stocks_view_{key}", extra=stock_view, locked=True)
        buttons("stocks", "stocks & funds")
    with tabs[1]:
        if editing == "estate":
            st.caption("Editing. The numbers are your own estimates: change any cell, add or delete rows. Equity and net yield are worked out by Finnie.")
            estate_now = properties_editor(estate, f"dash_estate_edit_{key}", extra=estate_view, disabled=ESTATE_READONLY)
        else:
            properties_editor(estate, f"dash_estate_view_{key}", extra=estate_view, locked=True)
        buttons("estate", "real estate")
    if not pressed:
        return
    now, problems = rows_by_portfolio(stocks_now, estate_now, ids)
    before, _unused = rows_by_portfolio(stocks, estate, ids)
    changed = [pid for pid in ids.values() if _comparable(now[pid]) != _comparable(before[pid])]
    for pid in changed:
        if not now[pid]["holdings"] and not now[pid]["properties"] and not problems:
            problems.append(f"'{labels[pid]}' would have nothing left. To remove a whole portfolio, use 'Delete this portfolio' below.")
    if problems:
        for problem in problems:
            st.warning(problem)
        return
    if not changed:
        st.info("Nothing to save: you haven't changed anything.")
        return
    done = []
    for pid in changed:
        try:
            client.update_portfolio(pid, stored[pid]["name"], now[pid]["holdings"], now[pid]["properties"])
        except ApiError as error:
            if error.code in ("invalid_request", "invalid_portfolio"):
                st.error(md(f"The server could not accept some rows of '{labels[pid]}':"))
                for line in server_problems_by_row(error, now[pid]["hrows"], now[pid]["prows"]):
                    st.caption(f"- {line}")
            else:
                show_error(error)
            if done:
                st.warning("Saved before this problem: " + ", ".join(md(labels[d]) for d in done) + ". The tables will show what is stored now.")
                st.session_state.pf_data_version = version + 1
            return
        done.append(pid)
    st.session_state.pf_editing_table = None
    st.session_state.pf_edit_version = st.session_state.get("pf_edit_version", 0) + 1
    st.session_state.pf_data_version = version + 1   # the cached summaries are stale now
    st.session_state.pop("pf_explain_for", None)
    st.session_state.pf_updated_notice = ", ".join(labels[d] for d in done)
    st.rerun()


def render_saved(client: FinnieClient) -> None:
    notice = st.session_state.pop("pf_saved_notice", None)
    if notice:
        st.success(md(f"Saved '{notice[0]}'. Its unique id is {notice[1]}."))
    updated = st.session_state.pop("pf_updated_notice", None)
    if updated:
        st.success(md(f"Saved your changes to: {updated}."))
    try:
        everything = client.list_portfolios()
    except ApiError as error:
        show_error(error)
        return
    if everything:
        render_dashboard(client, everything)
        render_mortgage_payoff(client, everything)
    st.divider()
    render_actions(client)


def render_actions(client: FinnieClient) -> None:
    """The buttons under the tables: they act on ONE portfolio, chosen here."""
    st.subheader("Actions for one portfolio")
    portfolio_id = choose_portfolio(client)
    if not portfolio_id:
        return
    st.session_state.selected_portfolio_id = portfolio_id
    user = st.session_state.auth["email"]
    version = st.session_state.get("pf_data_version", 0)
    try:
        with st.spinner("Loading prices…"):
            summary = guarded(("summary", user, portfolio_id, version), lambda: cached_summary(client, client.base_url, user, portfolio_id, version))
    except ApiError as error:
        if error.status == 404:
            st.error("No portfolio with that id was found in your account.")
        else:
            show_error(error)
        return
    st.caption("Portfolio id (unique, keep it to load this portfolio again):")
    st.code(portfolio_id, language="text")
    left, middle, right = st.columns(3)
    market_holdings = summary["holding_count"] - summary.get("property_count", 0)
    if left.button("Explain this portfolio", key="pf_explain", icon=":material/auto_awesome:", width="stretch", disabled=market_holdings == 0,
                   help="Asks Finnie to explain the market holdings in plain words (uses the AI model); real estate is not part of the explanation"
                   if market_holdings else "There are no stocks or funds in this portfolio to explain"):
        st.session_state.pf_explain_for = portfolio_id
    if middle.button("Use in Chat", key="pf_use_in_chat", icon=":material/chat:", width="stretch", help="Chat questions about 'my portfolio' will use these holdings"):
        st.session_state.portfolio_csv = chat_csv(summary)
        st.session_state.portfolio_id = portfolio_id   # lets chat answer "when will my mortgage be paid off?" from the saved properties
        st.success("Chat will now use this portfolio (cash is left out).")
    with right.popover("Delete this portfolio", icon=":material/delete:", width="stretch"):
        st.write("This permanently deletes the portfolio and all its holdings.")
        if st.button("Yes, delete it", key="pf_delete", type="primary", icon=":material/delete_forever:"):
            try:
                client.delete_portfolio(portfolio_id)
            except ApiError as error:
                show_error(error)
            else:
                st.session_state.pf_data_version = version + 1
                st.session_state.pop("selected_portfolio_id", None)
                st.session_state.pop("pf_explain_for", None)
                st.rerun()
    if st.session_state.get("pf_explain_for") == portfolio_id:
        try:
            with st.spinner("Explaining your portfolio…"):
                result = guarded(("saved-analysis", user, portfolio_id), lambda: cached_saved_analysis(client, client.base_url, user, portfolio_id, ""))
        except ApiError as error:
            show_error(error)
        else:
            show_result(result)
    if market_holdings or any(p["mortgage"] > 0 for p in (summary.get("real_estate") or {}).get("properties", [])):
        render_portfolio_chat(client, chat_csv(summary) if market_holdings else "", f"saved:{portfolio_id}")


def render(client: FinnieClient) -> None:
    if "pf_goto" in st.session_state:   # a widget's value cannot be changed after it exists in the same run, so switching mode is done here, before the radio
        st.session_state.portfolio_mode = st.session_state.pop("pf_goto")
    mode = st.radio("What would you like to do?", MODES, horizontal=True, key="portfolio_mode")
    if mode == MODES[0]:
        render_enter(client)
    elif mode == MODES[1]:
        render_saved(client)
    else:
        render_analyze(client)


def render_analyze(client: FinnieClient) -> None:
    st.markdown("See how a set of holdings is spread out. Nothing is stored in this mode: your file is sent for the analysis and kept only in this browser session. To keep a portfolio, use 'Enter my holdings'.")
    source, sample = pick_source()
    with st.form("portfolio"):
        csv_text, problem = read_input(source, sample)
        question = st.text_input("Anything specific to ask? (optional)", max_chars=500, placeholder="e.g. How can I grow this portfolio?",
                                 help="Ask how to grow it and Finnie also shows example what-ifs: moving part of your biggest sector into other kinds of holdings")
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
    st.session_state.portfolio_id = None        # a pasted file is not a saved portfolio
    st.success("Portfolio analyzed. Chat questions about 'my portfolio' will now use it.")
    show_result(result)
    render_portfolio_chat(client, csv_text, "analyze")
    if st.button("Forget this portfolio"):
        st.session_state.portfolio_csv = None
        st.session_state.portfolio_id = None
        st.session_state.portfolio_request = None
        st.rerun()
