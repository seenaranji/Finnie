"""Tab 3: Portfolio, in three modes (a radio decides which one runs, so the others make no API calls):
  Enter my holdings           type symbol, kind, shares, cost and date into a table, name it, save it (PostgreSQL gives it a unique id)
  My saved portfolios         pick or load by id; shown GROUPED by kind with cost, value, gain and share of the total; explain, use in chat, delete
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
         "mortgage": "Mortgage owed ($)", "rent": "Monthly rent ($)", "costs": "Monthly expenses ($)"}
PROPERTY_FIELD_NAMES = {"name": "name", "property_type": "kind", "purchase_price": "purchase price", "purchase_date": "purchase date", "current_value": "current value",
                        "mortgage_balance": "mortgage", "monthly_rent": "monthly rent", "monthly_expenses": "monthly expenses"}
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
                         PCOLS["rent"]: pd.Series([], dtype="float64"), PCOLS["costs"]: pd.Series([], dtype="float64")})


def rows_to_properties(table: pd.DataFrame) -> tuple[list[dict], list[int], list[str]]:
    """The real estate table -> (properties to send, the visible row number of each, plain-language problems). Empty rows are ignored. A property needs a name, a
    purchase price and a current value; mortgage, rent and expenses default to 0. Numbers travel as text so nothing is rounded on the way."""
    properties, row_numbers, problems = [], [], []
    for number, row in enumerate(table.to_dict("records"), start=1):
        raw_name = row[PCOLS["name"]]
        name = "" if pd.isna(raw_name) else str(raw_name).strip()
        money = {key: row[PCOLS[key]] for key in ("price", "value", "mortgage", "rent", "costs")}
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
        for key, label in (("mortgage", "mortgage"), ("rent", "monthly rent"), ("costs", "monthly expenses")):
            if not pd.isna(money[key]) and money[key] < 0:
                problems.append(f"Property row {number}: {label} cannot be negative.")
        if not pd.isna(bought) and bought.date() > date.today():
            problems.append(f"Property row {number}: the purchase date cannot be in the future.")
        if len(problems) > before:
            continue
        kind = row[PCOLS["type"]]
        prop = {"name": name, "property_type": PROPERTY_LABEL_TO_KEY.get(kind, "house" if pd.isna(kind) else "other"), "purchase_price": repr(float(money["price"])),
                "current_value": repr(float(money["value"])), "mortgage_balance": repr(float(0 if pd.isna(money["mortgage"]) else money["mortgage"])),
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


def render_enter(client: FinnieClient) -> None:
    st.markdown("Type in what you own, one holding per row, and save it. Each saved portfolio gets its own **unique id** and belongs only to your account.")
    version = st.session_state.setdefault("pf_version", 0)   # saving bumps this, which gives the widgets new keys and so clears them
    name = st.text_input("Portfolio name", max_chars=100, key=f"pf_name_{version}", placeholder="for example: Retirement account")
    table = st.data_editor(
        blank_table(), key=f"pf_table_{version}", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            COLUMNS["symbol"]: st.column_config.TextColumn("Symbol", help="The ticker, such as VTI, AAPL or BND. For cash use CASH.", max_chars=10, required=False),
            COLUMNS["type"]: st.column_config.SelectboxColumn("Type", help="What kind of holding it is", options=list(ASSET_TYPES.values()), required=True),
            COLUMNS["shares"]: st.column_config.NumberColumn("Shares", help="How many shares or units you own (for cash: the dollar amount)", min_value=0.0, format="%g"),
            COLUMNS["cost"]: st.column_config.NumberColumn("Avg cost per share ($)", help="What you paid per share on average (for cash: 1)", min_value=0.0, format="%.2f"),
            COLUMNS["date"]: st.column_config.DateColumn("Purchase date", help="Optional", max_value=date.today())})
    st.caption("Add rows with the + at the bottom of the table. Empty rows are ignored. The same symbol on two rows is combined into one holding.")
    st.markdown("#### Real estate (optional)")
    st.caption("Properties you own. Finnie looks nothing up: the current value, mortgage, rent and expenses are your own numbers, and only as fresh as your last edit. "
               "Expenses are everything except the mortgage payment (tax, insurance, upkeep, fees). To add a property, click the empty row under the headings and start typing.")
    properties_table = st.data_editor(
        blank_property_table(), key=f"pf_props_{version}", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            PCOLS["name"]: st.column_config.TextColumn("Property (name or address)", max_chars=120, help="For example: Maple St rental"),
            PCOLS["type"]: st.column_config.SelectboxColumn("Kind", options=list(PROPERTY_TYPES.values()), help="Defaults to House if left empty"),
            PCOLS["price"]: st.column_config.NumberColumn("Purchase price ($)", min_value=0.0, format="dollar", help="What you paid"),
            PCOLS["date"]: st.column_config.DateColumn("Purchase date", max_value=date.today(), help="Optional"),
            PCOLS["value"]: st.column_config.NumberColumn("Current value ($)", min_value=0.0, format="dollar", help="Your own estimate of what it is worth today"),
            PCOLS["mortgage"]: st.column_config.NumberColumn("Mortgage owed ($)", min_value=0.0, format="dollar", help="The balance still owed; leave empty for none"),
            PCOLS["rent"]: st.column_config.NumberColumn("Monthly rent ($)", min_value=0.0, format="dollar", help="Rent you receive each month; leave empty for none"),
            PCOLS["costs"]: st.column_config.NumberColumn("Monthly expenses ($)", min_value=0.0, format="dollar", help="Tax, insurance, upkeep, fees: not the mortgage payment")})
    if not st.button("Save portfolio", type="primary", key=f"pf_save_{version}"):
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


def type_allocation_figure(groups: list[dict]) -> go.Figure:
    priced = [g for g in groups if g["value"] > 0]
    return charts.donut([g["label"] for g in priced], [g["value"] for g in priced], "Share of the portfolio's value by type")


@st.cache_data(ttl=60, show_spinner=False)
def cached_summary(_client: FinnieClient, url: str, user: str, portfolio_id: str, version: int) -> dict:
    """`user` is part of the cache key on purpose: st.cache_data is shared by every visitor of the server, and a portfolio summary must never be served to someone else."""
    return _client.portfolio_summary(portfolio_id)


@st.cache_data(ttl=300, show_spinner=False)
def cached_saved_analysis(_client: FinnieClient, url: str, user: str, portfolio_id: str, question: str) -> dict:
    return _client.analyze_saved_portfolio(portfolio_id, question or None)


def show_grouped(summary: dict) -> None:
    st.subheader(summary["name"])
    st.caption("Portfolio id (unique, keep it to load this portfolio again):")
    st.code(summary["id"], language="text")
    top = st.columns(4)
    top[0].metric("Total cost", money(summary["total_cost"]))
    top[1].metric("Current value", money(summary["total_value"]))
    top[2].metric("Gain", pct(summary["gain_pct"], signed=True))
    top[3].metric("Holdings", summary["holding_count"], help="Market holdings plus properties")
    if summary.get("property_count"):
        owed = st.columns(4)
        estate = summary["real_estate"]
        owed[0].metric("Mortgage owed", money(summary["mortgage_owed"]))
        owed[1].metric("Net value", money(summary["net_value"]), help="Everything you own minus the mortgage")
        owed[2].metric("Rent per year", money(estate["annual_rent"]))
        owed[3].metric("Net yield", pct(estate["net_yield_pct"]), help="(Rent - expenses) / value of the properties, before any mortgage payment")
    if summary["prices_as_of"]:
        st.caption(f"Prices as of {summary['prices_as_of']}. Gain compares today's value with what you paid for the holdings that have a price.")
    if summary["unpriced"]:
        st.warning("No current price for: " + ", ".join(summary["unpriced"]) + ". They are shown with what you paid, but not counted in the value, gain or shares of the total.")
    if summary["total_value"] > 0:
        st.plotly_chart(type_allocation_figure(summary["groups"]), width="stretch")
    for group in summary["groups"]:
        if group["asset_type"] == "real_estate":
            show_real_estate(group, summary["real_estate"])
            continue
        weight = f" · {pct(group['weight_pct'])} of the total" if group["weight_pct"] is not None else ""
        header = f"{group['label']} ({group['count']}): {money(group['value'])}{weight} · gain {pct(group['gain_pct'], signed=True)}"
        with st.expander(md(header), expanded=True):
            rows = [{"Symbol": h["ticker"], "Shares": h["shares"], "Avg cost": h["avg_cost"], "Paid": h["cost"], "Price": h["price"], "Value": h["value"],
                     "Gain": h["gain_pct"], "Share of total": h["weight_pct"], "Bought": h["purchase_date"]} for h in group["holdings"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Shares": st.column_config.NumberColumn(format="%g"), "Avg cost": st.column_config.NumberColumn(format="dollar"),
                "Paid": st.column_config.NumberColumn(format="dollar"), "Price": st.column_config.NumberColumn(format="dollar"),
                "Value": st.column_config.NumberColumn(format="dollar"), "Gain": st.column_config.NumberColumn(format="%+.1f%%"),
                "Share of total": st.column_config.NumberColumn(format="%.1f%%")})
            if group["unpriced_count"]:
                st.caption(f"{group['unpriced_count']} holding(s) here have no current price.")


def show_real_estate(group: dict, estate: dict) -> None:
    weight = f" · {pct(group['weight_pct'])} of the total" if group["weight_pct"] is not None else ""
    with st.expander(md(f"Real estate ({group['count']}): {money(group['value'])}{weight} · gain {pct(group['gain_pct'], signed=True)}"), expanded=True):
        rows = [{"Property": p["name"], "Kind": PROPERTY_TYPES.get(p["property_type"], p["property_type"]), "Bought": p["purchase_date"], "Paid": p["purchase_price"],
                 "Value": p["value"], "Mortgage": p["mortgage"], "Equity": p["equity"], "Gain": p["gain_pct"], "Rent / yr": p["annual_rent"], "Expenses / yr": p["annual_expenses"],
                 "Net income / yr": p["net_operating_income"], "Gross yield": p["gross_yield_pct"], "Net yield": p["net_yield_pct"], "Loan-to-value": p["loan_to_value_pct"]}
                for p in estate["properties"]]
        dollars = {name: st.column_config.NumberColumn(format="dollar") for name in ("Paid", "Value", "Mortgage", "Equity", "Rent / yr", "Expenses / yr", "Net income / yr")}
        percents = {name: st.column_config.NumberColumn(format="%.1f%%") for name in ("Gross yield", "Net yield", "Loan-to-value")}
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={**dollars, **percents, "Gain": st.column_config.NumberColumn(format="%+.1f%%")})
        st.caption("Values are your own estimates. Net income is rent minus expenses, before your mortgage payment. Gain is on paper: before selling costs, taxes and improvements. "
                   "Yields are measured on what each property is worth today.")


def chat_csv(summary: dict) -> str:
    """The holdings as the CSV the chat expects. Cash has no market price, so it is left out of what chat analyzes."""
    lines = [f"{h['ticker']},{h['shares']:g},{h['avg_cost']}" for g in summary["groups"] if g["asset_type"] != "cash" for h in g["holdings"]]
    return "ticker,shares,avg_cost\n" + "\n".join(lines)


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


def render_saved(client: FinnieClient) -> None:
    notice = st.session_state.pop("pf_saved_notice", None)
    if notice:
        st.success(md(f"Saved '{notice[0]}'. Its unique id is {notice[1]}."))
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
    show_grouped(summary)
    left, middle, right = st.columns(3)
    market_holdings = summary["holding_count"] - summary.get("property_count", 0)
    if left.button("Explain this portfolio", key="pf_explain", disabled=market_holdings == 0,
                   help="Asks Finnie to explain the market holdings in plain words (uses the AI model); real estate is not part of the explanation"
                   if market_holdings else "There are no stocks or funds in this portfolio to explain"):
        st.session_state.pf_explain_for = portfolio_id
    if middle.button("Use in Chat", key="pf_use_in_chat", help="Chat questions about 'my portfolio' will use these holdings"):
        st.session_state.portfolio_csv = chat_csv(summary)
        st.success("Chat will now use this portfolio (cash is left out).")
    with right.popover("Delete this portfolio"):
        st.write("This permanently deletes the portfolio and all its holdings.")
        if st.button("Yes, delete it", key="pf_delete", type="primary"):
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
