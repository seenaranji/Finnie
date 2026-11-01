"""Tab 4: Goals. Streamlit concepts: st.form with typed inputs (number_input limits), results kept across reruns, a Plotly line chart."""
import plotly.graph_objects as go
import streamlit as st

from src.web_app.client import ApiError, FinnieClient
from src.web_app.ui_common import guarded, show_error

PROFILES = {"Low risk (mostly cash and bonds)": "low", "Medium risk (a mix of stocks and bonds)": "medium", "High risk (mostly stocks)": "high"}
COLORS = {"conservative": "#6b7280", "expected": "#2563eb", "optimistic": "#16a34a"}


@st.cache_data(ttl=300, show_spinner=False)
def cached_plan(_client: FinnieClient, url: str, goal: float, years: float, savings: float, profile: str) -> dict:
    return _client.plan_goal(goal, years, savings, profile)


def projection_figure(plan: dict) -> go.Figure:
    figure = go.Figure()
    for scenario in plan["scenarios"]:
        path = scenario["path_by_year"]
        figure.add_scatter(x=list(range(len(path))), y=path, mode="lines", name=f"{scenario['scenario'].title()} ({scenario['annual_return_pct']:g}% a year)",
                           line={"color": COLORS[scenario["scenario"]], "width": 2}, hovertemplate="Year %{x}<br>$%{y:,.0f}<extra></extra>")
    figure.add_hline(y=plan["goal"], line_dash="dash", annotation_text=f"Goal ${plan['goal']:,.0f}", annotation_position="top left")
    figure.update_layout(title="Projected balance if you save the monthly amount shown", xaxis_title="Years from now", yaxis_title="Balance (USD)",
                         margin={"l": 10, "r": 10, "t": 50, "b": 10}, height=400, legend={"orientation": "h", "y": -0.25})
    figure.update_yaxes(tickprefix="$", tickformat=",.0f")
    return figure


def scenario_rows(plan: dict) -> list[dict]:
    return [{"Scenario": s["scenario"].title(), "Assumed yearly return": f"{s['annual_return_pct']:g}%", "Monthly saving needed": f"${s['monthly_needed']:,.0f}",
             "…keeping today's buying power": f"${s['monthly_needed_inflation_adjusted']:,.0f}", "You put in": f"${s['total_contributed']:,.0f}",
             "Growth assumed from returns": f"${s['growth_from_returns']:,.0f}"} for s in plan["scenarios"]]


def show_plan(plan: dict) -> None:
    years = f"{plan['years']:g}"
    st.subheader(f"To reach ${plan['goal']:,.0f} in {years} years")
    for flag in plan["flags"]:
        st.warning(flag)
    st.dataframe(scenario_rows(plan), hide_index=True, width="stretch")
    if plan["years"] >= 1:
        st.plotly_chart(projection_figure(plan), width="stretch")
    else:
        st.caption("The growth chart needs a goal at least one year away; for a shorter goal the table above is the whole picture.")   # the path is drawn in whole years and would run past the goal date
    today_value = plan["goal"] ** 2 / plan["inflation_adjusted_goal"]   # goal / (1 + inflation)^years
    st.caption(f"With {plan['inflation_pct']:g}% yearly inflation, ${plan['goal']:,.0f} in {years} years will buy about what ${today_value:,.0f} buys today. "
               f"To keep today's buying power you would need about ${plan['inflation_adjusted_goal']:,.0f}.")
    st.info(plan["assumptions"])
    st.caption(plan["disclaimer"])


def render(client: FinnieClient) -> None:
    st.markdown("Work out what you would need to save each month. These are **what-if illustrations using assumed returns, not predictions.**")
    with st.form("goal"):
        left, right = st.columns(2)
        goal = left.number_input("Goal amount ($)", min_value=1.0, max_value=1_000_000_000.0, value=20000.0, step=1000.0, format="%.0f")
        years = right.number_input("Years to get there", min_value=0.25, max_value=60.0, value=5.0, step=0.5)
        savings = left.number_input("Already saved ($)", min_value=0.0, max_value=1_000_000_000.0, value=2000.0, step=500.0, format="%.0f")
        profile = right.selectbox("Risk level", list(PROFILES), index=1)
        submitted = st.form_submit_button("Make a plan")
    if submitted:
        st.session_state.goal_request = (goal, years, savings, PROFILES[profile])
        st.session_state.goal_forced = True
    request = st.session_state.get("goal_request")
    if not request:
        return
    try:
        plan = guarded(("goal", *request), lambda: cached_plan(client, client.base_url, *request), force=st.session_state.pop("goal_forced", False))
    except ApiError as error:
        show_error(error)
        return
    show_plan(plan)
