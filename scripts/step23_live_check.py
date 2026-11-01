"""Step 23: drive the REAL Streamlit script (headless) against the REAL running API and count what each interaction costs.

  python -m scripts.step23_live_check [tab ...]      (API must be running on :8000)
"""
import sys
import time
from collections import Counter

from streamlit.testing.v1 import AppTest

from src.web_app.client import FinnieClient
from tests.test_ui import APP


class CountingClient(FinnieClient):
    """A real client that records every API call, so 'did that click call the server again?' becomes a number."""
    def __init__(self):
        super().__init__("http://127.0.0.1:8000")
        self.calls = Counter()

    def _call(self, method, path, **kwargs):
        self.calls[f"{method} {path.split('/%5E')[0] if '/market/' in path else path}"] += 1
        return super()._call(method, path, **kwargs)


def fresh():
    client = CountingClient()
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["client"] = client
    return at, client


def timed(label, action, client):
    before = Counter(client.calls)
    start = time.perf_counter()
    result = action()
    delta = client.calls - before
    print(f"  {label:58} {time.perf_counter() - start:6.2f}s  API calls: {dict(delta) or 'none'}")
    return result


def chat_tab():
    print("\nCHAT TAB (real script, real API)")
    at, client = fresh()
    timed("first page load", lambda: at.run(), client)
    at = timed("first question: 'What is an ETF?'", lambda: at.chat_input[0].set_value("What is an ETF?").run(), client)
    print("    answer shown:", [m.value[:90] for m in at.chat_message[1].markdown][:1])
    timed("a plain rerun with no new input", lambda: at.run(), client)
    timed("second question (same session): 'And a mutual fund?'", lambda: at.chat_input[0].set_value("And what is a mutual fund?").run(), client)
    print("    history messages on screen:", len(at.chat_message), "  session:", at.session_state["session_id"])
    timed("an empty question", lambda: at.chat_input[0].set_value("   ").run(), client)
    print("    warning:", [w.value for w in at.warning])


def markets_tab():
    print("\nMARKETS TAB (real script, real API)")
    at, client = fresh()
    at = timed("first page load (overview fetched)", lambda: at.run(), client)
    print("    metrics:", [(m.label, m.value, m.delta) for m in at.metric])

    def lookup(ticker, period="1 month"):
        at.text_input[0].set_value(ticker)
        at.selectbox[0].set_value(period)
        next(b for b in at.button if b.label == "Look up").click()
        return at.run()
    timed("look up AAPL, 1 month (quote + history)", lambda: lookup("aapl"), client)
    print("    chart drawn:", len(at.get("plotly_chart")), " price:", [(m.label, m.value, m.delta) for m in at.metric if m.label == "AAPL"])
    timed("a plain rerun (should call nothing)", lambda: at.run(), client)
    timed("same ticker, different period (history only)", lambda: lookup("aapl", "3 months"), client)
    timed("back to 1 month (all cached)", lambda: lookup("aapl"), client)
    timed("unknown ticker ZZZZ", lambda: lookup("zzzz"), client)
    print("    error shown:", [e.value for e in at.error])
    timed("malformed ticker 'DROP TABLE'", lambda: lookup("drop table"), client)
    print("    error shown:", [e.value for e in at.error])
    timed("index ^GSPC", lambda: lookup("^gspc"), client)
    print("    price:", [(m.label, m.value) for m in at.metric if m.label == "^GSPC"])
    api_down = CountingClient()
    api_down._http.base_url = "http://127.0.0.1:9"
    api_down.base_url = "http://127.0.0.1:9"
    dead = AppTest.from_file(APP, default_timeout=30)
    dead.session_state["client"] = api_down
    timed("page load with the server DOWN", lambda: dead.run(), api_down)
    print("    exception:", bool(dead.exception), " errors:", [e.value for e in dead.error][:2])


def portfolio_tab():
    print("\nPORTFOLIO TAB (real script, real API)")
    at, client = fresh()
    at.run()

    def submit(source, text=None, sample=None):
        at.radio[0].set_value(source).run()
        if text is not None:
            at.text_area[0].set_value(text)
        if sample:
            next(b for b in at.selectbox if b.label == "Sample").set_value(sample)
        next(b for b in at.button if b.label == "Analyze").click()
        return at.run()
    timed("analyze the 'Diversified' sample", lambda: submit("Sample portfolio"), client)
    print("    metrics:", [(m.label, m.value) for m in at.metric if m.label in ("Total value", "Total gain", "Holdings", "Diversification score")], " charts:", len(at.get("plotly_chart")))
    timed("a plain rerun (should call nothing)", lambda: at.run(), client)
    timed("same sample again (cached for 5 min)", lambda: submit("Sample portfolio"), client)
    timed("the 'One big stock' sample", lambda: submit("Sample portfolio", sample="One big stock"), client)
    print("    metrics:", [(m.label, m.value) for m in at.metric if m.label in ("Holdings", "Diversification score")])
    timed("pasted CSV with a missing column", lambda: submit("Paste CSV text", "ticker,shares\nAAPL,5"), client)
    print("    error:", [e.value for e in at.error], " details:", [c.value for c in at.caption if c.value.startswith("-")])
    timed("pasted CSV with a made-up ticker", lambda: submit("Paste CSV text", "ticker,shares,avg_cost\nZZZZ,5,10\nAAPL,2,100"), client)
    print("    notes:", [i.value for i in at.info][:3])
    timed("empty paste", lambda: submit("Paste CSV text", ""), client)
    print("    warning:", [w.value for w in at.warning])
    print("    portfolio now remembered for chat:", bool(at.session_state["portfolio_csv"]) if "portfolio_csv" in at.session_state else False)


def goals_tab():
    print("\nGOALS TAB (real script, real API)")
    at, client = fresh()
    at.run()

    def plan(goal, years, savings, risk=None):
        box = {b.label: b for b in at.number_input}
        box["Goal amount ($)"].set_value(goal)
        box["Years to get there"].set_value(years)
        box["Already saved ($)"].set_value(savings)
        if risk:
            next(b for b in at.selectbox if b.label == "Risk level").set_value(risk)
        next(b for b in at.button if b.label == "Make a plan").click()
        return at.run()

    def show():
        print("    table:", [list(r.values())[:4] for r in at.dataframe[0].value.to_dict("records")] if len(at.dataframe) else "none")
        print("    warnings:", [w.value[:90] for w in at.warning], " errors:", [e.value for e in at.error], " charts:", len(at.get("plotly_chart")))
    timed("$20,000 in 5 years, $2,000 saved, medium", lambda: plan(20000.0, 5.0, 2000.0), client); show()
    print("    inflation note:", [c.value for c in at.caption if "buying power" in c.value])
    timed("a plain rerun (should call nothing)", lambda: at.run(), client)
    timed("same plan again (cached)", lambda: plan(20000.0, 5.0, 2000.0), client)
    timed("high risk", lambda: plan(20000.0, 5.0, 2000.0, "High risk (mostly stocks)"), client); show()
    timed("already saved MORE than the goal", lambda: plan(10000.0, 5.0, 12000.0), client); show()
    timed("only 3 months (0.25 years)", lambda: plan(5000.0, 0.25, 0.0), client); show()
    timed("a huge goal: $5,000,000 in 2 years", lambda: plan(5_000_000.0, 2.0, 0.0), client); show()


def knowledge_tab():
    print("\nKNOWLEDGE TAB (real script, real API)")
    at, client = fresh()
    timed("first page load", lambda: at.run(), client)

    def search(query, topic=None):
        next(r for r in at.radio if r.label == "How do you want to explore?").set_value("Search").run()
        next(t for t in at.text_input if t.label == "What do you want to learn about?").set_value(query)
        if topic:
            next(x for x in at.selectbox if x.key == "search_topic").set_value(topic)
        next(b for b in at.button if b.label == "Search").click()
        return at.run()
    timed("search: what is an expense ratio", lambda: search("what is an expense ratio"), client)
    print("    top result:", [m.value[:70] for m in at.markdown if m.value.startswith("**")][:2])
    timed("same search again (cached)", lambda: search("what is an expense ratio"), client)
    timed("search limited to Taxes: wash sale", lambda: search("wash sale", "Taxes"), client)
    print("    top result:", [m.value[:70] for m in at.markdown if m.value.startswith("**")][:2])
    timed("off-topic search: best pizza in town", lambda: search("best pizza in town"), client)
    print("    info:", [i.value for i in at.info if "close match" in i.value])
    timed("one-letter search", lambda: search("x"), client)
    print("    warning:", [w.value for w in at.warning])
    timed("switch to Browse", lambda: next(r for r in at.radio if r.label == "How do you want to explore?").set_value("Browse articles").run(), client)
    box = next(x for x in at.selectbox if x.key == "browse_article")
    print("    articles listed:", len(box.options), " first:", box.options[0])
    timed("pick another article", lambda: box.set_value(box.options[3]).run(), client)
    print("    article text starts:", [m.value[:60] for m in at.markdown if m.value.startswith("#")][:1])


def whole_app():
    print("\nWHOLE APP: what does opening it cost, and what does a click in any tab cost?")
    at, client = fresh()
    timed("COLD first page load (all five tabs run)", lambda: at.run(), client)
    timed("WARM second page load", lambda: at.run(), client)
    timed("click 'New conversation'", lambda: next(b for b in at.sidebar.button if b.label == "New conversation").click().run(), client)
    at2, client2 = fresh()
    at2.run()
    timed("first chat question", lambda: at2.chat_input[0].set_value("What is an ETF?").run(), client2)
    print("    time to first answer = the line above; includes the router, retrieval and answer model on the server")


if __name__ == "__main__":
    {"chat": chat_tab, "markets": markets_tab, "portfolio": portfolio_tab, "goals": goals_tab, "knowledge": knowledge_tab, "app": whole_app}.get(sys.argv[1] if len(sys.argv) > 1 else "chat", chat_tab)()
