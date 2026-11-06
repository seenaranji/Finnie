"""UI tests with Streamlit's headless AppTest: the real script runs against a stand-in for the API, so no server, no browser and no network."""
import re

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.web_app.client import ApiError

from pathlib import Path

APP = str(Path(__file__).resolve().parents[1] / "src" / "web_app" / "app.py")


class FakeApi:
    """Same methods as FinnieClient; each can be replaced per test. Records what the UI sent."""
    base_url = "http://fake-api"

    def __init__(self):
        self.sent, self.chats = [], []
        self.chat_reply = lambda message, session_id, csv: {
            "session_id": "sess-1", "turn": 1, "text": f"Answer to: {message}", "answered_by": "finance_qa", "route": "finance_qa",
            "sources": [{"id": "etf-basics#0", "title": "ETF Basics", "url": "https://x.example"}], "disclaimer": "General education only.", "cost_usd": 0.002,
            "latency_s": 1.5, "degraded": False}

    token = None

    def health(self):
        return {"status": "ok"}

    def login(self, email, password):
        self.sent.append(("login", email))
        if password != "right-password-123":
            raise ApiError("The email or password is incorrect.", 401, "invalid_credentials", "req-login")
        return {"access_token": f"token-for-{email}", "token_type": "bearer", "expires_in": 3600, "user": {"id": "u-1", "email": email.lower()}}

    def register(self, email, password):
        self.sent.append(("register", email))
        if email == "taken@example.com":
            raise ApiError("Please fix the problems below.", 422, "invalid_auth_input", "req-reg", [{"problem": "email: an account with this email already exists; log in instead"}])
        return {"access_token": f"token-for-{email}", "token_type": "bearer", "expires_in": 3600, "user": {"id": "u-2", "email": email.lower()}}

    def overview(self):
        self.sent.append("overview")
        return {"indices": [{"name": "S&P 500", "quote": quote("^GSPC", 7580.07, -1.82)}, {"name": "Nasdaq Composite", "quote": quote("^IXIC", 25437.0, 0.4)}],
                "unavailable": [], "note": ""}

    def market_news(self):
        self.sent.append("news_market")
        return news_batch("market")

    def ticker_news(self, ticker):
        self.sent.append(("news_ticker", ticker))
        return news_batch(ticker)

    def quote(self, ticker):
        self.sent.append(("quote", ticker))
        if ticker == "ZZZZ":
            raise ApiError("I couldn't find a ticker called ZZZZ.", 404, "ticker_not_found", "req-404")
        if ticker == "BAD!":
            raise ApiError("The request was not valid.", 422, "invalid_request", "req-422")
        return quote(ticker, 344.63, -1.23, stale=ticker == "OLD")

    def history(self, ticker, period="1mo"):
        self.sent.append(("history", ticker, period))
        return {"symbol": ticker, "period": period, "source": "fake", "is_stale": False, "age_seconds": 0.0, "fetched_at": "2026-11-01T17:00:00Z",
                "points": [{"date": f"2026-10-{d:02d}", "close": 340 + d} for d in range(1, 6)]}

    def analyze_portfolio(self, csv_text, question=None):
        self.sent.append(("analyze", csv_text, question))
        if "bad" in csv_text:
            raise ApiError("Your portfolio file could not be read. It needs the columns ticker, shares and avg_cost.", 422, "invalid_portfolio", "req-p",
                           [{"problem": "missing column(s): avg_cost"}])
        return {"text": "Your portfolio is mostly stocks.", "disclaimer": "Not advice.", "sources": [{"id": "fake", "title": "fake prices as of 2026-10-30", "url": None}],
                "cost_usd": 0.004, "latency_s": 2.0,
                "metrics": {"total_value_usd": 42094, "total_gain_pct": 43.1, "holding_count": 2, "diversification_score_out_of_100": {"score": 44.1, "built_from": "a heuristic"},
                            "asset_class_pct": {"US stock": 70.0, "Bond": 30.0}, "holdings": [{"ticker": "VTI", "weight_pct": 70.0}, {"ticker": "BND", "weight_pct": 30.0}],
                            "notes": ["no price for ['XYZ']: left out"],
                            **({"what_if_scenarios": [self.what_if(o, t) for o, t in (("Bond", "steadier, lower expected growth"), ("Cash", "stable, little growth"))]}
                               if question and "grow" in question else {})},
                "correlation": None if "nocorr" in csv_text else {"tickers": ["VTI", "BND"], "matrix": [[1.0, 0.2], [0.2, 1.0]]}}

    @staticmethod
    def what_if(option, trade_off):
        return {"option": option, "from_sector": "Technology", "moved_pct_of_portfolio": 10.0, "trade_off": trade_off, "sector_pct": {"before": 50.0, "after": 40.0},
                "option_asset_class_pct": {"before": 20.0, "after": 30.0}, "hhi_concentration_0_to_1": {"before": 0.5, "after": 0.42},
                "effective_number_of_holdings": {"before": 2.0, "after": 2.4}, "diversification_score_out_of_100": {"before": 44.1, "after": 52.0}}

    def plan_goal(self, goal, years, current_savings, risk_profile):
        self.sent.append(("plan", goal, years, current_savings, risk_profile))
        if goal == 1.0:
            raise ApiError("the goal amount is not valid", 422, "invalid_goal", "req-g")
        scenarios = [{"scenario": n, "annual_return_pct": r, "monthly_needed": m, "monthly_needed_inflation_adjusted": m * 1.1, "current_savings_grow_to": 2100.0,
                      "total_contributed": m * 60, "growth_from_returns": 500.0, "already_reached": False, "path_by_year": [2000, 5000, 9000, 13000, 17000, goal]}
                     for n, r, m in (("conservative", 2.0, 330.0), ("expected", 5.0, 300.0), ("optimistic", 8.0, 280.0))]
        return {"goal": goal, "years": years, "current_savings": current_savings, "risk_profile": risk_profile, "inflation_pct": 3.0,
                "inflation_adjusted_goal": goal * 1.159, "scenarios": scenarios, "flags": ["A flag from the server."] if goal > 90000 else [],
                "assumptions": "ASSUMPTIONS, not forecasts: yearly returns of 2% / 5% / 8%.", "disclaimer": "Educational illustration only."}

    saved_id = "3f2b8c1e-5d4a-4c7e-9b1a-0a1b2c3d4e5f"

    saved_properties = None

    def save_portfolio(self, name, holdings, properties=None):
        self.sent.append(("save", name, holdings))
        self.saved_properties = properties
        return {"id": self.saved_id}

    def list_portfolios(self):
        self.sent.append("list")
        return [p for p in self._all_portfolios() if p["id"] not in self.deleted]

    deleted: set = set()

    def _all_portfolios(self):
        return [{"id": self.saved_id, "name": "Retirement", "holding_count": 3, "created_at": "2026-11-01T10:00:00Z", "updated_at": "2026-11-01T10:00:00Z"},
                {"id": "aaaaaaaa-1111-4222-8333-444444444444", "name": "Kids college", "holding_count": 1, "created_at": "2026-11-01T11:00:00Z", "updated_at": "2026-11-01T11:00:00Z"}]

    def portfolio_summary(self, portfolio_id):
        self.sent.append(("summary", portfolio_id))
        holding = lambda t, k, sh, ac, cost, price, gain, w: {"ticker": t, "asset_type": k, "shares": sh, "avg_cost": ac, "purchase_date": None, "cost": cost, "price": price,
                                                              "value": None if price is None else sh * price, "gain_pct": gain, "weight_pct": w}
        return {"id": portfolio_id, "name": "Retirement", "prices_as_of": "2026-10-30", "total_cost": 1800.0, "total_value": 2500.0, "gain_pct": 38.89, "holding_count": 3, "unpriced": [],
                "groups": [{"asset_type": "etf", "label": "ETFs", "count": 1, "unpriced_count": 0, "cost": 1300.0, "value": 2000.0, "gain_pct": 53.85, "weight_pct": 80.0,
                            "holdings": [holding("VTI", "etf", 10, 130.0, 1300.0, 200.0, 53.85, 80.0)]},
                           {"asset_type": "cash", "label": "Cash", "count": 1, "unpriced_count": 0, "cost": 500.0, "value": 500.0, "gain_pct": 0.0, "weight_pct": 20.0,
                            "holdings": [holding("CASH", "cash", 500, 1.0, 500.0, 1.0, 0.0, 20.0)]}]}

    def analyze_saved_portfolio(self, portfolio_id, question=None):
        self.sent.append(("analyze_saved", portfolio_id))
        return self.analyze_portfolio("ticker,shares,avg_cost\nVTI,1,1", None)

    def get_portfolio(self, portfolio_id):
        self.sent.append(("get", portfolio_id))
        return {"id": portfolio_id, "name": "Retirement" if portfolio_id == self.saved_id else "Kids college", "created_at": "2026-11-01T10:00:00Z", "updated_at": "2026-11-01T10:00:00Z",
                "holdings": [{"ticker": "VTI", "shares": "10", "avg_cost": "130.00", "asset_type": "etf", "purchase_date": "2026-01-05"},
                             {"ticker": "CASH", "shares": "500", "avg_cost": "1", "asset_type": "cash", "purchase_date": None}],
                "properties": [{"name": "Maple St", "property_type": "house", "purchase_price": "300000", "purchase_date": None, "current_value": "350000", "mortgage_balance": "200000",
                                "monthly_rent": "2000", "monthly_expenses": "500", "monthly_mortgage_payment": "1500", "mortgage_rate_pct": "6.5"}]}

    def update_portfolio(self, portfolio_id, name, holdings, properties=None):
        self.sent.append(("update", portfolio_id, name, holdings, properties))

    def mortgage_payoff(self, portfolio_id, extra=0):
        self.sent.append(("payoff", portfolio_id, extra))
        return {"portfolio": "x", "extra": extra, "properties": [{"name": "Maple St", "balance": 200000.0, "monthly_payment": 1500.0, "rate_pct": 6.5, "extra": extra, "status": "ok",
                                                                   "regular": {"months": 221, "years": 18.4, "payoff_date": "2045-04-01", "total_interest": 100000.0}, "with_extra": None, "saved": None}]}

    def delete_portfolio(self, portfolio_id):
        self.sent.append(("delete", portfolio_id))
        self.deleted = self.deleted | {portfolio_id}

    def articles(self, category=None):
        self.sent.append("articles")
        return [{"id": "etf-basics", "title": "ETF Basics", "category": "etfs", "difficulty": "beginner", "source_name": "Investor.gov", "last_updated": "2026-10-30"},
                {"id": "bonds-basics", "title": "Bonds Basics", "category": "bonds", "difficulty": "beginner", "source_name": "Investor.gov", "last_updated": "2026-10-30"},
                {"id": "wash-sale", "title": "Wash Sales", "category": "taxes", "difficulty": None, "source_name": None, "last_updated": None}]

    def article(self, doc_id):
        self.sent.append(("article", doc_id))
        return {"id": doc_id, "title": doc_id, "category": "etfs", "text": f"# Body of {doc_id}", "source_name": "Investor.gov" if doc_id != "wash-sale" else None,
                "source_url": "https://investor.gov/x" if doc_id != "wash-sale" else None, "last_updated": "2026-10-30" if doc_id != "wash-sale" else None}

    def search(self, query, category=None, top_k=5):
        self.sent.append(("search", query, category))
        if query == "pizza":
            return {"query": query, "answerable": False, "results": []}
        if query == "boom":
            raise ApiError("The knowledge search index is being rebuilt. Please try again shortly.", 503, "knowledge_index_unavailable", "req-k")
        return {"query": query, "answerable": True, "results": [{"doc_id": "etf-basics", "title": "ETF Basics", "section": "Costs", "source_name": "Investor.gov",
                                                                 "source_url": "https://investor.gov/x", "last_updated": "2026-10-30", "text": "An expense ratio is a yearly fee.", "similarity": 0.512}]}

    def chat(self, message, session_id=None, portfolio_csv=None, portfolio_id=None):
        self.chats.append((message, session_id, portfolio_csv))
        self.chat_portfolio_ids = getattr(self, "chat_portfolio_ids", []) + [portfolio_id]
        return self.chat_reply(message, session_id, portfolio_csv)


def quote(symbol, price, change, stale=False):
    return {"symbol": symbol, "price": price, "previous_close": None, "change_pct": change, "currency": "USD", "source": "fake", "as_of": "2026-10-30",
            "fetched_at": "2026-11-01T17:00:00Z", "is_stale": stale, "age_seconds": 600.0 if stale else 0.0}


def news_batch(topic, items=None, withheld=0, feed_errors=()):
    items = [{"id": 1, "title": f"{topic} rallies on rate hopes", "summary": "Stocks rose.", "link": "https://news.example/1", "source": "CNBC Finance",
              "published": "2026-11-01T14:30:00Z"},
             {"id": 2, "title": "Fed holds rates", "summary": "", "link": "https://news.example/2", "source": "Federal Reserve Press Releases", "published": None}] \
        if items is None else items
    return {"topic": topic, "items": items, "fetched_at": "2026-11-01T17:00:00Z", "withheld": withheld, "feed_errors": list(feed_errors)}


def app(api=None, timeout=10, signed_in=True):
    st.cache_data.clear()   # cached values outlive a single AppTest run; each test must start cold
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.session_state["client"] = api or FakeApi()
    if signed_in:   # every test below the login tests starts as a logged-in user; the login tests pass signed_in=False
        at.session_state["auth"] = {"token": "test-token", "email": "tester@example.com"}
    return at.run()


def plain(text):
    """What the person reads: the app writes $ as \\$ so Streamlit does not take two amounts for a formula; the screen shows a plain $."""
    return text.replace("\\$", "$")


def tab_names(at):
    """The app's tab labels carry an icon (':material/forum: Chat'); the words are what matter."""
    return [re.sub(r"^:material/\w+:\s*", "", t.label) for t in at.tabs]


def texts(at):
    return [plain(x.value) for group in (at.markdown, at.caption, at.error, at.warning) for x in group]


# ---------- Chat tab ----------
def test_the_page_loads_with_a_disclaimer_and_no_exception():
    at = app()
    assert not at.exception and any("not financial advice" in m.value for m in at.markdown)   # a quiet line under the header, on every page


def test_a_chat_turn_shows_the_answer_badge_sources_and_disclaimer():
    api = FakeApi()
    at = app(api)
    at.chat_input[0].set_value("What is an ETF?").run()
    shown = " ".join(texts(at))
    assert not at.exception and "Answer to: What is an ETF?" in shown and "General education only." in shown
    assert ":blue-badge[Answered by: Finance Q&A]" in shown
    assert [e.label for e in at.expander] == ["Sources (1)"] and any("ETF Basics" in m.value for m in at.expander[0].markdown)


def test_the_session_id_from_the_first_reply_is_sent_with_the_second_message():
    api = FakeApi()
    at = app(api)
    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("second").run()
    assert api.chats[0][1] is None and api.chats[1][1] == "sess-1"


def test_history_survives_a_rerun():
    at = app()
    at.chat_input[0].set_value("hello there").run()
    at.run()   # any other interaction re-runs the whole script
    assert len([m for m in at.chat_message]) == 2


def test_an_api_failure_shows_a_friendly_error_with_the_request_id_and_no_answer():
    api = FakeApi()
    api.chat_reply = lambda *a: (_ for _ in ()).throw(ApiError("The server took too long to answer. Please try again.", 504, "timeout", "req-9"))
    at = app(api)
    at.chat_input[0].set_value("anything").run()
    assert not at.exception and any("took too long" in e.value for e in at.error) and any("req-9" in c.value for c in at.caption)
    assert "Answered by" not in " ".join(texts(at))


def test_a_starter_button_sends_its_question():
    api = FakeApi()
    at = app(api)
    at.button(key="starter_0").click().run()
    assert api.chats and api.chats[0][0] == "What is an ETF?"


def test_new_conversation_clears_history_and_session():
    at = app()
    at.chat_input[0].set_value("hi").run()
    at.button(key="new_conversation").click().run()
    assert len(at.chat_message) == 0 and at.session_state["session_id"] is None


def test_a_server_that_is_down_is_shown_in_the_header():
    api = FakeApi()
    api.health = lambda: (_ for _ in ()).throw(ApiError("I can't reach the Finnie server at http://x. Is it running?"))
    at = app(api)
    assert any("can't reach" in e.value for e in at.error) and not at.exception
    assert any("Server unavailable" in m.value for m in at.markdown)   # and the status pill says so


# ---------- Markets tab ----------
def lookup(at, ticker, period="1 month"):
    at.text_input[0].set_value(ticker)
    at.selectbox[0].set_value(period)
    next(b for b in at.button if b.label == "Look up").click()
    return at.run()


def test_overview_shows_each_index_as_a_metric_with_its_change():
    at = app()
    metrics = {m.label: (m.value, m.delta) for m in at.metric}
    assert metrics["S&P 500"] == ("7,580.07", "-1.82%") and metrics["Nasdaq Composite"][1] == "+0.40%"
    assert any("Closing price as of 2026-10-30" in c.value for c in at.caption)


def test_overview_partial_failure_is_listed_not_hidden():
    api = FakeApi()
    base = api.overview
    api.overview = lambda: {**base(), "unavailable": ["Dow Jones Industrial Average"]}
    at = app(api)
    assert any("Dow Jones" in w.value for w in at.warning)


def test_overview_error_is_shown_and_the_rest_of_the_page_still_works():
    api = FakeApi()
    api.overview = lambda: (_ for _ in ()).throw(ApiError("The data service is down.", 502, "data_provider_unavailable", "req-5"))
    at = app(api)
    assert not at.exception and any("data service is down" in e.value for e in at.error) and any("req-5" in c.value for c in at.caption)


def test_lookup_shows_price_freshness_and_a_chart():
    at = lookup(app(), "aapl")
    assert any(m.label == "AAPL" and m.value == "$344.63" for m in at.metric) and len(at.get("plotly_chart")) == 1


def test_lookup_of_a_stale_quote_warns_the_user():
    at = lookup(app(), "old")
    assert any("older saved value" in w.value for w in at.warning)


def test_unknown_ticker_shows_the_server_message_and_no_chart():
    at = lookup(app(), "zzzz")
    assert any("couldn't find a ticker" in e.value for e in at.error) and len(at.get("plotly_chart")) == 0


def test_malformed_ticker_gets_a_plain_explanation_not_a_validation_dump():
    at = lookup(app(), "bad!")
    assert any("doesn't look like a ticker" in e.value for e in at.error)


def test_empty_ticker_asks_for_one_and_calls_nothing():
    api = FakeApi()
    at = lookup(app(api), "")
    assert any("type a ticker" in w.value for w in at.warning) and not [s for s in api.sent if isinstance(s, tuple)]


def test_rerunning_does_not_refetch_prices_but_a_new_symbol_does():
    api = FakeApi()
    at = lookup(app(api), "aapl")
    at.run()
    at.run()
    assert api.sent.count("overview") == 1 and api.sent.count(("quote", "AAPL")) == 1 and api.sent.count(("history", "AAPL", "1mo")) == 1
    lookup(at, "msft")
    assert api.sent.count(("quote", "MSFT")) == 1 and api.sent.count("overview") == 1


def test_history_figure_has_one_line_with_the_right_points():
    from src.web_app.tabs.markets import history_figure
    figure = history_figure(FakeApi().history("AAPL", "5d"))
    assert len(figure.data) == 1 and list(figure.data[0].y) == [341, 342, 343, 344, 345] and "AAPL" in figure.layout.title.text


def test_an_index_level_is_not_shown_as_dollars():
    at = lookup(app(), "^gspc")
    assert any(m.label == "^GSPC" and m.value == "344.63" for m in at.metric)


# ---------- Portfolio tab ----------
def choose_source(at, pick):
    """The Portfolio tab has three modes; the analyze-a-file mode (the original tab) is reached by choosing it, then the source."""
    at.radio(key="portfolio_mode").set_value("Analyze a file or sample").run()
    next(r for r in at.radio if r.label == "Where are your holdings?").set_value(pick).run()


def analyze(at, pick="Sample portfolio"):
    choose_source(at, pick)
    next(b for b in at.button if b.label == "Analyze").click()
    return at.run()


def paste(at, text):
    """Switching the source reruns at once (it is outside the form); the pasted text is only read when the form is submitted, so set it and click in one run."""
    choose_source(at, "Paste CSV text")
    at.text_area[0].set_value(text)
    next(b for b in at.button if b.label == "Analyze").click()
    return at.run()


def test_a_sample_portfolio_is_analyzed_and_drawn():
    api = FakeApi()
    at = analyze(app(api))
    assert not at.exception and any(m.label == "Total value" and m.value == "$42,094" for m in at.metric)
    assert any(m.label == "Diversification score" and m.value == "44 / 100" for m in at.metric)
    assert len(at.get("plotly_chart")) == 3 and any("mostly stocks" in m.value for m in at.markdown)
    assert at.session_state["portfolio_csv"].startswith("ticker,shares,avg_cost") and any("no price for" in i.value for i in at.info)


def test_a_growth_question_shows_the_what_if_options_with_a_warning_and_a_table():
    api = FakeApi()
    choose_source(app_ := app(api), "Sample portfolio")
    next(t for t in app_.text_input if t.label.startswith("Anything specific")).set_value("How can I grow this portfolio?")
    next(b for b in app_.button if b.label == "Analyze").click()
    at = app_.run()
    assert not at.exception and len(at.get("plotly_chart")) == 4                                   # the three usual charts plus the before/after score
    assert any("illustrations, not advice" in w.value for w in at.warning)
    table = at.dataframe[0].value
    assert list(table["Option"]) == ["Bond", "Cash"] and table["Score"][0] == "44 → 52" and table["Technology share (now → after)"][0] == "50.0% → 40.0%"


def test_without_a_growth_question_there_is_no_what_if_section():
    at = analyze(app(FakeApi()))
    assert not any("illustrations, not advice" in w.value for w in at.warning) and not at.dataframe


def test_the_same_portfolio_is_not_sent_twice_when_the_page_reruns():
    api = FakeApi()
    at = analyze(app(api))
    at.run()
    at.run()
    assert len([s for s in api.sent if isinstance(s, tuple) and s[0] == "analyze"]) == 1


def test_pasted_text_that_the_server_rejects_shows_every_problem_and_an_example():
    at = paste(app(), "bad,data")
    assert any("could not be read" in e.value for e in at.error) and any("missing column(s): avg_cost" in c.value for c in at.caption)
    assert any("ticker,shares,avg_cost" in c.value for c in at.get("code")) and "portfolio_csv" not in at.session_state


def test_empty_paste_asks_for_input_and_calls_nothing():
    api = FakeApi()
    at = app(api)
    choose_source(at, "Paste CSV text")
    next(b for b in at.button if b.label == "Analyze").click()
    at.run()
    assert any("Paste some holdings" in w.value for w in at.warning) and not [s for s in api.sent if isinstance(s, tuple) and s[0] == "analyze"]


def test_missing_correlation_is_explained_instead_of_an_empty_chart():
    at = paste(app(), "ticker,shares,avg_cost\nnocorr,1,1")
    assert len(at.get("plotly_chart")) == 2 and any("enough overlapping price history" in i.value for i in at.info)


def test_an_oversized_paste_is_refused_before_the_server_is_asked():
    api = FakeApi()
    at = paste(app(api), "a" * 20001)
    assert any("too large" in e.value for e in at.error) and not [s for s in api.sent if isinstance(s, tuple) and s[0] == "analyze"]


def test_forget_clears_the_portfolio_used_by_chat():
    at = analyze(app())
    next(b for b in at.button if b.label == "Forget this portfolio").click()
    at.run()
    assert not at.session_state["portfolio_csv"]


def test_portfolio_figures_have_the_expected_shapes():
    from src.web_app.tabs.portfolio import allocation_figure, concentration_figure, correlation_figure
    assert list(allocation_figure({"A": 60.0, "B": 40.0}).data[0].values) == [60.0, 40.0]
    assert list(concentration_figure([{"ticker": "A", "weight_pct": 10.0}, {"ticker": "B", "weight_pct": 90.0}]).data[0].y) == ["A", "B"]   # smallest first, so the biggest is on top
    heat = correlation_figure({"tickers": ["A", "B"], "matrix": [[1.0, 0.5], [0.5, 1.0]]}).data[0]
    assert (heat.zmin, heat.zmax) == (-1, 1)


# ---------- failures must not be re-sent on every rerun ----------
def test_a_failed_lookup_is_not_re_sent_when_the_page_reruns_but_is_when_the_user_presses_the_button_again():
    api = FakeApi()
    at = lookup(app(api), "zzzz")
    at.run()
    at.run()
    assert api.sent.count(("quote", "ZZZZ")) == 1 and any("couldn't find" in e.value for e in at.error)
    next(b for b in at.button if b.label == "Look up").click()
    at.run()
    assert api.sent.count(("quote", "ZZZZ")) == 2


def test_a_failed_portfolio_analysis_is_not_re_sent_on_reruns():
    api = FakeApi()
    at = paste(app(api), "bad,data")
    at.run()
    at.run()
    assert len([s for s in api.sent if isinstance(s, tuple) and s[0] == "analyze"]) == 1 and any("could not be read" in e.value for e in at.error)


def test_a_failure_is_retried_after_the_wait_has_passed(monkeypatch):
    import time as real_time
    from src.web_app import ui_common
    clock = [real_time.time()]
    monkeypatch.setattr(ui_common.time, "time", lambda: clock[0])
    api = FakeApi()
    at = lookup(app(api), "zzzz")
    clock[0] += ui_common.RETRY_AFTER_SECONDS + 1
    at.run()
    assert api.sent.count(("quote", "ZZZZ")) == 2


# ---------- Goals tab ----------
def plan(at, goal=20000.0, years=5.0, savings=2000.0, risk=None):
    box = {b.label: b for b in at.number_input}
    box["Goal amount ($)"].set_value(goal)
    box["Years to get there"].set_value(years)
    box["Already saved ($)"].set_value(savings)
    if risk:
        next(b for b in at.selectbox if b.label == "Risk level").set_value(risk)
    next(b for b in at.button if b.label == "Make a plan").click()
    return at.run()


def test_a_goal_plan_shows_three_scenarios_a_chart_and_the_assumptions():
    api = FakeApi()
    at = plan(app(api))
    scenario_tables = [d for d in at.dataframe if len(d.value) == 3]   # (the Portfolio tab's entry table is also a dataframe, with one row)
    assert not at.exception and len(scenario_tables) == 1
    assert [r["Scenario"] for r in scenario_tables[0].value.to_dict("records")] == ["Conservative", "Expected", "Optimistic"]
    assert any("not forecasts" in i.value for i in at.info) and any("Educational illustration" in c.value for c in at.caption) and len(at.get("plotly_chart")) >= 1
    assert api.sent[-1] == ("plan", 20000.0, 5.0, 2000.0, "medium")


def test_the_chosen_risk_level_is_sent_as_the_servers_word_for_it():
    api = FakeApi()
    plan(app(api), risk="High risk (mostly stocks)")
    assert api.sent[-1][-1] == "high"


def test_server_warnings_about_the_plan_are_shown():
    at = plan(app(), goal=100000.0)
    assert any("A flag from the server." in w.value for w in at.warning)


def test_a_rejected_goal_shows_the_server_message_and_no_chart():
    at = plan(app(), goal=1.0)
    assert any("not valid" in e.value for e in at.error) and not any("Projected balance" in str(p.value) for p in at.get("plotly_chart"))


def test_rerunning_does_not_ask_the_server_again():
    api = FakeApi()
    at = plan(app(api))
    at.run()
    at.run()
    assert len([s for s in api.sent if isinstance(s, tuple) and s[0] == "plan"]) == 1


def test_number_inputs_stop_nonsense_before_it_is_sent():
    at = app()
    box = {b.label: b for b in at.number_input}
    assert box["Goal amount ($)"].min == 1.0 and box["Years to get there"].max == 60.0 and box["Already saved ($)"].min == 0.0


def test_projection_figure_has_three_lines_and_a_goal_marker():
    from src.web_app.tabs.goals import projection_figure
    figure = projection_figure(FakeApi().plan_goal(20000.0, 5.0, 2000.0, "medium"))
    assert len(figure.data) == 3 and figure.layout.shapes[0].y0 == 20000.0


def test_a_goal_under_a_year_shows_no_chart_that_would_run_past_the_goal_date():
    at = plan(app(), goal=5000.0, years=0.5, savings=0.0)
    assert len(at.get("plotly_chart")) == 0 and any("at least one year" in c.value for c in at.caption)


def test_the_inflation_sentence_gives_todays_value_of_the_goal():
    at = plan(app(), goal=20000.0, years=5.0)
    assert any("will buy about what $17,256 buys today" in plain(c.value) for c in at.caption)


# ---------- Knowledge tab ----------
def search(at, query, topic=None):
    knowledge_radio = next(r for r in at.radio if r.label == "How do you want to explore?")
    knowledge_radio.set_value("Search").run()
    next(t for t in at.text_input if t.label == "What do you want to learn about?").set_value(query)
    if topic:
        next(s for s in at.selectbox if s.key == "search_topic").set_value(topic)
    next(b for b in at.button if b.label == "Search").click()
    return at.run()


def test_a_search_shows_the_passage_its_source_and_its_date():
    at = search(app(), "expense ratio")
    shown = " ".join(texts(at))
    assert "An expense ratio is a yearly fee." in shown and "Investor.gov" in shown and "updated 2026-10-30" in shown and "ETF Basics" in shown


def test_a_topic_filter_is_sent_as_the_servers_category_name():
    api = FakeApi()
    search(app(api), "expense ratio", topic="Etfs")
    assert ("search", "expense ratio", "etfs") in api.sent


def test_a_search_with_no_match_says_so_plainly():
    at = search(app(), "pizza")
    assert any("close match" in i.value for i in at.info)


def test_a_search_server_failure_is_shown_with_a_retry_hint_and_not_repeated_on_reruns():
    api = FakeApi()
    at = search(app(api), "boom")
    at.run()
    at.run()
    assert any("being rebuilt" in e.value for e in at.error) and any("try again" in c.value.lower() for c in at.caption)
    assert len([s for s in api.sent if isinstance(s, tuple) and s[0] == "search"]) == 1


def test_a_one_letter_search_is_refused_before_the_server_is_asked():
    api = FakeApi()
    at = search(app(api), "x")
    assert any("at least two characters" in w.value for w in at.warning) and not [s for s in api.sent if isinstance(s, tuple) and s[0] == "search"]


def test_browse_lists_articles_by_topic_and_shows_the_chosen_one_with_its_source():
    api = FakeApi()
    at = app(api)
    next(r for r in at.radio if r.label == "How do you want to explore?").set_value("Browse articles").run()
    next(s for s in at.selectbox if s.key == "browse_topic").set_value("Bonds").run()
    chosen = next(s for s in at.selectbox if s.key == "browse_article")
    assert chosen.options == ["Bonds Basics  (Bonds, beginner)"]
    assert ("article", "bonds-basics") in api.sent and any("Body of bonds-basics" in m.value for m in at.markdown)
    assert any("Source: [Investor.gov](https://investor.gov/x)" in c.value for c in at.caption)


def test_an_article_with_no_source_recorded_says_so_instead_of_inventing_one():
    at = app()
    next(r for r in at.radio if r.label == "How do you want to explore?").set_value("Browse articles").run()
    next(s for s in at.selectbox if s.key == "browse_topic").set_value("Taxes").run()
    assert any("Source: not recorded" in c.value and "last updated unknown" in c.value for c in at.caption)


def test_the_article_list_is_fetched_once_however_often_the_page_reruns():
    api = FakeApi()
    at = app(api)
    at.run()
    at.run()
    assert api.sent.count("articles") == 1


# ---------- News tab ----------
def show_news(at, ticker):
    next(t for t in at.text_input if t.label.startswith("Ticker (leave empty")).set_value(ticker)
    next(b for b in at.button if b.label == "Show news").click()
    return at.run()


def test_market_news_lists_headlines_with_links_source_and_date():
    at = app()
    shown = texts(at)
    assert not at.exception and any(sub.value == "Market news" for sub in at.subheader)
    assert any("[market rallies on rate hopes](https://news.example/1)" in m for m in shown)
    assert any(m.value == "Stocks rose." for m in at.markdown)
    assert any("CNBC Finance · 2026-11-01" in c for c in shown)
    assert any(c.endswith("· date unknown") for c in shown)   # an item with no date says so, in full
    assert any("Finnie does not vouch" in c for c in shown)


def test_no_headlines_says_so_plainly():
    api = FakeApi()
    api.market_news = lambda: news_batch("market", items=[])
    at = app(api)
    assert not at.exception and any("No recent headlines" in i.value for i in at.info)


def test_a_news_failure_shows_the_server_message_and_request_id_and_the_other_tabs_still_work():
    api = FakeApi()
    api.market_news = lambda: (_ for _ in ()).throw(ApiError("The news feeds are not responding right now.", 502, "news_unavailable", "req-n"))
    at = app(api)
    assert not at.exception and any("news feeds are not responding" in e.value for e in at.error) and any("req-n" in c.value for c in at.caption)
    assert any(m.label == "S&P 500" for m in at.metric)   # the Markets tab was not broken by it


def test_withheld_headlines_are_counted_not_shown():
    api = FakeApi()
    api.market_news = lambda: news_batch("market", withheld=2)
    at = app(api)
    assert any("2 item(s) were left out because their text looked unsafe" in c.value for c in at.caption)
    assert not any("ignore previous" in m for m in texts(at))


def test_a_feed_that_did_not_respond_warns_that_the_list_may_be_incomplete():
    api = FakeApi()
    api.market_news = lambda: news_batch("market", feed_errors=["MarketWatch Top Stories"])
    at = app(api)
    assert any("may be incomplete" in w.value for w in at.warning)


def test_a_ticker_is_uppercased_and_asks_for_that_tickers_news():
    api = FakeApi()
    at = show_news(app(api), " aapl ")
    assert ("news_ticker", "AAPL") in api.sent and any(sub.value == "News: AAPL" for sub in at.subheader)


def test_emptying_the_ticker_goes_back_to_market_news():
    api = FakeApi()
    at = show_news(show_news(app(api), "aapl"), "")
    assert any(sub.value == "Market news" for sub in at.subheader)


def test_rerunning_does_not_refetch_the_news_but_a_failure_is_not_resent_either():
    api = FakeApi()
    at = app(api)
    at.run()
    assert api.sent.count("news_market") == 1
    failing = FakeApi()
    failing.market_news = lambda: (failing.sent.append("news_market"), (_ for _ in ()).throw(ApiError("down", 502, "news_unavailable", "r")))[1]
    bad = app(failing)
    bad.run()
    bad.run()
    assert failing.sent.count("news_market") == 1


# ---------- Login and registration (Step 27) ----------
def submit(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


def fill(at, key, value):
    at.text_input(key=key).set_value(value)


def test_nobody_sees_the_app_or_causes_a_single_api_call_before_logging_in():
    api = FakeApi()
    at = app(api, signed_in=False)
    assert not at.exception and api.sent == []                        # no health check, no market data, nothing
    assert [t.label for t in at.tabs] == ["Log in", "Create account"]  # the six app tabs are not even built
    assert not at.chat_input and not at.metric


def test_logging_in_shows_the_app_and_sends_the_token_with_every_later_call():
    api = FakeApi()
    at = app(api, signed_in=False)
    fill(at, "login_email", "Alice@Example.com")
    fill(at, "login_password", "right-password-123")
    at = submit(at, "Log in")
    assert not at.exception and ("login", "Alice@Example.com") in api.sent
    assert api.token == "token-for-Alice@Example.com" and any("alice@example.com" in t.value for t in at.get("text"))   # the account menu in the header shows it as plain text, never as a link
    assert tab_names(at)[:3] == ["Chat", "Markets", "News"]


def test_wrong_credentials_show_the_servers_message_and_keep_the_door_shut():
    api = FakeApi()
    at = app(api, signed_in=False)
    fill(at, "login_email", "alice@example.com")
    fill(at, "login_password", "nope")
    at = submit(at, "Log in")
    assert any("email or password is incorrect" in e.value for e in at.error) and api.token is None and "auth" not in at.session_state
    assert [t.label for t in at.tabs] == ["Log in", "Create account"]


def test_empty_fields_are_refused_before_the_server_is_asked():
    api = FakeApi()
    at = submit(app(api, signed_in=False), "Log in")
    assert any("Enter your email and password" in e.value for e in at.error) and api.sent == []


def test_creating_an_account_signs_the_new_user_in():
    api = FakeApi()
    at = app(api, signed_in=False)
    fill(at, "register_email", "new@example.com")
    fill(at, "register_password", "a-fine-password-1")
    fill(at, "register_confirm", "a-fine-password-1")
    at = submit(at, "Create account")
    assert not at.exception and ("register", "new@example.com") in api.sent and api.token == "token-for-new@example.com"
    assert tab_names(at)[:2] == ["Chat", "Markets"]


def test_mismatched_passwords_are_caught_before_the_server_is_asked():
    api = FakeApi()
    at = app(api, signed_in=False)
    fill(at, "register_email", "new@example.com")
    fill(at, "register_password", "a-fine-password-1")
    fill(at, "register_confirm", "a-different-one-2")
    at = submit(at, "Create account")
    assert any("not the same" in e.value for e in at.error) and api.sent == []


def test_a_taken_email_shows_the_servers_explanation():
    at = app(signed_in=False)
    fill(at, "register_email", "taken@example.com")
    fill(at, "register_password", "a-fine-password-1")
    fill(at, "register_confirm", "a-fine-password-1")
    at = submit(at, "Create account")
    assert any("Please fix the problems" in e.value for e in at.error) and any("already exists" in c.value for c in at.caption)


def chatted_and_loaded_a_portfolio(api):
    """A signed-in user who really asked something in chat (so `messages` and `session_id` exist, made by the real tab code) and has a portfolio loaded."""
    at = app(api)
    at.session_state["portfolio_csv"] = "ticker,shares,avg_cost\nVTI,1,1"
    at.chat_input[0].set_value("my private question").run()
    assert at.session_state["messages"] and at.session_state["session_id"] == "sess-1"   # guards the test: the state it checks for really existed
    return at


def test_logging_out_removes_the_token_and_the_previous_users_data():
    api = FakeApi()
    at = submit(chatted_and_loaded_a_portfolio(api), "Log out")
    assert api.token is None and "auth" not in at.session_state
    for key in ("messages", "session_id", "portfolio_csv"):
        assert key not in at.session_state, f"{key} survived the logout"
    assert [t.label for t in at.tabs] == ["Log in", "Create account"]
    assert "my private question" not in " ".join(texts(at))


def test_a_session_that_the_server_stopped_accepting_returns_to_the_login_screen_with_an_explanation():
    api = FakeApi()
    api.overview = lambda: (_ for _ in ()).throw(ApiError("Please log in to use Finnie.", 401, "not_authenticated", "req-401"))
    at = app(api)
    assert not at.exception and "auth" not in at.session_state and api.token is None
    assert any("Your session has ended" in w.value for w in at.warning) and [t.label for t in at.tabs] == ["Log in", "Create account"]


def test_the_next_user_on_the_same_browser_tab_starts_clean():
    api = FakeApi()
    at = submit(chatted_and_loaded_a_portfolio(api), "Log out")
    fill(at, "login_email", "bob@example.com")
    fill(at, "login_password", "right-password-123")
    at = submit(at, "Log in")
    assert at.session_state["auth"]["email"] == "bob@example.com"
    assert not at.session_state["messages"] and at.session_state["session_id"] is None and "portfolio_csv" not in at.session_state
    assert "my private question" not in " ".join(texts(at))
    assert "_failures" not in at.session_state or not at.session_state["_failures"]   # an earlier user's remembered failures must not block the next one


# ---------- Portfolio tab: enter, save, see grouped, load by id, delete ----------
import pandas as pd  # noqa: E402

from src.web_app.tabs.portfolio import rows_to_holdings, server_problems_by_row  # noqa: E402

SYMBOL, TYPE, SHARES, COST, BOUGHT = "Symbol", "Type", "Shares", "Avg cost per share ($)", "Purchase date"
ID = FakeApi.saved_id


def fill_table(at, rows, name="Retirement", version=0):
    """Put rows into the entry table the way the browser does (the editor's delta state) and name the portfolio. The first row is the blank one the table starts with."""
    first, *more = rows
    at.session_state[f"pf_table_{version}"] = {"edited_rows": {0: first}, "added_rows": more, "deleted_rows": []}
    at.text_input(key=f"pf_name_{version}").set_value(name)
    return at


def save(at, version=0):
    at.button(key=f"pf_save_{version}").click()
    return at.run()


VTI = {SYMBOL: "VTI", TYPE: "ETF", SHARES: 10.0, COST: 130.0}
BND = {SYMBOL: "bnd", TYPE: "Bond", SHARES: 5.0, COST: 74.5, BOUGHT: "2024-03-15"}


def test_the_portfolio_tab_opens_on_entering_holdings_with_a_table_a_name_and_a_save_button():
    at = app()
    assert at.radio(key="portfolio_mode").value == "Enter my holdings" and at.text_input(key="pf_name_0") and at.button(key="pf_save_0")
    assert not at.exception


def test_saving_sends_exact_text_numbers_kinds_and_dates_and_shows_the_new_unique_id():
    api = FakeApi()
    at = save(fill_table(app(api), [VTI, BND]))
    (_, name, holdings), = [s for s in api.sent if isinstance(s, tuple) and s[0] == "save"]
    assert name == "Retirement"
    assert holdings == [{"ticker": "VTI", "asset_type": "etf", "shares": "10.0", "avg_cost": "130.0"},
                        {"ticker": "bnd", "asset_type": "bond", "shares": "5.0", "avg_cost": "74.5", "purchase_date": "2024-03-15"}]   # the server upper-cases bnd
    assert not at.exception and at.radio(key="portfolio_mode").value == "My saved portfolios"            # taken straight to the grouped view
    assert any(ID in s.value for s in at.success) and any(c.value == ID for c in at.get("code"))
    assert at.session_state["selected_portfolio_id"] == ID


def test_the_form_is_reset_after_a_save_so_the_same_portfolio_is_not_saved_twice_by_accident():
    api = FakeApi()
    at = save(fill_table(app(api), [VTI]))
    at.radio(key="portfolio_mode").set_value("Enter my holdings").run()
    assert at.text_input(key="pf_name_1").value == "" and at.session_state["pf_version"] == 1


def test_missing_pieces_are_named_by_row_and_nothing_is_sent():
    api = FakeApi()
    at = save(fill_table(app(api), [{SYMBOL: "VTI", TYPE: "ETF", SHARES: 10.0}, {SYMBOL: "", TYPE: "Bond", SHARES: -3.0, COST: 70.0}], name="  "))
    warnings = " | ".join(w.value for w in at.warning)
    assert "Give the portfolio a name" in warnings and "Row 1: enter what you paid per share" in warnings
    assert "Row 2: enter the symbol" in warnings and "Row 2: shares must be a number greater than 0" in warnings
    assert not [s for s in api.sent if isinstance(s, tuple) and s[0] == "save"]


def test_an_empty_table_asks_for_a_holding():
    api = FakeApi()
    at = save(app(api).run() and fill_table(app(api), [{}]))
    assert any("Add at least one holding" in w.value for w in at.warning) and not [s for s in api.sent if isinstance(s, tuple) and s[0] == "save"]


def test_server_complaints_are_shown_against_the_rows_the_person_sees_even_when_a_blank_row_sits_between():
    api = FakeApi()
    api.save_portfolio = lambda name, holdings, properties=None: (_ for _ in ()).throw(ApiError("The request was not valid.", 422, "invalid_request", "req-s", [
        {"field": "holdings.1.ticker", "problem": "Value error, 'N/A!' is not a valid ticker"}, {"field": "name", "problem": "String should have at most 100 characters"}]))
    at = save(fill_table(app(api), [VTI, {}, {SYMBOL: "N/A!", TYPE: "Stock", SHARES: 1.0, COST: 1.0}]))
    lines = " | ".join(c.value for c in at.caption)
    assert "Row 3, symbol: 'N/A!' is not a valid ticker" in lines and "String should have at most 100 characters" in lines
    assert at.radio(key="portfolio_mode").value == "Enter my holdings"          # stays on the form so nothing typed is lost


def test_rows_to_holdings_ignores_blank_rows_and_keeps_the_number_text_exact():
    table = pd.DataFrame({SYMBOL: ["", " vti ", ""], TYPE: ["Stock", "ETF", "Stock"], SHARES: [None, 0.1, None], COST: [None, 220.1, None],
                          BOUGHT: pd.Series([pd.NaT, pd.Timestamp("2024-01-02"), pd.NaT], dtype="datetime64[ns]")})
    holdings, rows, problems = rows_to_holdings(table)
    assert problems == [] and rows == [2] and holdings == [{"ticker": "vti", "asset_type": "etf", "shares": "0.1", "avg_cost": "220.1", "purchase_date": "2024-01-02"}]


def test_a_future_purchase_date_is_refused_before_sending():
    table = pd.DataFrame({SYMBOL: ["VTI"], TYPE: ["ETF"], SHARES: [1.0], COST: [1.0], BOUGHT: pd.Series([pd.Timestamp.today().normalize() + pd.Timedelta(days=3)], dtype="datetime64[ns]")})
    assert "Row 1: the purchase date cannot be in the future." in rows_to_holdings(table)[2]


def test_server_problem_text_loses_its_technical_prefix_and_falls_back_for_unknown_rows():
    error = ApiError("x", 422, "invalid_request", "r", [{"field": "holdings.0.shares", "problem": "Value error, shares must be greater than 0"}, {"field": "holdings.9.shares", "problem": "odd"}])
    assert server_problems_by_row(error, [4]) == ["Row 4, shares: shares must be greater than 0", "odd"]


# ---- the saved portfolios view
def open_saved(api=None, **session):
    at = app(api or FakeApi())
    for key, value in session.items():
        at.session_state[key] = value
    at.radio(key="portfolio_mode").set_value("My saved portfolios").run()
    return at








def test_loading_by_id_asks_only_for_that_portfolio_and_a_bad_id_asks_nothing():
    api = FakeApi()
    at = open_saved(api)
    at.text_input(key="pf_load_id").set_value("aaaaaaaa-1111-4222-8333-444444444444").run()
    assert ("summary", "aaaaaaaa-1111-4222-8333-444444444444") in api.sent
    api2 = FakeApi()
    at = open_saved(api2)
    at.text_input(key="pf_load_id").set_value("../auth/me").run()
    assert any("doesn't look like a portfolio id" in w.value for w in at.warning) and not [s for s in api2.sent if isinstance(s, tuple) and s[0] == "summary" and "auth" in s[1]]


def test_an_id_that_is_not_in_my_account_says_so_plainly():
    api = FakeApi()
    api.portfolio_summary = lambda pid: (_ for _ in ()).throw(ApiError("No saved portfolio has that id.", 404, "portfolio_not_found", "r404"))
    at = open_saved(api)
    assert any("No portfolio with that id was found in your account" in e.value for e in at.error)


def test_with_nothing_saved_it_points_to_entering_holdings():
    api = FakeApi()
    api.list_portfolios = lambda: []
    assert any("haven't saved a portfolio yet" in i.value for i in open_saved(api).info)


def test_the_summary_is_fetched_once_however_often_the_page_reruns():
    api = FakeApi()
    at = open_saved(api)
    at.run(); at.run()
    assert len([s for s in api.sent if isinstance(s, tuple) and s[0] == "summary"]) == 2   # once per saved portfolio (the dashboard loads them all), never again on a rerun


def test_one_users_cached_summary_is_never_served_to_another_user():
    api = FakeApi()
    at = open_saved(api)
    at.session_state["auth"] = {"token": "other", "email": "someone-else@example.com"}   # same server cache, different person
    at.run()
    assert len([s for s in api.sent if isinstance(s, tuple) and s[0] == "summary"]) == 4   # 2 portfolios x 2 people


def test_use_in_chat_sends_the_holdings_without_cash():
    at = open_saved()
    at.button(key="pf_use_in_chat").click()
    at.run()
    assert at.session_state["portfolio_csv"] == "ticker,shares,avg_cost\nVTI,10,130.0"


def test_explain_this_portfolio_runs_the_analysis_only_when_asked():
    api = FakeApi()
    at = open_saved(api)
    assert not [s for s in api.sent if isinstance(s, tuple) and s[0] == "analyze_saved"]
    at.button(key="pf_explain").click()
    at.run()
    assert ("analyze_saved", ID) in api.sent and any("mostly stocks" in m.value for m in at.markdown)






# ---- the dashboard: ONE editable stocks table and ONE editable real estate table for every portfolio, buttons under them
from src.web_app.tabs import portfolio as pf   # noqa: E402

OTHER = "aaaaaaaa-1111-4222-8333-444444444444"
SAVED = [{"id": ID, "name": "Retirement"}, {"id": OTHER, "name": "Kids college"}]


def dashboard_frames(api=None):
    api = api or FakeApi()
    api = with_real_estate(api)
    stored = {p["id"]: api.get_portfolio(p["id"]) for p in SAVED}
    summaries = {p["id"]: api.portfolio_summary(p["id"]) for p in SAVED}
    labels = pf.portfolio_labels(SAVED)
    return pf.stocks_frame(stored, summaries, labels), pf.estate_frame(stored, summaries, labels), labels


def test_every_holding_of_every_portfolio_is_one_row_of_one_stocks_table_and_every_property_one_row_of_one_real_estate_table():
    stocks, estate, _ = dashboard_frames()
    assert stocks["Portfolio"].tolist() == ["Retirement", "Retirement", "Kids college", "Kids college"] and stocks["Symbol"].tolist() == ["VTI", "CASH", "VTI", "CASH"]
    assert stocks.iloc[0]["Value"] == 2000.0 and stocks.iloc[0]["Shares"] == 10.0 and str(stocks.iloc[0]["Purchase date"].date()) == "2026-01-05"
    assert estate["Portfolio"].tolist() == ["Retirement", "Kids college"] and estate["Property (name or address)"].tolist() == ["Maple St", "Maple St"]
    row = estate.iloc[0].to_dict()
    assert (row["Equity"], row["Net yield"], row["Monthly mortgage payment ($)"], row["Mortgage rate (%)"], row["Monthly rent ($)"]) == (200000.0, 5.1, 1500.0, 6.5, 2000.0)


def test_two_portfolios_with_the_same_name_are_told_apart_in_the_portfolio_column():
    labels = pf.portfolio_labels([{"id": ID, "name": "IRA"}, {"id": OTHER, "name": "IRA"}, {"id": "bbbbbbbb-1111-4222-8333-444444444444", "name": "Taxable"}])
    assert list(labels.values()) == ["IRA (3f2b8c1e)", "IRA (aaaaaaaa)", "Taxable"]


def test_the_tables_are_split_back_into_their_portfolios_and_a_row_with_no_portfolio_is_a_problem():
    stocks, estate, labels = dashboard_frames()
    ids = {label: pid for pid, label in labels.items()}
    per, problems = pf.rows_by_portfolio(stocks, estate, ids)
    assert not problems and [h["ticker"] for h in per[ID]["holdings"]] == ["VTI", "CASH"] and [p["name"] for p in per[OTHER]["properties"]] == ["Maple St"]
    stocks.loc[1, "Portfolio"] = None
    _, problems = pf.rows_by_portfolio(stocks, estate, ids)
    assert problems == ["Row 2: choose which portfolio it belongs to."]


def dashboard(api=None):
    api = with_real_estate(api or FakeApi())
    return api, open_saved(api)


def open_editing(which, api=None):
    """Press the Edit button under the table, as a person does; the table then accepts changes and Save appears."""
    api, at = dashboard(api)
    at.button(key=f"pf_edit_{which}").click()
    at.run()
    return api, at


def press_save(at, which="stocks"):
    at.button(key=f"pf_save_{which}").click()
    return at.run()


def updates(api):
    return [x for x in api.sent if isinstance(x, tuple) and x[0] == "update"]


def test_the_dashboard_shows_totals_for_everything_with_the_buttons_under_the_tables():
    api, at = dashboard()
    assert not at.exception and {m.label: m.value for m in at.metric if m.label in ("Portfolios", "Total value", "Mortgage owed", "Net worth")}["Portfolios"] == "2"
    assert [m.label for m in at.metric].count("Mortgage owed") == 1
    keys = [b.key for b in at.button]
    assert keys.index("pf_edit_stocks") < keys.index("pf_edit_estate") < keys.index("pf_explain")    # an Edit button under each table, then the per-portfolio actions
    assert not [k for k in keys if k and (k.startswith("pf_save_") or k.startswith("pf_cancel_"))]   # no Save or Cancel until Edit is pressed
    assert not [e for e in at.expander if "ETFs" in e.label]                                          # the one-portfolio tables are gone


def test_pressing_edit_opens_that_table_for_changes_and_only_then_shows_save_and_cancel():
    api, at = dashboard()
    at.button(key="pf_edit_stocks").click()
    at.run()
    keys = [b.key for b in at.button]
    assert not at.exception and at.session_state["pf_editing_table"] == "stocks" and "pf_save_stocks" in keys and "pf_cancel_stocks" in keys
    assert "pf_edit_stocks" not in keys and "pf_edit_estate" in keys and "pf_save_estate" not in keys    # the other table stays view-only


def test_changing_a_cell_saves_only_the_portfolio_it_belongs_to():
    api, at = open_editing("stocks")
    at.session_state["dash_stocks_edit_0_0"] = {"edited_rows": {0: {"Shares": 20.0}}, "added_rows": [], "deleted_rows": []}   # row 0 = VTI in Retirement
    press_save(at)
    (_, pid, name, holdings, properties), = updates(api)
    assert pid == ID and name == "Retirement" and holdings[0]["ticker"] == "VTI" and holdings[0]["shares"] == "20.0"
    assert properties[0]["name"] == "Maple St" and properties[0]["monthly_mortgage_payment"] == "1500.0"      # the portfolio's other table travels along unchanged
    assert not at.exception and any("Retirement" in x.value for x in at.success)


def test_changing_a_property_on_the_real_estate_table_saves_it():
    api, at = open_editing("estate")
    at.session_state["dash_estate_edit_0_0"] = {"edited_rows": {1: {"Monthly rent ($)": 2500.0, "Current value ($)": 360000.0}}, "added_rows": [], "deleted_rows": []}   # row 1 = Kids college
    press_save(at, "estate")
    (_, pid, _, _, properties), = updates(api)
    assert pid == OTHER and properties[0]["monthly_rent"] == "2500.0" and properties[0]["current_value"] == "360000.0"


def test_a_row_added_with_a_portfolio_chosen_is_saved_into_that_portfolio():
    api, at = open_editing("stocks")
    at.session_state["dash_stocks_edit_0_0"] = {"edited_rows": {}, "added_rows": [{"Portfolio": "Kids college", "Symbol": "bnd", "Type": "Bond", "Shares": 5.0, "Avg cost per share ($)": 74.5}], "deleted_rows": []}
    press_save(at)
    (_, pid, _, holdings, _), = updates(api)
    assert pid == OTHER and [h["ticker"] for h in holdings] == ["VTI", "CASH", "bnd"]


def test_deleting_a_row_removes_it_from_its_portfolio():
    api, at = open_editing("stocks")
    at.session_state["dash_stocks_edit_0_0"] = {"edited_rows": {}, "added_rows": [], "deleted_rows": [1]}   # CASH in Retirement
    press_save(at)
    (_, pid, _, holdings, _), = updates(api)
    assert pid == ID and [h["ticker"] for h in holdings] == ["VTI"]


def test_saving_without_changes_sends_nothing():
    api, at = open_editing("stocks")
    press_save(at)
    assert not updates(api) and any("Nothing to save" in i.value for i in at.info)


def test_emptying_a_portfolio_is_refused_and_nothing_is_sent_for_any_portfolio():
    api = FakeApi()
    stored = api.get_portfolio
    api.get_portfolio = lambda pid: {**stored(pid), "properties": []}      # portfolios that hold only stocks
    api, at = open_editing("stocks", api)
    at.session_state["dash_stocks_edit_0_0"] = {"edited_rows": {0: {"Shares": 99.0}}, "added_rows": [], "deleted_rows": [2, 3]}   # all of Kids college's stocks
    press_save(at)
    assert not updates(api) and any("Kids college" in w.value and "nothing left" in w.value for w in at.warning)


def test_a_row_with_no_portfolio_is_named_and_nothing_is_sent():
    api, at = open_editing("stocks")
    at.session_state["dash_stocks_edit_0_0"] = {"edited_rows": {}, "added_rows": [{"Symbol": "VXUS", "Type": "ETF", "Shares": 3.0, "Avg cost per share ($)": 60.0}], "deleted_rows": []}
    press_save(at)
    assert not updates(api) and any("Row 5: choose which portfolio" in w.value for w in at.warning)


def test_a_bad_value_is_named_by_its_row_and_nothing_is_sent():
    api, at = open_editing("stocks")
    at.session_state["dash_stocks_edit_0_0"] = {"edited_rows": {2: {"Shares": -1.0}}, "added_rows": [], "deleted_rows": []}
    press_save(at)
    assert not updates(api) and any("Row 3: shares must be a number greater than 0" in w.value for w in at.warning)


def test_cancel_throws_the_edits_away_and_closes_the_table():
    api, at = open_editing("stocks")
    at.session_state["dash_stocks_edit_0_0"] = {"edited_rows": {0: {"Shares": 20.0}}, "added_rows": [], "deleted_rows": []}
    at.button(key="pf_cancel_stocks").click()
    at.run()
    assert not updates(api) and at.session_state["pf_edit_version"] == 1 and at.session_state["pf_editing_table"] is None and "pf_edit_stocks" in [b.key for b in at.button]


def test_after_saving_the_table_is_view_only_again():
    api, at = open_editing("stocks")
    at.session_state["dash_stocks_edit_0_0"] = {"edited_rows": {0: {"Shares": 20.0}}, "added_rows": [], "deleted_rows": []}
    press_save(at)
    keys = [b.key for b in at.button]
    assert at.session_state["pf_editing_table"] is None and "pf_edit_stocks" in keys and "pf_save_stocks" not in keys


def test_a_server_refusal_names_the_portfolio_and_the_visible_row():
    api, at = open_editing("stocks")

    def refuse(portfolio_id, name, holdings, properties=None):
        raise ApiError("bad", 422, "invalid_portfolio", details=[{"field": "holdings.0.shares", "problem": "Value error, shares must be greater than 0"}])
    api.update_portfolio = refuse
    at.session_state["dash_stocks_edit_0_0"] = {"edited_rows": {2: {"Shares": 7.0}}, "added_rows": [], "deleted_rows": []}   # Kids college's VTI is visible row 3
    press_save(at)
    assert any("Kids college" in e.value for e in at.error) and any("Row 3, shares" in c.value for c in at.caption)


def test_the_mortgage_payoff_calculator_lists_the_mortgages_of_every_portfolio_and_works_out_each():
    api, at = dashboard()
    at.button(key="pf_payoff_open").click()
    at.run()
    assert not at.exception and any("2 mortgage(s) across your saved portfolios" in c.value for c in at.caption)
    assert len([n for n in at.number_input if n.label == "Monthly payment ($)"]) == 2                           # one row per mortgage, from both portfolios
    at.button(key="pf_payoff_go").click()
    at.run()
    assert {x[1] for x in api.sent if isinstance(x, tuple) and x[0] == "payoff"} == {ID, OTHER}
    assert any("Maple St (Retirement)" in m.value for m in at.markdown) and any("Maple St (Kids college)" in m.value for m in at.markdown)


def test_deleting_asks_the_server_and_then_shows_the_next_portfolio_not_the_deleted_one():
    api = FakeApi()
    api.deleted = set()
    at = open_saved(api)
    at.button(key="pf_delete").click()
    at.run()
    other = "aaaaaaaa-1111-4222-8333-444444444444"
    assert ("delete", ID) in api.sent and at.session_state["selected_portfolio_id"] == other and ("summary", other) in api.sent
    assert ID not in [o for o in at.selectbox(key="pf_pick").options]



def test_a_row_added_with_the_plus_button_and_left_untouched_is_ignored_not_saved_as_nan():
    blank_added = pd.DataFrame({SYMBOL: ["VTI", None], TYPE: ["ETF", None], SHARES: [1.0, float("nan")], COST: [1.0, float("nan")],
                                BOUGHT: pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]")})
    holdings, rows, problems = rows_to_holdings(blank_added)
    assert problems == [] and [h["ticker"] for h in holdings] == ["VTI"] and rows == [1]


# ---------- real estate in the Portfolio tab ----------
from src.web_app.tabs.portfolio import rows_to_properties  # noqa: E402

PNAME, PKIND, PPRICE, PDATE, PVALUE, PMORT, PRENT, PCOSTS = ("Property (name or address)", "Kind", "Purchase price ($)", "Purchase date", "Current value ($)",
                                                              "Mortgage owed ($)", "Monthly rent ($)", "Monthly expenses ($)")
MAPLE_ROW = {PNAME: "Maple St", PKIND: "House", PPRICE: 300000.0, PDATE: "2019-06-01", PVALUE: 400000.0, PMORT: 200000.0, PRENT: 2500.0, PCOSTS: 800.0}


def fill_properties(at, rows, version=0):
    """The real estate table starts empty, so every row is an 'added row' in the editor's delta state."""
    at.session_state[f"pf_props_{version}"] = {"edited_rows": {}, "added_rows": rows, "deleted_rows": []}
    return at


def test_the_real_estate_table_is_there_but_optional():
    at = app()
    assert any("Real estate (optional)" in m.value for m in at.markdown) and not at.exception


def test_a_property_is_sent_with_exact_numbers_and_sensible_defaults():
    api = FakeApi()
    at = save(fill_properties(fill_table(app(api), [VTI]), [MAPLE_ROW, {PNAME: "Lot 7", PPRICE: 50000.0, PVALUE: 65000.0}]))
    assert api.saved_properties == [
        {"name": "Maple St", "property_type": "house", "purchase_price": "300000.0", "current_value": "400000.0", "mortgage_balance": "200000.0", "monthly_mortgage_payment": "0.0",
         "mortgage_rate_pct": "0.0", "monthly_rent": "2500.0", "monthly_expenses": "800.0", "purchase_date": "2019-06-01"},
        {"name": "Lot 7", "property_type": "house", "purchase_price": "50000.0", "current_value": "65000.0", "mortgage_balance": "0.0", "monthly_mortgage_payment": "0.0",
             "mortgage_rate_pct": "0.0", "monthly_rent": "0.0", "monthly_expenses": "0.0"}]
    assert not at.exception and at.radio(key="portfolio_mode").value == "My saved portfolios"


def test_a_portfolio_of_only_a_property_can_be_saved_and_the_blank_holdings_row_is_ignored():
    api = FakeApi()
    save(fill_properties(fill_table(app(api), [{}]), [MAPLE_ROW]))
    (_, name, holdings), = [s for s in api.sent if isinstance(s, tuple) and s[0] == "save"]
    assert holdings == [] and len(api.saved_properties) == 1


def test_a_property_missing_its_price_or_value_is_named_by_row_and_nothing_is_sent():
    api = FakeApi()
    at = save(fill_properties(fill_table(app(api), [VTI]), [{PNAME: "Maple St", PRENT: 100.0}, {PNAME: "", PPRICE: 5.0, PVALUE: 6.0, PMORT: -1.0}]))
    warnings = " | ".join(w.value for w in at.warning)
    assert "Property row 1: enter what you paid" in warnings and "Property row 1: enter what you think it is worth today" in warnings
    assert "Property row 2: enter a name or address" in warnings and "Property row 2: mortgage cannot be negative" in warnings
    assert not [s for s in api.sent if isinstance(s, tuple) and s[0] == "save"]


def test_nothing_at_all_still_asks_for_something_to_save():
    api = FakeApi()
    at = save(fill_table(app(api), [{}]))
    assert any("Add at least one holding or property" in w.value for w in at.warning)


def test_server_complaints_about_a_property_name_the_property_row_the_person_sees():
    api = FakeApi()
    api.save_portfolio = lambda name, holdings, properties=None: (_ for _ in ()).throw(ApiError("The request was not valid.", 422, "invalid_request", "r", [
        {"field": "properties.1.monthly_rent", "problem": "Input should be greater than or equal to 0"}]))
    at = save(fill_properties(fill_table(app(api), [VTI]), [MAPLE_ROW, {}, {**MAPLE_ROW, PNAME: "Oak"}]))   # the blank middle row is skipped: Oak is the 2nd sent, the 3rd shown
    assert "Property row 3, monthly rent: Input should be greater than or equal to 0" in " | ".join(c.value for c in at.caption)


def test_rows_to_properties_pure():
    table = pd.DataFrame({PNAME: ["Maple St", None], PKIND: ["Apartment / condo", None], PPRICE: [300000.0, float("nan")], PDATE: pd.Series([pd.Timestamp("2019-06-01"), pd.NaT], dtype="datetime64[ns]"),
                          PVALUE: [400000.5, float("nan")], PMORT: [float("nan"), float("nan")], PRENT: [2500.0, float("nan")], PCOSTS: [float("nan"), float("nan")]})
    props, rows, problems = rows_to_properties(table)
    assert problems == [] and rows == [1] and props == [{"name": "Maple St", "property_type": "condo", "purchase_price": "300000.0", "current_value": "400000.5",
                                                          "mortgage_balance": "0.0", "monthly_mortgage_payment": "0.0", "mortgage_rate_pct": "0.0", "monthly_rent": "2500.0", "monthly_expenses": "0.0", "purchase_date": "2019-06-01"}]


# ---- the grouped view with real estate
RE_FIGURES = {"purchase_price": 450000.0, "value": 540000.0, "mortgage": 290000.0, "equity": 250000.0, "gain": 90000.0, "gain_pct": 20.0, "annual_rent": 42000.0, "annual_expenses": 15000.0,
              "net_operating_income": 27000.0, "gross_yield_pct": 7.78, "net_yield_pct": 5.0, "loan_to_value_pct": 53.7}


def with_real_estate(api):
    base = api.portfolio_summary

    def summary(pid):
        s = base(pid)
        props = [{"name": "Maple St", "property_type": "house", "purchase_date": None, **{**RE_FIGURES, "purchase_price": 300000.0, "value": 400000.0, "mortgage": 200000.0, "equity": 200000.0,
                                                                                         "gain": 100000.0, "gain_pct": 33.33, "annual_rent": 30000.0, "annual_expenses": 9600.0,
                                                                                         "net_operating_income": 20400.0, "gross_yield_pct": 7.5, "net_yield_pct": 5.1, "loan_to_value_pct": 50.0}}]
        group = {"asset_type": "real_estate", "label": "Real estate", "count": 1, "unpriced_count": 0, "cost": 300000.0, "value": 400000.0, "gain_pct": 33.33, "weight_pct": 99.4, "holdings": []}
        return {**s, "groups": [group] + s["groups"], "property_count": 1, "holding_count": s["holding_count"] + 1, "mortgage_owed": 200000.0, "net_value": 202500.0,
                "total_value": 402500.0, "total_cost": 301800.0, "real_estate": {"count": 1, **{**RE_FIGURES, "purchase_price": 300000.0, "value": 400000.0, "mortgage": 200000.0, "equity": 200000.0,
                "gain": 100000.0, "gain_pct": 33.33, "annual_rent": 30000.0, "annual_expenses": 9600.0, "net_operating_income": 20400.0, "gross_yield_pct": 7.5, "net_yield_pct": 5.1,
                "loan_to_value_pct": 50.0}, "properties": props}}
    api.portfolio_summary = summary
    return api






def test_a_portfolio_without_real_estate_shows_no_mortgage_row():
    at = open_saved(FakeApi())
    assert [m.label for m in at.metric].count("Mortgage owed") == 1 and not [d for d in at.dataframe if "Gross yield" in d.value.columns]   # only the dashboard's total, no per-portfolio row or table


def test_explain_is_switched_off_for_a_portfolio_that_holds_only_real_estate():
    api = FakeApi()
    only_property = with_real_estate(FakeApi()).portfolio_summary(FakeApi.saved_id)
    only_property = {**only_property, "groups": only_property["groups"][:1], "holding_count": 1}          # one group (the property) and nothing else
    api.portfolio_summary = lambda pid: only_property
    assert open_saved(api).button(key="pf_explain").disabled
    assert not open_saved(FakeApi()).button(key="pf_explain").disabled                                    # with stocks or funds in it, it is available


def test_saved_portfolios_in_the_picker_mention_their_properties():
    api = FakeApi()
    base = api._all_portfolios
    api._all_portfolios = lambda: [{**p, "property_count": 2} for p in base()]
    assert at_picker_labels(open_saved(api))[0] == "Retirement — 3 holding(s) + 2 properties — 3f2b8c1e…"
    api._all_portfolios = lambda: [{**p, "property_count": 1} for p in base()]
    assert "+ 1 property —" in at_picker_labels(open_saved(api))[0]                                       # singular


def at_picker_labels(at):
    return list(at.selectbox(key="pf_pick").options)


# ---------- chat box under the portfolio ----------
def test_the_saved_portfolio_has_its_own_chat_that_sends_the_holdings_and_keeps_its_own_session():
    api = FakeApi()
    at = open_saved(api)
    box = next(c for c in at.chat_input if c.key == "pf_chat_input")
    box.set_value("What if the market is up tomorrow?").run()
    next(c for c in at.chat_input if c.key == "pf_chat_input").set_value("And if I buy NVDA?").run()
    assert [c[0] for c in api.chats] == ["What if the market is up tomorrow?", "And if I buy NVDA?"]
    assert api.chats[0][1] is None and api.chats[1][1] == "sess-1"                      # second question continues the first one's conversation
    assert "ticker,shares,avg_cost" in api.chats[0][2] and "VTI" in api.chats[0][2]     # the portfolio travels with the question; cash is left out
    assert "CASH" not in api.chats[0][2] and not at.session_state["messages"]           # the Chat tab's own conversation is untouched
    assert all(pid for pid in api.chat_portfolio_ids)                                   # the saved portfolio's id travels too, so mortgage questions can use it


def test_the_analyze_view_has_a_chat_box_too():
    api = FakeApi()
    at = analyze(app(api))
    next(c for c in at.chat_input if c.key == "pf_chat_input").set_value("How risky is this?").run()
    assert len(api.chats) == 1 and api.chats[0][2] == at.session_state["portfolio_csv"]
