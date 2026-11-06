"""Tab 1: Chat. Streamlit concepts: st.session_state (history survives reruns), st.chat_input / st.chat_message, spinner, badges."""
import streamlit as st

from src.web_app.client import ApiError, FinnieClient
from src.web_app.ui_common import md, show_error

LABELS = {"finance_qa": ("Finance Q&A", "blue"), "market_analysis": ("Market data", "green"), "portfolio_analysis": ("Portfolio", "violet"),
          "goal_planning": ("Goal planning", "orange"), "news_synthesizer": ("News", "red"), "tax_education": ("Tax education", "yellow"),
          "multi_agent": ("Several agents", "primary"), "scope_reply": ("Out of scope", "gray"), "clarify": ("Needs more detail", "gray"), "fallback": ("Error reply", "gray")}
# (short label on the chip, the full question it sends)
STARTERS = [("What is an ETF?", "What is an ETF?"), ("How is Microsoft doing today?", "How is Microsoft doing today?"),
            ("Save $20,000 in 4 years", "I want to save $20,000 in 4 years. I have $2,000 and a medium risk level."), ("What is a wash sale?", "What is a wash sale?")]


def new_conversation() -> None:
    st.session_state.messages, st.session_state.session_id = [], None


AVATARS = {"user": ":material/person:", "assistant": ":material/insights:"}


def render_message(message: dict) -> None:
    with st.chat_message(message["role"], avatar=AVATARS.get(message["role"])):
        if message.get("error"):
            show_error(message["error"])
            return
        st.markdown(md(message["content"]))   # md: a pair of $ amounts in an answer must not turn into a formula
        if message["role"] == "assistant" and message.get("meta"):
            meta = message["meta"]
            label, color = LABELS.get(meta["answered_by"], (meta["answered_by"], "gray"))
            st.badge(f"Answered by: {label}", color=color)
            if meta["sources"]:
                with st.expander(f"Sources ({len(meta['sources'])})"):
                    for source in meta["sources"]:
                        st.markdown(f"- [{md(source['title'])}]({source['url']})" if source.get("url") else f"- {md(source['title'])}")
            st.caption(md(f"{meta['disclaimer']}  ·  {meta['latency_s']:.1f}s  ·  ${meta['cost_usd']:.4f}"))


def saved_portfolio_kwargs(portfolio_id: str | None) -> dict:
    """Send the saved portfolio's id only when one was picked (so mortgage questions can use it)."""
    return {"portfolio_id": portfolio_id} if portfolio_id else {}


def toolbar() -> None:
    """A thin row above the conversation: what the chat is using, and a way to start over."""
    left, right = st.columns([4, 1.4], vertical_alignment="center")
    with left:
        if st.session_state.get("portfolio_id"):
            st.caption(":material/check_circle: Answers about 'my portfolio' and 'my mortgage' use the saved portfolio you picked.")
        elif st.session_state.get("portfolio_csv"):
            st.caption(":material/check_circle: Answers about 'my portfolio' use the portfolio you loaded.")
        elif st.session_state.get("session_id"):
            st.caption(f"Conversation `{st.session_state.session_id}`")
    with right:
        if st.session_state.get("messages") and st.button("New conversation", icon=":material/add:", width="stretch", key="new_conversation"):
            new_conversation()
            st.rerun()


def render(client: FinnieClient) -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", None)
    toolbar()
    if not st.session_state.messages:
        st.markdown('<div class="welcome"><h2>What would you like to understand today?</h2>'
                    '<p>Ask about a concept, a stock price, your savings goal, the news or taxes. Or start with one of these:</p></div>', unsafe_allow_html=True)   # fixed text
        columns = st.columns(len(STARTERS))
        for number, (column, (label, question)) in enumerate(zip(columns, STARTERS)):
            if column.button(md(label), width="stretch", key=f"starter_{number}", help=question):
                st.session_state.pending = question
                st.rerun()
    for message in st.session_state.messages:
        render_message(message)

    typed = st.chat_input("Ask Finnie…", max_chars=2000)
    prompt = st.session_state.pop("pending", None) or typed
    if prompt is None:
        return
    if not prompt.strip():
        st.warning("Please type a question first.")
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message(st.session_state.messages[-1])
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Thinking…"):
                reply = client.chat(prompt, st.session_state.session_id, st.session_state.get("portfolio_csv"), **saved_portfolio_kwargs(st.session_state.get("portfolio_id")))
        except ApiError as error:
            st.session_state.messages.append({"role": "assistant", "error": error})
            st.rerun()
    st.session_state.session_id = reply["session_id"]
    st.session_state.messages.append({"role": "assistant", "content": reply["text"], "meta": reply})
    st.rerun()
