"""Tab 1: Chat. Streamlit concepts: st.session_state (history survives reruns), st.chat_input / st.chat_message, spinner, badges."""
import streamlit as st

from src.web_app.client import ApiError, FinnieClient
from src.web_app.ui_common import show_error

LABELS = {"finance_qa": ("Finance Q&A", "blue"), "market_analysis": ("Market data", "green"), "portfolio_analysis": ("Portfolio", "violet"),
          "goal_planning": ("Goal planning", "orange"), "news_synthesizer": ("News", "red"), "tax_education": ("Tax education", "yellow"),
          "multi_agent": ("Several agents", "primary"), "scope_reply": ("Out of scope", "gray"), "clarify": ("Needs more detail", "gray"), "fallback": ("Error reply", "gray")}
STARTERS = ["What is an ETF?", "How is Microsoft doing today?", "I want to save $20,000 in 4 years. I have $2,000 and a medium risk level.", "What is a wash sale?"]


def new_conversation() -> None:
    st.session_state.messages, st.session_state.session_id = [], None


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            show_error(message["error"])
            return
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("meta"):
            meta = message["meta"]
            label, color = LABELS.get(meta["answered_by"], (meta["answered_by"], "gray"))
            st.badge(f"Answered by: {label}", color=color)
            if meta["sources"]:
                with st.expander(f"Sources ({len(meta['sources'])})"):
                    for source in meta["sources"]:
                        st.markdown(f"- [{source['title']}]({source['url']})" if source.get("url") else f"- {source['title']}")
            st.caption(f"{meta['disclaimer']}  ·  {meta['latency_s']:.1f}s  ·  ${meta['cost_usd']:.4f}")


def render(client: FinnieClient) -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", None)
    if not st.session_state.messages:
        st.markdown("Ask about a concept, a stock price, your savings goal, the news or taxes. Try one:")
        columns = st.columns(len(STARTERS))
        for column, text in zip(columns, STARTERS):
            if column.button(text, width="stretch"):
                st.session_state.pending = text
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
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                reply = client.chat(prompt, st.session_state.session_id, st.session_state.get("portfolio_csv"))
        except ApiError as error:
            st.session_state.messages.append({"role": "assistant", "error": error})
            st.rerun()
    st.session_state.session_id = reply["session_id"]
    st.session_state.messages.append({"role": "assistant", "content": reply["text"], "meta": reply})
    st.rerun()
