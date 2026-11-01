"""Small building blocks for the graph that are not agents: the "it/that" resolver, the news-topic extractor, and fallback answers."""
import json
import re

from src.core.guardrails import DISCLAIMER_STANDARD
from src.core.llm import LLMClient, LLMResult
from src.core.logging import get_logger
from src.core.schemas import AgentResponse

log = get_logger(__name__)

REWRITE_SCHEMA = {"type": "object", "properties": {"standalone": {"type": "string"}}, "required": ["standalone"], "additionalProperties": False}
REWRITE_FORMAT = {"type": "json_schema", "json_schema": {"name": "standalone", "strict": True, "schema": REWRITE_SCHEMA}}
REWRITE_PROMPTS = {"v1": """Rewrite the user's LATEST message as a standalone message that can be understood without the conversation.
Replace pronouns and omitted subjects (it, that, they, the other one, "and ...") with what they refer to, using the conversation.
If the message is already standalone, or nothing in the conversation says what it refers to, return it unchanged.
Do not answer the message. Do not add facts or details that are not in the conversation. Keep it short.""",
"v2": """Rewrite the user's LATEST message as a complete, standalone message that can be understood without the conversation.

When the latest message CONTINUES the conversation, say in full what it continues:
- "What about X?" or "And X?" repeats the previous question's task with the new subject.
- A short answer or detail (a number, an amount, a risk level, "aggressive") states what it is answering, together with what was said earlier.
- A correction ("I meant X") repeats the previous question with X in place of what it replaces.
- A pronoun or omitted subject (it, they, that, "its volatility") is replaced by what it refers to.

Leave the message UNCHANGED when it is already a complete standalone question, when nothing in the conversation says what it refers to, or when it is only a greeting, thanks or other pleasantry.
Do not answer the message. Do not add facts or details that are not in the conversation. Keep it short.

Examples:
Conversation: "How is Shell doing?" ... Latest: "What about BP?" -> "How is BP doing?"
Conversation: "I want to save $30,000 in 4 years" ... Latest: "I have $1,000 saved" -> "I want to save $30,000 in 4 years and I have $1,000 saved."
Conversation: "What are the headlines on Ford?" ... Latest: "I meant GM" -> "What are the headlines on GM?"
Conversation: "What is a mutual fund?" ... Latest: "And a REIT?" -> "What is a REIT?"
Conversation: "How is Shell doing?" ... Latest: "Great, thanks" -> (unchanged)"""}

TOPIC_SCHEMA = {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"], "additionalProperties": False}
TOPIC_FORMAT = {"type": "json_schema", "json_schema": {"name": "news_topic", "strict": True, "schema": TOPIC_SCHEMA}}
TOPIC_PROMPT = """Extract the news topic from the message. If it names a company or a ticker, reply with the stock ticker symbol in UPPERCASE (Apple -> AAPL, Tesla -> TSLA).
If it is about the markets, the economy, the Fed or finance in general, reply with MARKETS."""
_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


def history_text(history: list[dict], turns: int) -> str:
    """The last `turns` exchanges as plain text. Assistant answers are cut short: only what they were ABOUT matters for resolving 'it'."""
    lines = [f"{'USER' if h['role'] == 'user' else 'ASSISTANT'}: {h['content'][:200]}" for h in history[-2 * turns:]]
    return "\n".join(lines)


def rewrite_message(llm: LLMClient, message: str, history: list[dict], turns: int, version: str = "v1") -> tuple[str, LLMResult | None]:
    """Returns (standalone message, the model call or None). Any problem returns the original message: resolving is a convenience, never a gate."""
    prompt = f"CONVERSATION SO FAR:\n{history_text(history, turns)}\n\nLATEST MESSAGE: {message}"
    try:
        result = llm.chat(prompt, system=REWRITE_PROMPTS[version], model_key="router_model", temperature=0, max_output_tokens=120, response_format=REWRITE_FORMAT)
        standalone = json.loads(result.text)["standalone"].strip()
    except Exception as error:   # model error, bad JSON, missing key
        log.warning("could not resolve the message (%s); using it as written", type(error).__name__)
        return message, None
    if not standalone or len(standalone) > 3 * len(message) + 200:   # empty, or the model wrote an essay instead of a rewrite
        return message, result
    return standalone, result


def extract_news_topic(llm: LLMClient, message: str) -> tuple[str, LLMResult | None]:
    """A ticker (uppercase) or 'markets'. Anything unusable becomes 'markets', which is always a safe thing to summarize."""
    try:
        result = llm.chat(f"MESSAGE: {message}", system=TOPIC_PROMPT, model_key="router_model", temperature=0, max_output_tokens=12, response_format=TOPIC_FORMAT)
        topic = json.loads(result.text)["topic"].strip().upper()
    except Exception:
        return "markets", None
    return ("markets", result) if topic == "MARKETS" or not _TICKER.match(topic) else (topic, result)


def user_safe_text(error: Exception, fallback_reply: str) -> str:
    """Our own errors carry a message meant for users; anything else gets the generic fixed reply (never a stack trace or a raw exception text)."""
    message = getattr(error, "user_message", None)
    return f"{message} Please try again in a few minutes." if message else fallback_reply


def fixed_response(text: str, agent_name: str, model_note: str, disclaimer: str = DISCLAIMER_STANDARD, **metadata) -> AgentResponse:
    return AgentResponse(text=text, agent_name=agent_name, disclaimer=disclaimer,
                         metadata={"model": f"none ({model_note})", "cost_usd": 0.0, "latency_s": 0.0, "input_tokens": 0, "output_tokens": 0, **metadata})
