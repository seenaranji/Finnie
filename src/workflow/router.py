"""The router: a cheap model that reads a message and picks ONE route. It is a classifier, not an agent: it answers nothing.

Three choices that make it dependable:
  - a small, cheap model (router_model) at temperature 0, so the same message gets the same route and the cost stays tiny
  - STRUCTURED OUTPUT: a strict JSON schema whose only field is an enum of the allowed routes, so the model cannot invent a route
  - a safe default: if the reply is somehow unusable, fall back to `finance_qa` (its retrieval gate refuses what it cannot ground), and say so
"""
import json
from dataclasses import asdict, dataclass

from src.core.llm import LLMClient
from src.core.logging import get_logger

log = get_logger(__name__)
ROUTES = ("finance_qa", "market")
DEFAULT_ROUTE = "finance_qa"

SCHEMA = {
    "type": "object",
    "properties": {"route": {"type": "string", "enum": list(ROUTES)}},
    "required": ["route"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "route", "strict": True, "schema": SCHEMA}}

SYSTEM_PROMPT = """You route a user's message to exactly one agent. Reply with the route only.

Routes:
- finance_qa: the user wants a concept, definition, comparison or rule EXPLAINED (what is an ETF, how does a Roth IRA work, how are capital gains taxed, what does an expense ratio mean). It is answered from a knowledge base.
- market: the user wants current or recent MARKET DATA: a price or quote, how a named stock, fund or index is doing, a change today, a trend, a moving average, volatility, a 52-week range, or a comparison of tickers' performance.

Rules:
- A question that names a ticker or company AND asks how it is doing or what it costs is market.
- A question that asks what something IS or MEANS is finance_qa, even if it uses a ticker as an example.

Examples:
"What is an ETF?" -> finance_qa
"How is VOO doing today?" -> market
"What does the expense ratio on VOO mean?" -> finance_qa
"Is Apple above its 50-day moving average?" -> market
"How are short-term capital gains taxed?" -> finance_qa
"Compare Microsoft and Apple this week" -> market"""


@dataclass
class RouteDecision:
    route: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    fell_back: bool = False   # True = the reply was unusable and the default route was used

    def as_dict(self) -> dict:
        return asdict(self)


class Router:
    def __init__(self, llm: LLMClient, model_key: str = "router_model") -> None:
        self.llm, self.model_key = llm, model_key

    def classify(self, message: str) -> RouteDecision:
        result = self.llm.chat(f"MESSAGE: {message}", system=SYSTEM_PROMPT, model_key=self.model_key, temperature=0,
                               max_output_tokens=20, response_format=RESPONSE_FORMAT)
        try:
            route, fell_back = json.loads(result.text)["route"], False
            if route not in ROUTES:
                raise ValueError(route)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            log.warning("unusable router reply %r; falling back to %s", result.text[:80], DEFAULT_ROUTE)
            route, fell_back = DEFAULT_ROUTE, True
        return RouteDecision(route, result.model, result.input_tokens, result.output_tokens, result.cost_usd, result.latency_s, fell_back)
