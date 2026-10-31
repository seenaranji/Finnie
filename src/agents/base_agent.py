"""Base class every Finnie agent inherits.

handle() is the fixed pipeline all agents share (the "franchise rules"):
    1. check the request   2. build the system prompt   3. call the model
    4. check the output    5. attach the disclaimer
Subclasses only fill in what is different (their prompt and how they get an answer).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.core.guardrails import (
    ADVICE_REDIRECT,
    DISCLAIMER_STANDARD,
    DISCLAIMER_STRONG,
    find_advice_phrases,
    is_advice_request,
)
from src.core.llm import LLMClient, LLMResult
from src.core.logging import get_logger
from src.core.schemas import AgentRequest, AgentResponse, Source

log = get_logger(__name__)

# Added to the system prompt ONLY when the user is asking for a personal recommendation.
ADVICE_INSTRUCTION = (
    "The user is asking for a personal recommendation. Do NOT tell them what to buy, "
    "sell, or hold, and do not say what is best for them. Instead explain how people "
    "think about this kind of decision, the main tradeoffs and risks, and what "
    "information would matter. Suggest talking to a licensed professional."
)


@dataclass
class AgentAnswer(LLMResult):
    """An LLMResult plus what a RAG or tool agent also knows: the sources it used and extra details.
    answer() may return a plain LLMResult (as before) or this."""
    sources: list[Source] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class BaseAgent(ABC):
    name: str = "base"
    system_prompt: str = ""
    default_disclaimer: str = DISCLAIMER_STANDARD

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @abstractmethod
    def answer(self, request: AgentRequest, system: str) -> LLMResult:
        """Produce the raw model answer. Each agent decides how (plain LLM, RAG, tools...)."""

    def handle(self, request: AgentRequest) -> AgentResponse:
        advice_requested = is_advice_request(request.message)         # 1. check request
        system = self.system_prompt                                    # 2. build prompt
        if advice_requested:
            system = f"{system}\n\n{ADVICE_INSTRUCTION}".strip()

        result = self.answer(request, system)                          # 3. call model

        leaks = find_advice_phrases(result.text)                       # 4. check output
        text = ADVICE_REDIRECT if leaks else result.text

        stronger = advice_requested or bool(leaks)                     # 5. disclaimer
        log.info("agent=%s advice_requested=%s output_flagged=%s",
                 self.name, advice_requested, bool(leaks))
        return AgentResponse(
            text=text,
            agent_name=self.name,
            sources=getattr(result, "sources", []),
            disclaimer=DISCLAIMER_STRONG if stronger else self.default_disclaimer,
            metadata={
                "advice_requested": advice_requested,
                "output_flagged": bool(leaks),
                "flagged_phrases": leaks,
                "original_text": result.text if leaks else None,
                "model": result.model,
                "latency_s": result.latency_s,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                **getattr(result, "extra", {}),
            },
        )
