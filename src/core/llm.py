"""Thin wrapper around the OpenAI SDK.

What it adds on top of a raw SDK call:
  - a timeout (never wait forever)
  - retry with exponential backoff + jitter, for TRANSIENT errors only
  - one log line per call: model, latency, tokens, estimated cost
"""
import json
from dataclasses import dataclass, field

import openai
from openai import OpenAI
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.core.config import Settings, load_settings
from src.core.logging import get_logger
from src.core.metrics import timer

log = get_logger(__name__)

# Errors where trying again a moment later can succeed. Anything else (401 bad key,
# 400 bad request) is permanent: retrying only wastes time, so it fails immediately.
TRANSIENT_ERRORS = (
    openai.RateLimitError,       # 429
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,  # 5xx
)


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float   # total wall time including retries and waits
    cost_usd: float
    attempts: int      # 1 = succeeded first try


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None   # None = the model's arguments were not valid JSON
    raw_arguments: str = ""


@dataclass
class LLMTurn(LLMResult):
    """One model turn in a tool conversation: text (the final answer) or tool_calls (data requests)."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict = field(default_factory=dict)   # append this to the message list before the tool results


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        cfg = self.settings.llm
        # max_retries=0: the SDK retries silently by default. We turn that off so
        # OUR retry logic is the only one, and we can see and count it.
        self._client = OpenAI(
            api_key=self.settings.secrets.require_openai_key(),  # clear error if missing
            timeout=cfg.timeout_seconds,
            max_retries=0,
        )

    def _cost(self, model_key: str, input_tokens: int, output_tokens: int) -> float:
        price = self.settings.llm.prices_per_million_tokens[model_key]
        return (input_tokens * price.input + output_tokens * price.output) / 1_000_000

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        model_key: str = "answer_model",   # "answer_model" or "router_model"
        temperature: float | None = None,  # None = let the model use its default
        max_output_tokens: int | None = None,
        response_format: dict | None = None,   # e.g. {"type": "json_schema", "json_schema": {...}}: the reply must match the schema
    ) -> LLMResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._complete(messages, model_key, temperature, max_output_tokens, response_format=response_format)[1]

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model_key: str = "answer_model",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> "LLMTurn":
        """One model turn over a full message list, optionally offering tools. The model either answers (text) or asks for
        tool calls; the caller runs them, appends the results as 'tool' messages and calls complete() again."""
        response, result = self._complete(messages, model_key, temperature, max_output_tokens, tools)
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = None   # the model sent broken JSON; the caller turns this into a tool error
            calls.append(ToolCall(call.id, call.function.name, arguments, call.function.arguments))
        assistant_message = {"role": "assistant", "content": message.content}
        if calls:
            assistant_message["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments}} for c in calls]
        return LLMTurn(**vars(result), tool_calls=calls, assistant_message=assistant_message)

    def _complete(self, messages, model_key, temperature, max_output_tokens, tools=None, response_format=None):
        cfg = self.settings.llm
        model = getattr(cfg, model_key)

        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_completion_tokens"] = max_output_tokens

        def log_retry(state: RetryCallState) -> None:
            log.warning("transient error (%s); retry %d in %.1fs",
                        type(state.outcome.exception()).__name__,
                        state.attempt_number, state.next_action.sleep)

        retrying = Retrying(
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            stop=stop_after_attempt(cfg.max_retries + 1),
            wait=wait_exponential_jitter(initial=1, max=20),
            before_sleep=log_retry,
            reraise=True,   # after the last attempt, raise the real error
        )

        with timer() as timing:
            response = retrying(self._client.chat.completions.create, **kwargs)

        usage = response.usage
        result = LLMResult(
            text=response.choices[0].message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_s=timing["seconds"],
            cost_usd=self._cost(model_key, usage.prompt_tokens, usage.completion_tokens),
            attempts=retrying.statistics.get("attempt_number", 1),
        )
        log.info("llm call model=%s latency=%.2fs in=%d out=%d cost=$%.5f attempts=%d",
                 result.model, result.latency_s, result.input_tokens,
                 result.output_tokens, result.cost_usd, result.attempts)
        return response, result
