"""Market Analysis agent: answers questions about prices and trends by CALLING TOOLS, never from memory.

Loop:  messages -> model -> (tool requests? run them, append results, ask the model again) -> final text.
The model chooses which tool to call; our code runs it. The numbers in the answer come from the tool results, which come
from the market data layer (Steps 11-12), and the statistics are computed in Python (src/data/stats.py), not by the model.
"""
import json
from datetime import date

from src.agents.base_agent import AgentAnswer, BaseAgent
from src.agents.market_tools import TOOL_SCHEMAS, MarketTools
from src.core.guardrails import DISCLAIMER_STANDARD
from src.core.llm import LLMClient
from src.core.logging import get_logger
from src.core.schemas import AgentRequest, Source

log = get_logger(__name__)
MAX_STEPS = 5   # model turns per question; stops a model that keeps asking for tools

SYSTEM_PROMPT = """You are Finnie, a friendly finance tutor for beginners. You explain market data; you never give personal investment advice.

Rules:
1. Every price, percentage, average, range or volatility figure MUST come from a tool result in this conversation. Call a tool for each symbol you discuss. Never use numbers from memory, and never calculate anything yourself.
2. Copy numbers exactly as the tool returns them (you may add $ or %). Do not round them differently or derive new numbers.
3. Say when the data is from: "as of <as_of> (source: <source>)". If is_stale is true, say it may be out of date. If as_of is before today's date, say it is the last closing price, not live.
4. If a tool returns an error, or a value is null or unavailable, say so plainly. Never fill the gap with a guess.
5. After the numbers, explain in one or two simple sentences what they mean for a beginner. Give facts and context only: do not predict prices or tell the user to buy, sell, or hold.
6. Use simple words, under 150 words, and define jargon the first time you use it."""


class MarketAnalysisAgent(BaseAgent):
    name = "market_analysis"
    system_prompt = SYSTEM_PROMPT
    default_disclaimer = DISCLAIMER_STANDARD
    temperature = 0   # numbers should not vary run to run

    def __init__(self, llm: LLMClient, tools: MarketTools) -> None:
        super().__init__(llm)
        self.tools = tools

    def answer(self, request: AgentRequest, system: str) -> AgentAnswer:
        messages = [{"role": "system", "content": f"{system}\n\nToday's date is {date.today().isoformat()}."},
                    {"role": "user", "content": request.message}]
        trace, llm_seconds, tokens_in, tokens_out, cost, model, text = [], 0.0, 0, 0, 0.0, "", ""
        for step in range(1, MAX_STEPS + 1):
            turn = self.llm.complete(messages, tools=TOOL_SCHEMAS, temperature=self.temperature)
            llm_seconds += turn.latency_s
            tokens_in, tokens_out, cost, model = tokens_in + turn.input_tokens, tokens_out + turn.output_tokens, cost + turn.cost_usd, turn.model
            if not turn.tool_calls:
                text = turn.text
                break
            messages.append(turn.assistant_message)
            for call in turn.tool_calls:
                result, entry = self.tools.run(call.name, call.arguments)
                entry["step"] = step
                trace.append(entry)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        else:
            log.warning("agent hit MAX_STEPS=%d without a final answer", MAX_STEPS)
            text = "I couldn't finish looking that up. Please try a simpler question."

        tool_seconds = sum(t["seconds"] for t in trace)
        sources, seen = [], set()
        for entry in trace:
            result = entry["result"]
            key = (result.get("source"), result.get("symbol"))
            if "error" not in result and key not in seen:
                seen.add(key)
                sources.append(Source(id=f"{key[0]}:{key[1]}", title=f"{key[0]} market data for {key[1]} (as of {result.get('as_of')})"))
        return AgentAnswer(text, model, tokens_in, tokens_out, llm_seconds + tool_seconds, cost, 1, sources=sources,
                           extra={"tool_trace": trace, "tool_names": [t["tool"] for t in trace], "llm_calls": step if text else MAX_STEPS,
                                  "llm_seconds": round(llm_seconds, 3), "tool_seconds": round(tool_seconds, 4)})
