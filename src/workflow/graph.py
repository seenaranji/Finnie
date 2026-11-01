"""The Finnie graph, part 2 (Step 21):

  START -> contextualize -> router -> (conditional edge) -> finance_qa | market | portfolio | goal | news | tax | clarify | scope_reply | fallback -> finalize -> END

contextualize  turns "and how much is it up this year?" into a standalone question using the saved conversation (no model call on the first turn)
router         one cheap classification (Step 20/21); a failure sends the turn to `fallback`
agent nodes    thin wrappers around the agents; ANY exception becomes a safe fixed reply plus an `error` record (nothing escapes the graph)
clarify        a fixed reply asking what the user means (no model call)
scope_reply    a fixed reply for messages that are not about finance, sent BEFORE any retrieval, tool or answering model (Step 21a); every decision is logged
finalize       sets the disclaimer by route and saves the assistant's turn to the history
Compile with a checkpointer and call with config={"configurable": {"thread_id": session_id}} to get memory.
"""
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.goal_extraction import GoalState
from src.core.config import WorkflowConfig
from src.core.guardrails import DISCLAIMER_STANDARD, DISCLAIMER_STRONG
from src.core.llm import LLMClient
from src.core.logging import get_logger
from src.core.schemas import AgentRequest, AgentResponse
from src.workflow.nodes import extract_news_topic, fixed_response, rewrite_message, user_safe_text
from src.workflow.router import ROUTES, Router
from src.workflow.state import FinnieState

log = get_logger(__name__)
STRONG_DISCLAIMER_ROUTES = ("portfolio", "goal", "tax")


@dataclass
class Agents:
    finance_qa: Any   # .handle(AgentRequest)
    market: Any       # .handle(AgentRequest)
    portfolio: Any    # .run(csv_text, question)
    goal: Any         # .run(message, GoalState) -> GoalTurn
    news: Any         # .run(topic)
    tax: Any          # .handle(AgentRequest)


def build_graph(llm: LLMClient, router: Router, agents: Agents, config: WorkflowConfig, checkpointer=None, resolver_version: str = "v2"):
    def request(state: FinnieState) -> AgentRequest:
        return AgentRequest(message=state["resolved_message"], session_id=state.get("session_id", "default"))

    # ---------------- nodes ----------------
    def contextualize(state: FinnieState) -> dict:
        history, message = state.get("history", []), state["message"]
        turn = sum(h["role"] == "user" for h in history) + 1
        resolved, call = (message, None) if not history else rewrite_message(llm, message, history, config.history_turns_for_rewrite, resolver_version)
        return {"turn": turn, "resolved_message": resolved, "error": None, "history": [{"role": "user", "content": message}],
                "trace": [{"turn": turn, "node": "contextualize", "rewritten": resolved != message, "resolved": resolved if resolved != message else None,
                           "cost_usd": call.cost_usd if call else 0.0, "seconds": round(call.latency_s, 3) if call else 0.0}]}

    def router_node(state: FinnieState) -> dict:
        try:
            d = router.classify(state["resolved_message"])
        except Exception as error:   # the router's model call failed: nothing else will work either, so answer with the fixed fallback
            log.exception("router failed")
            return {"route": "fallback", "router": {"error": type(error).__name__}, "error": {"node": "router", "type": type(error).__name__, "message": str(error)[:200]},
                    "trace": [{"turn": state["turn"], "node": "router", "route": "fallback", "error": type(error).__name__}]}
        route, guard = d.route, None
        if route == "out_of_scope" and state.get("portfolio_csv"):   # a user who uploaded their holdings is never told "that is not finance": ask instead
            route, guard = "clarify", "has_uploaded_portfolio"
        return {"route": route, "router": d.as_dict() | {"guard": guard},
                "trace": [{"turn": state["turn"], "node": "router", "route": route, "raw_route": d.raw_route, "confidence": d.confidence, "overridden": d.overridden, "guard": guard,
                           "seconds": round(d.latency_s, 3), "tokens": d.input_tokens + d.output_tokens, "cost_usd": d.cost_usd, "fell_back": d.fell_back}]}

    def agent_node(name: str, call):
        """`call(state)` returns (AgentResponse, extra state keys). Any exception becomes a safe fixed reply."""
        def run(state: FinnieState) -> dict:
            start = time.perf_counter()
            try:
                response, extra = call(state)
            except Exception as error:
                log.exception("node %s failed", name)
                reply = fixed_response(user_safe_text(error, config.fallback_reply), "fallback", "fallback", error=type(error).__name__)
                return {"response": reply.model_dump(mode="json"), "error": {"node": name, "type": type(error).__name__, "message": str(error)[:200]},
                        "trace": [{"turn": state["turn"], "node": name, "error": type(error).__name__, "seconds": round(time.perf_counter() - start, 3)}]}
            meta = response.metadata
            return {"response": response.model_dump(mode="json"), **extra,
                    "trace": [{"turn": state["turn"], "node": name, "agent": response.agent_name, "seconds": round(time.perf_counter() - start, 3),
                               "model": meta.get("model"), "cost_usd": meta.get("cost_usd"), "tools": meta.get("tool_names"), **meta.get("trace_extra", {})}]}
        return run

    def call_portfolio(state):
        csv_text = state.get("portfolio_csv")
        if not csv_text:
            return fixed_response(config.needs_portfolio_reply, "portfolio_analysis", "needs the portfolio", DISCLAIMER_STRONG), {}
        return agents.portfolio.run(csv_text, state["resolved_message"]), {}

    def call_goal(state):
        turn = agents.goal.run(state["resolved_message"], GoalState(**(state.get("goal_state") or {})))
        return turn.response, {"goal_state": {} if turn.plan_made else turn.state.model_dump()}   # a finished plan clears the slots so the next goal starts fresh

    def call_news(state):
        topic, call = extract_news_topic(llm, state["resolved_message"])
        response = agents.news.run(topic)
        response.metadata["trace_extra"] = {"topic": topic, "topic_cost_usd": call.cost_usd if call else 0.0}
        return response, {}

    def clarify(state: FinnieState) -> dict:
        reply = fixed_response(config.clarify_reply, "clarify", "clarify")
        return {"response": reply.model_dump(mode="json"), "trace": [{"turn": state["turn"], "node": "clarify", "agent": "clarify", "seconds": 0.0, "cost_usd": 0.0}]}

    def scope_reply(state: FinnieState) -> dict:
        """The router said the message is not about finance: a FIXED reply, no model, no search, no tool. The decision is logged for review."""
        entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "session_id": state.get("session_id"), "query": state["message"],
                 "resolved": state["resolved_message"], "confidence": state["router"].get("confidence"), "turn": state["turn"]}
        log.info("out_of_scope decision: %r (confidence %s)", state["message"], entry["confidence"])
        try:
            path = Path(config.scope_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:   # a full disk must not stop the polite reply
            log.warning("could not write the out-of-scope log to %s", config.scope_log_path)
        reply = fixed_response(config.scope_reply, "scope_reply", "out of scope")
        return {"response": reply.model_dump(mode="json"), "trace": [{"turn": state["turn"], "node": "scope_reply", "agent": "scope_reply", "seconds": 0.0, "cost_usd": 0.0}]}

    def fallback(state: FinnieState) -> dict:
        reply = fixed_response(config.fallback_reply, "fallback", "fallback")
        return {"response": reply.model_dump(mode="json"), "trace": [{"turn": state["turn"], "node": "fallback", "agent": "fallback", "seconds": 0.0, "cost_usd": 0.0}]}

    def finalize(state: FinnieState) -> dict:
        response = dict(state["response"])
        if state.get("route") in STRONG_DISCLAIMER_ROUTES:
            response["disclaimer"] = DISCLAIMER_STRONG
        elif not response.get("disclaimer"):
            response["disclaimer"] = DISCLAIMER_STANDARD
        return {"response": response, "history": [{"role": "assistant", "content": response["text"], "route": state.get("route")}],
                "trace": [{"turn": state["turn"], "node": "finalize", "disclaimer": "strong" if response["disclaimer"] == DISCLAIMER_STRONG else "standard"}]}

    # ---------------- wiring ----------------
    nodes = {"finance_qa": lambda s: (agents.finance_qa.handle(request(s)), {}), "market": lambda s: (agents.market.handle(request(s)), {}),
             "tax": lambda s: (agents.tax.handle(request(s)), {}), "portfolio": call_portfolio, "goal": call_goal, "news": call_news}
    graph = StateGraph(FinnieState)
    graph.add_node("contextualize", contextualize)
    graph.add_node("router", router_node)
    for name, call in nodes.items():
        graph.add_node(name, agent_node(name, call))
    graph.add_node("clarify", clarify)
    graph.add_node("fallback", fallback)
    graph.add_node("scope_reply", scope_reply)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "router")
    targets = {route: route for route in ROUTES if route != "out_of_scope"} | {"out_of_scope": "scope_reply", "fallback": "fallback"}
    graph.add_conditional_edges("router", lambda state: state["route"], targets)   # the conditional edge: follow the route the router wrote into the state
    for name in (*nodes, "clarify", "fallback", "scope_reply"):
        graph.add_edge(name, "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
