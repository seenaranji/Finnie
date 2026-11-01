"""The Finnie graph, part 2 (Step 21):

  START -> contextualize -> router -> (conditional edge) -> finance_qa | market | portfolio | goal | news | tax | clarify | scope_reply | fallback -> finalize -> END
                                                             +-> multi -> planner -> [run_task x N in parallel] -> after_tasks -> (next wave | synthesize) -> finalize

contextualize  turns "and how much is it up this year?" into a standalone question using the saved conversation (no model call on the first turn)
router         one cheap classification (Step 20/21); a failure sends the turn to `fallback`
agent nodes    thin wrappers around the agents; ANY exception becomes a safe fixed reply plus an `error` record (nothing escapes the graph)
clarify        a fixed reply asking what the user means (no model call)
scope_reply    a fixed reply for messages that are not about finance, sent BEFORE any retrieval, tool or answering model (Step 21a); every decision is logged
planner        (route `multi` only, Step 21b) splits the message into tasks with dependencies
run_task       one sub-task, launched with LangGraph's Send; sibling tasks run at the same time; each has its own timeout and records when it started and ended
after_tasks    runs when a wave finishes; the conditional edge then starts the next wave of ready tasks, or goes to synthesize
synthesize     merges the branches into one answer (checked in code), lists what is missing, sets one disclaimer
finalize       sets the disclaimer by route and saves the assistant's turn to the history
Compile with a checkpointer and call with config={"configurable": {"thread_id": session_id}} to get memory.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.goal_extraction import GoalState
from src.core.config import WorkflowConfig
from src.core.guardrails import DISCLAIMER_STANDARD, DISCLAIMER_STRONG
from src.core.llm import LLMClient
from src.core.logging import get_logger
from src.core.schemas import AgentRequest, AgentResponse
from src.workflow.nodes import extract_news_topic, fixed_response, rewrite_message, user_safe_text
from src.workflow.planner import Planner, levels, Task
from src.workflow.router import ROUTES, Router
from src.workflow.state import FinnieState
from src.workflow.synthesizer import Synthesizer, missing_note
from src.workflow.timeline import timing_summary

log = get_logger(__name__)
STRONG_DISCLAIMER_ROUTES = ("portfolio", "goal", "tax")


def degraded_reason(metadata: dict) -> str | None:
    """An agent can return NORMALLY and still not have done the job: every tool call failed, there was nothing in the knowledge base, no news, or no portfolio.
    Such a branch is left out of the merge and listed as missing, with a fixed user-safe reason. Returns None when the branch did its job."""
    trace = metadata.get("tool_trace") or []
    note = str(metadata.get("model", ""))
    if trace and all("error" in entry["result"] for entry in trace):
        return "that part isn't available right now"
    if note.startswith("none (no context)"):
        return "I don't have information on that in my knowledge base"
    if note.startswith("none (needs the portfolio)"):
        return "I need your portfolio holdings for that"
    if note.startswith("none (no usable news)"):
        return "no recent news was found"
    if note.startswith("none (no prices)"):
        return "your holdings' prices aren't available right now"
    if note.startswith("none (bad input)"):
        return "I couldn't read your portfolio file"
    return None


def run_with_timeout(fn, timeout: float):
    """('ok', value) | ('error', exception) | ('timeout', None). A thread cannot be killed: on a timeout the call is ABANDONED (its result is discarded)
    and keeps running in the background until it finishes, so its model calls still cost money."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return "ok", future.result(timeout=timeout)
    except Exception as error:
        return ("error", error) if future.done() else ("timeout", None)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class Agents:
    finance_qa: Any   # .handle(AgentRequest)
    market: Any       # .handle(AgentRequest)
    portfolio: Any    # .run(csv_text, question)
    goal: Any         # .run(message, GoalState) -> GoalTurn
    news: Any         # .run(topic)
    tax: Any          # .handle(AgentRequest)


def build_graph(llm: LLMClient, router: Router, agents: Agents, config: WorkflowConfig, checkpointer=None, resolver_version: str = "v2",
                planner: Planner | None = None, synthesizer: Synthesizer | None = None):
    planner = planner or Planner(llm, config.max_tasks)
    synthesizer = synthesizer or Synthesizer(llm)

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

    # ---------------- Step 21b: several agents for one message ----------------
    calls: dict = {}   # filled below with the same agent callables the single-route nodes use, so a sub-task runs exactly what a single route would
    def planner_node(state: FinnieState) -> dict:
        try:
            plan = planner.plan(state["resolved_message"])
        except Exception as error:   # PlanError, or the planner's model call failed
            log.exception("planner failed")
            return {"plan": {"tasks": [], "error": type(error).__name__}, "sub_results": {"__reset__": True},
                    "error": {"node": "planner", "type": type(error).__name__, "message": str(error)[:200]},
                    "trace": [{"turn": state["turn"], "node": "planner", "error": type(error).__name__}]}
        tasks = [t.as_dict() for t in plan.tasks]
        return {"plan": {"tasks": tasks, "error": None, "cost_usd": plan.cost_usd, "seconds": plan.latency_s, "tokens": plan.input_tokens + plan.output_tokens, "dropped_dependencies": plan.dropped},
                "sub_results": {"__reset__": True},   # start this turn's results empty (the checkpointer keeps last turn's)
                "trace": [{"turn": state["turn"], "node": "planner", "tasks": [f"{t['id']}:{t['agent']}" for t in tasks], "waves": len(levels(plan.tasks)), "dropped_dependencies": plan.dropped,
                           "seconds": round(plan.latency_s, 3), "cost_usd": plan.cost_usd}]}

    def dispatch(state: FinnieState):
        """The conditional edge after the planner and after every wave: start every READY task (all its dependencies are finished), or go on to synthesize.
        In sequential mode only one ready task is started at a time; the graph is otherwise identical."""
        tasks, done = (state.get("plan") or {}).get("tasks", []), state.get("sub_results") or {}
        ready = [t for t in tasks if t["id"] not in done and all(dep in done for dep in t["depends_on"])]
        if not ready:
            return "synthesize"
        if (state.get("execution") or config.execution) == "sequential":
            ready = ready[:1]
        env = {"session_id": state.get("session_id", "default"), "portfolio_csv": state.get("portfolio_csv"), "turn": state["turn"]}
        return [Send("run_task", {"task": t, "deps": {dep: done[dep] for dep in t["depends_on"]}, "env": env}) for t in ready]

    def run_task(payload: dict) -> dict:
        task, deps, env = payload["task"], payload["deps"], payload["env"]
        started = time.perf_counter()
        base = {"id": task["id"], "agent": task["agent"], "question": task["question"], "started": started}
        failed_deps = [d for d, r in deps.items() if r["status"] != "ok"]
        if failed_deps:   # a task that needs a result that does not exist is skipped, not attempted
            result = {**base, "status": "skipped", "text": "", "cost_usd": 0.0, "ended": time.perf_counter(), "skipped_because": failed_deps}
        else:
            question = task["question"] + "".join(f"\n\nRESULT OF AN EARLIER STEP ({r['agent']}):\n{r['text'][:1500]}" for r in deps.values())
            pseudo = {"resolved_message": question, "session_id": env["session_id"], "portfolio_csv": env.get("portfolio_csv"), "turn": env["turn"], "goal_state": {}}
            status, outcome = run_with_timeout(lambda: calls[task["agent"]](pseudo), config.branch_timeout_seconds)
            ended = time.perf_counter()
            if status == "ok":
                response = outcome[0]
                degraded = degraded_reason(response.metadata)
                result = {**base, "status": "degraded" if degraded else "ok", "degraded_reason": degraded, "text": response.text, "sources": [s.model_dump() for s in response.sources], "cost_usd": response.metadata.get("cost_usd") or 0.0,
                          "response": response.model_dump(mode="json"), "ended": ended}
            else:
                log.warning("task %s (%s) %s: %s", task["id"], task["agent"], status, outcome)
                result = {**base, "status": status, "text": "", "cost_usd": 0.0, "ended": ended, "error_type": type(outcome).__name__ if outcome else "Timeout",
                          "user_text": user_safe_text(outcome, config.fallback_reply) if outcome else None}
        result["seconds"] = round(result["ended"] - started, 3)
        return {"sub_results": {task["id"]: result},
                "trace": [{"turn": env["turn"], "node": f"task:{task['id']}", "agent": task["agent"], "status": result["status"], "seconds": result["seconds"], "cost_usd": result["cost_usd"]}]}

    def synthesize(state: FinnieState) -> dict:
        plan, results, turn = state.get("plan") or {}, state.get("sub_results") or {}, state["turn"]
        if plan.get("error") or not plan.get("tasks"):
            reply = fixed_response(config.fallback_reply, "fallback", "fallback", error=plan.get("error"))
            return {"response": reply.model_dump(mode="json"), "trace": [{"turn": turn, "node": "synthesize", "error": plan.get("error")}]}
        ordered = [results[t["id"]] for t in plan["tasks"] if t["id"] in results]
        ok, missing = [r for r in ordered if r["status"] == "ok"], [r for r in ordered if r["status"] != "ok"]
        timing = timing_summary(results)
        branch_cost, planner_cost = sum(r["cost_usd"] for r in ordered), plan.get("cost_usd", 0.0)
        if not ok:
            reply = fixed_response(missing_note(missing, everything=True), "multi_agent", "all branches failed", partial=True, missing=[m["agent"] for m in missing])
            return {"response": reply.model_dump(mode="json"), "trace": [{"turn": turn, "node": "synthesize", "ok": 0, "missing": len(missing)}]}
        if len(plan["tasks"]) == 1:   # the planner decided ONE agent is enough: its answer is returned unchanged
            response = dict(ok[0]["response"])
            response["metadata"] = {**response["metadata"], "plan": plan["tasks"], "planner_cost_usd": planner_cost, "cost_usd": (response["metadata"].get("cost_usd") or 0) + planner_cost}
            return {"response": response, "trace": [{"turn": turn, "node": "synthesize", "ok": 1, "single_task": True}]}
        merged = synthesizer.merge(state["resolved_message"], ok, missing)
        note = missing_note(missing)
        sources = list({s["id"]: s for r in ok for s in r["sources"]}.values())
        strong = any(t["agent"] in STRONG_DISCLAIMER_ROUTES for t in plan["tasks"])
        response = AgentResponse(text=merged.text + (f"\n\n{note}" if note else ""), agent_name="multi_agent", sources=sources,
                                 disclaimer=DISCLAIMER_STRONG if strong else DISCLAIMER_STANDARD,
                                 metadata={"model": merged.model, "cost_usd": branch_cost + planner_cost + merged.cost_usd, "latency_s": timing["wall_s"], "plan": plan["tasks"],
                                           "tasks": {r["id"]: {"agent": r["agent"], "status": r["status"], "seconds": r["seconds"], "cost_usd": r["cost_usd"]} for r in ordered},
                                           "timing": timing, "partial": bool(missing), "missing": [m["agent"] for m in missing], "synthesis": merged.mode,
                                           "synthesis_issues": merged.issues, "synthesis_cost_usd": merged.cost_usd, "planner_cost_usd": planner_cost, "branch_cost_usd": branch_cost})
        return {"response": response.model_dump(mode="json"), "trace": [{"turn": turn, "node": "synthesize", "ok": len(ok), "missing": len(missing), "synthesis": merged.mode,
                                                                    "seconds": round(merged.latency_s, 3), "cost_usd": merged.cost_usd}]}

    # ---------------- wiring ----------------
    nodes = {"finance_qa": lambda s: (agents.finance_qa.handle(request(s)), {}), "market": lambda s: (agents.market.handle(request(s)), {}),
             "tax": lambda s: (agents.tax.handle(request(s)), {}), "portfolio": call_portfolio, "goal": call_goal, "news": call_news}
    calls.update(nodes)
    graph = StateGraph(FinnieState)
    graph.add_node("contextualize", contextualize)
    graph.add_node("router", router_node)
    for name, call in nodes.items():
        graph.add_node(name, agent_node(name, call))
    graph.add_node("clarify", clarify)
    graph.add_node("fallback", fallback)
    graph.add_node("scope_reply", scope_reply)
    graph.add_node("finalize", finalize)
    graph.add_node("planner", planner_node)
    graph.add_node("run_task", run_task)
    graph.add_node("after_tasks", lambda state: {})   # a join point: runs once when a wave of run_task branches has finished
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "router")
    targets = {route: route for route in ROUTES if route not in ("out_of_scope", "multi")} | {"out_of_scope": "scope_reply", "multi": "planner", "fallback": "fallback"}
    graph.add_conditional_edges("router", lambda state: state["route"], targets)   # the conditional edge: follow the route the router wrote into the state
    for name in (*nodes, "clarify", "fallback", "scope_reply"):
        graph.add_edge(name, "finalize")
    graph.add_conditional_edges("planner", dispatch, ["run_task", "synthesize"])
    graph.add_edge("run_task", "after_tasks")
    graph.add_conditional_edges("after_tasks", dispatch, ["run_task", "synthesize"])
    graph.add_edge("synthesize", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
