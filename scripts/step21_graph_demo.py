"""Step 21: the full graph with real agents and memory. Four sessions: the spec's three-turn dialog, goal slot filling, a remembered portfolio,
and a session that starts with an unresolvable "it".

Run:  .venv/bin/python -m scripts.step21_graph_demo
"""
import json
from datetime import datetime
from pathlib import Path

from src.core.logging import setup_logging
from src.workflow.factory import build_default_graph, config_for

setup_logging("ERROR")
graph = build_default_graph()   # real LLM, real retrieval, real market data and news; an in-memory checkpointer
report, total_cost = [], 0.0
CSV = "ticker,shares,avg_cost\nVTI,60,220\nBND,100,74\nVXUS,80,58\nAAPL,40,150\n"


def say(session, message, **extra):
    global total_cost
    state = graph.invoke({"message": message, "session_id": session, **extra}, config_for(session))
    turn = [t for t in state["trace"] if t["turn"] == state["turn"]]
    cost = sum(t.get("cost_usd") or 0 for t in turn) + sum(t.get("topic_cost_usd") or 0 for t in turn)
    seconds = sum(t.get("seconds") or 0 for t in turn)
    total_cost += cost
    r = state["response"]
    print(f"\n[{session} / turn {state['turn']}] USER: {message}")
    if state["resolved_message"] != message:
        print(f"    resolved as : {state['resolved_message']}")
    print(f"    route       : {state['route']} (confidence {state['router'].get('confidence')}, overridden={state['router'].get('overridden')})   error: {state.get('error')}")
    print(f"    nodes       : {' -> '.join(t['node'] for t in turn)}   ({seconds:.1f}s, ${cost:.5f})")
    print(f"    disclaimer  : {'STRONG' if 'personalized financial, investment, or tax advice' in r['disclaimer'] else 'standard'}")
    print(f"    FINNIE      : {r['text'][:520]}")
    report.append({"session": session, "turn": state["turn"], "message": message, "resolved": state["resolved_message"], "route": state["route"], "answer": r["text"], "seconds": seconds, "cost": cost})
    return state


print("=" * 100 + "\nSESSION A: the spec's dialog. 'it' must become VOO, and the question must reach the market agent\n" + "=" * 100)
say("A", "What's an ETF?")
say("A", "How is VOO doing today?")
state = say("A", "And how much is it up this year?")

print("\n" + "=" * 100 + "\nSESSION B: goal planning. The slots are remembered between turns and cleared once the plan is made\n" + "=" * 100)
say("B", "I want to save $50,000 in 5 years")
say("B", "I have $2,000 saved")
state_b = say("B", "medium risk")
print(f"\n    goal_state after the plan: {state_b['goal_state']}   (empty = cleared, ready for a new goal)")

print("\n" + "=" * 100 + "\nSESSION C: the portfolio is supplied once and remembered\n" + "=" * 100)
say("C", "Analyze my portfolio", portfolio_csv=CSV)
state_c = say("C", "Is it too concentrated?")
print(f"\n    portfolio_csv still in the saved state for turn 2: {bool(state_c.get('portfolio_csv'))}")

print("\n" + "=" * 100 + "\nSESSION D: no portfolio supplied, a vague opener, and a fresh session that has no memory of session A\n" + "=" * 100)
say("D", "Analyze my portfolio")
say("D", "What about that one?")
say("E", "And how much is it up this year?")

print("\n" + "=" * 100)
checkpoints = list(graph.get_state_history(config_for("A")))
print(f"checkpoints saved for session A: {len(checkpoints)}    history in session A: {[h['role'] + ':' + (h.get('route') or '') for h in state['history']]}")
print(f"TOTAL COST of {len(report)} turns: ${total_cost:.4f}")
out = Path(f"benchmarks/results/step21_graph_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
out.write_text(json.dumps(report, indent=2, default=str))
print(f"saved: {out}")
