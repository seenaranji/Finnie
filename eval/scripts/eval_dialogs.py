"""Step 21: does memory resolve "it / that / and X?" correctly? 10 scripted dialogs, 25 turns, through contextualize + router (real model calls).

The agents are STUBS that answer "[name] answer to: <resolved question>", so this measures the part memory affects (the rewrite and the route) at a few cents,
not the agents themselves. A turn passes if the route is acceptable AND the resolved question contains every `must_contain` word and none of `must_not_contain`.
A dialog passes only if every turn passes. Limitation: stub answers are shorter than real ones, which makes the rewrite's job a little easier.

Run:  .venv/bin/python -m eval.scripts.eval_dialogs
"""
import json
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from src.core.config import load_settings
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.schemas import AgentRequest, AgentResponse
from src.workflow.graph import Agents, build_graph
from src.workflow.router import Router

import argparse


class Stub:
    def __init__(self, name):
        self.name = name

    def _response(self, text):
        return AgentResponse(text=f"[{self.name}] answer to: {text}", agent_name=self.name, disclaimer="", metadata={"model": "stub", "cost_usd": 0.0, "tool_names": []})

    def handle(self, request: AgentRequest):
        return self._response(request.message)

    def run(self, *args):
        return self._response(args[-1] if isinstance(args[-1], str) else args[0])


class StubGoal(Stub):
    def run(self, message, state):
        from types import SimpleNamespace
        return SimpleNamespace(response=self._response(message), state=state, plan_made=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dialogs", "dialogs2"], default="dialogs")
    parser.add_argument("--resolver", default="v1")
    args = parser.parse_args()
    dataset = Path(f"eval/datasets/{args.dataset}.jsonl")
    setup_logging("ERROR")
    settings, llm = load_settings(), LLMClient()
    agents = Agents(Stub("finance_qa"), Stub("market"), Stub("portfolio"), StubGoal("goal"), Stub("news"), Stub("tax"))
    graph = build_graph(llm, Router(llm), agents, settings.workflow, checkpointer=InMemorySaver(), resolver_version=args.resolver)
    # the news node asks the model for a topic; everything else is the real router and the real resolver
    dialogs = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    results, turn_total, turn_ok, cost = [], 0, 0, 0.0
    print("=" * 100 + f"\nMULTI-TURN CONTEXT RESOLUTION: dataset={args.dataset}, resolver prompt={args.resolver} ({len(dialogs)} dialogs)\n" + "=" * 100)
    for dialog in dialogs:
        session, dialog_ok, rows = f"eval-{dialog['id']}", True, []
        for number, turn in enumerate(dialog["turns"], start=1):
            extra = {"portfolio_csv": dialog["portfolio_csv"]} if dialog.get("portfolio_csv") and number == 1 else {}
            state = graph.invoke({"message": turn["say"], "session_id": session, **extra}, {"configurable": {"thread_id": session}})
            resolved = state["resolved_message"]
            low = resolved.lower()
            route_ok = state["route"] in turn["routes"]
            has_ok = all(w in low for w in turn.get("must_contain", []))
            not_ok = not any(w in low for w in turn.get("must_not_contain", []))
            ok = route_ok and has_ok and not_ok
            turn_total += 1; turn_ok += ok; dialog_ok &= ok
            cost += sum(t.get("cost_usd") or 0 for t in state["trace"] if t["turn"] == state["turn"] and t["node"] in ("contextualize", "router"))
            rows.append({"say": turn["say"], "resolved": resolved, "route": state["route"], "ok": ok, "route_ok": route_ok, "has_ok": has_ok, "not_ok": not_ok})
            print(f"  {dialog['id']} turn {number} {'ok  ' if ok else 'FAIL'} route={state['route']:11} said={turn['say']!r}\n          resolved={resolved!r}"
                  + ("" if ok else f"\n          expected route in {turn['routes']}, must contain {turn.get('must_contain', [])}, must not contain {turn.get('must_not_contain', [])}"))
        results.append({"id": dialog["id"], "note": dialog["note"], "ok": dialog_ok, "turns": rows})
        print(f"  -> {dialog['id']} {'PASSED' if dialog_ok else 'FAILED'}   ({dialog['note']})\n")
    print("=" * 100)
    print(f"TURNS passed: {turn_ok}/{turn_total} = {turn_ok / turn_total:.0%}      DIALOGS passed: {sum(r['ok'] for r in results)}/{len(results)}      resolver+router cost: ${cost:.5f}")
    out = Path(f"eval/results/{args.dataset}_{args.resolver}_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps({"turns_ok": turn_ok, "turns": turn_total, "dialogs": results}, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
