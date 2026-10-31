"""Step 17: the Goal Planning agent on a complete request, a vague one (completed over several turns), and an unrealistic one.

Run:  .venv/bin/python -m scripts.step17_goal_agent_demo
"""
import json
from datetime import datetime
from pathlib import Path

from src.agents.goal_agent import GoalPlanningAgent
from src.agents.goal_extraction import GoalState
from src.core.config import load_settings
from src.core.llm import LLMClient
from src.core.logging import setup_logging

setup_logging("WARNING")
settings = load_settings()
agent = GoalPlanningAgent(LLMClient(settings), settings.goal_planning)
transcript = []


def conversation(title, messages):
    print("=" * 100, f"\n{title}\n" + "=" * 100)
    state = GoalState()
    for number, message in enumerate(messages, start=1):
        turn = agent.run(message, state)
        state, meta = turn.state, turn.response.metadata
        kind = "PLAN" if turn.plan_made else "FOLLOW-UP QUESTION" if turn.needs else "NOTICE"
        print(f"\n[turn {number}] USER: {message}\n[{kind}] model={meta['model']}  cost=${meta['cost_usd']:.5f}  needs={turn.needs}")
        print(f"  extracted this turn: {meta.get('extracted')}")
        print(f"  state after: {turn.state.model_dump()}\n")
        print(turn.response.text)
        if turn.plan_made:
            print(f"\n  disclaimer: {turn.response.disclaimer}")
            print(f"  flags in METRICS: {meta['metrics']['flags']}")
        transcript.append({"title": title, "user": message, "kind": kind, "text": turn.response.text, "metadata": meta})


conversation("A. COMPLETE REQUEST", ["I want to save $50,000 in 5 years. I already have $2,000 and I'm fine with medium risk."])
conversation("B. VAGUE REQUEST, completed over turns", ["I want to retire comfortably.", "about 1.5 million, in 30 years", "I've got 20k saved so far", "balanced I guess"])
conversation("C. UNREALISTIC REQUEST", ["I want to turn my $1,000 into $1,000,000 in one year. High risk is fine."])
conversation("D. ADVICE-SEEKING REQUEST", ["Should I put my $2,000 into stocks to reach $50,000 in 5 years? I'm medium risk."])

output_file = Path(f"benchmarks/results/step17_goal_agent_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(transcript, indent=2, default=str))
print(f"\nsaved: {output_file}")
