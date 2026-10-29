"""Step 2 demo: guarded agent vs an unguarded plain LLM call.

Run:  .venv/bin/python -m scripts.step2_agent_demo
"""
import json

from src.agents.test_agent import TutorTestAgent
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.schemas import AgentRequest

setup_logging("WARNING")
llm = LLMClient()
agent = TutorTestAgent(llm)

for q in ["What is a bond?", "Should I put my savings into Tesla?"]:
    print("=" * 70, f"\nQUESTION: {q}\n")
    raw = llm.chat(q, temperature=0.2)  # no system prompt, no guardrails
    print(f"--- UNGUARDED (plain LLM call) ---\n{raw.text}\n")
    resp = agent.handle(AgentRequest(message=q))
    print(f"--- GUARDED (agent) ---\n{resp.text}\n")
    print(f"disclaimer: {resp.disclaimer}\n")
    print("metadata:", json.dumps(resp.metadata, indent=2))
