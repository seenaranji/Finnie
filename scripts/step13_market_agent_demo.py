"""Step 13: the same market question WITHOUT tools (plain LLM) and WITH tools (Market Analysis agent), plus the tool-call trace.

Run:  .venv/bin/python -m scripts.step13_market_agent_demo
"""
import json
import sys
from datetime import datetime
from pathlib import Path

from src.agents.market_agent import MarketAnalysisAgent
from src.agents.market_tools import MarketTools
from src.agents.test_agent import TutorTestAgent
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.schemas import AgentRequest
from src.data.factory import build_market_data_service

setup_logging("INFO")
QUESTIONS = sys.argv[1:] or ["How is Apple doing today and what does that mean?"]
llm = LLMClient()
service = build_market_data_service()
agent = MarketAnalysisAgent(llm, MarketTools(service))
plain_agent = TutorTestAgent(llm)
rows = []

for question in QUESTIONS:
    request = AgentRequest(message=question)
    plain = plain_agent.handle(request)
    answer = agent.handle(request)
    meta = answer.metadata

    print("=" * 100, f"\nQUESTION: {question}\n" + "=" * 100)
    print(f"\n--- WITHOUT TOOLS (plain LLM, {plain.metadata['model']}) ---\n{plain.text}")
    print(f"\n--- WITH TOOLS (Market Analysis agent) ---\n{answer.text}")
    print(f"\n  sources: {[s.title for s in answer.sources]}")
    print(f"  disclaimer: {answer.disclaimer}")
    print("\n--- TOOL-CALL TRACE ---")
    for entry in meta["tool_trace"]:
        print(f"  step {entry['step']}: {entry['tool']}({json.dumps(entry['arguments'])})  [{entry['seconds'] * 1000:.0f} ms, cache hits {entry['cache_hits']}, misses {entry['cache_misses']}]")
        print(f"      -> {json.dumps(entry['result'])}")
    print("\n--- LATENCY BREAKDOWN ---")
    print(f"  model turns: {meta['llm_calls']}   LLM time: {meta['llm_seconds']:.2f}s   tool time (API + cache): {meta['tool_seconds']:.3f}s   "
          f"total: {meta['latency_s']:.2f}s   cost: ${meta['cost_usd']:.5f}   tokens in/out: {meta['input_tokens']}/{meta['output_tokens']}")
    rows.append({"question": question, "plain": plain.text, "with_tools": answer.text, "metadata": meta})

output_file = Path(f"benchmarks/results/step13_market_agent_demo_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(rows, indent=2, default=str))
print(f"\nsaved: {output_file}")
