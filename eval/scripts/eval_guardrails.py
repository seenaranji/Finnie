"""Step 2 eval: advice-leak rate and over-refusal rate on eval/datasets/guardrails.jsonl.

Run:  .venv/bin/python -m eval.scripts.eval_guardrails

Per prompt we record three things:
  request_flagged  - did is_advice_request() fire on the user message?
  model_flagged    - did the model's RAW answer contain advice-like phrases?
  redirected       - did the user end up seeing the canned redirect instead?
Rules can miss things, so read the printed answers yourself too.
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.agents.test_agent import TutorTestAgent
from src.core.guardrails import ADVICE_REDIRECT
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.schemas import AgentRequest

setup_logging("WARNING")
rows = [json.loads(l) for l in Path("eval/datasets/guardrails.jsonl").read_text().splitlines() if l]

agent = TutorTestAgent(LLMClient())
agent.temperature = 0  # repeatability

results = []
for row in rows:
    resp = agent.handle(AgentRequest(message=row["prompt"]))
    results.append({
        **row,
        "request_flagged": resp.metadata["advice_requested"],
        "model_flagged": resp.metadata["output_flagged"],
        "redirected": resp.text == ADVICE_REDIRECT,
        "final_text": resp.text,
        "raw_model_text": resp.metadata["original_text"] or resp.text,
        "cost_usd": resp.metadata["cost_usd"],
    })

adv = [r for r in results if r["label"] == "advice"]
ben = [r for r in results if r["label"] == "benign"]
metrics = {
    "advice_prompts": len(adv),
    "benign_prompts": len(ben),
    "advice_request_detected": sum(r["request_flagged"] for r in adv),
    "advice_model_flagged_by_rules": sum(r["model_flagged"] for r in adv),
    "benign_wrongly_flagged_request": sum(r["request_flagged"] for r in ben),
    "benign_redirected": sum(r["redirected"] for r in ben),
    "total_cost_usd": sum(r["cost_usd"] for r in results),
}

try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    commit = "unknown"
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = Path(f"eval/results/guardrails_{stamp}.json")
out.write_text(json.dumps({"commit": commit, "temperature": 0, "metrics": metrics,
                           "results": results}, indent=2))

print("##### FULL RAW OUTPUT #####")
for r in results:
    print(f"\n=== {r['id']} [{r['label']}] request_flagged={r['request_flagged']} "
          f"model_flagged={r['model_flagged']} redirected={r['redirected']}")
    print(f"Q: {r['prompt']}")
    if r["model_flagged"]:
        print(f"RAW MODEL ANSWER (before guard):\n{r['raw_model_text']}\nUSER SAW:")
    print(r["final_text"])

print("\n##### METRICS #####")
print(json.dumps(metrics, indent=2))
print(f"saved: {out}")
