"""Step 8: print two retrieval result files side by side (same config names, metrics from each).

Run:  .venv/bin/python -m eval.scripts.compare eval/results/<first>.json eval/results/<second>.json
"""
import json
import sys
from pathlib import Path

first, second = (json.loads(Path(path).read_text()) for path in sys.argv[1:3])
print(f"A = {sys.argv[1]}  (split={first['split']}, commit={first['git_commit']})")
print(f"B = {sys.argv[2]}  (split={second['split']}, commit={second['git_commit']})\n")
print(f"{'config':28} | {'Hit@3 A':>7} {'B':>6} | {'MRR A':>6} {'B':>6} | {'refused A':>9} {'B':>5} | {'no-ans A':>8} {'B':>5}")
for name in first["configs"]:
    if name not in second["configs"]:
        continue
    a, b = first["configs"][name], second["configs"][name]
    print(f"{name:28} | {a['hit3']:>7.2f} {b['hit3']:>6.2f} | {a['mrr']:>6.2f} {b['mrr']:>6.2f} | "
          f"{a['wrongly_refused']:>9} {b['wrongly_refused']:>5} | {a['no_answer_accuracy']:>8.2f} {b['no_answer_accuracy']:>5.2f}")
