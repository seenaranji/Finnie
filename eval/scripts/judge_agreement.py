"""Step 10: how often does the LLM judge agree with YOUR hand grades?

Run:  .venv/bin/python -m eval.scripts.judge_agreement eval/results/answers_improved_<time>.json eval/results/grading_sheet.csv
"""
import csv
import json
import sys
from pathlib import Path

judge_items = {i["question"]: i for i in json.loads(Path(sys.argv[1]).read_text())["items"]}
key = {k["question"]: k["kind"] for k in json.loads(Path("eval/results/grading_key.json").read_text())}
fields = [("faithful", 4), ("relevant", 5), ("refused", 6), ("gives_advice", 7)]
stats = {kind: {name: [0, 0] for name, _ in fields} for kind in ("random", "disputed")}
disagreements = []
for row in csv.reader(Path(sys.argv[2]).open()):
    if not row or row[0] == "n":
        continue
    question = row[1]
    verdict = judge_items[question]["verdict"]
    judge = {"faithful": not any(not c.get("supported") for c in verdict["claims"]), "relevant": verdict["relevant"],
             "refused": verdict["refused"], "gives_advice": verdict["gives_advice"]}
    for name, column in fields:
        human = row[column].strip().upper().startswith("Y")
        stats[key[question]][name][1] += 1
        stats[key[question]][name][0] += int(human == judge[name])
        if human != judge[name]:
            disagreements.append(f"  [{key[question]}] {name}: human={'Y' if human else 'N'} judge={'Y' if judge[name] else 'N'} | {question}")
for kind, per_field in stats.items():
    total_agree = sum(a for a, _ in per_field.values()); total = sum(n for _, n in per_field.values())
    print(f"\n{kind.upper()} answers ({total // 4} answers): overall agreement {total_agree}/{total}"
          f" = {total_agree / total:.0%}" if total else f"\n{kind}: no rows")
    for name, (agree, n) in per_field.items():
        print(f"   {name:13} {agree}/{n}")
print("\nDISAGREEMENTS:" if disagreements else "\nNo disagreements.")
print("\n".join(disagreements))
