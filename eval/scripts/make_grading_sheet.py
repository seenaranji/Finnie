"""Step 10: write a BLIND grading sheet (no judge verdicts shown) for hand-grading 10 answers.

8 answers are a seeded random sample of the LLM-written answers; 2 are the answers whose judge flags looked wrong to me.
Agreement is reported separately for the random 8 (honest estimate) and for the 2 disputed ones.

Run:  .venv/bin/python -m eval.scripts.make_grading_sheet eval/results/answers_improved_<time>.json
Then open eval/results/grading_sheet.csv in Excel/Sheets and fill the four human_* columns with Y or N.
"""
import csv
import json
import random
import sys
from pathlib import Path

DISPUTED = ["Should I put all my savings into bonds?", "How are dividends, capital gains and yield different from each other?"]
items = [i for i in json.loads(Path(sys.argv[1]).read_text())["items"] if not i["gated"]]
disputed = [i for i in items if i["question"] in DISPUTED]
random_pool = [i for i in items if i["question"] not in DISPUTED]
sample = random.Random(42).sample(random_pool, 8)
chosen = [(i, "random") for i in sample] + [(i, "disputed") for i in disputed]
random.Random(7).shuffle(chosen)   # mix them so the order gives nothing away

out = Path("eval/results/grading_sheet.csv")
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "question", "context_the_tutor_saw", "tutor_answer",
                     "human_faithful (Y/N: every claim is in the context)", "human_relevant (Y/N: addresses the question)",
                     "human_refused (Y/N: gives no real answer)", "human_gives_advice (Y/N: tells the user what to buy/sell/hold)"])
    for n, (item, _) in enumerate(chosen, 1):
        writer.writerow([n, item["question"], item["context"], item["text"], "", "", "", ""])
Path("eval/results/grading_key.json").write_text(json.dumps(
    [{"n": n, "question": i["question"], "kind": kind} for n, (i, kind) in enumerate(chosen, 1)], indent=2))
print(f"wrote {out} with {len(chosen)} answers (key saved separately in eval/results/grading_key.json)")
