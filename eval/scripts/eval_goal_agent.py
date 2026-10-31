"""Step 17: evaluate the Goal Planning agent. Code checks only (no LLM judge).

PART 1  Extraction accuracy: 20 hand-labelled phrasings (eval/datasets/goals.jsonl), four fields each = 80 fields, for both models.
          correct = same value, or null when the user did not say. The two ways to be wrong are different:
          HALLUCINATED = the model filled a field the user never gave (dangerous: a silent wrong plan)
          MISSED       = the model left null what the user did say (harmless: the agent just asks again)
          WRONG VALUE  = both non-null but different
PART 2  Follow-up behaviour: from the extraction alone, does the agent ask for exactly the missing fields, and never ask when nothing is missing?
PART 3  Explanations for the complete requests (real plan, default agent = answer-model extraction, real explanation model): numeric faithfulness, three scenarios shown, assumptions stated,
          "not a prediction" stated, flags stated when present, advice leak, within 200 words, and one SEMANTIC check for a mistake the
          number check cannot see: calling the future-dollar amount a value "in today's dollars".
PART 4  Canaries for the number checker.

Run:  .venv/bin/python -m eval.scripts.eval_goal_agent
"""
import json
import re
from datetime import date, datetime
from pathlib import Path

from eval.scripts.eval_market_numbers import grade, numbers_in
from eval.scripts.eval_portfolio_agent import _DIRECTIVE
from src.agents.goal_agent import GoalPlanningAgent
from src.agents.goal_extraction import FIELDS, extract_goal_fields
from src.core.config import load_settings
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import summarize

DATASET = Path("eval/datasets/goals.jsonl")
MODELS = ["router_model", "answer_model"]
_SCENARIOS = ("conservative", "expected", "optimistic")
_ASSUMPTION = re.compile(r"assum", re.I)
_NOT_PREDICTION = re.compile(r"not (a )?(predict|forecast|guarantee)|aren't (predict|forecast|guarantee)|isn't (a )?(predict|forecast|guarantee)|not guaranteed", re.I)
_FLAG_STATED = re.compile(r"very large|large commitment|unrealistic|stretch|hard to|difficult|ambitious|very uncertain|far from these scenarios", re.I)   # "uncertain" added after reading #11
_TODAYS_DOLLARS = re.compile(r"today'?s dollars|today'?s money|current dollars|in today", re.I)


def same(expected, got) -> bool:
    if expected is None or got is None:
        return expected is None and got is None
    if isinstance(expected, str) or isinstance(got, str):
        return expected == got
    return abs(expected - got) <= 1e-6 * max(1.0, abs(expected))


def payload_numbers(obj) -> list[float]:
    """Every number the model was shown: numeric values AND numbers written inside strings (the flags)."""
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [v for v, _ in numbers_in(obj)]
    if isinstance(obj, dict):
        return [x for value in obj.values() for x in payload_numbers(value)]
    if isinstance(obj, list):
        return [x for value in obj for x in payload_numbers(value)]
    return []


def main() -> None:
    setup_logging("WARNING")
    settings = load_settings()
    llm = LLMClient(settings)
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line]
    report: dict = {}

    print("=" * 100 + "\nPART 1. EXTRACTION ACCURACY (20 phrasings x 4 fields = 80 fields)\n" + "=" * 100)
    extraction: dict[str, list] = {}
    for model in MODELS:
        results = []
        for row in rows:
            state, result = extract_goal_fields(llm, row["text"], None, date.today(), model)
            results.append((row, state, result))
        extraction[model] = results
        correct = hallucinated = missed = wrong = 0
        per_field = {f: 0 for f in FIELDS}
        exact = 0
        problems = []
        for row, state, _ in results:
            item_ok = True
            for f in FIELDS:
                expected, got = row[f], getattr(state, f)
                if same(expected, got):
                    correct += 1
                    per_field[f] += 1
                    continue
                item_ok = False
                kind = "HALLUCINATED" if expected is None else "MISSED" if got is None else "WRONG VALUE"
                hallucinated += kind == "HALLUCINATED"; missed += kind == "MISSED"; wrong += kind == "WRONG VALUE"
                problems.append(f"   #{row['id']:2d} {f}: expected {expected!r}, got {got!r}  [{kind}]  {row['text'][:70]!r}")
            exact += item_ok
        latencies = [r.latency_s for _, _, r in results]
        cost = sum(r.cost_usd for _, _, r in results)
        print(f"\n  MODEL {model} ({results[0][2].model})")
        print(f"   fields correct: {correct}/80 = {correct / 80:.1%}   items with all 4 fields right: {exact}/20")
        print(f"   per field: { {f: f'{n}/20' for f, n in per_field.items()} }")
        print(f"   errors: hallucinated {hallucinated}, missed {missed}, wrong value {wrong}")
        print(f"   latency per extraction: {summarize(latencies)}   total cost ${cost:.5f}")
        for line in problems:
            print(line)
        report[model] = {"correct": correct, "hallucinated": hallucinated, "missed": missed, "wrong": wrong, "exact_items": exact, "cost": cost}

    print("\n" + "=" * 100 + "\nPART 2. FOLLOW-UP BEHAVIOUR (from the extraction alone, no plan made)\n" + "=" * 100)
    for model in MODELS:
        incomplete = [(row, state) for row, state, _ in extraction[model] if any(row[f] is None for f in FIELDS)]
        complete = [(row, state) for row, state, _ in extraction[model] if all(row[f] is not None for f in FIELDS)]
        asked_exactly = sum(set(state.missing()) == {f for f in FIELDS if row[f] is None} for row, state in incomplete)
        asked_something = sum(bool(state.missing()) for _, state in incomplete)
        false_asks = sum(bool(state.missing()) for _, state in complete)
        print(f"  {model:13} incomplete requests: asks for something {asked_something}/{len(incomplete)}, asks for exactly the right fields {asked_exactly}/{len(incomplete)};"
              f"   complete requests: asks needlessly {false_asks}/{len(complete)}")
        report[model]["follow_up"] = {"incomplete": len(incomplete), "exact": asked_exactly, "false_asks": false_asks, "complete": len(complete)}

    print("\n" + "=" * 100 + "\nPART 3. EXPLANATIONS for the complete requests (default agent: answer-model extraction, real plan, real explanation)\n" + "=" * 100)
    agent = GoalPlanningAgent(llm, settings.goal_planning)
    explained = []
    for row in rows:
        if any(row[f] is None for f in FIELDS):
            continue
        turn = agent.run(row["text"])
        if not turn.plan_made:
            print(f"  #{row['id']:2d} NO PLAN MADE: {turn.response.text[:100]!r}")
            continue
        meta, text = turn.response.metadata, turn.response.text
        payload = meta["metrics"]
        numbers = grade(text, row["text"], payload_numbers(payload))
        adjusted = f"{payload['future_dollars_needed_to_match_the_goals_buying_power_today_usd']:,}"
        mislabelled = [s.strip()[:90] for s in re.split(r"(?<=[.!?])\s+|\n", text) if adjusted in s and _TODAYS_DOLLARS.search(s)]
        record = {"id": row["id"], "numbers": numbers["numbers"], "ungrounded": numbers["ungrounded"], "words": len(text.split()),
                  "three_scenarios": all(w in text.lower() for w in _SCENARIOS), "assumptions": bool(_ASSUMPTION.search(text)),
                  "not_prediction": bool(_NOT_PREDICTION.search(text)), "has_flags": bool(payload["flags"]),
                  "flag_stated": bool(_FLAG_STATED.search(text)), "mislabelled_inflation": mislabelled,
                  "rule_flag": bool(meta["output_flagged"]), "strict": [m.group(0) for m in _DIRECTIVE.finditer(text)],
                  "seconds": meta["extraction_seconds"] + meta["latency_s"], "cost": meta["cost_usd"], "text": text}
        explained.append(record)
        issues = []
        if record["ungrounded"]: issues.append(f"UNGROUNDED {record['ungrounded']}")
        if record["words"] > 200: issues.append(f"{record['words']} WORDS")
        if not record["three_scenarios"]: issues.append("MISSING SCENARIO")
        if not record["assumptions"]: issues.append("NO 'assumption'")
        if not record["not_prediction"]: issues.append("NO 'not a prediction'")
        if record["has_flags"] and not record["flag_stated"]: issues.append("FLAG NOT STATED")
        if record["mislabelled_inflation"]: issues.append(f"MISLABELLED INFLATION {record['mislabelled_inflation']}")
        if record["rule_flag"] or record["strict"]: issues.append(f"ADVICE rules={record['rule_flag']} strict={record['strict']}")
        print(f"  #{row['id']:2d} numbers={record['numbers']:2d} words={record['words']:3d} {record['seconds']:5.2f}s  {'; '.join(issues) or 'ok'}")
    n = len(explained)
    print(f"\n  SUMMARY ({n} plans, one run each, temperature 0)")
    total, bad = sum(r["numbers"] for r in explained), sum(len(r["ungrounded"]) for r in explained)
    print(f"   numeric faithfulness: {total - bad}/{total} numbers grounded; {sum(not r['ungrounded'] for r in explained)}/{n} answers with no invented number")
    print(f"   shows all three scenarios: {sum(r['three_scenarios'] for r in explained)}/{n}   states the assumptions: {sum(r['assumptions'] for r in explained)}/{n}   says 'not a prediction': {sum(r['not_prediction'] for r in explained)}/{n}")
    flagged = [r for r in explained if r["has_flags"]]
    print(f"   flags stated when the plan has one: {sum(r['flag_stated'] for r in flagged)}/{len(flagged)}")
    print(f"   mislabelled the inflation-adjusted amount as 'today's dollars': {sum(bool(r['mislabelled_inflation']) for r in explained)}/{n}")
    print(f"   advice leak: Step 2 rules {sum(r['rule_flag'] for r in explained)}/{n}, stricter phrase list {sum(bool(r['strict']) for r in explained)}/{n}")
    print(f"   within 200 words: {sum(r['words'] <= 200 for r in explained)}/{n}   (words {summarize([float(r['words']) for r in explained])})")
    print(f"   end-to-end seconds (extraction + explanation): {summarize([r['seconds'] for r in explained])}   cost per plan: ${sum(r['cost'] for r in explained) / n:.5f}")
    report["explanations"] = explained

    print("\n" + "=" * 100 + "\nPART 4. CANARIES for the number checker\n" + "=" * 100)
    sample = next(r for r in explained if r["id"] == 1)
    row1 = next(r for r in rows if r["id"] == 1)
    turn = agent.run(row1["text"])
    truth = payload_numbers(turn.response.metadata["metrics"])
    for label, text in (("untouched answer", turn.response.text),
                        ("invented statistic added (73.4%)", turn.response.text + "\n\nThere is a 73.4% chance you will reach the goal early."),
                        ("real number altered ($700 -> $750)", turn.response.text.replace("$700", "$750", 1))):
        result = grade(text, row1["text"], truth)
        print(f"  {label:40} ungrounded={result['ungrounded']}  -> {'FLAGGED' if result['ungrounded'] else 'not flagged'}")

    output_file = Path(f"eval/results/goal_agent_{datetime.now():%Y%m%d_%H%M%S}.json")
    output_file.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  saved: {output_file}")


if __name__ == "__main__":
    main()
