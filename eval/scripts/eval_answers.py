"""Step 10: evaluate Finance Q&A ANSWERS (not just retrieval), as LangSmith experiments.

Two kinds of checks:
  CODE (deterministic, free): citation_valid (every cited id was retrieved), retrieval_hit (expected article was
        retrieved: this separates a retrieval failure from a generation failure), advice_rule_flag (Step 2 regexes).
  JUDGE (LLM, fuzzy):  faithful (no claim unsupported by the context), relevant, refused, states_gap, gives_advice.

Summaries per question type are computed after the run and saved to eval/results/answers_<prompt>_<time>.json.

Run locally (nothing uploaded):  .venv/bin/python -m eval.scripts.eval_answers --prompts weak improved
Upload to LangSmith:             .venv/bin/python -m eval.scripts.eval_answers --prompts weak improved --upload
"""
import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from langsmith import Client, evaluate
from langsmith.schemas import Example

from eval.code_checks import check_answer
from eval.judge import judge_answer
from eval.scripts.eval_retrieval import CachingEmbedder
from src.agents.finance_qa_agent import SYSTEM_PROMPT, FinanceQAAgent
from src.core.config import load_settings
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.schemas import AgentRequest
from src.rag.hybrid_retriever import HybridRetriever

DATASET_FILE = Path("eval/datasets/answers.jsonl")
RESULTS_DIR = Path("eval/results")

# The "weak" prompt has no rules about sources, gaps or citations: it is the before-picture.
WEAK_PROMPT = "You are a finance tutor. Use the context below to answer the user's question."
PROMPTS = {"weak": WEAK_PROMPT, "improved": SYSTEM_PROMPT}

JUDGE_LOG: dict[str, dict] = {}   # question -> judge verdict + context, kept so we can save and read failures


def load_rows() -> list[dict]:
    return [json.loads(line) for line in DATASET_FILE.read_text().splitlines() if line]


def make_target(agent: FinanceQAAgent):
    def target(inputs: dict) -> dict:
        response = agent.handle(AgentRequest(message=inputs["question"]))
        meta = response.metadata
        return {"text": response.text, "gated": str(meta["model"]).startswith("none"),
                "retrieved_ids": meta["retrieved_ids"], "cited_ids": meta["cited_ids"],
                "invalid_citations": meta["invalid_citations"], "context": meta.get("context", ""),
                "output_flagged": meta["output_flagged"], "cost_usd": meta["cost_usd"]}
    return target


def make_evaluator(llm: LLMClient | None):
    """llm=None -> code checks only (the default). With an llm, the LLM judge runs in addition."""
    def answer_quality(outputs: dict, reference_outputs: dict) -> dict:
        results = []
        expected = reference_outputs["expected_ids"]
        if expected:   # CODE: was the right article retrieved?
            hit = any(rid.split("#")[0] in expected for rid in outputs["retrieved_ids"])
            results.append({"key": "retrieval_hit", "score": int(hit)})
        results.append({"key": "advice_rule_flag", "score": int(outputs["output_flagged"])})   # CODE (Step 2 rules)

        question = reference_outputs["question"]
        if outputs["gated"]:   # retrieval returned nothing -> the fixed "I don't have that" text; no judge needed
            results += [{"key": "refused_rule", "score": 1}, {"key": "refused", "score": 1}]
            JUDGE_LOG[question] = {"gated": True, "text": outputs["text"]}
            return {"results": results}

        checks = check_answer(outputs["text"], outputs["context"], outputs["cited_ids"], outputs["invalid_citations"], expected)   # CODE
        results += [{"key": k, "score": v} for k, v in checks.items() if isinstance(v, (int, float)) and v is not None and k != "overlap"]
        results.append({"key": "words_overlap", "score": checks["overlap"]})
        if llm is None:
            JUDGE_LOG[question] = {"gated": False, "text": outputs["text"], "context": outputs["context"],
                                   "cited_ids": outputs["cited_ids"], "checks": checks}
            return {"results": results}

        verdict = judge_answer(llm, question, outputs["context"], outputs["text"])                  # JUDGE (only with --judge)
        claims = verdict["claims"]
        unsupported = [c["claim"] for c in claims if not c.get("supported")]
        JUDGE_LOG[question] = {"gated": False, "text": outputs["text"], "context": outputs["context"],
                               "cited_ids": outputs["cited_ids"], "checks": checks, "verdict": verdict}
        results += [
            {"key": "faithful", "score": int(not unsupported), "comment": f"{verdict.get('reason', '')} UNSUPPORTED: {unsupported}"},
            {"key": "claim_support", "score": (len(claims) - len(unsupported)) / len(claims) if claims else 1.0},
            {"key": "relevant", "score": int(bool(verdict.get("relevant")))},
            {"key": "refused", "score": int(bool(verdict.get("refused")))},
            {"key": "states_gap", "score": int(bool(verdict.get("states_gap")))},
            {"key": "gives_advice", "score": int(bool(verdict.get("gives_advice")))},
        ]
        return {"results": results}
    return answer_quality


def rate(frame: pd.DataFrame, column: str) -> tuple[str, float | None]:
    if f"feedback.{column}" not in frame:
        return "n/a", None
    values = pd.to_numeric(frame[f"feedback.{column}"], errors="coerce").dropna()
    return (f"{int(values.sum())}/{len(values)}", values.mean()) if len(values) else ("n/a", None)


def summarize(df: pd.DataFrame, use_judge: bool) -> dict:
    """Each metric is computed only on the question types where it makes sense. Code checks always; judge rows only with --judge."""
    by = lambda *cats: df[df["category"].isin(cats)]
    answered = df[pd.to_numeric(df["feedback.refused_rule"], errors="coerce") == 0]   # the gate or a refusal phrase means no real answer
    spec = {
        "CODE  numbers all in context (answered)": (answered, "numbers_grounded"),
        "CODE  words mostly in context (answered)": (answered, "words_grounded"),
        "CODE  every sentence cited (answered)": (answered, "every_sentence_cited"),
        "CODE  has a citation (answered)": (answered, "has_citation"),
        "CODE  citation valid (answered)": (answered, "citation_valid"),
        "CODE  cites the expected article (answerable)": (by("answerable", "jargon")[lambda f: pd.to_numeric(f["feedback.refused_rule"], errors="coerce") == 0], "cited_expected"),
        "CODE  within 150 words (answered)": (answered, "length_ok"),
        "CODE  correct refusal (unanswerable)": (by("unanswerable"), "refused_rule"),
        "CODE  over-refusal (answerable + jargon)": (by("answerable", "jargon"), "refused_rule"),
        "CODE  states the gap, phrase match (partial)": (by("partial"), "states_gap_rule"),
        "CODE  advice leak, Step 2 rules (advice)": (by("advice"), "advice_rule_flag"),
        "CODE  retrieval hit (has expected article)": (df, "retrieval_hit"),
    }
    if use_judge:
        spec.update({
            "JUDGE faithful (all claims supported)": (df, "faithful"),
            "JUDGE claim support (share of claims)": (df, "claim_support"),
            "JUDGE relevant (answerable + jargon)": (by("answerable", "jargon"), "relevant"),
            "JUDGE correct refusal (unanswerable)": (by("unanswerable"), "refused"),
            "JUDGE over-refusal (answerable + jargon)": (by("answerable", "jargon"), "refused"),
            "JUDGE states the gap (partial)": (by("partial"), "states_gap"),
            "JUDGE advice leak (advice)": (by("advice"), "gives_advice"),
        })
    return {name: dict(zip(("count", "rate"), rate(frame, col))) for name, (frame, col) in spec.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="*", default=["weak", "improved"], choices=list(PROMPTS))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--judge", action="store_true", help="also run the LLM judge (paused by default: code checks only)")
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default="dense",
                        help="hybrid = BM25 + dense with the Step 8 gate (threshold 0.465, bm25 >= 3.3)")
    args = parser.parse_args()

    setup_logging("WARNING")
    settings = load_settings()
    rows = load_rows()
    category_of = {r["question"]: r["category"] for r in rows}
    examples = [{"inputs": {"question": r["question"]},
                 "outputs": {"question": r["question"], "category": r["category"], "expected_ids": r["expected_ids"]},
                 "metadata": {"id": r["id"], "category": r["category"]}} for r in rows]

    llm = LLMClient(settings)
    retriever = HybridRetriever(settings=settings)
    retriever.dense.embedder = CachingEmbedder(retriever.dense.embedder)

    data, client = [Example(id=uuid.uuid4(), **e) for e in examples], None
    if args.upload:
        if not os.environ.get("LANGSMITH_API_KEY"):
            raise SystemExit("LANGSMITH_API_KEY is not set (see .env.example), or run without --upload.")
        client = Client()
        if not client.has_dataset(dataset_name="finnie-answers"):
            client.create_dataset("finnie-answers", description="Finnie Q&A answer-quality questions (30)")
            client.create_examples(dataset_name="finnie-answers", examples=examples)
            print("created LangSmith dataset 'finnie-answers' with 30 examples")
        data = "finnie-answers"

    retriever.settings.rag.retrieval_mode = args.retrieval
    if args.retrieval == "hybrid":
        retriever.settings.rag.bm25_min_score = 3.3
    suffix = "" if args.retrieval == "dense" else f"-{args.retrieval}"

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    for name in args.prompts:
        JUDGE_LOG.clear()
        agent = FinanceQAAgent(llm, retriever)
        agent.temperature = 0          # repeatability
        agent.system_prompt = PROMPTS[name]
        results = evaluate(make_target(agent), data=data, evaluators=[make_evaluator(llm if args.judge else None)],
                           experiment_prefix=f"answers-{name}{suffix}", metadata={"prompt": name, "retrieval": args.retrieval, "git_commit": commit},
                           client=client, upload_results=args.upload, max_concurrency=4)
        df = results.to_pandas()
        if df.empty:
            raise SystemExit("0 examples ran; check the data argument")
        df["category"] = df["inputs.question"].map(category_of)
        summary = summarize(df, args.judge)
        agent_cost = pd.to_numeric(df.get("outputs.cost_usd"), errors="coerce").sum()
        judge_cost = sum(v["verdict"]["judge_cost_usd"] for v in JUDGE_LOG.values() if "verdict" in v)

        mode = "code checks + LLM judge" if args.judge else "code checks only (LLM judge paused)"
        print(f"\n{'=' * 90}\nPROMPT = {name}, retrieval = {args.retrieval}, {mode}   ({len(df)} questions; agent cost ${agent_cost:.4f}, judge cost ${judge_cost:.4f})\n{'=' * 90}")
        for metric, s in summary.items():
            percent = "" if s["rate"] is None else f"{s['rate']:.0%}"
            print(f"  {metric:48} {s['count']:>7}   {percent}")

        print("\n  FAILURES:")
        for _, row in df.iterrows():
            question, category = row["inputs.question"], row["category"]
            log = JUDGE_LOG.get(question, {})
            verdict, checks = log.get("verdict", {}), log.get("checks", {})
            problems = []
            if category in ("answerable", "jargon") and (row.get("feedback.refused_rule") == 1):
                problems.append("OVER-REFUSED")
            if category == "unanswerable" and row.get("feedback.refused_rule") != 1:
                problems.append("ANSWERED AN UNANSWERABLE QUESTION")
            if checks.get("missing_numbers"):
                problems.append(f"NUMBERS NOT IN CONTEXT: {checks['missing_numbers']}")
            if checks and not checks["words_grounded"]:
                problems.append(f"LOW WORD OVERLAP WITH CONTEXT: {checks['overlap']:.0%}")
            if checks and not checks["every_sentence_cited"]:
                problems.append(f"UNCITED SENTENCES: {len(checks['uncited_sentences'])}")
            if checks and not checks["length_ok"]:
                problems.append("OVER 150 WORDS")
            if row.get("feedback.faithful") == 0:
                problems.append(f"JUDGE: UNSUPPORTED CLAIMS: {[c['claim'] for c in verdict.get('claims', []) if not c.get('supported')]}")
            if row.get("feedback.citation_valid") == 0:
                problems.append("INVALID CITATION")
            if category == "partial" and checks and not checks["states_gap_rule"]:
                problems.append("DID NOT STATE THE GAP (phrase match)")
            if category == "advice" and (row.get("feedback.advice_rule_flag") == 1 or row.get("feedback.gives_advice") == 1):
                problems.append("ADVICE LEAK")
            if problems:
                where = "retrieval MISSED the article" if row.get("feedback.retrieval_hit") == 0 else "retrieval ok"
                print(f"   [{category}] {question}\n      {'; '.join(problems)}   ({where})")

        rows_out = [{"question": q, "category": category_of[q], **log} for q, log in JUDGE_LOG.items()]
        output_file = RESULTS_DIR / f"answers_{name}{suffix}_{datetime.now():%Y%m%d_%H%M%S}.json"
        output_file.write_text(json.dumps({"prompt": name, "git_commit": commit, "summary": summary, "agent_cost_usd": agent_cost,
                                           "judge_cost_usd": judge_cost, "judge_model": settings.llm.answer_model,
                                           "items": rows_out}, indent=2, default=str))
        print(f"\n  saved: {output_file}")


if __name__ == "__main__":
    main()
