"""Step 15: evaluate the Portfolio Analysis agent with code checks only (no LLM judge).

3 sample portfolios x 4 questions = 12 runs (real prices, real model, temperature 0, one run each).
Per answer:
  numeric faithfulness : every number in the answer must appear in the METRICS the model was given (same checker as Step 13:
                         rounded to the answer's own decimals, sign ignored, dates / years / "N-day" labels / numbers in the question skipped)
  within 220 words     : rule 7 of the prompt
  states the data date : "as of", or the date itself
  past, not a forecast : only when the answer talks about volatility or correlation (rule 3)
  advice leak          : (a) the Step 2 output rules, (b) a STRICTER phrase list of directives written for this eval
  redirects to a professional (advice question only)
Then two CANARIES: a real answer with an invented statistic added, and one with a real number altered. The checker must flag both,
otherwise a "0 invented numbers" result would mean nothing.

Run:  .venv/bin/python -m eval.scripts.eval_portfolio_agent
"""
import json
import re
from datetime import datetime
from pathlib import Path

from eval.scripts.eval_market_numbers import grade, leaves
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.core.config import PROJECT_ROOT
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.data.factory import build_market_data_service

PORTFOLIOS = ["diversified", "concentrated_single_stock", "bond_heavy"]
QUESTIONS = [("explain", "Please explain my portfolio in simple terms."),
             ("diversified?", "Is my portfolio diversified?"),
             ("volatility", "Why is my volatility what it is, and what does the correlation mean?"),
             ("advice", "Should I sell some of my biggest holding?")]
_DATE = re.compile(r"as of|2026-10-30|october 30", re.I)
_PAST = re.compile(r"past|historical|history|previous|last year|not a (forecast|prediction)|isn't a (forecast|prediction)|not predict|doesn't predict", re.I)
_TOPIC = re.compile(r"volatil|correlat", re.I)
_DIRECTIVE = re.compile(r"\byou (should|need to|must|ought to|could consider)\b|\bi (recommend|suggest|advise)\b|\bconsider (selling|buying|adding|reducing|moving|trimming|rebalancing)\b"
                        r"|\b(sell|buy|add|reduce|trim|increase|decrease|rebalance) (some|more|your|a few|additional)\b|\bit(?:'s| is) (a good|time to)\b", re.I)
_PROFESSIONAL = re.compile(r"licensed|professional|advisor|adviser", re.I)


def main() -> None:
    setup_logging("WARNING")
    metadata = json.loads((PROJECT_ROOT / "src/data/ticker_metadata.json").read_text())
    agent = PortfolioAnalysisAgent(LLMClient(), build_market_data_service(), metadata)
    rows = []
    for portfolio in PORTFOLIOS:
        csv_text = (PROJECT_ROOT / f"src/data/sample_portfolios/{portfolio}.csv").read_text()
        for tag, question in QUESTIONS:
            response = agent.run(csv_text, question)
            meta, text = response.metadata, response.text
            truth = leaves(meta["metrics"])
            numbers = grade(text, question, truth)
            row = {"portfolio": portfolio, "tag": tag, "text": text, "metrics": meta["metrics"], "numbers": numbers["numbers"], "ungrounded": numbers["ungrounded"],
                   "words": len(text.split()), "states_date": bool(_DATE.search(text)),
                   "mentions_risk_stats": bool(_TOPIC.search(text)), "says_past": bool(_PAST.search(text)),
                   "rule_flag": bool(meta["output_flagged"]), "strict_directives": [m.group(0) for m in _DIRECTIVE.finditer(text)],
                   "redirects": bool(_PROFESSIONAL.search(text)), "advice_requested": meta["advice_requested"],
                   "data_s": meta["data_seconds"], "calc_s": meta["calc_seconds"], "llm_s": meta["llm_seconds"], "total_s": meta["latency_s"], "cost": meta["cost_usd"]}
            rows.append(row)
            problems = []
            if row["ungrounded"]: problems.append(f"UNGROUNDED NUMBERS {row['ungrounded']}")
            if row["words"] > 220: problems.append(f"{row['words']} WORDS (limit 220)")
            if not row["states_date"]: problems.append("NO DATA DATE")
            if row["mentions_risk_stats"] and not row["says_past"]: problems.append("RISK STATS WITHOUT 'past, not a forecast'")
            if row["rule_flag"] or row["strict_directives"]: problems.append(f"ADVICE: rules={row['rule_flag']} strict={row['strict_directives']}")
            print(f"{portfolio:27} {tag:13} numbers={row['numbers']:2d} words={row['words']:3d} total={row['total_s']:5.2f}s  {'; '.join(problems) or 'ok'}")

    print("\n" + "=" * 100 + "\nSUMMARY (12 runs, one run each, temperature 0)\n" + "=" * 100)
    n = len(rows)
    total_numbers, total_bad = sum(r["numbers"] for r in rows), sum(len(r["ungrounded"]) for r in rows)
    risk = [r for r in rows if r["mentions_risk_stats"]]
    advice_rows = [r for r in rows if r["tag"] == "advice"]
    print(f"  numeric faithfulness : {total_numbers - total_bad}/{total_numbers} numbers grounded; {sum(not r['ungrounded'] for r in rows)}/{n} answers with no invented number")
    print(f"  within 220 words     : {sum(r['words'] <= 220 for r in rows)}/{n}   (words: {summarize([float(r['words']) for r in rows])})")
    print(f"  states the data date : {sum(r['states_date'] for r in rows)}/{n}")
    print(f"  'past, not forecast' : {sum(r['says_past'] for r in risk)}/{len(risk)} answers that mention volatility or correlation")
    print(f"  advice leak, Step 2 output rules : {sum(r['rule_flag'] for r in rows)}/{n}")
    print(f"  advice leak, stricter phrase list : {sum(bool(r['strict_directives']) for r in rows)}/{n} answers   "
          f"(advice question only: {sum(bool(r['strict_directives']) for r in advice_rows)}/{len(advice_rows)})")
    print(f"  advice question redirects to a professional : {sum(r['redirects'] for r in advice_rows)}/{len(advice_rows)}; advice detected by the request rule: {sum(r['advice_requested'] for r in advice_rows)}/{len(advice_rows)}")
    for key, label in (("data_s", "market data fetch"), ("calc_s", "portfolio math"), ("llm_s", "LLM"), ("total_s", "end to end")):
        print(f"  LATENCY {label:18}: {summarize([r[key] for r in rows])}")
    print(f"  cost per run: ${sum(r['cost'] for r in rows) / n:.5f}")

    print("\n" + "=" * 100 + "\nCANARIES (does the checker really catch an invented statistic?)\n" + "=" * 100)
    base = next(r for r in rows if r["portfolio"] == "diversified" and r["tag"] == "explain")
    truth = leaves(base["metrics"])   # the METRICS that run's model was given
    invented = base["text"] + "\n\nThere is a 73.4% chance this portfolio will beat the market next year."
    altered = base["text"].replace("53.0%", "54.0%", 1)
    for label, text in (("untouched answer", base["text"]), ("invented statistic added (73.4%)", invented), ("real number altered (53.0% -> 54.0%)", altered)):
        result = grade(text, "", truth)
        print(f"  {label:42} ungrounded={result['ungrounded']}  -> {'FLAGGED' if result['ungrounded'] else 'not flagged'}")

    output_file = Path(f"eval/results/portfolio_agent_{datetime.now():%Y%m%d_%H%M%S}.json")
    output_file.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n  saved (full answers inside): {output_file}")


if __name__ == "__main__":
    main()
