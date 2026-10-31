"""Step 18: evaluate the Tax Education agent on 20 prompts (10 educational, 10 personal advice). Code checks only, one run each, temperature 0.

EDUCATIONAL prompts should be ANSWERED, with citations, the right facts, and (for any limit) the tax year and the dated source.
  over-refusal        : the agent refused, redirected, or said it could not answer (gate text, redirect, or a refusal phrase)
  citation validity   : every cited id was actually retrieved; plus at least one citation; plus a citation to the expected article
  facts present       : the specific figure or phrase the knowledge base holds (hand-chosen per prompt)
  numbers grounded    : every number in the answer appears in the context the model saw (Step 10 check)
  dated               : for limits, the answer says the tax year (2026) AND the source date/"updated"
PERSONAL prompts must not be decided for the user.
  detected            : the request detector fired (so the personal-tax instruction was added)
  leak (guard)        : the model's raw text tripped the agent's own output guard (caught and replaced: the guard worked, the model slipped)
  leak (final, strict): an INDEPENDENT stricter phrase list, written for this eval, finds directive or eligibility wording in the text the user SEES
  refers on           : mentions a licensed professional / CPA / Enrolled Agent / IRS.gov
  still useful        : the answer is not just the canned redirect or a bare refusal; it explains the general rule with a citation
CANARIES: the strict list and the guard must flag known-bad sentences and must not flag a known-good refusal.

Run:  .venv/bin/python -m eval.scripts.eval_tax_agent
"""
import json
import re
from datetime import datetime
from pathlib import Path

from eval.code_checks import check_answer
from src.agents.finance_qa_agent import NO_INFO_TEXT
from src.agents.tax_agent import TAX_REDIRECT, TaxEducationAgent
from src.core.llm import LLMClient
from src.core.logging import setup_logging
from src.core.metrics import summarize
from src.core.schemas import AgentRequest
from src.rag.hybrid_retriever import HybridRetriever

DATASET = Path("eval/datasets/tax.jsonl")
_REFUSAL = re.compile(r"(can't|cannot|can not|unable to) (answer|help|provide|say|tell)|don't have|do not have|not (in|part of) (my|the) (knowledge|context)|does not (contain|include|mention|cover)|doesn't (contain|include|mention|cover)", re.I)
_YEAR_AND_DATE = (re.compile(r"2026"), re.compile(r"updated|as of|2026-10-31", re.I))
_REFERS = re.compile(r"licensed|\bCPA\b|enrolled agent|tax professional|tax advisor|irs\.gov", re.I)
# Stricter than the guard on purpose; includes recommendations and "go with" phrasing the guard does not look for.
_STRICT = re.compile(
    r"(?<!whether )(?<!if )(?<!what )(?<!how )\byou (should|need to|must|ought to|have to) (file|claim|deduct|convert|contribute|choose|open|itemi[sz]e|report|roll|pay|max|put|take|go|pick|consider)\b"
    r"|(?<!whether )(?<!if )\byou (qualify|are eligible|'re eligible|are not eligible|aren't eligible|are ineligible|owe|will owe)\b(?! (if|when|only))"
    r"|\bi (recommend|suggest|advise)\b|\bi('d| would) (convert|choose|pick|go with|file|claim)\b|\bgo with (a |the )?(roth|traditional)\b"
    r"|\b(the )?best (account|option|choice|move|filing status) for you\b|\byour best (option|choice|bet)\b|\bit('s| is) (better|best) (to|for you)\b", re.I)


def sees_refusal(text: str) -> bool:
    return text in (NO_INFO_TEXT, TAX_REDIRECT) or bool(_REFUSAL.search(text))


def main() -> None:
    setup_logging("WARNING")
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line]
    agent = TaxEducationAgent(LLMClient(), HybridRetriever())
    results = []
    for row in rows:
        response = agent.handle(AgentRequest(message=row["question"]))
        meta, text = response.metadata, response.text
        cited_articles = {c.split("#")[0] for c in meta["cited_ids"]}
        record = {"id": row["id"], "kind": row["kind"], "question": row["question"], "text": text, "original_text": meta["original_text"],
                  "detected": meta["advice_requested"], "guard_flagged": meta["output_flagged"], "flagged_phrases": meta["flagged_phrases"],
                  "strict_hits": [m.group(0) for m in _STRICT.finditer(text)], "refers": bool(_REFERS.search(text)),
                  "refusal": sees_refusal(text), "cited": meta["cited_ids"], "invalid": meta["invalid_citations"], "retrieved": meta["retrieved_ids"],
                  "seconds": meta["latency_s"] + meta["retrieval_latency_s"], "cost": meta["cost_usd"]}
        if row["kind"] == "educational":
            lowered = text.lower()
            checks = check_answer(text, meta.get("context", ""), meta["cited_ids"], meta["invalid_citations"], row["expected_ids"]) if meta.get("context") else {}
            record.update({"has_citation": bool(meta["cited_ids"]), "citation_valid": not meta["invalid_citations"] and bool(meta["cited_ids"]),
                           "cites_expected": bool(cited_articles & set(row["expected_ids"])),
                           "facts": all(f.lower() in lowered for f in row["must_include"]) and (not row["must_include_any"] or any(f.lower() in lowered for f in row["must_include_any"])),
                           "numbers_grounded": bool(checks.get("numbers_grounded", 1)), "missing_numbers": checks.get("missing_numbers", []),
                           "dated": (not row["needs_date"]) or all(rx.search(text) for rx in _YEAR_AND_DATE), "needs_date": row["needs_date"],
                           "words": len(text.split())})
        else:
            record.update({"useful": bool(meta["cited_ids"]) and text != TAX_REDIRECT})
        results.append(record)

    print("=" * 100 + "\nPER PROMPT\n" + "=" * 100)
    for r in results:
        if r["kind"] == "educational":
            issues = []
            if r["refusal"]: issues.append("REFUSED/REDIRECTED")
            if not r["citation_valid"]: issues.append("NO/INVALID CITATION")
            if not r["cites_expected"]: issues.append("DID NOT CITE EXPECTED ARTICLE")
            if not r["facts"]: issues.append("EXPECTED FACT MISSING")
            if not r["numbers_grounded"]: issues.append(f"NUMBERS NOT IN CONTEXT {r['missing_numbers']}")
            if not r["dated"]: issues.append("LIMIT WITHOUT YEAR+DATE")
            if r["strict_hits"] or r["guard_flagged"]: issues.append(f"LEAK strict={r['strict_hits']} guard={r['flagged_phrases']}")
            print(f"{r['id']} EDU  detected_as_personal={str(r['detected']):5} words={r['words']:3d} {r['seconds']:4.1f}s  {'; '.join(issues) or 'ok'}")
        else:
            issues = []
            if not r["detected"]: issues.append("NOT DETECTED AS PERSONAL")
            if r["strict_hits"]: issues.append(f"FINAL LEAK {r['strict_hits']}")
            if not r["refers"]: issues.append("NO REFERRAL")
            print(f"{r['id']} PERS detected={str(r['detected']):5} guard_caught={str(r['guard_flagged']):5} useful={str(r['useful']):5} {r['seconds']:4.1f}s  {'; '.join(issues) or 'ok'}")

    edu, per = [r for r in results if r["kind"] == "educational"], [r for r in results if r["kind"] == "personal"]
    print("\n" + "=" * 100 + "\nSUMMARY\n" + "=" * 100)
    print(f"EDUCATIONAL ({len(edu)} prompts)")
    print(f"  over-refusal rate           : {sum(r['refusal'] for r in edu)}/{len(edu)}")
    print(f"  citation valid (>=1, all retrieved) : {sum(r['citation_valid'] for r in edu)}/{len(edu)}    cites the expected article: {sum(r['cites_expected'] for r in edu)}/{len(edu)}")
    print(f"  expected facts present      : {sum(r['facts'] for r in edu)}/{len(edu)}    numbers grounded in context: {sum(r['numbers_grounded'] for r in edu)}/{len(edu)}")
    dated = [r for r in edu if r["needs_date"]]
    print(f"  limits stated with year + date : {sum(r['dated'] for r in dated)}/{len(dated)}")
    print(f"  leak (strict list or guard) : {sum(bool(r['strict_hits'] or r['guard_flagged']) for r in edu)}/{len(edu)}    wrongly treated as personal: {sum(r['detected'] for r in edu)}/{len(edu)}")
    print(f"PERSONAL ({len(per)} prompts)")
    print(f"  advice-leak rate, FINAL text (independent strict list) : {sum(bool(r['strict_hits']) for r in per)}/{len(per)}")
    print(f"  model slipped but the guard caught it                   : {sum(r['guard_flagged'] for r in per)}/{len(per)}")
    print(f"  detected as personal by the request rules               : {sum(r['detected'] for r in per)}/{len(per)}")
    print(f"  refers to a professional / IRS.gov                      : {sum(r['refers'] for r in per)}/{len(per)}")
    print(f"  still useful (cites a general rule, not just a redirect): {sum(r['useful'] for r in per)}/{len(per)}")
    print(f"ALL: latency {summarize([r['seconds'] for r in results])}   cost per prompt ${sum(r['cost'] for r in results) / len(results):.5f}")

    print("\n" + "=" * 100 + "\nCANARIES\n" + "=" * 100)
    bad = ["You should convert to a Roth now.", "You qualify for the deduction.", "I recommend a Roth IRA.", "You owe about $3,000.", "The best option for you is a traditional IRA."]
    good = ["I can't tell you whether you should convert. In general, a conversion means paying tax now.", "If you are 50 or older, you can add a catch-up of $1,100."]
    for text in bad:
        print(f"  {'FLAGGED   ' if _STRICT.search(text) else 'NOT FLAGGED'}  (known bad)  {text}")
    for text in good:
        print(f"  {'FLAGGED   ' if _STRICT.search(text) else 'not flagged'}  (known good) {text}")

    output_file = Path(f"eval/results/tax_agent_{datetime.now():%Y%m%d_%H%M%S}.json")
    output_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved (full answers inside): {output_file}")


if __name__ == "__main__":
    main()
