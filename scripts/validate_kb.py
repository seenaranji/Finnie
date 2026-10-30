"""Validate knowledge-base articles and print a coverage table.

Run:  .venv/bin/python -m scripts.validate_kb [directory]      (default: knowledge_base)
Exit code is 1 if any article has a problem, so it can be used in tests/CI.

Handles .md articles (front matter) and .pdf files (sidecar .meta.yaml + extracted text).
Checks per article: required front-matter fields present, category and difficulty
are allowed values, source_url looks like a URL, last_updated is a date, and the
body is long enough to be useful.
"""
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import frontmatter
import yaml
from pypdf import PdfReader

CATEGORIES = ["basics", "stocks", "bonds", "etfs", "mutual_funds", "retirement_accounts",
              "taxes", "risk", "portfolio", "planning", "markets"]
DIFFICULTIES = ["beginner", "intermediate", "advanced"]
REQUIRED = ["title", "category", "difficulty", "source_name", "source_url", "last_updated"]
MIN_BODY_CHARS = 300


def read_document(path: Path) -> tuple[dict, str, list[str]]:
    """Return (metadata, text, problems found while reading) for a .md or .pdf file.

    .md  : metadata is the front matter at the top of the file.
    .pdf : PDFs have built-in metadata (Title, Author, Subject, Keywords, dates) but not
           our custom fields (category, difficulty, source_url), and in practice those
           built-ins are often blank or wrong. So ours live in a sidecar file with the
           same name: guide.pdf + guide.meta.yaml. The text is extracted with pypdf.
    """
    if path.suffix == ".md":
        post = frontmatter.load(path)
        return post.metadata, post.content, []

    problems = []
    sidecar = path.with_suffix(".meta.yaml")
    meta = {}
    if sidecar.exists():
        meta = yaml.safe_load(sidecar.read_text()) or {}
    else:
        problems.append(f"missing sidecar metadata file: {sidecar.name}")
    try:
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages).strip()
    except Exception as e:  # corrupt or encrypted PDF
        return meta, "", problems + [f"could not read PDF: {type(e).__name__}: {e}"]
    if not text:
        problems.append("no extractable text (scanned/image-only PDF? needs OCR)")
    return meta, text, problems


def check(path: Path) -> tuple[list[str], int, str | None]:
    """Return (problems, body length, category) for one article."""
    meta, body, problems = read_document(path)
    problems += [f"missing field: {f}" for f in REQUIRED if not meta.get(f)]
    cat = meta.get("category")
    if cat and cat not in CATEGORIES:
        problems.append(f"unknown category: {cat!r}")
    if meta.get("difficulty") and meta["difficulty"] not in DIFFICULTIES:
        problems.append(f"unknown difficulty: {meta['difficulty']!r}")
    url = meta.get("source_url")
    if url and not str(url).startswith("http"):
        problems.append(f"source_url is not a URL: {url!r}")
    updated = meta.get("last_updated")
    if updated and not isinstance(updated, date):
        problems.append(f"last_updated is not a date: {updated!r}")
    if body and len(body) < MIN_BODY_CHARS:  # empty PDF text already reported above
        problems.append(f"body too short ({len(body)} chars, min {MIN_BODY_CHARS})")
    elif not body and path.suffix == ".md":
        problems.append("body is empty")
    return problems, len(body), cat


def main(directory: str) -> int:
    files = sorted(p for p in Path(directory).iterdir() if p.suffix in (".md", ".pdf"))
    per_cat, lengths, complete, failed = Counter(), [], 0, 0
    for f in files:
        problems, length, cat = check(f)
        lengths.append(length)
        if cat in CATEGORIES:
            per_cat[cat] += 1
        if problems:
            failed += 1
            print(f"PROBLEM  {f.name}")
            for p in problems:
                print(f"           - {p}")
        else:
            complete += 1
            print(f"ok       {f.name}")

    print(f"\n{'category':22} {'articles':>8}")
    for c in CATEGORIES:
        print(f"{c:22} {per_cat[c]:>8}{'   <-- EMPTY' if per_cat[c] == 0 else ''}")
    n = len(files)
    if n:
        print(f"\narticles: {n} | valid: {complete} ({100*complete/n:.0f}% complete metadata) "
              f"| avg body: {sum(lengths)/n:.0f} chars")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "knowledge_base"))
