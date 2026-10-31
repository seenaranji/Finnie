"""Step 4: my hand-written chunkers vs LangChain's ready-made text splitters.

Run:  .venv/bin/python -m scripts.step4_library_compare
Same metrics as step4_compare. LangGraph itself has no splitters; they live in the
separate langchain-text-splitters package.
"""
import re
from datetime import datetime
import json
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.rag.chunker import chunk_fixed, chunk_structured
from src.rag.loader import load_documents

docs = load_documents()
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def lib_recursive(doc, size, overlap=50):
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [c.page_content for c in sp.create_documents([doc.text])]


def lib_markdown_then_recursive(doc, size):
    # strip_headers=False keeps the heading text inside each chunk, like my chunker does
    sections = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(doc.text)
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=0)
    return [c.page_content for c in sp.split_documents(sections)]


def spans(doc, texts):
    """(start, end) of each chunk text inside the article, found by searching for it."""
    out = []
    for t in texts:
        s = doc.text.find(t)
        out.append((s, s + len(t)) if s >= 0 else (None, None))
    return out


def sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if len(p.strip()) > 10]


def stats(name, size, to_texts):
    n = cuts = midword = multi = whole = total = lost = 0
    lens = []
    for d in docs:
        texts = to_texts(d)
        sp = spans(d, texts)
        n += len(texts)
        lens += [len(t) for t in texts]
        for i, (t, (s, e)) in enumerate(zip(texts, sp)):
            if s is None:
                lost += 1
                continue
            if i < len(texts) - 1 and not (t.rstrip()[-1:] in ".!?" or d.text[e:e + 1] in ("\n", "")):
                cuts += 1
            if s > 0 and d.text[s - 1].isalnum() and d.text[s].isalnum():
                midword += 1
            if len(re.findall(r"^#{1,6}\s", t, re.M)) >= 2:
                multi += 1
        for sent in sentences(d.text):
            total += 1
            whole += any(sent in t for t in texts)
    return {"method": name, "size": size, "chunks": n, "avg": round(sum(lens) / n), "min": min(lens),
            "max": max(lens), "end_mid_sentence_pct": round(100 * cuts / n, 1), "start_mid_word": midword,
            "multi_heading": multi, "sentences_whole_pct": round(100 * whole / total, 1)}


rows = []
for size in (200, 500, 1000):
    rows += [
        stats("mine: fixed(ovlp50)", size, lambda d: [c.text for c in chunk_fixed(d, size, 50)]),
        stats("mine: structured", size, lambda d: [c.text for c in chunk_structured(d, size)]),
        stats("lib: Recursive(ovlp50)", size, lambda d: lib_recursive(d, size)),
        stats("lib: MdHeader+Recursive", size, lambda d: lib_markdown_then_recursive(d, size)),
    ]
print(f"{'method':24} {'size':>5} {'chunks':>6} {'avg':>4} {'min':>4} {'max':>5} {'end mid-sent':>12} "
      f"{'start mid-word':>14} {'multi-head':>10} {'sent whole':>10}")
for r in rows:
    print(f"{r['method']:24} {r['size']:>5} {r['chunks']:>6} {r['avg']:>4} {r['min']:>4} {r['max']:>5} "
          f"{r['end_mid_sentence_pct']:>11}% {r['start_mid_word']:>14} {r['multi_heading']:>10} "
          f"{r['sentences_whole_pct']:>9}%")
out = Path(f"benchmarks/results/step4_library_compare_{datetime.now():%Y%m%d_%H%M%S}.json")
out.write_text(json.dumps(rows, indent=2))
print(f"\nsaved: {out}")

doc = next(d for d in docs if d.id == "stocks-basics")
for name, fn in (("lib: Recursive(ovlp50)", lib_recursive), ("lib: MdHeader+Recursive", lib_markdown_then_recursive)):
    texts = fn(doc, 500)
    print(f"\n{'#' * 8} {name}, stocks-basics, size=500: {len(texts)} chunks {'#' * 8}")
    for i, t in enumerate(texts):
        print(f"\n--- chunk {i} ({len(t)} chars) ---\n{t}")
