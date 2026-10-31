"""Step 4: fixed-size vs structure-aware chunking, side by side.

Run:  .venv/bin/python -m scripts.step4_compare
"""
import json
import re
from datetime import datetime
from pathlib import Path

from src.rag.chunker import chunk_fixed, chunk_structured
from src.rag.loader import load_documents

SIZES = [200, 500, 1000]
HEADING_PATTERN = r"^#{1,6}\s"
documents = load_documents()


def split_sentences(text):
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [piece.strip() for piece in pieces if len(piece.strip()) > 10]


def ends_mid_sentence(chunk, doc):
    """True if the chunk stops where the article's sentence or line does not."""
    last_character = chunk.text.rstrip()[-1:]
    next_character = doc.text[chunk.end:chunk.end + 1]
    return not (last_character in ".!?" or next_character in ("\n", ""))


def starts_mid_word(chunk, doc):
    return (chunk.start > 0
            and doc.text[chunk.start - 1].isalnum()
            and doc.text[chunk.start].isalnum())


def measure(method_name, size, chunker):
    """Chunk every article with `chunker` and count the problems the lesson talks about."""
    all_chunks = []
    cut_mid_sentence = 0
    cut_mid_word = 0
    multi_heading = 0
    sentences_total = 0
    sentences_whole = 0

    for doc in documents:
        chunks = chunker(doc)
        all_chunks += chunks
        cut_mid_sentence += sum(ends_mid_sentence(chunk, doc) for chunk in chunks[:-1])
        cut_mid_word += sum(starts_mid_word(chunk, doc) for chunk in chunks)
        multi_heading += sum(len(re.findall(HEADING_PATTERN, chunk.text, re.M)) >= 2
                             for chunk in chunks)
        for sentence in split_sentences(doc.text):
            sentences_total += 1
            sentences_whole += any(sentence in chunk.text for chunk in chunks)

    lengths = [len(chunk.text) for chunk in all_chunks]
    chunk_count = len(all_chunks)
    return {"method": method_name, "size": size, "chunks": chunk_count,
            "avg": round(sum(lengths) / chunk_count), "min": min(lengths), "max": max(lengths),
            "end_mid_sentence_pct": round(100 * cut_mid_sentence / chunk_count, 1),
            "start_mid_word": cut_mid_word, "multi_heading_chunks": multi_heading,
            "sentences_whole_pct": round(100 * sentences_whole / sentences_total, 1)}


results = []
for size in SIZES:
    results.append(measure("fixed(ovlp 50)", size, lambda doc: chunk_fixed(doc, size, 50)))
    results.append(measure("structured", size, lambda doc: chunk_structured(doc, size)))

print(f"{'method':15} {'size':>5} {'chunks':>6} {'avg':>4} {'min':>4} {'max':>5} "
      f"{'end mid-sent':>12} {'start mid-word':>14} {'multi-head':>10} {'sent whole':>10}")
for row in results:
    print(f"{row['method']:15} {row['size']:>5} {row['chunks']:>6} {row['avg']:>4} {row['min']:>4} "
          f"{row['max']:>5} {row['end_mid_sentence_pct']:>11}% {row['start_mid_word']:>14} "
          f"{row['multi_heading_chunks']:>10} {row['sentences_whole_pct']:>9}%")
output_file = Path(f"benchmarks/results/step4_compare_{datetime.now():%Y%m%d_%H%M%S}.json")
output_file.write_text(json.dumps(results, indent=2))
print(f"\nsaved: {output_file}")

for doc_id, size in (("stocks-basics", 500), ("stocks-basics", 200), ("glossary", 500)):
    doc = next(doc for doc in documents if doc.id == doc_id)
    chunks = chunk_structured(doc, size)
    shown = chunks if doc_id == "stocks-basics" else chunks[:3]
    note = "(first 3 shown)" if doc_id == "glossary" else ""
    print(f"\n{'#' * 8} STRUCTURED {doc_id} size={size}: {len(chunks)} chunks {note} {'#' * 8}")
    for chunk in shown:
        print(f"\n--- {chunk.chunk_id} section={chunk.metadata['section']!r} "
              f"({len(chunk.text)} chars) ---\n{chunk.text}")
