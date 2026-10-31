"""Split documents into chunks.

Two methods:
- Fixed-size: slide a window of `size` characters along the text, moving forward by
  (size - overlap) each step. The overlap repeats the end of one chunk at the start of the
  next, so a sentence cut at a boundary still appears whole in one of the two chunks.
- Structure-aware: cut at headings first, then at lines, then at sentences, only as needed.

Sizes are in CHARACTERS (config.yaml). In English one token is about 4 characters.
"""
import re
from dataclasses import dataclass

from src.rag.loader import Document

Span = tuple[int, int]   # (start, end) character positions in the article text


@dataclass
class Chunk:
    chunk_id: str    # "<doc id>#<n>", e.g. "stocks-basics#2"
    doc_id: str
    text: str
    metadata: dict   # copied from the article, so each chunk can be filtered and cited
    index: int       # position within the article
    start: int       # character positions in the article text
    end: int


def chunk_fixed(doc: Document, size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be >= 0 and < size (got size={size}, overlap={overlap})")
    chunks = []
    start = 0
    step = size - overlap
    while start < len(doc.text):
        end = min(start + size, len(doc.text))
        index = len(chunks)
        chunks.append(Chunk(f"{doc.id}#{index}", doc.id, doc.text[start:end], dict(doc.metadata),
                            index, start, end))
        if end == len(doc.text):
            break   # the last window reached the end; don't emit a tail that is all overlap
        start += step
    return chunks


# --------------------------------------------------------------------------------------
# Structure-aware chunking: split at headings first, then by size only when needed.
# --------------------------------------------------------------------------------------

def _split_into_lines(text: str, start: int, end: int) -> list[Span]:
    """Span of each non-blank line in text[start:end]."""
    lines = []
    line_start = start
    while line_start < end:
        newline = text.find("\n", line_start, end)
        line_end = end if newline == -1 else newline
        if text[line_start:line_end].strip():
            lines.append((line_start, line_end))
        line_start = line_end + 1
    return lines


def _split_into_sentences(text: str, start: int, end: int) -> list[Span]:
    """Split one long line into sentences."""
    boundaries = [start]
    boundaries += [start + match.end() for match in re.finditer(r"(?<=[.!?])\s+", text[start:end])]
    boundaries.append(end)
    sentences = []
    for sentence_start, next_start in zip(boundaries, boundaries[1:]):
        sentence_end = sentence_start + len(text[sentence_start:next_start].rstrip())
        if sentence_end > sentence_start:
            sentences.append((sentence_start, sentence_end))
    return sentences


def _is_heading_only(text: str, span: Span) -> bool:
    piece = text[span[0]:span[1]]
    return piece.startswith("#") and "\n" not in piece


def _split_into_pieces(text: str, section: Span, size: int) -> list[Span]:
    """Break a section that is too long into pieces that each fit in `size`."""
    pieces = []
    for line_start, line_end in _split_into_lines(text, *section):
        if line_end - line_start <= size:
            pieces.append((line_start, line_end))
            continue
        for sentence_start, sentence_end in _split_into_sentences(text, line_start, line_end):
            if sentence_end - sentence_start <= size:
                pieces.append((sentence_start, sentence_end))
            else:   # last resort: a single sentence longer than `size`
                for cut in range(sentence_start, sentence_end, size):
                    pieces.append((cut, min(cut + size, sentence_end)))
    return pieces


def _pack_pieces(text: str, pieces: list[Span], size: int) -> list[Span]:
    """Greedily join neighbouring pieces until adding another would exceed `size`."""
    packed = []
    current = None
    for piece in pieces:
        if current is None:
            current = piece
        elif piece[1] - current[0] <= size or _is_heading_only(text, current):
            # A heading is never left alone: it joins the line it introduces
            # (so `size` is a soft cap by at most one heading line).
            current = (current[0], piece[1])
        else:
            packed.append(current)
            current = piece
    if current is not None:
        packed.append(current)
    return packed


def chunk_structured(doc: Document, size: int) -> list[Chunk]:
    text = doc.text
    heading_starts = [match.start() for match in re.finditer(r"^#{1,6}\s", text, re.M)]
    section_starts = sorted({0, *heading_starts})
    section_ends = section_starts[1:] + [len(text)]

    spans = []   # (start, end, section heading) for every chunk
    for section_start, section_end in zip(section_starts, section_ends):
        section_end = section_start + len(text[section_start:section_end].rstrip())
        if section_end <= section_start:
            continue
        first_line = text[section_start:section_end].split("\n", 1)[0]
        heading = first_line.lstrip("# ").strip() if first_line.startswith("#") else ""

        section = (section_start, section_end)
        if section_end - section_start <= size:   # whole section fits: one clean chunk
            chunk_spans = [section]
        else:
            chunk_spans = _pack_pieces(text, _split_into_pieces(text, section, size), size)
        spans += [(start, end, heading) for start, end in chunk_spans]

    return [Chunk(f"{doc.id}#{index}", doc.id, text[start:end], {**doc.metadata, "section": heading},
                  index, start, end)
            for index, (start, end, heading) in enumerate(spans)]
