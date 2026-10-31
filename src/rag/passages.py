"""What retrieval hands to the answer step: passage text plus the information to cite it."""
import re
from dataclasses import dataclass

from src.rag.vector_store import SearchHit


@dataclass
class Citation:
    doc_id: str
    title: str
    section: str
    source_name: str
    source_url: str
    last_updated: str

    def label(self) -> str:
        """Short human-readable reference, e.g. 'ETF Basics > Main risks (Investor.gov, updated 2026-10-30)'."""
        return f"{self.title} > {self.section} ({self.source_name}, updated {self.last_updated})"


@dataclass
class Passage:
    chunk_id: str
    text: str
    similarity: float                  # embedding cosine similarity; NOT a probability
    citation: Citation
    rerank_score: float | None = None  # 0-10 from the reranker, if one ran


def passage_from_hit(hit: SearchHit) -> Passage:
    metadata = hit.metadata
    citation = Citation(
        doc_id=metadata.get("doc_id", ""), title=metadata.get("title", ""), section=metadata.get("section", ""),
        source_name=metadata.get("source_name", ""), source_url=metadata.get("source_url", ""),
        last_updated=metadata.get("last_updated", ""))
    return Passage(hit.chunk_id, hit.text, hit.similarity, citation)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def dedupe_passages(passages: list[Passage], overlap: float) -> tuple[list[Passage], int]:
    """Drop passages that mostly repeat a better-ranked one. Input must be best first.

    Two passages count as duplicates when the shorter one has at least `overlap` of its
    words inside the other (so a chunk that is a slice of a bigger chunk is dropped too).
    Very short passages (under 5 words) are only dropped if their words are identical.
    Returns (kept passages, number dropped).
    """
    kept: list[Passage] = []
    kept_words: list[set[str]] = []
    for passage in passages:
        words = _words(passage.text)
        is_duplicate = False
        for other_words in kept_words:
            shorter = min(len(words), len(other_words))
            shared = len(words & other_words)
            if shorter < 5:
                is_duplicate = words == other_words
            else:
                is_duplicate = shared / shorter >= overlap
            if is_duplicate:
                break
        if not is_duplicate:
            kept.append(passage)
            kept_words.append(words)
    return kept, len(passages) - len(kept)
