"""Keyword (BM25) search over the same chunks the vector store holds.

BM25 scores a chunk higher when it contains the query words (term frequency, with
diminishing returns), when those words are rare across the knowledge base (inverse document
frequency), and it adjusts for chunk length so long chunks don't win by size alone.

Chunk ids and metadata are copied from the chunks, so a keyword hit and a vector hit for the
same chunk share one id. That shared id is what lets RRF recognise "found by both".

The index is built from a snapshot of the chunks. After the knowledge base changes it must be
rebuilt (scripts.index_knowledge_base does it together with the vector index).
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np

from src.rag.chunker import Chunk

# Words so common that they only add noise to a keyword match. Applied to the chunks AND the
# query, so both sides are treated the same way (a mismatch here silently breaks matching).
STOPWORDS = frozenset(
    "a an the is are was were be been of to in on at by for with and or as it its this that "
    "what how do does did i my me you your can".split())

_KEPT_PATTERN = re.compile(r"[a-z0-9]+(?:&[a-z0-9]+|\.\d+)*")   # keeps "s&p" and "0.03" whole


def tokenize(text: str) -> list[str]:
    """Turn text into BM25 tokens. Deliberate rules, each one fixing a naive-tokenizer problem:

    1. lowercase                                    "ETF" == "etf"
    2. "401(k)" / "403(b)" -> "401k" / "403b"        a naive \\w+ split gives "401" + "k"
    3. "s&p" stays one token, "0.03" stays one       a naive split gives "s", "p" and "0", "03"
    4. possessive "'s" dropped, hyphens split        "investor's" -> "investor", "tax-free" -> "tax", "free"
    5. stopwords removed                             "what is HHI" -> ["hhi"]
    6. crude plural fold ("stocks" -> "stock")       no stemmer is installed; consistent on both sides.
       Only for words longer than 3 letters not ending in "ss" ("bonds" -> "bond", "loss" stays).
    """
    text = text.lower()
    text = re.sub(r"\b(\d{3})\s*\(\s*([a-z])\s*\)", r"\1\2", text)
    text = re.sub(r"['’]s\b", "", text)
    tokens = []
    for token in _KEPT_PATTERN.findall(text):
        if token in STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


@dataclass
class KeywordHit:
    chunk_id: str
    text: str
    metadata: dict
    score: float   # BM25 score: unbounded, depends on this corpus. NOT comparable to cosine similarity.


def _matches(actual, wanted) -> bool:
    """Exact match, or membership when `wanted` is {"$in": [...]} (the same operator Chroma uses)."""
    return actual in wanted["$in"] if isinstance(wanted, dict) and "$in" in wanted else actual == wanted


class KeywordIndex:
    def __init__(self, chunk_ids: list[str], texts: list[str], metadatas: list[dict],
                 bm25: bm25s.BM25, fingerprint: str) -> None:
        self.chunk_ids, self.texts, self.metadatas = chunk_ids, texts, metadatas
        self._bm25 = bm25
        self.fingerprint = fingerprint

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @classmethod
    def build(cls, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> "KeywordIndex":
        """k1 = how fast repeated words stop mattering; b = how strongly to punish long chunks."""
        if not chunks:
            raise ValueError("cannot build a keyword index from zero chunks")
        bm25 = bm25s.BM25(k1=k1, b=b)
        bm25.index([tokenize(chunk.text) for chunk in chunks], show_progress=False)
        metadatas = [{**chunk.metadata, "doc_id": chunk.doc_id} for chunk in chunks]   # same shape the store keeps
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        return cls(ids, texts, metadatas, bm25, _fingerprint(ids, texts))

    def search(self, query: str, top_k: int, where: dict | None = None) -> list[KeywordHit]:
        """Best-first chunks with a score above 0. `where` is an exact-match metadata filter,
        the same shape the vector store takes, so both retrievers see the same chunks."""
        tokens = tokenize(query)
        if not tokens:   # bm25s raises IndexError on an empty query
            return []
        scores = self._bm25.get_scores(tokens)
        if where:
            allowed = np.array([all(_matches(meta.get(key), value) for key, value in where.items())
                                for meta in self.metadatas])
            scores = np.where(allowed, scores, 0.0)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [KeywordHit(self.chunk_ids[i], self.texts[i], self.metadatas[i], float(scores[i]))
                for i in order if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        folder = Path(path)
        self._bm25.save(str(folder))
        (folder / "chunks.json").write_text(json.dumps(
            {"fingerprint": self.fingerprint, "chunk_ids": self.chunk_ids,
             "texts": self.texts, "metadatas": self.metadatas}))

    @classmethod
    def load(cls, path: str | Path) -> "KeywordIndex":
        folder = Path(path)
        if not (folder / "chunks.json").exists():
            raise FileNotFoundError(
                f"no keyword index at '{folder}'. Build it with: .venv/bin/python -m scripts.build_keyword_index")
        data = json.loads((folder / "chunks.json").read_text())
        return cls(data["chunk_ids"], data["texts"], data["metadatas"],
                   bm25s.BM25.load(str(folder)), data["fingerprint"])


def _fingerprint(ids: list[str], texts: list[str]) -> str:
    """Short hash of the indexed content, so a stale index can be recognised."""
    digest = hashlib.sha256()
    for chunk_id, text in zip(ids, texts):
        digest.update(chunk_id.encode() + b"\0" + text.encode() + b"\0")
    return digest.hexdigest()[:12]
