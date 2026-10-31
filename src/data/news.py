"""News fetching (Step 19). Everything in here is UNTRUSTED DATA: third parties write it and nobody on our side reads it.

Pipeline per feed:  download (timeout, size cap, identifying user agent) -> parse -> sanitize each item -> flag instruction-like text
                    -> drop old items -> dedupe -> newest first -> cap -> number 1..n (the numbers the answer will cite).

Sanitizing is NOT a prompt-injection fix by itself. It removes the cheap tricks (HTML, hidden characters, fake delimiters, huge text) and
flags the blatant instructions. A clever or foreign-language injection passes the flag, which is why the news agent adds more layers.
"""
import html
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

import feedparser
import requests
from pydantic import BaseModel, Field

from src.core.config import NewsConfig
from src.core.logging import get_logger
from src.data.cache import TTLCache

log = get_logger(__name__)
MAX_FEED_BYTES = 2_000_000
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


class NewsUnavailable(RuntimeError):
    """No feed could be read. The message is safe to show a user."""


class NewsItem(BaseModel):
    title: str
    summary: str = ""
    link: str
    source: str
    published: datetime | None = None
    id: int = 0                                           # 1-based position in the batch; what the answer cites as [n]
    injection_markers: list[str] = Field(default_factory=list)   # why the item looks like it contains instructions (empty = nothing found)


class NewsBatch(BaseModel):
    topic: str
    items: list[NewsItem]
    fetched_at: datetime
    feed_errors: list[str] = Field(default_factory=list)


# ---------- sanitizing ----------
_TAGS = re.compile(r"<[^>]*>")
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿­]")   # zero-width and bidi characters hide text from humans, not models
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_FAKE_DELIMITER = re.compile(r"(={2,}|-{3,}|#{3,}|<{2,}|>{2,}|\[{2,}|\]{2,})?\s*(BEGIN|END)\s+(UNTRUSTED\s+)?NEWS\s+DATA[^\n]*", re.I)


def clean_text(text: str, limit: int) -> str:
    text = html.unescape(_TAGS.sub(" ", text or ""))
    text = _FAKE_DELIMITER.sub(" ", _INVISIBLE.sub("", _CONTROL.sub(" ", text)))
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# Instruction-like wording. Rules only see the wording they were written for: English, direct phrasing. Measured in eval/scripts/eval_news_injection.py.
_INJECTION = {rx: re.compile(p, re.I) for rx, p in {
    "tells the AI to ignore/override instructions": r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|these)\b.{0,40}\b(instruction|prompt|rule|direction|guideline)s?\b",
    "mentions the system/developer prompt": r"\b(system|developer) (prompt|message|instruction)s?\b",
    "re-assigns the AI's role": r"\byou are (now|no longer)\b|\bpretend (to be|you)\b|\bact as (a|an|if)\b|\bnew instructions?\b",
    "dictates the reply": r"\b(reply|respond|answer|output|print|say|write)\b.{0,15}\b(only|exactly|verbatim|with the (phrase|word|text))\b",
    "addresses the AI directly": r"\b(note|message|instruction)s? (to|for) (the )?(ai|assistant|model|llm|chatbot|summari[sz]er)s?\b|\b(dear|attention|hey) (ai|assistant|model|llm)\b",
    "markdown image or script (data exfiltration)": r"!\[[^\]]*\]\(\s*https?://|<\s*(script|iframe|img)\b",
}.items()}


def find_injection_markers(text: str) -> list[str]:
    return [reason for reason, rx in _INJECTION.items() if rx.search(text)]


def sanitize_item(item: NewsItem, title_chars: int = 200, summary_chars: int = 500) -> NewsItem:
    """Clean and flag. The flags are computed on the RAW text too (before hidden characters are removed), so hiding a phrase does not hide the flag."""
    raw = f"{item.title} {item.summary}"
    markers = sorted(set(find_injection_markers(raw)) | set(find_injection_markers(clean_text(raw, 10_000))))
    return item.model_copy(update={"title": clean_text(item.title, title_chars), "summary": clean_text(item.summary, summary_chars),
                                   "injection_markers": markers})


# ---------- dedupe ----------
def canonical_link(link: str) -> str:
    parts = urlsplit(link.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))   # query strings carry tracking junk


def _title_tokens(title: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", title.lower())) - {"the", "a", "an", "of", "to", "in", "and", "for", "on", "is", "as", "at"}


def dedupe(items: list[NewsItem], similarity: float = 0.8) -> list[NewsItem]:
    """Drop repeats: same canonical link, or a title whose word set overlaps an already-kept title by at least `similarity` (Jaccard).
    Keeps the first occurrence, so sort newest-first BEFORE calling."""
    kept, links, token_sets = [], set(), []
    for item in items:
        link, tokens = canonical_link(item.link), _title_tokens(item.title)
        if link in links:
            continue
        if tokens and any(len(tokens & other) / len(tokens | other) >= similarity for other in token_sets):
            continue
        links.add(link)
        token_sets.append(tokens)
        kept.append(item)
    return kept


# ---------- fetching ----------
def parse_feed(content: bytes, source_name: str) -> list[NewsItem]:
    feed = feedparser.parse(content)
    items = []
    for entry in feed.entries:
        link, title = entry.get("link", ""), entry.get("title", "")
        if not link.startswith(("http://", "https://")) or not title.strip():
            continue   # an item we cannot link to cannot be cited
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        items.append(NewsItem(title=title, summary=entry.get("summary", ""), link=link, source=source_name,
                              published=datetime(*published[:6], tzinfo=timezone.utc) if published else None))
    return items


class NewsService:
    def __init__(self, config: NewsConfig, cache: TTLCache | None = None, get: Callable = requests.get,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.config, self.cache, self._get, self._now = config, cache or TTLCache(), get, now

    def _download(self, name: str, url: str) -> list[NewsItem]:
        try:
            response = self._get(url, timeout=self.config.timeout_seconds, headers={"User-Agent": self.config.user_agent})
            response.raise_for_status()
        except requests.RequestException as error:
            raise NewsUnavailable(f"{name}: {type(error).__name__}") from error
        if len(response.content) > MAX_FEED_BYTES:
            raise NewsUnavailable(f"{name}: feed larger than {MAX_FEED_BYTES} bytes")
        return parse_feed(response.content, name)

    def _batch(self, topic: str, feeds: list[tuple[str, str]], per_feed: int | None = None) -> NewsBatch:
        key = f"news:{topic}"
        cached = self.cache.get(key)
        if cached is not None:
            log.info("news cache HIT %s", key)
            return cached
        raw, errors = [], []
        for name, url in feeds:
            try:
                raw += self._download(name, url)
            except NewsUnavailable as error:
                errors.append(str(error))
                log.warning("news feed failed: %s", error)
        if not raw:
            stale = self.cache.get_stale(key)
            if stale:   # all feeds down: an old digest, flagged by its own fetched_at, beats nothing
                log.warning("all news feeds failed; serving a stale batch (%.0fs old)", stale[1])
                return stale[0].model_copy(update={"feed_errors": errors})
            raise NewsUnavailable("News is not available right now. " + "; ".join(errors))
        cfg, cutoff = self.config, self._now() - timedelta(days=self.config.max_age_days)
        items = [sanitize_item(i, cfg.max_title_chars, cfg.max_summary_chars) for i in raw]
        items = [i for i in items if i.published is None or i.published >= cutoff]
        items.sort(key=lambda i: i.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        items, per_source, kept = dedupe(items), {}, []
        for i in items:   # newest first, but no feed may supply more than per_feed items
            if per_feed is None or per_source.get(i.source, 0) < per_feed:
                per_source[i.source] = per_source.get(i.source, 0) + 1
                kept.append(i)
        items = [i.model_copy(update={"id": n}) for n, i in enumerate(sorted(kept, key=lambda i: i.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)[: cfg.max_items], start=1)]
        batch = NewsBatch(topic=topic, items=items, fetched_at=self._now(), feed_errors=errors)
        self.cache.set(key, batch, cfg.ttl_seconds)
        return batch

    def get_ticker_news(self, symbol: str) -> NewsBatch:
        symbol = (symbol or "").strip().upper()
        if not _SYMBOL.match(symbol):
            raise NewsUnavailable(f"'{symbol}' doesn't look like a valid ticker symbol.")
        return self._batch(symbol, [(f"Yahoo Finance ({symbol})", self.config.ticker_feed_url.format(symbol=symbol))])

    def get_market_news(self) -> NewsBatch:
        return self._batch("markets", [(f.name, f.url) for f in self.config.market_feeds], per_feed=self.config.max_per_feed)
