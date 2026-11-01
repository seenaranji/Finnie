"""Chunker: properties that must hold for ANY article (nothing lost, nothing oversized, ids in order), plus the specific rules."""
import pytest

from src.rag.chunker import chunk_fixed, chunk_structured
from src.rag.loader import Document

TEXT = ("# Title\n\nIntro sentence one. Intro sentence two.\n\n## First\n\n" + "A fairly ordinary sentence about money. " * 12 +
        "\n\n## Second\n\nShort section.\n\n- bullet one\n- bullet two\n")
DOC = Document("demo", TEXT, {"title": "Demo", "category": "basics"})


def test_fixed_windows_overlap_and_cover_the_whole_text():
    chunks = chunk_fixed(DOC, size=100, overlap=20)
    assert chunks[0].text == TEXT[:100] and chunks[1].start == 80   # the next window starts 20 characters before the previous one ended
    assert chunks[-1].end == len(TEXT) and all(len(c.text) <= 100 for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"demo#{i}" for i in range(len(chunks))]


def test_fixed_rejects_an_overlap_that_would_never_advance():
    for overlap in (-1, 100, 150):
        with pytest.raises(ValueError):
            chunk_fixed(DOC, size=100, overlap=overlap)


def test_fixed_emits_no_tail_that_is_pure_overlap():
    doc = Document("d", "x" * 100, {})
    assert len(chunk_fixed(doc, size=100, overlap=20)) == 1


@pytest.mark.parametrize("size", [60, 150, 300, 1000])
def test_structured_chunks_cover_every_character_of_content_and_stay_near_the_size_limit(size):
    chunks = chunk_structured(DOC, size)
    assert chunks and [c.chunk_id for c in chunks] == [f"demo#{i}" for i in range(len(chunks))]
    covered = "".join(c.text for c in chunks)
    assert "".join(covered.split()) == "".join(TEXT.split())   # nothing lost or duplicated, ignoring whitespace between chunks
    assert all(c.text.strip() for c in chunks) and all(c.text == TEXT[c.start:c.end] for c in chunks)
    # size is a soft cap by at most one heading line (documented in _pack_pieces)
    assert all(len(c.text) <= size + 40 for c in chunks)


def test_structured_starts_a_new_chunk_at_each_heading_when_sections_are_small():
    chunks = chunk_structured(Document("d", "# A\nalpha\n\n# B\nbeta\n", {"category": "x"}), size=500)
    assert [c.text for c in chunks] == ["# A\nalpha", "# B\nbeta"] and [c.metadata["section"] for c in chunks] == ["A", "B"]


def test_structured_copies_metadata_and_records_the_section():
    chunks = chunk_structured(DOC, 150)
    assert all(c.metadata["title"] == "Demo" and c.metadata["category"] == "basics" for c in chunks)
    assert {c.metadata["section"] for c in chunks} >= {"Title", "First", "Second"}


def test_a_heading_is_never_left_alone_at_the_end_of_a_chunk():
    for size in (40, 80, 120):
        for chunk in chunk_structured(DOC, size):
            assert not chunk.text.rstrip().split("\n")[-1].startswith("#")


def test_a_single_unbreakable_run_is_cut_at_the_size():
    chunks = chunk_structured(Document("d", "x" * 250, {}), size=100)
    assert [len(c.text) for c in chunks] == [100, 100, 50]


def test_empty_and_blank_text_make_no_chunks():
    assert chunk_structured(Document("d", "", {}), 100) == [] and chunk_structured(Document("d", "  \n\n ", {}), 100) == []


def test_structured_is_deterministic():
    assert [c.text for c in chunk_structured(DOC, 150)] == [c.text for c in chunk_structured(DOC, 150)]
