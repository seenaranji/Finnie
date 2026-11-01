"""PgVectorStore and PgKeywordIndex against a real (throwaway) PostgreSQL + pgvector database."""
import numpy as np
import pytest

from src.rag.chunker import Chunk
from src.rag.pg_keyword_index import PgKeywordIndex
from src.rag.pg_vector_store import PgVectorStore, where_sql
from src.rag.vector_store import EmbeddingModelMismatchError

DIMENSIONS = 1536


def unit(*hot: int, size=DIMENSIONS) -> np.ndarray:
    """A vector pointing mostly along the given axes (so we know exactly which chunk is closest)."""
    vector = np.zeros(size, dtype=np.float32)
    for axis in hot:
        vector[axis] = 1.0
    return vector / np.linalg.norm(vector)


def chunk(chunk_id: str, text: str, category: str = "etfs", **meta) -> Chunk:
    doc_id, _, index = chunk_id.partition("#")
    return Chunk(chunk_id, doc_id, text, {"title": doc_id.title(), "category": category, "section": "Intro", "source_name": "Src",
                                          "source_url": "https://x.example", "last_updated": "2026-10-30", "difficulty": "beginner", **meta},
                 int(index), 0, len(text))


@pytest.fixture
def store(pg):
    return PgVectorStore("test-model", pg)


def test_search_orders_by_cosine_similarity_and_returns_similarity_not_distance(store):
    store.add([chunk("a#0", "alpha"), chunk("b#0", "beta"), chunk("c#0", "gamma")], np.array([unit(0), unit(0, 1), unit(2)]))
    hits = store.search(unit(0), 3)
    assert [h.chunk_id for h in hits] == ["a#0", "b#0", "c#0"]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-6)          # same direction: similarity 1 (distance 0)
    assert hits[1].similarity == pytest.approx(0.7071, abs=1e-3)       # 45 degrees
    assert hits[2].similarity == pytest.approx(0.0, abs=1e-6)          # orthogonal
    assert hits[0].metadata["title"] == "A" and hits[0].metadata["doc_id"] == "a"   # the citation fields come back unchanged


def test_category_filter_single_and_list(store):
    store.add([chunk("a#0", "x", "etfs"), chunk("b#0", "y", "bonds"), chunk("c#0", "z", "taxes")], np.array([unit(0), unit(0), unit(0)]))
    assert [h.chunk_id for h in store.search(unit(0), 5, {"category": "bonds"})] == ["b#0"]
    assert sorted(h.chunk_id for h in store.search(unit(0), 5, {"category": {"$in": ["etfs", "taxes"]}})) == ["a#0", "c#0"]
    assert store.search(unit(0), 5, {"category": "nothing"}) == []


def test_filter_on_a_metadata_field_that_is_not_a_column(store):
    store.add([chunk("a#0", "x", difficulty="beginner"), chunk("b#0", "y", difficulty="advanced")], np.array([unit(0), unit(0)]))
    assert [h.chunk_id for h in store.search(unit(0), 5, {"difficulty": "advanced"})] == ["b#0"]


def test_a_filter_key_can_never_carry_sql():
    with pytest.raises(ValueError):
        where_sql({"category = 'x' OR 1=1 --": "etfs"})


def test_adding_twice_does_not_duplicate_and_replaces_changed_text(store):
    store.add([chunk("a#0", "old text")], np.array([unit(0)]))
    store.add([chunk("a#0", "new text")], np.array([unit(1)]))
    assert store.count() == 1
    assert store.search(unit(1), 1)[0].text == "new text"


def test_an_article_that_got_shorter_loses_its_stale_chunks(store):
    store.add([chunk("a#0", "one"), chunk("a#1", "two"), chunk("b#0", "other")], np.array([unit(0), unit(1), unit(2)]))
    store.add([chunk("a#0", "one")], np.array([unit(0)]))   # article a re-indexed with one chunk
    assert sorted(store.all_chunk_ids()) == ["a#0", "b#0"]   # a#1 is gone, b untouched


def test_a_failed_add_leaves_nothing_behind(store):
    with pytest.raises(Exception):   # the second vector has the wrong number of dimensions: PostgreSQL rejects it
        store.add([chunk("a#0", "ok"), chunk("b#0", "bad")], [unit(0), np.zeros(10, dtype=np.float32)])
    assert store.count() == 0


def test_mismatched_lengths_are_refused(store):
    with pytest.raises(ValueError):
        store.add([chunk("a#0", "x")], np.array([unit(0), unit(1)]))


def test_reset_empties_everything(store):
    store.add([chunk("a#0", "x")], np.array([unit(0)]))
    store.reset()
    assert store.count() == 0


def test_a_different_embedding_model_is_refused(pg):
    PgVectorStore("model-one", pg)
    with pytest.raises(EmbeddingModelMismatchError):
        PgVectorStore("model-two", pg)
    PgVectorStore("model-one", pg)   # the same model is fine


def test_deleting_an_article_removes_its_chunks(store, pg):
    store.add([chunk("a#0", "x")], np.array([unit(0)]))
    with pg.connection() as connection:
        connection.execute("DELETE FROM articles WHERE id = 'a'")
    assert store.count() == 0


def test_checksum_is_stable_and_changes_with_content(store):
    store.add([chunk("a#0", "x")], np.array([unit(0)]))
    first = store.content_checksum()
    assert first == store.content_checksum()
    store.add([chunk("a#0", "y")], np.array([unit(0)]))
    assert store.content_checksum() != first


# ---------- keyword search ----------
@pytest.fixture
def keywords(store, pg):
    store.add([chunk("etf-basics#0", "An ETF is a basket of investments you can trade like a stock.", "etfs"),
               chunk("ira#0", "A Roth IRA is a retirement account with tax-free growth.", "retirement"),
               chunk("k401#0", "The 401(k) contribution limit is set by the IRS each year.", "retirement"),
               chunk("bonds#0", "Bonds pay interest. Investors buy bonds to earn steady income.", "bonds")],
              np.array([unit(0), unit(1), unit(2), unit(3)]))
    return PgKeywordIndex(pg)


def test_finds_exact_terms_and_ranks_them(keywords):
    assert keywords.search("Roth IRA", 5)[0].chunk_id == "ira#0"
    assert keywords.search("401(k) limit", 5)[0].chunk_id == "k401#0"
    assert keywords.search("pizza", 5) == []


def test_stems_plurals(keywords):
    assert keywords.search("bond", 5)[0].chunk_id == "bonds#0"


def test_any_mode_matches_a_long_natural_question_and_all_mode_does_not(pg, keywords):
    question = "what is a basket of stocks I can trade and also zebra"
    assert keywords.search(question, 5)[0].chunk_id == "etf-basics#0"
    assert PgKeywordIndex(pg, match="all").search(question, 5) == []   # 'zebra' is in no chunk, so AND matches nothing


def test_same_category_filter_as_the_vector_store(keywords):
    assert {h.chunk_id for h in keywords.search("IRS limit retirement", 5, {"category": "retirement"})} == {"k401#0", "ira#0"}
    assert keywords.search("bonds", 5, {"category": "etfs"}) == []


def test_hit_has_the_same_shape_as_a_bm25_hit(keywords):
    hit = keywords.search("ETF", 1)[0]
    assert (hit.chunk_id, hit.metadata["title"]) == ("etf-basics#0", "Etf-Basics") and hit.score > 0 and hit.text.startswith("An ETF")


def test_odd_input_is_just_text(keywords):
    assert keywords.search("'; DROP TABLE chunks; --", 5) == []   # parsed as plain words, matches nothing, and the table is still there
    assert keywords.search("", 5) == []
    assert keywords.search("the a of", 5) == []   # only stopwords
    assert len(keywords) == 4


def test_unknown_match_mode_is_rejected(pg):
    with pytest.raises(ValueError):
        PgKeywordIndex(pg, match="some")


def test_every_pooled_connection_has_iterative_hnsw_scans_on(pg):
    # Without this a tight metadata filter on an HNSW scan can return fewer than k rows (measured in scripts/step6_pgvector_check.py)
    with pg.connection() as connection:
        assert connection.execute("SHOW hnsw.iterative_scan").fetchone()[0] == "strict_order"
