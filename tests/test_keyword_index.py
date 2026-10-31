from src.rag.chunker import Chunk
from src.rag.keyword_index import KeywordIndex, tokenize


def make_chunk(chunk_id: str, text: str, category: str = "stocks") -> Chunk:
    return Chunk(chunk_id, chunk_id.split("#")[0], text, {"category": category}, 0, 0, len(text))


def test_tokenizer_keeps_401k_whole():
    assert tokenize("401(k) match") == ["401k", "match"]
    assert tokenize("a 403(b) plan") == ["403b", "plan"]


def test_tokenizer_keeps_s_and_p_and_decimals():
    assert tokenize("S&P 500 returned 0.03") == ["s&p", "500", "returned", "0.03"]


def test_tokenizer_possessive_hyphen_stopwords_plural():
    assert tokenize("investor's tax-free bonds") == ["investor", "tax", "free", "bond"]
    assert tokenize("what is HHI") == ["hhi"]
    assert tokenize("loss") == ["loss"]   # "ss" ending is not a plural


def test_search_finds_exact_term_and_ignores_missing_term():
    index = KeywordIndex.build([make_chunk("a#0", "The HHI measures concentration."),
                                make_chunk("b#0", "Bonds pay interest.")])
    assert [h.chunk_id for h in index.search("what is HHI", 5)] == ["a#0"]
    assert index.search("VOO", 5) == []          # unseen word -> no hits, no crash
    assert index.search("what is it", 5) == []   # only stopwords -> empty query -> no crash


def test_category_filter_applies_to_keyword_search():
    index = KeywordIndex.build([make_chunk("a#0", "stock shares", "stocks"), make_chunk("b#0", "stock fund", "etfs")])
    assert [h.chunk_id for h in index.search("stock", 5, {"category": "etfs"})] == ["b#0"]


def test_save_and_load_round_trip(tmp_path):
    index = KeywordIndex.build([make_chunk("a#0", "expense ratio matters"), make_chunk("b#0", "bonds pay interest")])
    index.save(tmp_path)
    loaded = KeywordIndex.load(tmp_path)
    assert loaded.fingerprint == index.fingerprint
    assert [h.chunk_id for h in loaded.search("expense ratio", 5)] == ["a#0"]
