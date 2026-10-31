import pytest

from src.rag.fusion import reciprocal_rank_fusion


def test_worked_example_from_the_lesson():
    # doc "a": 1st in dense, 3rd in BM25  -> 1/61 + 1/63;  doc "b": 1st in BM25 only -> 1/61
    fused = dict(reciprocal_rank_fusion([["a", "x", "y"], ["b", "z", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["b"] == pytest.approx(1 / 61)


def test_item_found_by_both_lists_rises_to_the_top():
    fused = reciprocal_rank_fusion([["a", "shared"], ["b", "shared"]])
    assert fused[0][0] == "shared"


def test_single_list_keeps_its_order():
    assert [item for item, _ in reciprocal_rank_fusion([["a", "b", "c"]])] == ["a", "b", "c"]


def test_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_ties_are_deterministic():
    # "a" and "b" each appear once at rank 1 -> same score; first list wins the tie
    assert [item for item, _ in reciprocal_rank_fusion([["a"], ["b"]])] == ["a", "b"]


def test_duplicate_inside_one_list_counts_once():
    fused = dict(reciprocal_rank_fusion([["a", "a"]]))
    assert fused["a"] == pytest.approx(1 / 61)


def test_bigger_k_flattens_the_gap_between_ranks():
    gap_small_k = 1 / (1 + 1) - 1 / (1 + 2)
    gap_big_k = 1 / (100 + 1) - 1 / (100 + 2)
    assert gap_big_k < gap_small_k
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)
