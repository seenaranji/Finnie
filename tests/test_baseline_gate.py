"""The regression gate's decision rule (pure function; the measuring itself costs money and runs via `make eval-gate`)."""
from eval.scripts.check_baseline import DEFAULT_TOLERANCE, compare

BASE = {"retrieval_hit3_test": 1.0, "router_accuracy_test": 0.9792}


def verdicts(current, tolerance=DEFAULT_TOLERANCE):
    return {r["metric"]: r["ok"] for r in compare(BASE, current, tolerance)}


def test_equal_or_better_passes():
    assert verdicts({"retrieval_hit3_test": 1.0, "router_accuracy_test": 1.0}) == {"retrieval_hit3_test": True, "router_accuracy_test": True}


def test_one_lost_retrieval_question_fails_because_retrieval_is_deterministic():
    assert verdicts({"retrieval_hit3_test": 0.9375, "router_accuracy_test": 0.9792})["retrieval_hit3_test"] is False


def test_one_flipped_router_query_is_tolerated_but_two_are_not():
    assert verdicts({"retrieval_hit3_test": 1.0, "router_accuracy_test": 0.9583})["router_accuracy_test"] is True    # 46/48
    assert verdicts({"retrieval_hit3_test": 1.0, "router_accuracy_test": 0.9375})["router_accuracy_test"] is False   # 45/48


def test_a_metric_that_was_not_measured_fails_instead_of_being_ignored():
    assert verdicts({"retrieval_hit3_test": 1.0})["router_accuracy_test"] is False


def test_with_zero_tolerance_the_baseline_itself_passes_despite_float_noise():
    assert all(verdicts({"retrieval_hit3_test": 1.0, "router_accuracy_test": 0.9792}, {}).values())
