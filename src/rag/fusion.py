"""Reciprocal Rank Fusion (RRF): merge several ranked lists into one, using only RANKS.

Why ranks and not scores? Cosine similarity (about 0-1) and BM25 (unbounded, depends on the
corpus) live on different scales, so adding or averaging them is fragile. A rank ("2nd best")
means the same thing in every list.

    score(item) = sum over every list containing the item of  1 / (k + rank_in_that_list)

Ranks start at 1. `k` (default 60, the value from the original RRF paper) softens the gap
between rank 1 and rank 2: a big k makes the top spots matter less.

Pure function: no I/O, no dependencies, so it is trivial to unit test.
"""


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge best-first lists of ids into one list of (id, fused_score), best first.

    An id repeated inside one list only counts at its best (first) position.
    Ties are broken by the best rank the id had in any list, then by order of first appearance,
    so the result is deterministic.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0 (got {k})")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for ranked in ranked_lists:
        seen_in_this_list: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            if item in seen_in_this_list:
                continue
            seen_in_this_list.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(((item, scores[item]) for item in scores),
                  key=lambda pair: (-pair[1], best_rank[pair[0]], first_seen[pair[0]]))
