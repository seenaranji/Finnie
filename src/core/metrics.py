"""Tiny timing and statistics helpers, reused by every later step.

Why percentiles? The mean hides slow outliers. If 19 calls take 1 s and one takes
20 s, the mean is ~2 s (looks fine) but a real user sometimes waits 20 s. p95 shows
the slow tail: "95% of calls were faster than this".
"""
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def timer():
    """Usage:  with timer() as t: do_work()   then   t['seconds']"""
    result = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def summarize(values: list[float]) -> dict:
    """Count, mean, p50 (typical), p95 (slow tail), min and max for a list of numbers."""
    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


class Recorder:
    """Collects named measurements (e.g. 'latency', 'cost') across many calls."""

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}

    def add(self, name: str, value: float) -> None:
        self._data.setdefault(name, []).append(value)

    def summary(self) -> dict[str, dict]:
        return {name: summarize(vals) for name, vals in self._data.items()}
