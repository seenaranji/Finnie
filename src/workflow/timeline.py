"""A text picture of when each branch ran, so overlap (or the lack of it) can be SEEN. Works on the `sub_results` the executor records."""


def timing_summary(sub_results: dict) -> dict:
    ran = [r for r in sub_results.values() if r.get("started") is not None and r.get("ended") is not None and r["status"] != "skipped"]
    if not ran:
        return {"wall_s": 0.0, "busy_s": 0.0, "overlap_factor": 1.0, "slowest": None}
    wall = max(r["ended"] for r in ran) - min(r["started"] for r in ran)
    busy = sum(r["ended"] - r["started"] for r in ran)
    slowest = max(ran, key=lambda r: r["ended"] - r["started"])
    return {"wall_s": wall, "busy_s": busy, "overlap_factor": busy / wall if wall else 1.0, "slowest": slowest["id"]}


def render_timeline(sub_results: dict, width: int = 48) -> str:
    rows = sorted((r for r in sub_results.values() if r.get("started") is not None), key=lambda r: r["started"])
    if not rows:
        return "(no branches ran)"
    t0 = min(r["started"] for r in rows)
    span = max(max(r["ended"] for r in rows) - t0, 1e-9)
    lines = []
    for r in rows:
        start, end = r["started"] - t0, r["ended"] - t0
        lead, length = int(start / span * width), max(1, int((end - start) / span * width))
        bar = " " * lead + ("#" if r["status"] == "ok" else "x" if r["status"] in ("error", "timeout", "degraded") else "-") * length
        lines.append(f"  {r['id']:3}{r['agent']:11}|{bar:<{width + 1}}| {start:5.2f}s -> {end:5.2f}s  ({end - start:4.2f}s, {r['status']})")
    s = timing_summary(sub_results)
    lines.append(f"  wall-clock {s['wall_s']:.2f}s, sum of branch times {s['busy_s']:.2f}s, overlap factor {s['overlap_factor']:.2f}x, slowest branch {s['slowest']}")
    return "\n".join(lines)
