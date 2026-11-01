"""Step 25, part 3: load. N simulated users each send a chat request, wait for the reply, and immediately send the next (no think time, so N users
means N requests in flight: harsher than real people, who pause). A fresh server process for every level, so memory is attributed to that level.

Request mix (what a beginner-finance app would see): 40% Q&A, 15% market, 10% portfolio, 10% goal, 10% tax, 5% news, 10% out-of-scope-ish short question.

  python -m benchmarks.bench_load [--users 1 5 10 25] [--seconds 20] [--target-p95 8] [--label baseline]

Costs real money (about $0.004 per request; 41 user-streams x 20 s is a few hundred requests).
"""
import argparse
import asyncio
import random
import time

import httpx
import psutil

from benchmarks.common import BASE, ROUTES, Server, save, summarize, percentile

MIX = [("qa", 40), ("market", 15), ("portfolio", 10), ("goal", 10), ("tax", 10), ("news", 5), ("qa", 10)]


def pick(rng):
    route = rng.choices([r for r, _ in MIX], weights=[w for _, w in MIX])[0]
    return route, rng.choice(ROUTES[route])


async def user(client, deadline, rng, results):
    while time.perf_counter() < deadline:
        route, (message, csv) = pick(rng)
        body = {"message": message, **({"portfolio_csv": csv} if csv else {})}
        start = time.perf_counter()
        try:
            response = await client.post("/chat", json=body)
            status = response.status_code
            data = response.json() if status == 200 else {}
        except httpx.HTTPError as error:
            status, data = type(error).__name__, {}
        results.append({"route": route, "seconds": time.perf_counter() - start, "status": status, "cost_usd": data.get("cost_usd", 0.0), "degraded": data.get("degraded")})


async def sample_memory(pid, stop, samples):
    process = psutil.Process(pid)
    process.cpu_percent(None)
    while not stop.is_set():
        try:
            family = [process] + process.children(recursive=True)
            samples.append((sum(p.memory_info().rss for p in family) / 2**20, process.cpu_percent(None)))
        except psutil.Error:
            pass
        await asyncio.sleep(0.5)


async def level(users, seconds, pid, seed):
    results, samples, stop = [], [], asyncio.Event()
    rng = random.Random(seed)
    async with httpx.AsyncClient(base_url=BASE, timeout=120, limits=httpx.Limits(max_connections=users + 5)) as client:
        for route in ("qa", "market", "portfolio", "goal", "tax", "news"):   # warm-up, not counted: lazy builders, connection pools, data caches
            message, csv = ROUTES[route][0]
            await client.post("/chat", json={"message": message, **({"portfolio_csv": csv} if csv else {})})
        watcher = asyncio.create_task(sample_memory(pid, stop, samples))
        start = time.perf_counter()
        await asyncio.gather(*(user(client, start + seconds, random.Random(rng.random()), results) for _ in range(users)))
        wall = time.perf_counter() - start
        stop.set()
        await watcher
    ok = [r for r in results if r["status"] == 200]
    return {"users": users, "seconds": round(wall, 1), "requests": len(results), "errors": len(results) - len(ok), "error_rate": (len(results) - len(ok)) / max(len(results), 1),
            "error_kinds": sorted({str(r["status"]) for r in results if r["status"] != 200}), "degraded": sum(1 for r in ok if r["degraded"]),
            "throughput_rps": len(ok) / wall, "latency_s": summarize([r["seconds"] for r in ok]),
            "p99_s": percentile([r["seconds"] for r in ok], 99), "cost_usd": sum(r["cost_usd"] for r in ok),
            "rss_mb_peak": max((m for m, _ in samples), default=None), "cpu_percent_mean": sum(c for _, c in samples) / max(len(samples), 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, nargs="*", default=[1, 5, 10, 25])
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--target-p95", type=float, default=8.0)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    report = {"label": args.label, "target_p95_s": args.target_p95, "workers": args.workers, "levels": []}
    print(f"{'users':>5} {'reqs':>5} {'req/s':>6} {'p50':>6} {'p95':>6} {'p99':>6} {'max':>6} {'err%':>5} {'degr':>4} {'RSS MB':>7} {'CPU%':>5} {'cost':>7}")
    for users in args.users:
        with Server(workers=args.workers) as server:
            row = asyncio.run(level(users, args.seconds, server.process.pid, seed=users))
        report["levels"].append(row)
        s = row["latency_s"]
        print(f"{users:5d} {row['requests']:5d} {row['throughput_rps']:6.2f} {s['p50']:6.2f} {s['p95']:6.2f} {row['p99_s']:6.2f} {s['max']:6.2f} {100 * row['error_rate']:5.1f} "
              f"{row['degraded']:4d} {row['rss_mb_peak'] or 0:7.0f} {row['cpu_percent_mean']:5.0f} ${row['cost_usd']:.3f}   {row['error_kinds'] or ''}")
    within = [r["users"] for r in report["levels"] if r["latency_s"]["p95"] <= args.target_p95 and r["error_rate"] == 0]
    report["max_users_within_target"] = max(within) if within else 0
    print(f"\nlargest tested level with p95 <= {args.target_p95}s and no errors: {report['max_users_within_target']} users")
    print(f"raw results: {save(f'step25_load_{args.label}', report)}")


if __name__ == "__main__":
    main()
