"""Step 25, part 1: latency baseline per query type over real HTTP (POST /chat), cold and warm.

  WARM: one server, a throwaway warm-up request, then 30 timed runs per route (ten different requests, three passes), one at a time, a fresh session each.
        Data caches (market quotes 60 s, news 15 min) are allowed to work, as they do for a real user.
  COLD: a brand new server process for each sample and ONE request to it: the first request pays for lazy imports, index loading and connection setup.

  python -m benchmarks.bench_latency [--routes qa market ...] [--warm 30] [--cold 3] [--label baseline]

Costs real money (about $0.004 per request: 180 warm + 18 cold requests is under $1).
"""
import argparse
import random

import httpx

from benchmarks.common import BASE, ROUTES, Server, chat, save, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", nargs="*", default=[r for r in ROUTES if r != "portfolio_new"], choices=list(ROUTES))
    parser.add_argument("--env", nargs="*", default=[], help="KEY=VALUE for the server process, e.g. FINNIE_FETCH_THREADS=1")
    parser.add_argument("--warm", type=int, default=30)
    parser.add_argument("--cold", type=int, default=3)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    random.seed(11)
    env = dict(item.split("=", 1) for item in args.env)
    report = {"label": args.label, "warm": {}, "cold": {}}

    print(f"WARM: {args.warm} runs per route, one at a time")
    with Server(env=env) as server, httpx.Client(base_url=BASE, timeout=90) as client:
        for route in args.routes:
            chat(client, *ROUTES[route][0]) if route != "portfolio_new" else chat(client, *ROUTES["portfolio"][0])   # warm-up (its own request: lazy builders, connection pools); not counted
            requests = (ROUTES[route] * 3)[:args.warm]
            if route != "portfolio_new":
                random.shuffle(requests)
            runs = [chat(client, message, csv) for message, csv in requests]
            ok = [r for r in runs if r["status"] == 200]
            seconds = [r["seconds"] for r in ok]
            routed = {}
            for r in ok:
                routed[r["route"]] = routed.get(r["route"], 0) + 1
            report["warm"][route] = {"latency_s": summarize(seconds), "errors": len(runs) - len(ok), "degraded": sum(1 for r in ok if r["degraded"]),
                                     "routed_to": routed, "cost_usd_mean": sum(r["cost_usd"] for r in ok) / max(len(ok), 1), "runs": runs}
            s = report["warm"][route]["latency_s"]
            print(f"  {route:10} n={s['n']:<3} p50 {s['p50']:6.2f}s  p95 {s['p95']:6.2f}s  mean {s['mean']:6.2f}s  max {s['max']:6.2f}s  errors {len(runs) - len(ok)}  "
                  f"degraded {report['warm'][route]['degraded']}  routed {routed}  cost ${report['warm'][route]['cost_usd_mean']:.5f}/query")

    print(f"\nCOLD: first request to a fresh server process, {args.cold} per route")
    for route in args.routes:
        samples = []
        for i in range(args.cold):
            message, csv = ROUTES[route][i % len(ROUTES[route])]
            with Server(env=env) as server, httpx.Client(base_url=BASE, timeout=90) as client:
                samples.append(chat(client, message, csv))
        seconds = [r["seconds"] for r in samples if r["status"] == 200]
        report["cold"][route] = {"latency_s": summarize(seconds), "runs": samples}
        print(f"  {route:10} " + "  ".join(f"{x:6.2f}s" for x in seconds) + f"   (warm p50 {report['warm'][route]['latency_s']['p50']:.2f}s)" if route in report["warm"] else "")

    path = save(f"step25_latency_{args.label}", report)
    print(f"\nraw results: {path}")


if __name__ == "__main__":
    main()
