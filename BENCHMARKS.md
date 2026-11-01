# Finnie benchmarks (Step 25)

Everything here was measured on one Mac, on 2026-11-01, against the real services: OpenAI (`gpt-4o` for answers, `gpt-4o-mini` for routing and `text-embedding-3-small`),
Yahoo Finance through `yfinance`, public RSS feeds, and the Docker PostgreSQL from Step 23b. One `uvicorn` worker. Numbers move with the network and the model
provider's mood, so each table says how many runs it rests on, and Section 7 lists what they cannot tell you. Raw results for every table are in
`benchmarks/results/` (file names in Section 8); the scripts are `benchmarks/bench_latency.py`, `profile_run.py`, `bench_load.py` and the `exp_*.py` experiments.

## 1. Summary

* **Almost all of the time is spent waiting for OpenAI.** Model calls (the router plus the answering model) are 86-100% of every request; Python, the database and
  the market-data cache together are a few percent. cProfile agrees: the top of the profile is the OpenAI HTTP call, nothing of ours.
* **The router alone costs 0.7-0.8 s on every request** (20-40% of the total). Shortening its prompt from 1,300 to 100 tokens would save only about 0.12 s (Section 5.1), so it was not touched.
* **Three changes made it into the code**, each measured separately: prices for a portfolio are fetched side by side (a portfolio of new tickers: **4.20 s to 3.14 s**),
  the question embedding starts while the router is still deciding (Q&A **-0.29 s**, tax **-0.25 s** in the isolated test), and recent question embeddings are remembered (repeat questions only, **-0.17 s**).
* **Load is not our bottleneck at the sizes tested.** From 1 to 50 simultaneous users p95 stayed between 3.7 and 4.4 s with no errors; memory grew from 209-233 MB to 255-266 MB.
  I did not find the limit; Section 6 says what it is likely to be.
* **Cost per query is $0.002 (Q&A) to $0.006 (goal).** Nothing here changed it.

## 2. Baseline: latency per query type

POST `/chat` over real HTTP, one request at a time, a fresh session each. **Warm** = 30 runs per route after a throwaway warm-up request (ten different requests, three passes, shuffled;
data caches work as they would for a user: market quotes 60 s, price history 15 min, news 15 min). **Cold** = the first request ever sent to a brand-new server process (3 per route).

| Route | warm p50 | warm p95 | mean | max | cold (mean of 3) | cost / query |
|---|---|---|---|---|---|---|
| Q&A | 1.99 s | 2.53 s | 2.00 s | 2.65 s | 2.32 s | $0.0021 |
| Market | 2.50 s | 2.90 s | 2.51 s | 3.43 s | 3.10 s | $0.0042 |
| Portfolio | 3.19 s | 3.98 s | 3.28 s | 4.19 s | 4.41 s | $0.0050 |
| Goal | 3.54 s | 4.01 s | 3.56 s | 4.16 s | 3.51 s | $0.0060 |
| News | 3.62 s | 4.71 s | 3.71 s | 5.46 s | 3.68 s | $0.0047 |
| Tax | 2.05 s | 2.80 s | 2.10 s | 3.02 s | 2.73 s | $0.0034 |

Tax questions are routed to the Tax agent 18 times in 30 and to Finance Q&A 12 times (the router's call; both answer from the knowledge base). Errors: 0 in 180 warm requests. File: `step25_latency_baseline_20261101_142524.json`.

## 3. Where the time goes (per stage)

`benchmarks/profile_run.py` runs the real graph in one process with timers around the expensive calls. Times are **exclusive** (a stage excludes stages nested inside it) and are means of 10 warm requests, in seconds.
`other` is what remains of the wall time: graph plumbing, prompt building, validation.

| Route | wall | router | embedding | vector search | market data | news fetch | answer LLM | other | model calls | tokens in / out |
|---|---|---|---|---|---|---|---|---|---|---|
| Q&A | 1.96 | 0.78 | 0.24 | 0.018 | 0.00 | 0.00 | 0.91 | 0.007 | 2 | 1807 / 75 |
| Market | 2.72 | 0.72 | 0.00 | 0.000 | 0.12 | 0.00 | 1.88 | 0.007 | 3 | 2433 / 122 |
| Portfolio | 3.00 | 0.75 | 0.00 | 0.000 | 0.10 | 0.00 | 2.14 | 0.008 | 2 | 2193 / 269 |
| Goal | 3.57 | 0.84 | 0.00 | 0.000 | 0.00 | 0.00 | 2.72 | 0.008 | 3 | 2493 / 303 |
| News | 3.69 | 0.72 | 0.00 | 0.000 | 0.00 | 0.21 | 2.75 | 0.006 | 3 | 2347 / 230 |
| Tax | 2.17 | 0.72 | 0.23 | 0.012 | 0.00 | 0.00 | 1.21 | 0.008 | 2 | 2123 / 125 |

* The **router** takes 0.72-0.84 s whatever the question; the **answering model** takes 0.9 s (a 75-token Q&A answer) up to 2.7 s (a 300-token goal explanation).
  Output length matters: the portfolio answer (269 tokens) spends 2.1 s decoding, about 125 tokens per second.
* **Embedding** (0.23 s) is the only retrieval cost that shows; the pgvector query is 12-18 ms. Market quotes cost 0.10-0.12 s when cached or fast, news feeds 0.21 s, the portfolio math 1 ms.
* **Goal, News and Market make three model calls** (router, a tool-calling or extraction turn, then the answer); Q&A, Tax and Portfolio make two.
* Tokens and cost per query come from the same runs: Q&A 1,807 in / 75 out ($0.0021), Market 2,433 / 122 ($0.0042), Portfolio 2,193 / 269 ($0.0050), Goal 2,493 / 303 ($0.0061), News 2,347 / 230 ($0.0047), Tax 2,123 / 125 ($0.0034).
  About 1,300 of every request's input tokens are the router's prompt.
* Two cProfile runs (Q&A and Portfolio, in the raw file) show the same thing as the table: the cumulative time sits in `openai/_base_client.py` request code.

File: `step25_stages_baseline_20261101_142850.json`.

## 4. Baseline: load

N simulated users, each sending a request and, as soon as the reply arrives, the next one (no think time, so N users means N requests in flight: harsher than real people, who pause).
Mix: Q&A 50%, market 15%, portfolio 10%, goal 10%, tax 10%, news 5%. A fresh server per level; 20 s per level (15 s for the 50-user row). Errors count every non-200 or transport failure.

| Users | requests | req/s | p50 | p95 | p99 | max | errors | peak memory | mean CPU |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 9 | 0.4 | 1.99 s | 3.52 s | 3.65 s | 3.68 s | 0 | 233 MB | 1% |
| 5 | 41 | 1.9 | 2.39 s | 3.92 s | 4.24 s | 4.39 s | 0 | 240 MB | 6% |
| 10 | 86 | 4.0 | 2.24 s | 3.79 s | 4.11 s | 4.15 s | 0 | 241 MB | 9% |
| 25 | 215 | 9.7 | 2.34 s | 3.70 s | 4.55 s | 4.83 s | 0 | 258 MB | 18% |
| 50 | 279 | 15.1 | 2.72 s | 4.26 s | 4.95 s | 5.47 s | 0 | 266 MB | 23% |

Largest level tested with p95 under 8 s and no errors: **50 users**. Throughput scaled almost linearly (0.4 to 15 requests/s), CPU never passed 25% of one core, memory rose about 33 MB over the range.
Files: `step25_load_baseline_20261101_143215.json`, `step25_load_baseline_50_20261101_143308.json`.

## 5. Optimizing, one change at a time

### 5.1 Tried and rejected: a shorter router prompt
`exp_router_prompt.py`: same messages, same model and schema, prompts of different length called in shuffled order, n=20 each.

| Router prompt | tokens in | p50 | p95 |
|---|---|---|---|
| full v6 | 1,300 | 0.645 s | 0.885 s |
| first half | 691 | 0.546 s | 0.789 s |
| first quarter | 377 | 0.586 s | 0.792 s |
| one line | 101 | 0.527 s | 0.718 s |

Cutting 92% of the tokens saves about 0.12 s at p50, and a shorter prompt would route worse (the v1 to v6 history is a record of that). The router's cost is the fixed price of one model call, not its prompt. **Not changed.**

### 5.2 Fetch portfolio prices side by side
The Portfolio agent fetched a quote and a year of history for each holding one after another. With a cold market-data cache each holding costs about 0.2 s, so it grows with portfolio size.
`exp_portfolio_fetch.py` times only the fetch and the math (the model is faked), 10 runs per cell, a fresh service every run, different random tickers every run, the two variants alternating:

| Portfolio | 1 thread (before) p50 / p95 | 8 threads (after) p50 / p95 |
|---|---|---|
| 5 holdings | 1.62 s / 1.90 s | 0.47 s / 0.57 s |
| 12 holdings | 3.16 s / 3.74 s | 0.69 s / 0.84 s |

End to end through the real server (`bench_latency --routes portfolio_new`, 30 portfolios of 5 tickers none of which repeats, so the cache is cold every time; the thread count switched with `FINNIE_FETCH_THREADS`):
**p50 4.20 s to 3.14 s (-1.06 s), p95 4.94 s to 3.83 s (-1.11 s).**
My first version of this experiment reused the same 5 tickers every run and showed almost no gain (0.35 s to 0.36 s): the data provider answers faster for tickers it has just served, which flattered the sequential code. The warm routes in Section 2 hide this effect, because three sample portfolios repeat and the cache serves them.

### 5.3 Start the question embedding while the router runs
Retrieval routes embed the question (0.23 s) only after routing. The two do not depend on each other, so the graph's first node now starts the embedding in a background thread (`CachedEmbedder.prefetch`); if the route turns out not to search (market, goal, news, portfolio),
the result is simply unused (cost: about $0.0000004 per request). A second, independent switch keeps recent embeddings so that the same question again needs no embedding call. Q&A and tax routes, 30 runs each, four server configurations (`FINNIE_EMBEDDING_PREFETCH`, `FINNIE_EMBEDDING_CACHE_SIZE`):

| Configuration | Q&A p50 | Q&A p95 | Tax p50 | Tax p95 |
|---|---|---|---|---|
| A: neither | 1.86 s | 2.32 s | 2.05 s | 3.06 s |
| B: prefetch only | 1.57 s | 2.07 s | 1.80 s (1.77 and 1.97 in two repeats) | 2.50 s |
| C: cache only | 1.69 s | 2.00 s | 2.05 s | 2.85 s |
| D: both (shipped) | 1.59 s | 1.96 s | 1.73 s | 2.48 s |

Prefetch is worth about **0.25-0.29 s** on the routes that search. The cache helps only questions asked more than once (here two thirds of the Q&A requests were repeats): -0.17 s on Q&A, nothing visible on Tax, within the noise.
The stage table after the change shows the overlap directly: stage times now add up to about 0.2 s more than the wall time, because the embedding runs beside the router (Section 3's `other` column goes negative).
**One unexplained outlier:** in the first run of configuration B one tax request took 19.2 s (cost and result normal, not degraded). Sixty further requests in the same configuration with the server log kept (max 2.6 s, no retries, timeouts or errors in the log) did not reproduce it.
I cannot say whether it was OpenAI, the network or my code; it is in the raw file and I would watch for it.

## 6. After all three changes

Same scripts, same requests, defaults on. Warm, 30 runs per route:

| Route | p50 before | p50 after | change | p95 before | p95 after | change |
|---|---|---|---|---|---|---|
| Q&A | 1.99 s | 1.53 s | -0.46 s | 2.53 s | 2.00 s | -0.53 s |
| Market | 2.50 s | 2.35 s | -0.15 s | 2.90 s | 2.62 s | -0.27 s |
| Portfolio | 3.19 s | 2.76 s | -0.43 s | 3.98 s | 3.38 s | -0.60 s |
| Goal | 3.54 s | 3.47 s | -0.07 s | 4.01 s | 3.74 s | -0.27 s |
| News | 3.62 s | 3.47 s | -0.15 s | 4.71 s | 3.88 s | -0.84 s |
| Tax | 2.05 s | 1.85 s | -0.20 s | 2.80 s | 2.77 s | -0.03 s |

* **Changed routes:** Q&A -0.46 s, Portfolio -0.43 s, Tax -0.20 s at p50.
* **Routes I did not touch** (Market, Goal, News) moved by -0.07 to -0.15 s at p50 and up to -0.84 s at p95.
  That is the **noise floor** of this setup (run-to-run drift of the model and data providers), so differences smaller than about 0.15 s at p50 and about 0.8 s at p95 in these tables should not be believed (the untouched News route's p95 fell by 0.84 s by itself: a p95 from 30 runs is the second-slowest request). That means the p50 gains on Q&A and Portfolio are real but their p95 gains are not proven. The Q&A figure (-0.46 s) is larger than the isolated tests (-0.29 s) partly for that reason.
* **Cold starts** (3 samples per route) did not improve and cannot be distinguished from noise: Goal 3.51 to 3.91 s, News 3.68 to 3.92 s, Portfolio 4.41 to 4.20 s, Q&A 2.32 to 2.01 s. Cold costs 0 to 1.2 s more than warm at p50 (Portfolio most, Goal and News nothing); the first request to a server pays for lazy imports and connections.
* **Cost per query** is unchanged to the fifth decimal.

Load after the changes (same script; 20 s per level including 50 users, so the 50-user row ran longer than the baseline's 15 s and saw more of the tail):

| Users | requests | req/s | p50 | p95 | p99 | max | errors | peak memory | mean CPU |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 0.4 | 2.09 s | 3.73 s | 4.00 s | 4.07 s | 0 | 209 MB | 1% |
| 5 | 46 | 2.0 | 2.33 s | 3.93 s | 4.21 s | 4.22 s | 0 | 239 MB | 6% |
| 10 | 94 | 4.3 | 1.81 s | 3.80 s | 4.93 s | 6.28 s | 0 | 232 MB | 10% |
| 25 | 232 | 10.3 | 2.06 s | 3.70 s | 4.29 s | 5.27 s | 0 | 244 MB | 18% |
| 50 | 390 | 17.0 | 2.50 s | 4.38 s | 6.45 s | 10.24 s | 0 | 255 MB | 24% |

Load is where nothing changed, as expected: the changes remove 0.3-1.1 s from specific routes, not model capacity. **Largest level tested with p95 under 8 s and no errors: 50 users. The limit was not reached.**
The likely first limits, in order, are untested guesses: OpenAI rate limits for the account (429s would appear as retries, then errors), FastAPI's thread pool for blocking endpoints (about 40 threads by default; the 50-user row is the first beyond it, and its p50 rose from 2.06 to 2.50 s and max to 10.2 s),
and the PostgreSQL pool (10 connections, held for about 15 ms per search). Memory is not a concern: 255 MB at 50 users.

## 7. What was not done, and what these numbers cannot tell you

* **Streaming** would improve *perceived* speed (the first words after the router and retrieval, about 1.0-1.5 s, instead of the whole answer at 2-3.5 s) without changing total time. It needs the streaming API through the graph, FastAPI and the UI. Not built.
* **A smaller answering model, or shorter answers**, would cut the largest stage (the answer is 1-2.7 s). Portfolio, Goal and News answers decode 230-300 tokens at about 125 tokens/s. Both change answer quality, so they need the Step 10 evaluation before anyone adopts them; I only measured where the time is.
* **Skipping the router** for obvious messages (rules before the model) would save 0.7 s on those requests and risks the 98-100% routing accuracy; the regression gate (`make eval-gate`) is the safety net if anyone tries.
* **Cache hit rates over HTTP were not measured.** The counters exist (`TTLCache.hit_rate`, `CachedEmbedder.hits`) but the API does not expose them. The only hit rate I know is by construction: two thirds of the Q&A requests in Section 5.3 were repeats.
* The load test is closed-loop with zero think time and a fixed mix; real traffic is burstier and has idle gaps. The 1-user row has only 9-10 requests, so its p95 is close to a max.
* Single machine, one network, one afternoon; a different hour or region would shift every absolute number. The data-provider part (yfinance, unofficial) is the least stable.
* Each latency cell is 30 runs (cold: 3). A p95 from 30 runs is the second-worst request, not a stable percentile.
* **Spend:** about **$8.1** of API cost reported by the app across all of Step 25 (latency runs $2.8 including the repeated A/B variants, load $4.7, stage profiling $0.6, plus about $0.01 for the router-prompt experiment). More than the $3 I estimated at the start: the repeated A/B runs and the 50-user levels were the difference.
* Tests added with the changes: concurrent fetching (a barrier that only opens if three price calls are in flight at once), order of failure notes, the embedding wrapper (12 tests, including a deadlock I introduced and fixed, and a cache-eviction bug found before measuring), and the graph's prefetch hook.

## 8. Reproducing

```
.venv/bin/python -m benchmarks.bench_latency --label baseline         # about 8 minutes, about $0.9
.venv/bin/python -m benchmarks.profile_run --label baseline --cprofile  # about 2 minutes, about $0.3
.venv/bin/python -m benchmarks.bench_load --users 1 5 10 25 50 --label baseline   # about 5 minutes, about $2.3
.venv/bin/python -m benchmarks.exp_router_prompt                       # about $0.01
.venv/bin/python -m benchmarks.exp_portfolio_fetch                     # real yfinance calls, no model cost
```
Switches used for the A/B runs: `FINNIE_FETCH_THREADS`, `FINNIE_EMBEDDING_PREFETCH`, `FINNIE_EMBEDDING_CACHE_SIZE` (pass with `--env KEY=VALUE`). Set `BENCH_SERVER_LOG=file` to keep the server's log.

Raw results: baseline latency `step25_latency_baseline_20261101_142524.json`, after `step25_latency_after_20261101_150608.json`, stages `step25_stages_baseline_20261101_142850.json` / `step25_stages_after_20261101_150953.json`,
load `step25_load_baseline_20261101_143215.json` + `step25_load_baseline_50_20261101_143308.json` / `step25_load_after_20261101_151340.json`, portfolio fetch `step25_exp_portfolio_fetch_20261101_143644.json`, router prompt `step25_exp_router_prompt_20261101_143411.json`.
