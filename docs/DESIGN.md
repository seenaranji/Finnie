# Finnie: technical design

Audience: an engineer who has to maintain or extend Finnie. Everything measured here was measured on one Mac between 2026-10-29 and 2026-11-02 with the real services (OpenAI, Yahoo Finance, public RSS feeds, a Docker PostgreSQL). Datasets are small and labelled by the author, so treat differences of a question or two as noise. Setup is in the [README](../README.md); performance detail is in [BENCHMARKS.md](../BENCHMARKS.md).

## 1. Principles
1. **The model explains; code calculates and fetches.** Prices come from data providers, arithmetic from Python (`portfolio_calc`, `goal_calc`, `stats`, `real_estate`), facts from retrieved documents. Numbers in answers are checked against tool output.
2. **Education, not advice.** Enforced in code (an advice-request detector, output checks, mandatory disclaimers, a fixed refusal for out-of-scope questions) and tested, not only requested in prompts.
3. **Fail soft and say so.** Every agent node catches any exception and returns a fixed, safe reply; stale prices are served flagged (`is_stale`) rather than as errors; the user never sees a stack trace.
4. **Measure, change one thing, re-measure.** Thresholds and prompts were chosen from datasets with dev/test splits; the regression gate (`make eval-gate`, `eval/baseline.json`) lives on the `main` branch.

## 2. Architecture
```
Streamlit UI -> FastAPI -> LangGraph graph -> agents -> tools (market, news, retriever, calculators)
                   |                                         |
                   +-- auth, portfolios ---> PostgreSQL <----+  (users, portfolios, holdings, properties, chunks + vectors + full-text index)
```
- **UI** (`src/web_app/app.py`, `tabs/`): Streamlit; talks only to the API (never the database); keeps the login token in the tab's session state.
- **API** (`src/web_app/api.py`): FastAPI. Request validation with Pydantic, one error shape, request ids, CORS limited to the UI origin, every route except health/register/login behind a bearer-token dependency. Heavy objects (retriever, graph, market service) are built lazily once (`services.py`).
- **Graph** (`src/workflow/`): LangGraph, section 3.
- **Data layer** (`src/data/`): market providers behind one interface, TTL cache, retry with jitter, fallback; news; repositories; Alembic migrations in `migrations/`.
- **MCP server** (`src/mcp_server/`): the same functions exposed to desktop AI clients (see [MCP.md](../MCP.md)).

## 3. Agent communication (the graph)
State is one typed dictionary (`state.py`) passed between nodes: message, history, route, per-agent slots, `sub_results`, `trace`, `error`. A checkpointer keyed by `session_id` gives memory (in memory only: restarting the API loses conversations).

`contextualize -> router -> (conditional edge) -> one agent | clarify | scope_reply | fallback | multi -> finalize`
- **contextualize** rewrites "and how much is it up this year?" into a standalone question from the last three exchanges (no model call on the first turn).
- **router**: one cheap `gpt-4o-mini` call with a strict output schema: a label from nine (`finance_qa, market, portfolio, goal, news, tax, multi, clarify, out_of_scope`) plus a confidence. Policy lives in code: low confidence becomes `clarify`; an error becomes `fallback`; a user with an uploaded portfolio is never blocked as out of scope.
- **scope_reply**: off-topic questions get a fixed template before any retrieval, tool or answering model runs.
- **multi** (several intents in one message): a planner (`gpt-4o-mini`) splits it into tasks; sibling tasks start together with LangGraph `Send`, each with its own timeout; the only dependency allowed is "needs the portfolio first" (invented dependencies had serialized 2 of 20 plans, so others are dropped in code). A synthesizer merges the branches, checks the merge in code (no invented content, lists what is missing) and sets one disclaimer. Partial failure returns the branches that worked, marked `degraded`.
- **Agents** (`src/agents/`, all subclass `BaseAgent`, which owns the guardrails, disclaimer and cost/latency accounting): Finance Q&A and Tax (retrieval + cited answer), Market (tool calling over quotes and history; percentages computed in Python), Portfolio (CSV or saved portfolio -> prices fetched in parallel -> metrics in Python -> model explains the METRICS block), Goal (strict-schema slot extraction over several turns -> Python plan -> explanation), News (feeds -> sanitize -> numbered items -> summary with citations).

## 4. RAG details
- **Corpus:** 51 markdown articles from primary sources (SEC, IRS, Investor.gov, Fed, ...), each with front matter (source, date, category, difficulty); 272 chunks. Tax limits live in dated articles, not in prompts.
- **Chunking:** structured (headings first, then a 500-character size limit, 50 overlap). Sizes 200/500/1000 gave differences of about one question, so 500 stayed.
- **Embeddings:** `text-embedding-3-small` (1536 dimensions). The store records the model name and refuses a mismatch (a silent mix-up gives plausible but meaningless scores). Recent question embeddings are cached; the next question's embedding starts while the router runs.
- **Store:** PostgreSQL with pgvector for vectors and PostgreSQL full-text search for keywords; both are rebuilt by `scripts/index_knowledge_base.py` (full rebuild, about $0.001).
- **Retrieval:** dense search takes 20 candidates, applies the similarity threshold **0.465** (the midpoint of the gap between answerable and unanswerable questions in the tuning set), drops near-duplicate chunks, returns the top 5. If nothing passes, the agent answers with a fixed "I don't have that" text and **no model call**. Hybrid (BM25 + reciprocal rank fusion) is built and switchable (`rag.retrieval_mode`) but off: on the test split it was identical to dense.
- **Answering:** rules in the system message, context in the user message, passages labelled with chunk ids; citations are parsed and each cited id must have been retrieved (checked in code).
- **Untrusted text:** news and documents are fenced in a labelled block with a random boundary, sanitized (HTML, hidden and bidirectional characters, fake delimiters), instruction-like items are quarantined, and the output is validated (only supplied links, only real item numbers).

## 5. Data model
Tables: `users` (Argon2id hash), `portfolios` (owner, UUID id), `holdings` (asset type, purchase date, exact NUMERIC), `properties` (value, mortgage, rent, expenses; UNIQUE name per portfolio, CHECKs), and the knowledge-base chunks with vector and generated full-text columns. Money uses `Decimal`/NUMERIC. Every portfolio query takes `owner_id`; another user's id gives 404. Schema changes are Alembic migrations 0001-0005.

## 6. Security
Passwords: Argon2id, 10-128 characters, one error message for wrong password and unknown email, per-email-and-address lockout (5 failures, 15 minutes). Tokens: signed JWT with issuer, subject and expiry (60 minutes); algorithm `none` and tampered tokens are rejected. Secrets only in `.env`/environment; the container image contains none. SQL is parameterized. Text from models, feeds and the database is escaped before display (a pair of `$` signs would otherwise render as a formula).

## 7. Results (all numbers are from the repository's saved runs)
| Area | Result | Note |
|---|---|---|
| Retrieval, test split (21 q, 16 answerable) | Hit@3 0.94, MRR 0.94, no-answer accuracy 5/5 | targets 0.85 / 0.7 / 0.9; the one miss is "what is a dividend" (short "what is X" scores below the threshold) |
| Retrieval, dev split | Hit@3 0.75 dense at the shipped threshold (0.88 at 0.40, at the cost of one wrong answer) | threshold tuned on dev; dev numbers are not a fair estimate |
| Answer quality, improved prompt (30 q) | faithful 21/21, cited 14/14, valid citations 21/21, correct refusal 6/6, advice leak 0/6 by rules | weak prompt: faithful 12/21, no citations. The LLM judge is paused; rules are partly miscalibrated |
| Router (128 queries) | 98-100% on held-out sets (regression gate: test 0.979, test2 1.0) | labels are the author's; the original test set is contaminated by prompt tuning |
| Out-of-scope guard | 0 false blocks in 50 in-scope questions; 3/4 to 6/6 catch on stress sets | zero false blocks bounds the true rate only to about 6% |
| Multi-agent | end-to-end speedup 1.43x p50 (expected 1.6-1.8x, missed); partial-failure scenarios 10/10 after fixes | fixes were shaped by the first run |
| Prompt injection (news, 90 runs) | obeyed (strict) 3/30 naive, 0/30 with prompt defense, 0/30 full | the undefended model already resisted most attacks |
| Fault injection through the full graph | 11/11 graceful | e.g. provider killed mid-run: last saved price served, flagged |
| Latency, warm p50 / p95 (30 runs per route, after optimizations) | Q&A 1.53 / 2.00 s, Market 2.35 / 2.62, Portfolio 2.76 / 3.38, Goal 3.47 / 3.74, News 3.47 / 3.88, Tax 1.85 / 2.77 | 86-100% of time is waiting for OpenAI |
| Cost per query | $0.002 (Q&A) to $0.006 (goal) | |
| Load | 50 simultaneous users: p95 4.4 s, 0 errors, 255 MB, CPU 24% | limit not found; closed loop with no think time |
| Tests | 716 passing, coverage 95% | `make test`, `make cov` |
| Docker | image 1.26 GB; healthy API 18 s after `up` (images cached); idle memory api 155 MiB, ui 66 MiB, db 22-33 MiB | volume survives `down`, not `down -v` |
| MCP | 10/10 calls succeeded per tool; protocol adds about 1-5 ms | |

## 8. Decisions and trade-offs
- **PostgreSQL for everything** (vectors, keyword search, portfolios, users) instead of Chroma/SQLite: one engine locally and in production, transactions across related rows. Cost: a database is needed to run anything.
- **Dense retrieval over hybrid:** hybrid's dev gain did not repeat on test; added complexity without proof.
- **A threshold gate with no model call on a miss:** cheaper and safer than asking the model to refuse; cost is some wrongly refused short questions.
- **One image, several commands:** simpler to build and version; the image carries libraries the UI does not need.
- **Optimizations kept only when measured:** parallel price fetching (-1.06 s on new portfolios), embedding prefetch (-0.25 to -0.29 s), embedding cache (repeat questions only). Rejected: a shorter router prompt (saves 0.12 s, risks routing quality).

## 9. Known limitations and next steps
1. No TLS; put a proxy with HTTPS in front before any public use. `JWT_SECRET` belongs in a secrets manager.
2. Sessions (conversation memory) and the login throttles live in process memory: lost on restart, per worker. A shared store (Redis or a table) fixes both.
3. Tokens cannot be revoked; a deleted user's token works until it expires. No password reset, email verification or two-factor sign-in.
4. Prices come from unofficial Yahoo data; check each news publisher's terms before a public deployment.
5. Streaming responses (better perceived speed) were not built. Load limits were not reached; likely first limits are OpenAI rate limits, the API's thread pool and the 10-connection database pool.
6. Knowledge-base articles were written from primary sources by the assistant and still need a human accuracy review.
7. The asset type on a holding is the user's label and is not checked against the ticker.
