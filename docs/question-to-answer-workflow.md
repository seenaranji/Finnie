# Finnie: from a user's question to the answer

Based on the code as of 2026-11-02. Main files: `src/workflow/graph.py`, `router.py`, `planner.py`, `synthesizer.py`, `src/web_app/api.py`. Read from the code, not from a run.

## 1. End-to-end picture

```
Streamlit (chat.py) --HTTP--> FastAPI POST /chat --> LangGraph (checkpointed per session)
   FinnieClient                 auth gate, session lock         |
                                                                v
 contextualize -> router -+- finance_qa | market | portfolio | goal | news | tax --+
   (resolve "it")         +- clarify (fixed reply) -------------------------------+
   + start embedding      +- out_of_scope -> scope_reply (fixed, logged) ----------+-> finalize -> END
                          +- fallback (router failed) -----------------------------+  (disclaimer +
                          +- multi -> planner -> run_task xN (parallel waves) -----+   save history)
                                          ^            |
                                          +- after_tasks -> synthesize
```

Only the router, the agents, the planner and the synthesizer call a model. `clarify`, `scope_reply` and `fallback` are fixed text with no model call.

## 2. Before the graph (UI and API)

| Step | Where | What happens | If it fails |
|---|---|---|---|
| 1 | `app.py` | Without a login the user sees only the login screen. After login the token is attached to every call. | No tabs, no API calls |
| 2 | `tabs/chat.py` | Input capped at 2000 characters. Empty message shows a warning and nothing is sent. The message is shown at once and a spinner starts. | Warning shown |
| 3 | `client.py` | `POST /chat` with `message`, `session_id` (None on first turn), `portfolio_csv`. Timeout 60 s. | Timeout, no connection or HTTP error becomes one `ApiError` with a safe message. Connection errors and 5xx are marked retryable. |
| 4 | `api.py` middleware | Request id assigned (caller's accepted if short and plain); timing starts. | - |
| 5 | `require_user` | Verifies the bearer token (signature, expiry). | Missing/invalid token gives 401 |
| 6 | `ChatRequest` | `message` 1-2000 chars; `session_id` matches `[A-Za-z0-9_-]{1,64}`; `portfolio_csv` at most 20 KB. | 422 with per-field details |
| 7 | `chat()` | `session_id` generated if absent. Internal id is `"<user_id>:<session_id>"`, so another user sending your session id starts a separate conversation. | - |
| 8 | `session_lock` | One turn per conversation at a time; different sessions never wait. | A second simultaneous turn in the same session waits |
| 9 | `graph.invoke(...)` | `thread_id = internal_id`, `max_concurrency` 4. The checkpointer restores history, `goal_state`, `portfolio_csv`. | Unexpected exception becomes a 500 `internal_error` with the request id |

## 3. Inside the graph: shared steps

**3.1 `contextualize`**
- Turn 1: no history, message used as written, no model call.
- Later turns: a cheap model (`gpt-4o-mini`, temperature 0, strict JSON) rewrites the message as a standalone question using the last 3 exchanges ("and BP?" becomes "How is BP doing?").
- If the call fails, the JSON is bad, or the result is empty or over 3x the message + 200 chars, the original message is used. Resolving never blocks a turn.
- Fires `prefetch(resolved)`, which starts embedding the question while the router decides. Unused if the route needs no search.

**3.2 `router`**
- One classification with a strict schema: route (`finance_qa`, `market`, `portfolio`, `goal`, `news`, `tax`, `clarify`, `out_of_scope`, `multi`) and confidence (high/medium/low). Prompt v6 is the default.
- Code policies:
  - Low confidence becomes `clarify`.
  - Unusable reply (`fell_back`) becomes `clarify`.
  - `out_of_scope` with an uploaded portfolio becomes `clarify`.
  - The finance-word veto exists but is off by default.
- If the router's model call throws, the route is `fallback`.

## 4. Per-route flows

Every route except `goal` and `portfolio` entry points goes through `BaseAgent.handle()`: detect an advice request (adds a "no personal recommendation" instruction), call the model, scan the output for directive phrases, replace a leaking answer with a fixed redirect. Disclaimer is strong if advice was requested or the output leaked.

**`finance_qa`** (`agents/finance_qa_agent.py`)
1. Retrieve from the knowledge base: cached/prefetched question embedding, pgvector search (dense default; hybrid optional), similarity gate 0.465, dedupe, optional rerank, top 5.
2. Nothing passes the gate: fixed "I don't have information on that" reply, **no model call**.
3. Otherwise the model answers from the passages only, with `[chunk-id]` citations.
4. Code builds `sources` from cited ids that were really retrieved; invented ids are logged and dropped.

**`tax`** (`agents/tax_agent.py`)
- Same pipeline limited to `taxes` and `retirement_accounts`. Passages carry source and last-updated date; dollar limits must state the tax year.
- A personal-situation detector adds a "explain generally, decide nothing" instruction. A tax leak check ("you qualify", "you owe") swaps the answer for a CPA/IRS redirect, except for first-person refusals.
- Always the strong disclaimer.

**`market`** (`agents/market_agent.py`)
- Tool loop, at most 5 model turns, with `get_quote`, `get_history`, `get_stats`. Tools validate arguments and never raise; errors return as `{"error": ...}`. Statistics are computed in Python.
- Each call goes through `MarketDataService`: fresh cache hit returns with no network; a miss goes to the provider chain with retries and backoff, falling to the next provider only on `ProviderUnavailable`; an unknown ticker fails at once with no retry; if all providers fail the last cached value is served with `is_stale=True`, or a clear error if nothing is cached.
- Hitting 5 steps gives "I couldn't finish looking that up."
- Sources are built from successful tool results.

**`portfolio`** (`agents/portfolio_agent.py`)
- No `portfolio_csv`: fixed "I need your holdings" reply, no model call.
- Bad CSV: fixed message listing the problems, no model call.
- Otherwise one quote and one year of history per holding are fetched in parallel (up to 8 threads). Python computes weights, HHI, volatility, correlation and the diversification score.
- A holding that fails to fetch becomes a note in the metrics. If **no** holding can be priced: fixed "no prices" message, no model call.
- The model sees only the rounded METRICS block. The correlation matrix goes to the UI heatmap, not to the model.
- Always the strong disclaimer.

**`goal`** (`agents/goal_agent.py`)
1. Extract four slots (amount, years, current savings, risk) with a strict schema using the stronger model. Null means not stated; nothing is guessed.
2. Merge with slots saved from earlier turns (corrections overwrite, nulls never erase). Stated but absurd values are dropped and asked again.
3. Any slot missing: a **templated follow-up question** (no second model call); partial `goal_state` is saved.
4. All four known: `goal_calc` computes three scenarios and flags in Python, the model explains the METRICS block, and `goal_state` is cleared.
5. Fixed messages for empty/over-long input, unreadable extraction, or invalid numbers.

**`news`** (`agents/news_agent.py`)
1. A cheap model extracts the topic (uppercase ticker or `MARKETS`); anything unusable becomes `markets`.
2. Headlines are fetched. Items that look like instructions are **quarantined** and reported.
3. No usable items: fixed "couldn't find recent news" reply, no model call.
4. Otherwise the model summarizes inside a random-boundary UNTRUSTED block. Code validates: citations must be real item numbers, links must be ones the articles supplied, no images, a length cap, at least one citation.
5. Validation failure: a **code-built headline list** replaces the model text. Sources and links come from code.

**`clarify`, `scope_reply`, `fallback`**: fixed replies at zero cost. `scope_reply` appends the decision (time, session, query, confidence) to `logs/out_of_scope.jsonl`; a full disk doesn't stop the reply.

**Any agent exception (single route):** `agent_node` catches it. The user gets a safe fixed reply and `state.error` is set.

## 5. The `multi` route

```
planner -> dispatch -> [run_task A || run_task B] -> after_tasks -> dispatch -> [run_task C] -> ... -> synthesize
```

1. **Planner** (cheap model, strict schema) produces 1-4 tasks with self-contained questions and `depends_on`. Code validates unique ids, known agents, existing dependencies, no cycles, task count, then drops every dependency except one on `portfolio`.
2. **Dispatch** starts every task whose dependencies are done, via LangGraph `Send`, up to 4 at once (one at a time in `sequential` mode). `after_tasks` re-dispatches until nothing is ready.
3. **`run_task`** runs the same agent callables as the single routes.
   - A failed dependency means the task is **skipped**.
   - Otherwise it runs under a 25 s timeout, with the dependency's result (up to 1500 chars) appended to its question.
   - Status is `ok`, `degraded`, `error`, `timeout` or `skipped`. `degraded` means the agent returned normally but did no real work.
   - A timed-out thread is abandoned but keeps running, so its model calls still cost money.
4. **Synthesize**:
   - Planner failure or empty plan: fixed fallback.
   - No successful branch: fixed "couldn't get any part" note.
   - Single-task plan: that agent's answer unchanged.
   - Otherwise the model merges. Code rejects the merge if it has numbers or citation markers absent from the branches, a link, ungrounded sentences, or over 2400 chars; then the code-built merge (each answer under a heading) is used.
   - Code always appends the missing-parts note.
   - Strong disclaimer if any task was portfolio, goal or tax.

## 6. Finishing the turn

**`finalize`:** strong disclaimer for portfolio, goal and tax routes, otherwise standard if none set; the assistant turn is appended to history.

**`chat()`** returns `session_id`, `turn`, `text`, `answered_by`, `route`, `sources`, `disclaimer`, `cost_usd` (agent + rewrite + router), `latency_s`, `degraded` (true only if `state.error` is set).

**UI:** saved to `session_state` and rendered as markdown with an "Answered by" badge, expandable sources and a caption with disclaimer, latency and cost.

## 7. Scenario matrix

| Scenario | Path | Model calls | What the user sees |
|---|---|---|---|
| Not logged in / bad token | `require_user` | 0 | 401, login screen |
| Empty or over-length message | UI check / `ChatRequest` | 0 | Warning or 422 |
| Concept question, KB covers it | contextualize, router, finance_qa | rewrite (turn 2+), router, answer | Cited answer plus sources |
| Concept question, KB doesn't cover it | gate empty | router only | "I don't have information on that..." |
| Follow-up ("what about BP?") | rewrite resolves, then normal route | +1 rewrite | Normal answer |
| Rewrite fails | original text used | - | Normal answer |
| Off-topic | router, `scope_reply` | router | Fixed scope message, logged |
| Off-topic, portfolio loaded | `out_of_scope` becomes `clarify` | router | Clarify prompt |
| Vague / low confidence | router, `clarify` | router | Clarify prompt |
| Router model down | `fallback` | 0 | Fixed sorry message, `degraded=true` |
| "Should I buy X?" | any agent; advice instruction and leak check | normal | Educational answer plus strong disclaimer, or redirect if output leaked |
| Price question, provider fine | market tool loop | router, 1-5 answer turns | Numbers with as-of and source |
| Price question, provider flaky | retry/backoff, then fallback provider | same | Normal answer |
| Price question, all providers down | stale cache | same | Answer flagged out of date |
| Price question, no cache | tool returns error | same | Model says plainly it's unavailable |
| Invalid/unknown ticker | no retry | same | "Ticker not found" |
| Portfolio question, no CSV | `call_portfolio` guard | router | "I need your holdings" |
| Portfolio question, malformed CSV | parse error | router | Problems listed |
| Portfolio question, some prices fail | note in metrics | router, answer | Answer plus "could not fetch..." note |
| Portfolio question, no prices | `no prices` | router | Try again later |
| Goal question, slots missing | follow-up template, state saved | router, extraction | Asks for the missing field |
| Goal question, all slots known | calc, then explanation | router, extraction, answer | Three-scenario plan |
| News, clean feed | fetch, summarize, validate | router, topic, answer | Cited summary, links from code |
| News, injected text | item quarantined | same | Summary of the safe items |
| News, output fails validation | code digest | same | Headline list |
| News, nothing found | `no usable news` | router, topic | "Couldn't find recent news..." |
| Two-part question | multi, planner, parallel tasks, synthesize | router, planner, N agents, merge | One merged answer |
| Multi, one part fails/times out | task not `ok` | same | Partial answer plus missing-parts note |
| Multi, dependency fails | dependent task skipped | same | Same, skipped part listed |
| Multi, all parts fail | `missing_note(everything=True)` | same | "I couldn't get any part..." |
| Multi, bad plan | `PlanError` | router, planner | Fixed fallback, `degraded=true` |
| Multi, merge fails or invents | concat fallback | same | Plain sections per agent |
| Agent throws mid-turn | `agent_node` catch | - | Safe fixed reply, `degraded=true` |
| Two simultaneous turns, same session | session lock | - | Second turn waits |
| Server slow/down | `FinnieClient` | - | `ApiError` message with retry hint |

## 8. Behaviors worth knowing

- **`degraded` and partial answers:** a partial multi-agent answer reports `degraded=false`, because the flag reflects only `state.error`. Partial status appears only in response metadata (`partial`, `missing`).
- **Timeouts and cost:** the branch timeout is 25 s, but the abandoned thread keeps running and is still billed. Cost is only counted for branches that finish.
- **Memory:** the default checkpointer is an in-process `InMemorySaver`, so conversation memory is lost on restart. Session locks are also in-process, which matters with more than one worker.
- **Guardrails are regex rules**, so rephrased requests or answers can slip through. They are measured, not trusted.
