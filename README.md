# Finnie

An educational personal-finance assistant. You chat with it, and a router sends each question to one of six specialist agents (finance Q&A over a knowledge base, market data, portfolio analysis, savings goals, news, tax education). You can also save portfolios (stocks, funds, bonds, cash and real estate) and see them grouped by type.

**Finnie explains; it does not advise.** It never says "buy this" or "file it this way", every answer carries a disclaimer, and numbers come from code and data feeds, not from the language model. See [docs/DESIGN.md](docs/DESIGN.md) for how it works and what was measured.

```
Streamlit UI :8501 --HTTP--> FastAPI :8000 --> LangGraph (router -> agents) --> OpenAI, Yahoo Finance, news feeds
                                   |
                                   +--> PostgreSQL + pgvector (users, portfolios, knowledge-base vectors)
```

## What you need
- **Docker Desktop** (or Docker Engine with the compose plugin) for the quick start, **or** Python 3.11 plus Docker for the database if you run from source.
- An **OpenAI API key** (required: answers, routing and embeddings). Optional: an Alpha Vantage key (backup price source).
- Cost: indexing the knowledge base is about $0.001; a chat question costs $0.002-$0.006.

## Quick start (everything in Docker)
```bash
cp .env.example .env
# edit .env: set OPENAI_API_KEY, POSTGRES_PASSWORD and JWT_SECRET, and put the same password in DATABASE_URL
# in place of <password> (see "Settings" below)
docker compose up -d --build                                   # database, migrations, API, UI (first build takes a few minutes)
docker compose run --rm api python -m scripts.index_knowledge_base   # fills the knowledge base once (about $0.001)
```
Open http://localhost:8501, open the **Create account** tab, make an account, and ask "What is an expense ratio?".
`docker compose ps` should show `db`, `api` and `ui` healthy (the `migrate` job exits by design). Stop with `docker compose down`; your data stays in a Docker volume. **`docker compose down -v` deletes the volume and every saved row.**

## Day-to-day: start, stop, restart
The code is copied into the Docker images, so **after you change any code, rebuild** or you will still see the old version.
| I want to… | Command |
|---|---|
| Start everything | `docker compose up -d` |
| Restart with my latest code changes | `docker compose up -d --build` |
| Restart without rebuilding (e.g. after editing `.env`) | `docker compose restart api ui` |
| Check it is healthy | `docker compose ps` (`db` and `api` should say healthy), or `curl localhost:8000/health` |
| See logs | `docker compose logs -f api` (or `ui`) |
| Stop, keeping saved data | `docker compose down` |
| Wipe everything, including saved data | `docker compose down -v` (cannot be undone) |

The UI is at http://localhost:8501 and the API docs at http://localhost:8000/docs. If you start from a clean checkout, do the one-time setup in "Quick start" first (`.env` and the knowledge-base index).

## Run from source (for development)
```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                       # fill it in, as above
docker compose up -d db                    # only the database
.venv/bin/alembic upgrade head             # create the tables
.venv/bin/python -m scripts.index_knowledge_base
.venv/bin/uvicorn src.web_app.api:app --port 8000            # terminal 1: the API
.venv/bin/streamlit run src/web_app/app.py                   # terminal 2: the UI (http://localhost:8501)
```

## Settings (`.env`, never committed)
| Variable | What it is |
|---|---|
| `OPENAI_API_KEY` | required |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT` | database account for the compose database (port 5433 on your machine) |
| `DATABASE_URL` | `postgresql://USER:PASSWORD@127.0.0.1:5433/DB`, used when running from source. In Docker, compose replaces it with the internal address. |
| `JWT_SECRET` | signs login tokens, at least 32 characters: `python -c "import secrets;print(secrets.token_urlsafe(48))"` |
| `ALPHA_VANTAGE_API_KEY` | optional backup for prices (free tier: 25 requests a day) |
| `LANGSMITH_API_KEY` | optional, only for uploading evaluation runs |
| `API_PORT`, `UI_PORT` | optional host ports for Docker (defaults 8000, 8501) |

Behaviour (models, thresholds, cache times, login limits, planning assumptions) lives in [config.yaml](config.yaml); it contains no secrets.

## Using it
- **Chat:** ask in plain words. Follow-ups work ("and how much is it up this year?"). A question with several parts ("price of VOO and explain my portfolio") runs the agents side by side.
- **Markets:** quotes, history charts, index overview. **News:** summaries with sources. **Goals:** how much to save per month under three return scenarios. **Knowledge:** search the articles.
- **Portfolio chat:** under a saved portfolio (or an analyzed file) there is an *Ask about this portfolio* box. Questions are answered with that portfolio's holdings attached, in their own conversation. Hypotheticals such as "what if the market is up tomorrow?" or "what if I buy X?" are not yet computed; see [docs/DESIGN.md](docs/DESIGN.md) for limits.
- **Portfolio:** (1) *Enter my holdings* in a table (symbol, type, shares, average cost, date; plus real-estate properties with mortgage and rent), save it and get a unique id; (2) *My saved portfolios*: grouped by type with cost, value, gain, share of total; (3) *Analyze a file or sample*: upload a CSV (`ticker,shares,avg_cost`); samples are in [src/data/sample_portfolios/](src/data/sample_portfolios/).

## API
Interactive documentation with every field: http://localhost:8000/docs. Everything except `/health`, `/auth/register` and `/auth/login` needs `Authorization: Bearer <token>`.
| Endpoint | Purpose |
|---|---|
| `POST /auth/register`, `POST /auth/login`, `GET /auth/me` | accounts and tokens (valid 60 minutes) |
| `POST /chat` | the full assistant; keeps a conversation per `session_id` |
| `GET /market/quote/{ticker}`, `/market/history/{ticker}`, `/market/overview` | prices; each carries its source and an `is_stale` flag |
| `GET /news/market`, `/news/ticker/{ticker}` | news with source links |
| `POST /goals/plan` | savings plan (arithmetic only, no model) |
| `GET /knowledge/search`, `/knowledge/articles[/{id}]` | the knowledge base |
| `POST /portfolio/analyze` | explain a portfolio from CSV text |
| `POST/GET /portfolios`, `GET/PUT/DELETE /portfolios/{id}`, `GET /portfolios/{id}/summary`, `GET /portfolios/{id}/analysis` | saved portfolios (yours only) |

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login -H 'content-type: application/json' \
  -d '{"email":"you@example.com","password":"your-password"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -s -H "Authorization: Bearer $TOKEN" localhost:8000/market/quote/AAPL
curl -s -X POST localhost:8000/goals/plan -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"goal":100000,"years":10,"current_savings":5000,"risk_profile":"medium"}'
curl -s -X POST localhost:8000/chat -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"message":"What is an expense ratio?"}'
```
Errors always have the same shape (`error.code`, `error.message`, `request_id`); every response carries an `X-Request-ID` header to quote when reporting a problem.

## MCP server (use Finnie from Claude Desktop)
Four read-only tools (`get_quote`, `analyze_portfolio`, `plan_goal`, `search_knowledge`) over stdio. Setup and protocol notes: [MCP.md](MCP.md).

## Tests and evaluation
```bash
make test        # unit and integration tests, free and offline (database tests are skipped if PostgreSQL is not running)
make cov         # the same with coverage (95% at last run)
make eval        # REAL models, about $0.10: retrieval, answer quality, router accuracy, then a regression gate against eval/baseline.json
make mcp-check   # live check of the MCP server
```
Performance numbers and how to reproduce them: [BENCHMARKS.md](BENCHMARKS.md).

## Troubleshooting
| Symptom | Cause and fix |
|---|---|
| `DATABASE_URL is not set` | `.env` missing or not filled in (run from source). |
| `connection refused` on port 5433 | database not running: `docker compose up -d db`. |
| Chat answers "I don't have that in my knowledge base" for everything | the knowledge base was never indexed: run the `index_knowledge_base` command. |
| Register/login says the secret is missing or too short | set `JWT_SECRET` (32+ characters) and restart the API. |
| `429` / "too many attempts" | five wrong passwords per email and address locks sign-in for 15 minutes (limits are in memory, so restarting the API clears them). |
| Prices missing or "may be out of date" | Yahoo Finance is an unofficial source and sometimes fails; Finnie then shows the last saved price, flagged. Add an Alpha Vantage key for a backup. |
| After a page reload you are logged out | by design: the token lives only in the browser tab's session. |
| `docker compose up` says port in use | set `API_PORT`/`UI_PORT` (or `POSTGRES_PORT`) in `.env`. |
| Docker: saved data vanished | the volume was deleted (`down -v`); restore is not possible. |
| A Docker build is slow the first time | about 1.3 GB of Python packages; later builds reuse the cached layer unless `requirements.txt` changes. |

## Known limits
No HTTPS (put a TLS proxy in front before exposing it to the internet; the compose file publishes to localhost only). No password reset or email verification. Login throttles are per process. Prices come from an unofficial Yahoo feed. Educational use only, not financial advice. More in [docs/DESIGN.md](docs/DESIGN.md).
