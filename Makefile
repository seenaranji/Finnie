# Finnie task runner (production branch). The evaluation suite, benchmarks and step demos live on the main branch.
#   make test / make cov   free, fast, offline-safe: fake models and fake markets (database tests skip if PostgreSQL is not running)
PY := .venv/bin/python

.PHONY: mcp mcp-check test cov db-up

mcp:                 ## Step 28: run the MCP server on stdio (a client normally starts it; this is for a manual look)
	$(PY) scripts/run_mcp_server.py

mcp-check:           ## Step 28: live check over real stdio: success rate and latency vs direct calls (about $0.005)
	$(PY) -m scripts.step28_mcp_check

db-up:                ## start PostgreSQL (needed by the database tests)
	docker compose up -d

test:                 ## unit + integration tests (no network, no model calls, no cost)
	$(PY) -m pytest -q

cov:                  ## the same tests with a coverage report (target: 80%+)
	$(PY) -m pytest -q --cov=src --cov-report=term-missing:skip-covered
