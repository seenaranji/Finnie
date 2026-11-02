# Finnie task runner. Two worlds, kept apart on purpose:
#   make test / make cov   free, fast, offline-safe: fake models and fake markets (database tests skip if PostgreSQL is not running)
#   make eval ...          REAL models, REAL embeddings, the real database: costs money (about $0.10 for `make eval`) and takes minutes
PY := .venv/bin/python

.PHONY: mcp mcp-check test cov eval eval-retrieval eval-answers eval-router eval-gate eval-gate-update eval-all db-up

mcp:                 ## Step 28: run the MCP server on stdio (a client normally starts it; this is for a manual look)
	$(PY) scripts/run_mcp_server.py

mcp-check:           ## Step 28: live check over real stdio: success rate and latency vs direct calls (about $0.005)
	$(PY) -m scripts.step28_mcp_check

db-up:                ## start PostgreSQL (needed by the database tests and every eval)
	docker compose up -d

test:                 ## unit + integration tests (no network, no model calls, no cost)
	$(PY) -m pytest -q

cov:                  ## the same tests with a coverage report (target: 80%+)
	$(PY) -m pytest -q --cov=src --cov-report=term-missing:skip-covered

eval-retrieval:       ## Step 8: retrieval quality on the dev and test splits (about $0.001)
	$(PY) -m eval.scripts.eval_retrieval --split dev
	$(PY) -m eval.scripts.eval_retrieval --split test

eval-answers:         ## Step 10: answer quality, rule-based checks, improved prompt (about $0.05)
	$(PY) -m eval.scripts.eval_answers --prompts improved

eval-router:          ## Steps 20-21: router accuracy on the held-out splits with the default prompt (about $0.01)
	$(PY) -m eval.scripts.eval_router --prompt v6 --split test
	$(PY) -m eval.scripts.eval_router --prompt v6 --split test2

eval-gate:            ## regression gate: exit 1 if retrieval Hit@3 or router accuracy fell below eval/baseline.json (about $0.01)
	$(PY) -m eval.scripts.check_baseline

eval-gate-update:     ## record today's numbers as the new baseline (do this on purpose, never to turn a red gate green)
	$(PY) -m eval.scripts.check_baseline --update

eval: eval-retrieval eval-answers eval-router eval-gate   ## everything above, in order; stops at the first failure

eval-all: eval        ## plus the slower Step 21 suites: scope guard, multi-agent, dialogs and fault injection (several more cents)
	$(PY) -m eval.scripts.eval_scope --split test --versions v5
	$(PY) -m eval.scripts.eval_multi_intent --split test
	$(PY) -m eval.scripts.eval_dialogs --dataset dialogs
	$(PY) -m eval.scripts.eval_faults
