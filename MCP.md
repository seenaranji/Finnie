# Finnie as an MCP server (Step 28)

**MCP (Model Context Protocol)** is an open standard that lets an AI desktop app (Claude Desktop, Cursor...) discover and call tools in a separate program. Without it, every app needs its own glue code for every tool; with it, Finnie writes its tools once and any MCP client can use them.

## Tools (all read-only; same code as the HTTP API)
| Tool | Inputs | Notes |
|---|---|---|
| `get_quote` | `ticker` | Price, source, `as_of`, `is_stale`. Ticker format checked before any lookup. |
| `analyze_portfolio` | `portfolio_id` **or** `holdings_csv` (columns ticker,shares,avg_cost), `question` | Exactly one source. Uses an AI model (about half a cent). Saved portfolios need the account below. |
| `plan_goal` | `goal`, `years`, `current_savings`=0, `risk_profile`=medium | Pure arithmetic, no model. |
| `search_knowledge` | `query`, `category`, `top_k`=5 | Passages with sources; `answerable:false` means nothing relevant. |

Failures come back as MCP tool errors (`isError: true`) with a plain message, never a stack trace or a secret.

## Connect Claude Desktop
Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (Settings > Developer > Edit Config), then quit and reopen Claude Desktop:
```json
{
  "mcpServers": {
    "finnie": {
      "command": "/Users/seenamani/Documents/Interview Kickstart/Finnie/.venv/bin/python",
      "args": ["/Users/seenamani/Documents/Interview Kickstart/Finnie/scripts/run_mcp_server.py"],
      "env": { "FINNIE_EMAIL": "you@example.com", "FINNIE_PASSWORD": "your-finnie-password" }
    }
  }
}
```
`env` is only needed for `analyze_portfolio` with a `portfolio_id` (the server signs in as that user, so ownership rules still apply). Quotes, goals, knowledge search and CSV analysis need no account. The `.env` file (API keys, database URL) is loaded by Finnie itself. PostgreSQL must be running for `search_knowledge` and saved portfolios (`make db-up`). The password sits in the client's config file in plain text: use a dedicated Finnie account.

## The protocol, briefly
- **Transport: stdio.** The client starts the server as a child process; messages are JSON-RPC 2.0, one per line, over its stdin/stdout.
- **Handshake:** client sends `initialize` (its protocol version and capabilities), the server answers with its own plus `instructions`; the client then sends `notifications/initialized`.
- **Discovery:** `tools/list` returns each tool's name, description and a JSON Schema for its inputs (generated from the Python type hints). The assistant reads these to decide when to call a tool.
- **Calling:** `tools/call` with a name and arguments; the arguments are validated against the schema, the result is text content (here JSON). Failures set `isError`.
- **stdout is the protocol.** Anything else printed there corrupts the stream. Finnie's logger writes to stdout by default, so `main()` redirects it to stderr first (a test checks this). Clients show stderr in their logs (Claude Desktop: `~/Library/Logs/Claude/mcp-server-finnie.log`).
- The launcher script finds the project by its own path because clients start servers from an unknown working directory.

## Run and measure
```
make mcp-check      # .venv/bin/python -m scripts.step28_mcp_check  (real subprocess over stdio, about $0.005)
.venv/bin/python -m pytest tests/test_mcp_server.py
```
Results are recorded in PROGRESS.md.

## Limits
Local stdio only (no remote/HTTP transport, so no multi-user auth beyond the one configured account); read-only (no saving or deleting portfolios); not clicked through in the real Claude Desktop app (verified with the official MCP client over a real subprocess).
