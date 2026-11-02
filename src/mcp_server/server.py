"""Finnie as an MCP server (stdio). Run: python -m src.mcp_server

STDIO RULE: stdout carries the protocol, so nothing else may print to it. Finnie's logger writes to stdout by default; main() points it at stderr first.
"""
import logging
import sys
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from src.core.logging import setup_logging
from src.mcp_server.tools import FinnieTools

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)


def build_server(tools: FinnieTools | None = None) -> MCPServer:
    tools = tools or FinnieTools()
    server = MCPServer("finnie", instructions="Finnie: educational personal-finance tools (quotes, portfolio explanation, savings-goal plans, a finance knowledge base). "
                                              "Everything is educational, not financial advice; always pass the disclaimer on to the user.")

    @server.tool(annotations=READ_ONLY)
    def get_quote(ticker: str) -> dict:
        """Latest price for a stock, ETF or index (e.g. AAPL, BRK-B, ^GSPC). Includes source, as_of date and an is_stale flag."""
        return tools.get_quote(ticker)

    @server.tool(annotations=READ_ONLY)
    def analyze_portfolio(portfolio_id: str | None = None, holdings_csv: str | None = None, question: str = "Please explain my portfolio in simple terms.") -> dict:
        """Explain a portfolio in plain language with its metrics. Give EITHER portfolio_id (a saved portfolio of the configured account) OR holdings_csv (CSV text with columns ticker,shares,avg_cost). Calls an AI model, so it costs a fraction of a cent."""
        return tools.analyze_portfolio(portfolio_id, holdings_csv, question)

    @server.tool(annotations=READ_ONLY)
    def plan_goal(goal: float, years: float, current_savings: float = 0, risk_profile: Literal["low", "medium", "high"] = "medium") -> dict:
        """Savings plan for a target amount (in future dollars) in a number of years: monthly amount needed under three return scenarios. Pure arithmetic, no AI model, repeatable."""
        return tools.plan_goal(goal, years, current_savings, risk_profile)

    @server.tool(annotations=READ_ONLY)
    def search_knowledge(query: str, category: str | None = None, top_k: int = 5) -> dict:
        """Search Finnie's finance knowledge base (ETFs, bonds, taxes, retirement accounts...). Returns passages with their sources; answerable=false means nothing relevant was found."""
        return tools.search_knowledge(query, category, top_k)

    return server


def main() -> None:
    setup_logging("INFO")
    for handler in logging.getLogger().handlers:   # protect the protocol channel
        if getattr(handler, "stream", None) is sys.stdout:
            handler.setStream(sys.stderr)
    build_server().run("stdio")
