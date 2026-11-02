"""Launcher for MCP clients (Claude Desktop starts servers from an unknown working directory, so this file finds the project itself).

Run: .venv/bin/python scripts/run_mcp_server.py   (normally started by the client, not by hand)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mcp_server.server import main  # noqa: E402

main()
