#!/usr/bin/env python3
"""
Revit BIM MCP Server (P3)
=========================
Exposes Revit session management + family library tools via the Model Context
Protocol (MCP).  This server can be used by Claude Desktop, Claude Code, or
any MCP-compatible client.

Usage — stdio (Claude Desktop integration):
    python backend/mcp/server.py

Usage — from the repo root so imports resolve:
    cd /path/to/mcc-amplify-ai
    PYTHONPATH=backend python backend/mcp/server.py

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "revit-bim": {
          "command": "python",
          "args": ["/abs/path/to/backend/mcp/server.py"],
          "env": {
            "PYTHONPATH": "/abs/path/to/backend",
            "WINDOWS_REVIT_SERVER": "http://LT-HQ-277:5000"
          }
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Resolve import paths regardless of working directory
_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from loguru import logger

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
    )
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    logger.error(
        "mcp package not installed. Run: pip install mcp\n"
        "The MCP server cannot start without it."
    )
    sys.exit(1)

from mcp.tools import TOOL_REGISTRY, call_tool

# ── Revit docs resource paths (P4) ────────────────────────────────────────────

_DOCS_DIR = _ROOT / "data" / "revit_docs"
_DOC_FILES = {
    "units":            "units.md",
    "family_categories": "family_categories.md",
    "placement_guide":  "placement_guide.md",
}


# ── Server setup ──────────────────────────────────────────────────────────────

server = Server("revit-bim")


@server.list_resources()
async def list_resources() -> list[Resource]:
    """Expose Revit 2023 reference docs as MCP resources (P4)."""
    resources = []
    for slug, filename in _DOC_FILES.items():
        path = _DOCS_DIR / filename
        if path.exists():
            resources.append(
                Resource(
                    uri=f"revit://docs/{slug}",
                    name=f"Revit 2023 — {slug.replace('_', ' ').title()}",
                    description=f"Revit 2023 API reference: {slug}",
                    mimeType="text/markdown",
                )
            )
    return resources


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Return the content of a Revit doc resource."""
    # uri format: revit://docs/{slug}
    slug = str(uri).split("/")[-1]
    filename = _DOC_FILES.get(slug)
    if not filename:
        raise ValueError(f"Unknown resource: {uri}")

    path = _DOCS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Doc file not found: {path}")

    return path.read_text(encoding="utf-8")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose all Revit tools from the registry."""
    tools = []
    for name, (fn, schema) in TOOL_REGISTRY.items():
        tools.append(
            Tool(
                name=name,
                description=(fn.__doc__ or "").strip().split("\n")[0],
                inputSchema=schema,
            )
        )
    return tools


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call and return JSON-encoded result."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name!r}. Available: {list(TOOL_REGISTRY)}")

    try:
        result = await call_tool(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as exc:
        logger.error(f"Tool {name!r} failed: {exc}")
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    logger.info("Starting Revit BIM MCP server (stdio)…")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
