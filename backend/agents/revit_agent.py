"""
Revit BIM Agent (P5)
====================
A Claude-powered agent that reads a geometry transaction JSON produced by the
pipeline (stage 6) and builds a Revit model step-by-step using the session API.

This is an embedded tool-use loop — it does NOT require a running MCP server
process.  The same tool functions from backend/mcp/tools.py are called directly
as Python coroutines.

Environment variables
---------------------
REVIT_AGENT_MODEL     Claude model to use (default: claude-sonnet-4-6)
REVIT_AGENT_MAX_TURNS Maximum tool-call rounds before giving up (default: 60)
ANTHROPIC_API_KEY     Required — Anthropic API key
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import anthropic
from loguru import logger

from mcp.tools import TOOL_REGISTRY, call_tool

# ── Configuration ─────────────────────────────────────────────────────────────

_MODEL     = os.getenv("REVIT_AGENT_MODEL", "claude-sonnet-4-6")
_MAX_TURNS = int(os.getenv("REVIT_AGENT_MAX_TURNS", "60"))

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Revit BIM construction agent.  Your task is to take a floor plan
geometry transaction JSON and build a complete Revit 2023 model using the
provided tools.

## Your workflow (follow this order strictly)

1. **Parse the transaction**: Identify all element types present — columns,
   walls, doors, windows.  Note their coordinates (mm) and sizes. *[Identify element type, determine the size, determine the height]*

2. **Search the family library**: For each unique category (columns, doors,
   windows), call `search_family_library` to find the best matching .rfa family.
   Match on size annotations if available. *[Identify Revit Family]*

3. **Open a Revit session**: Call `revit_new_session` once.  Save the session_id
   for all subsequent calls.

4. **Load families**: Call `revit_load_family` for each unique .rfa needed.
   Use the `windows_rfa_path` field from the search results. []

5. **Place elements** (in this order — respect Revit hosting constraints):
   a. Structural columns (`top_level`: "Level 1")
   b. Structural framing / beams if present
   c. Doors (wall-hosted — place AFTER walls; use wall position for x/y)
   d. Windows (wall-hosted — same rule)
   Walls, floors, and ceilings are NOT placed via this agent — they come from
   the batch transaction JSON handled by the main pipeline.

5a. **Discover parameters before setting them**: After placing any element whose
    size needs adjusting, call `revit_get_parameters` on it first.  Use the exact
    `name` values returned — never guess names like "b", "Width", "Depth" because
    they vary by family.  Only call `revit_set_parameter` with names you have
    confirmed via `revit_get_parameters`.

5b. **Join walls**: Once ALL elements are placed, call `revit_wall_join_all` once.
    This fixes T-junction display gaps and corner intersections automatically.

6. **Verify**: Call `revit_get_state` once to confirm placed element count.

7. **Export**: Call `revit_export_session` with the provided job_id.  This closes
   the session and saves the .rvt file.  Return the rvt_path from the result.

## Rules

- ALWAYS call `revit_new_session` before any other session tool.
- NEVER make up family names.  Only use names returned by `search_family_library`
  or `revit_list_families`.
- If `search_family_library` returns `"not_found": true`, that family does not
  exist in either the primary index or the user folder.  DO NOT call
  `revit_load_family` or `revit_place_instance` for that element.  Record it in
  your skipped list with reason `"family_not_found"` and move on.
- Coordinates (x_mm, y_mm) come directly from the geometry JSON — do not modify
  them unless correcting a placement error.
- If a tool returns an error, log the problem and continue with remaining elements.
  Never abort mid-session; always export whatever was placed.
- After `revit_export_session` succeeds, output ONLY the following JSON and stop:
  {"status": "done", "rvt_path": "<path>", "placed_count": <n>, "skipped": [{"element": "<desc>", "reason": "<reason>"}]}

## Column family mapping
Each column in the transaction has a `family_type` field and a `shape` field.

| `shape`      | `family_type` example | Search keyword               | Expected Revit family               |
|--------------|-----------------------|------------------------------|-------------------------------------|
| rectangular  | RECT200x250           | "200x250" + "concrete"       | CJY_Concrete-Rectangular-Column     |
| rectangular  | RECT300x300           | "300x300" + "concrete"       | CJY_Concrete-Rectangular-Column     |
| circular     | CIRC300               | "round" or "circular" + "300"| CJY_RC Round Column                 |

- Square columns are stored as RECT{n}x{n} — they use the same rectangular concrete family.
- NEVER load or place `M_W Shapes-Column` for a concrete column — that is a steel I-beam.
  If only W-shapes appear in the search results, search again with keyword "concrete" or "CJY".

## Coordinate notes
- All coordinates are in millimetres relative to the structural grid origin.
- x_mm → east, y_mm → north, z_mm → elevation above base level.
- Structural columns: set `top_level` = "Level 1" (or the highest level available).
"""

# ── Anthropic tool definitions ─────────────────────────────────────────────────

def _build_anthropic_tools() -> list[dict]:
    """Convert TOOL_REGISTRY to Anthropic SDK tool format."""
    tools = []
    for name, (fn, schema) in TOOL_REGISTRY.items():
        doc = (fn.__doc__ or "").strip()
        # Use only the first paragraph as description (stay within token budget)
        description = doc.split("\n\n")[0].replace("\n", " ").strip()
        tools.append({
            "name":         name,
            "description":  description,
            "input_schema": schema,
        })
    return tools


_TOOLS = _build_anthropic_tools()


# ── Agent loop ────────────────────────────────────────────────────────────────

class RevitAgent:
    """
    Async Claude agent that builds a Revit session model from a transaction JSON.

    Usage:
        agent = RevitAgent()
        result = await agent.run(transaction_json_str, job_id="abc123",
                                 on_progress=lambda msg: print(msg))
    """

    def __init__(self):
        self._client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY") or _load_api_key_from_file()
        )

    async def run(
        self,
        transaction_json: str | dict,
        job_id: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Run the agent loop until it exports the model or exceeds MAX_TURNS.

        Parameters
        ----------
        transaction_json : str or dict
            The geometry transaction produced by stage 6 (pipeline orchestrator).
        job_id           : str
            Pipeline job ID — passed to revit_export_session for file naming.
        on_progress      : callable, optional
            Called with human-readable status messages during the loop.

        Returns
        -------
        dict  {"status": "done"|"failed", "rvt_path": str|None, "placed_count": int,
               "turns": int, "error": str|None}
        """
        if isinstance(transaction_json, dict):
            tx_str = json.dumps(transaction_json, indent=2)
        else:
            tx_str = transaction_json

        _emit(on_progress, "Revit agent starting — parsing geometry transaction…")

        initial_message = (
            f"job_id: {job_id}\n\n"
            f"Geometry transaction JSON (stage 6 output):\n```json\n{tx_str}\n```\n\n"
            "Build the Revit model now following your workflow instructions."
        )

        messages: list[dict] = [{"role": "user", "content": initial_message}]

        turns        = 0
        placed_count = 0
        rvt_path     = None

        while turns < _MAX_TURNS:
            turns += 1
            _emit(on_progress, f"Agent turn {turns}/{_MAX_TURNS}…")

            # ── Call Claude ────────────────────────────────────────────────────
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.messages.create(
                    model=_MODEL,
                    max_tokens=4096,
                    system=_SYSTEM_PROMPT,
                    tools=_TOOLS,
                    messages=messages,
                ),
            )

            # Append assistant message to history
            messages.append({"role": "assistant", "content": response.content})

            # ── Check stop condition ───────────────────────────────────────────
            if response.stop_reason == "end_turn":
                # Agent declared it is done — extract final JSON if present
                final_text = _extract_text(response.content)
                logger.info(f"[RevitAgent] end_turn after {turns} turns: {final_text[:200]}")
                result = _try_parse_result(final_text)
                if result:
                    rvt_path     = result.get("rvt_path")
                    placed_count = result.get("placed_count", placed_count)
                    if result.get("skipped"):
                        _emit(on_progress, f"Skipped elements: {result['skipped']}")
                _emit(on_progress, f"Agent complete — {placed_count} elements placed.")
                break

            # ── Process tool calls ─────────────────────────────────────────────
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                logger.warning("[RevitAgent] No tool calls and stop_reason != end_turn — stopping.")
                break

            tool_results = []
            for block in tool_use_blocks:
                tool_name = block.name
                tool_args = block.input or {}
                _emit(on_progress, f"  → {tool_name}({_summarise_args(tool_args)})")

                try:
                    result = await call_tool(tool_name, tool_args)
                    result_json = json.dumps(result, indent=2, default=str)

                    # Track placed elements
                    if tool_name == "revit_place_instance" and isinstance(result, dict):
                        if result.get("element_id"):
                            placed_count += 1

                    # Capture rvt_path from export call
                    if tool_name == "revit_export_session" and isinstance(result, dict):
                        rvt_path = result.get("rvt_path")

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_json,
                    })

                except Exception as exc:
                    logger.error(f"[RevitAgent] Tool {tool_name!r} failed: {exc}")
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     json.dumps({"error": str(exc)}),
                        "is_error":    True,
                    })

            messages.append({"role": "user", "content": tool_results})

            # ── Stop after successful export ───────────────────────────────────
            if rvt_path:
                _emit(on_progress, f"Export complete → {rvt_path}")
                break

        else:
            logger.warning(f"[RevitAgent] MAX_TURNS ({_MAX_TURNS}) reached without export.")

        success = rvt_path is not None
        return {
            "status":        "done" if success else "failed",
            "rvt_path":      rvt_path,
            "placed_count":  placed_count,
            "turns":         turns,
            "error":         None if success else "Agent did not export the model.",
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _emit(callback: Optional[Callable], message: str) -> None:
    logger.info(f"[RevitAgent] {message}")
    if callback:
        callback(message)


def _extract_text(content: list) -> str:
    """Concatenate all TextBlock content from a response."""
    parts = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return " ".join(parts)


def _try_parse_result(text: str) -> dict | None:
    """Try to extract the final JSON result from the agent's last message."""
    import re
    # Look for {"status": ...} JSON object in the text
    m = re.search(r'\{[^{}]*"status"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _summarise_args(args: dict) -> str:
    """Compact single-line summary of tool arguments for logging."""
    parts = []
    for k, v in args.items():
        if k == "session_id":
            parts.append(f"session=…{str(v)[-6:]}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.0f}")
        elif isinstance(v, str) and len(v) > 40:
            parts.append(f"{k}='{v[:37]}…'")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts[:4])  # cap at 4 args for readability


def _load_api_key_from_file() -> str | None:
    """Fallback: read Anthropic key from backend/anthropic_key.txt if it exists."""
    for candidate in [
        Path(__file__).resolve().parents[1] / "anthropic_key.txt",
        Path(__file__).resolve().parents[2] / "anthropic_key.txt",
    ]:
        if candidate.exists():
            key = candidate.read_text().strip()
            if key:
                return key
    return None
