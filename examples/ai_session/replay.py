"""Replay a recorded phasor MCP session over the real protocol (stdio).

    uv run python examples/ai_session/replay.py examples/ai_session/session.json WORKDIR

Every call in ``session.json`` is a tool call an AI made, in order, to ``phasor mcp``;
this prints each result (images summarized). WORKDIR is the folder the server may write to.
"""

import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client


async def replay(calls: list[dict], workdir: str, show: int = 600) -> list[dict]:
    """Run ``calls`` in one server session; returns ``[{"tool", "args", "data", "error"}]``."""
    # The session was recorded when new scenes were plain; since 2026-11-22 they get the
    # prism look by default, so replay it plain, as it was made.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "phasor.cli", "mcp", "--root", workdir],
        env={**get_default_environment(), "PHASOR_DEFAULT_LOOK": "none"},
    )
    out = []
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        for call in calls:
            call = dict(call)
            tool = call.pop("tool")
            res = await session.call_tool(tool, call)
            text = next((p.text for p in res.content if p.type == "text"), "")
            images = sum(p.type == "image" for p in res.content)
            try:
                data = json.loads(text)
            except ValueError:
                data = text
            out.append({"tool": tool, "args": call, "data": data, "error": res.is_error,
                        "images": images})  # fmt: skip
    return out


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        recorded = json.load(f)
    steps = asyncio.run(replay(recorded, sys.argv[2]))
    for s in steps:
        print(f"\n### {s['tool']}{'  (ERROR)' if s['error'] else ''}")
        print(
            json.dumps(s["data"], indent=1)[:600]
            if not isinstance(s["data"], str)
            else s["data"][:600]
        )
        if s["images"]:
            print(f"[{s['images']} image(s) returned]")
