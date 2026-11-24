"""The generated ``llms.txt``: a short, always-current summary an AI can read first.

Built from the live tool docstrings and object registry, so it cannot drift from the code.
``tests/test_mcp.py`` fails when the committed copy is stale; regenerate with
``uv run python -m phasor.mcp.llms``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from phasor.mcp.server import TOOL_NAMES
from phasor.workspace.tools import Workspace

PATH = Path(__file__).resolve().parents[2] / "llms.txt"


def _first(doc: str | None) -> str:
    return " ".join((doc or "").strip().split("\n\n")[0].split()).replace("``", "`")


def build() -> str:
    ws = Workspace()
    lines = [
        "# phasor",
        "",
        "> GPU-rendered math animations that an AI can see, check and fix. You build a scene with",
        "> patches over MCP (`phasor mcp`), check it with `review` (lint + inspector + storyboard",
        "> picture), and fix what it finds. Read docs/ai/MCP.md first.",
        "",
        "## Tools",
        "",
    ]
    for name in TOOL_NAMES:
        fn = getattr(ws, name, None)
        lines.append(f"- `{name}`: {_first(inspect.getdoc(fn)) if fn else ''}")
    lines += ["", "## Object types (use these names in add_object)", ""]
    for name, t in ws.object_types()["types"].items():
        if t.get("scene_local"):
            continue  # only the types phasor ships
        params = ", ".join(p for p in t["params"] if p not in ("opacity", "dim", "z_index"))
        lines.append(f"- `{name}`: {params}")
    lines += [
        "",
        "## Docs",
        "",
        "- docs/ai/MCP.md: the loop, every tool, every patch operation (all examples are tested)",
        "- docs/ai/API.md: the Python API, one example per name",
        "- docs/ai/PATTERNS.md: idioms (label a pole, tie parameters, narrate and animate)",
        "- docs/ai/PITFALLS.md: known traps and how the linter reports them",
        "- examples/: small tested scenes",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    PATH.write_text(build(), encoding="utf-8")
    print(f"wrote {PATH}")
