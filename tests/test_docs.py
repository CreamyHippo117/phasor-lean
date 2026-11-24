"""Docs written for AI: every ```python block in docs/ai/ runs in CI.

Blocks in one file run in order in a shared namespace, so later examples may build
on earlier ones. Mark a block that must not run with ```python no-run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs" / "ai"
BLOCK = re.compile(r"^```python[ \t]*\n(.*?)^```", re.S | re.M)


def _doc_files():
    return sorted(DOCS.glob("**/*.md"))


@pytest.mark.parametrize("path", _doc_files(), ids=lambda p: p.name)
def test_doc_code_blocks_run(path: Path):
    text = path.read_text(encoding="utf-8")
    ns: dict = {"__name__": f"docs_{path.stem}"}
    for i, m in enumerate(BLOCK.finditer(text)):
        code = m.group(1)
        exec(compile(code, f"{path.name}[block {i}]", "exec"), ns)


MCP_BLOCK = re.compile(r"^```json (patch|calls)[ \t]*\n(.*?)^```", re.S | re.M)


def test_mcp_doc_examples_run_in_order(tmp_path):
    """Every ```json patch block (a patch) and ```json calls block (a tool call per line)
    in docs/ai/MCP.md runs, in order, against one scene."""
    import json

    from phasor.workspace.tools import Workspace

    ws = Workspace(tmp_path)
    ws.scene_new("doc")
    blocks = MCP_BLOCK.findall((DOCS / "MCP.md").read_text(encoding="utf-8"))
    assert len(blocks) >= 5
    for i, (kind, body) in enumerate(blocks):
        if kind == "patch":
            r = ws.scene_edit(json.loads(body))
            assert r["ok"], f"MCP.md block {i}: {r['errors']}"
            continue
        for line in filter(str.strip, body.splitlines()):
            call = json.loads(line)
            getattr(ws, call.pop("tool"))(**call)
