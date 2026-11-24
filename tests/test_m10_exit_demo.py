"""One scene edited by hand and by AI in one session.

The roadmap names the reverb scene, which Milestone 8 skipped (by decision); the owner's
reference scene, ``examples/prism_primes.py``, stands in. The person has the editor open (a
script scene, so edits become overrides); an AI connects with the real ``phasor mcp
--attach`` (a subprocess speaking MCP over stdio); each sees the other's edits as parameter
changes; saving keeps both, and re-running the script keeps both (contract 2)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets", reason="the gui extra is not installed")
pytest.importorskip("mcp", reason="the mcp extra is not installed")

from phasor import load
from phasor.gui.window import open_editor

ROOT = Path(__file__).parents[1]


async def _ai(root: Path, script):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "phasor.cli", "mcp", "--attach", "--root", str(root)]
    )
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        return await script(session)


def _data(result):
    if result.is_error:
        raise AssertionError(result.content[0].text)
    return json.loads(next(p.text for p in result.content if p.type == "text"))


def test_one_scene_edited_by_hand_and_by_ai_in_one_session(tmp_path):
    scene = tmp_path / "prism_primes.py"
    shutil.copy(ROOT / "examples" / "prism_primes.py", scene)
    _app, win, server = open_editor(str(scene), show=False)
    try:
        ed = win.ed
        # 1. By hand: in the inspector, the curve runs its hue 1.5 times; drag an equation.
        ed.select("truth")
        win.inspector.refresh()
        win.inspector.fields["hue_along"].setText("1.5")
        win.inspector.fields["hue_along"].editingFinished.emit()
        r = ed.drag_to("error_eq", 400, 60, (800, 450))
        assert r["ok"]
        moved = ed.engine.document.object("error_eq").params["pos"]

        # 2. By AI, attached to the same live scene through the real MCP server.
        async def ai(s):
            seen = _data(await s.call_tool("inspect", {"t": 12.0, "id": "truth",
                                                        "detail": "full"}))  # fmt: skip
            edit = _data(await s.call_tool("scene_edit", {"patch": [
                {"op": "set_style", "id": "zeros_60", "key": "glow", "role": "strong"},
                {"op": "set_param", "path": "zeros_eq.pos", "value": [0.0, 0.4, 0.0]},
            ]}))  # fmt: skip
            lint = _data(await s.call_tool("lint", {}))
            return seen, edit, lint

        seen, edit, lint = asyncio.run(_ai(tmp_path, ai))
        assert edit["ok"] and "issues" in lint
        # the AI saw the person's edit as a parameter, not as pixels
        assert seen["objects"][0]["params"]["hue_along"] == 1.5
        assert server.calls >= 3  # the AI's calls ran in the editor's process

        # 3. The open window notices the AI's edit and shows it.
        ed.select("zeros_60")
        win.tick()
        assert win.inspector.roles["glow"].currentText() == "strong"
        assert ed.engine.document.object("zeros_eq").params["pos"] == (0.0, 0.4, 0.0)

        # 4. Save: a script scene keeps both editors' tweaks in its overrides file...
        saved = ed.save()
        overrides = json.loads(
            (tmp_path / "prism_primes.overrides.json").read_text(encoding="utf-8")
        )
        text = json.dumps(overrides)
        assert "hue_along" in text and "zeros_eq" in text and "error_eq" in text
        # ...and re-running the script keeps them all (contract 2).
        again = load(scene, passes=None)
        assert again.document.object("truth").params["hue_along"] == 1.5
        assert again.document.object("error_eq").params["pos"] == pytest.approx(moved)
        assert again.state_at(0.0)["zeros_60"].roles["glow"] == "strong"
        assert saved["saved"].endswith(".phasor.json")
    finally:
        win.timer.stop()
        server.stop()
        win.close()
    assert not (tmp_path / ".phasor" / "gui.json").exists()  # closing withdraws the invitation
