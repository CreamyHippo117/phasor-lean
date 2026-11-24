"""Every program phasor starts is started without a console window (``core/proc.py``).

On Windows the desktop app has no console; a console program it starts without
``creationflags=NO_WINDOW`` flashes a window of its own (checked in the packaged app,
2026-11-20). This test reads the source, so it holds on every platform.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import phasor
from phasor.core.proc import NO_WINDOW

ROOT = Path(phasor.__file__).parent
STARTERS = {"run", "Popen", "call", "check_call", "check_output"}


def _calls_without_flags() -> list[str]:
    missing = []
    for path in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            f = node.func if isinstance(node, ast.Call) else None
            if (isinstance(f, ast.Attribute) and f.attr in STARTERS
                    and isinstance(f.value, ast.Name) and f.value.id == "subprocess"
                    and not any(k.arg == "creationflags" for k in node.keywords)):  # fmt: skip
                missing.append(f"{path.relative_to(ROOT.parent)}:{node.lineno}")
    return missing


def test_every_subprocess_call_hides_its_console_window():
    assert _calls_without_flags() == []


def test_no_window_is_the_windows_flag_there_and_nothing_elsewhere():
    want = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    assert want == NO_WINDOW
