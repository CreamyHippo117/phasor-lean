"""Contract 6: layer order and GPU-free lower layers.

import-linter checks static imports; the runtime test catches lazy/dynamic imports
that a static check cannot see.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GPU_FREE = ["phasor.core", "phasor.style", "phasor.time", "phasor.layout"]
LAYER_PACKAGES = [
    "core", "time", "layout", "inspect", "render", "render/gpu",
    "style", "domains", "sketch", "gui", "mcp",
]  # fmt: skip
GPU_MODULES = {"cupy", "cuda", "moderngl", "moderngl_window", "av", "glfw"}


def test_import_linter_contracts_hold():
    exe = shutil.which("lint-imports")
    if exe is None:
        pytest.skip("import-linter not installed")
    r = subprocess.run([exe], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("module", [*GPU_FREE, "phasor", "phasor.engine", "phasor.cli"])
def test_importing_does_not_load_gpu_libraries(module):
    code = (
        f"import sys, {module}; "
        f"bad = sorted(m for m in sys.modules if m.split('.')[0] in {sorted(GPU_MODULES)!r}); "
        "print(','.join(bad))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "", f"{module} pulled in GPU libraries: {r.stdout.strip()}"


def test_repo_layout():
    for path in [
        "phasor/engine.py",
        "phasor/cli.py",
        *(f"phasor/{p}/__init__.py" for p in LAYER_PACKAGES),
        "phasor/render/gpu/kernels",
        "docs/ai/API.md",
        "docs/ai/PATTERNS.md",
        "docs/ai/PITFALLS.md",
        "README.md",
        "CONTRIBUTING.md",
        "STYLE.md",
    ]:
        assert (ROOT / path).exists(), f"missing {path}"
