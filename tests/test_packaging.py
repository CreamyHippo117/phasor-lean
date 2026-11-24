"""The packaged desktop app (PyInstaller, ``packaging/``).

Building takes about a minute and ~480 MB, so the build is opt-in:
``PHASOR_TEST_PACKAGE=1 uv run pytest tests/test_packaging.py`` (needs ``--group package``).
It builds for the system it runs on (Linux here; Windows and macOS untested)."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.conftest import gl_available

ROOT = Path(__file__).parents[1]
COMMANDS = ("doctor", "render", "storyboard", "lint", "inspect", "web", "mcp", "gui", "manim",
            "sketch")  # fmt: skip


def test_the_entry_point_runs_the_command_line_and_survives_having_no_console(monkeypatch,
                                                                               capsys):  # fmt: skip
    """A windowed app (no console, as on Windows) has no stdout: printing must not crash."""
    monkeypatch.setattr(sys, "argv", ["phasor", "--cli", "--help"])
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(ROOT / "packaging" / "entry.py"), run_name="__main__")
    assert done.value.code in (0, None)


def test_the_icon_and_the_recipe_are_there():
    assert (ROOT / "phasor" / "gui" / "assets" / "icon.png").exists()
    assert (ROOT / "phasor" / "gui" / "assets" / "icon.ico").exists()
    spec = (ROOT / "packaging" / "phasor.spec").read_text(encoding="utf-8")
    assert 'copy_metadata("phasor")' in spec  # entry points (commands, tabs) need it
    assert '"examples"' in spec and "mpl-data" in spec  # examples and math fonts


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not os.environ.get("PHASOR_TEST_PACKAGE"):
        pytest.skip("building the app is opt-in: PHASOR_TEST_PACKAGE=1")
    pytest.importorskip("PyInstaller", reason="uv sync --group package")
    out = tmp_path_factory.mktemp("pkg")
    r = subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "phasor.spec"),
         "--noconfirm", "--distpath", str(out / "dist"), "--workpath", str(out / "build")],
        capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr[-3000:]
    exe = out / "dist" / "phasor" / ("phasor.exe" if sys.platform == "win32" else "phasor")
    assert exe.exists()
    return exe


def _run(exe: Path, *args: str, tmp: Path, timeout: float = 300) -> subprocess.CompletedProcess:
    """Run the app with a bare environment: nothing from this Python leaks in."""
    env = {"HOME": str(tmp), "PATH": "/usr/bin:/bin", "QT_QPA_PLATFORM": "offscreen",
           "PHASOR_CONFIG_DIR": str(tmp / "settings")}  # fmt: skip
    if sys.platform == "win32":
        env = {**os.environ, **env}
    return subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=timeout,
                          env=env, cwd=tmp)  # fmt: skip


@pytest.mark.slow
def test_every_command_is_inside_the_app(built, tmp_path):
    for c in COMMANDS:  # the add-on commands are found through the bundled metadata
        assert _run(built, "--cli", c, "--help", tmp=tmp_path).returncode == 0, c
    r = _run(built, "--cli", "doctor", tmp=tmp_path)
    assert "tier" in r.stdout


@pytest.mark.slow
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_app_renders_a_scene_with_typeset_maths(built, tmp_path):
    """Rendering works inside the app, and maths typesets (matplotlib's fonts came along)."""
    import json

    from phasor import Scene, Text

    s = Scene("maths", passes=None)
    s.add(Text(id="eq", text=r"$\oint f\,dz = 2\pi i$"))
    (tmp_path / "m.phasor.json").write_text(
        json.dumps(s.engine.document.model_dump(mode="json")), encoding="utf-8"
    )
    r = _run(built, "--cli", "render", "m.phasor.json", "--frame", "0", "-o", "m.png",
             tmp=tmp_path)  # fmt: skip
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "m.png").read_bytes()[:4] == b"\x89PNG"
    r = _run(built, "--cli", "lint", "m.phasor.json", tmp=tmp_path)
    assert r.returncode == 0, r.stdout[-2000:]


@pytest.mark.slow
def test_the_app_bundles_its_examples(built):
    ex = built.parent / "_internal" / "examples"
    assert (ex / "julia_explorer.py").exists() and (ex / "residue_theorem.py").exists()


@pytest.mark.slow
def test_the_window_starts_and_stays_up(built, tmp_path):
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "QT_QPA_PLATFORM": "offscreen",
           "PHASOR_CONFIG_DIR": str(tmp_path / "settings")}  # fmt: skip
    p = subprocess.Popen([str(built)], env=env, cwd=tmp_path, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)  # fmt: skip
    try:
        time.sleep(15)
        assert p.poll() is None, p.stdout.read()[-2000:] if p.stdout else ""  # no crash
    finally:
        p.terminate()
        out = p.communicate(timeout=20)[0]
    assert "did not load" not in out  # the Sketch tab came along too


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PHASOR_PACKAGE_GPU"), reason="PHASOR_PACKAGE_GPU=1")
def test_the_gpu_app_runs_kernels_on_the_card(built, tmp_path):
    """B7: built with PHASOR_PACKAGE_GPU=1 on an NVIDIA machine, the app finds the card."""
    from tests.conftest import cuda_available

    if not cuda_available():
        pytest.skip("no CUDA device on this machine")
    r = _run(built, "--cli", "doctor", tmp=tmp_path)
    assert "[ok  ] rawkernel" in r.stdout and "tier:    A" in r.stdout, r.stdout
