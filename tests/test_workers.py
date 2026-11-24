"""Parallel render workers.

The parent ``Engine`` bakes once and hands ``worker_payload()`` to each worker, whose
``Engine.from_payload`` applies that bake and never bakes itself. Workers render every Nth
stretch of frames; ``scene.parallel_frames`` yields them in order, so a parallel render is
the serial render, byte for byte.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Scene, Text
from phasor.engine import Engine
from tests.conftest import gl_available

needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _scene(title="workers"):
    """Motion (blurred in finals), a label placed by the layout solver, living glow, text;
    no lit surfaces (their last bits are not repeatable on every card)."""
    s = Scene(title, look="glow")
    s.set("camera.height", 4)
    s.add(Dot(id="p", pos=(-2.0, -0.5)))
    s.add(Text(id="p.label", text="p"))
    s.constrain("label(p) near p")
    s.add(Text(id="t", text="workers", pos=(0.0, 1.5)))
    s.at(0).to("p.pos", (2.0, 0.5, 0.0), duration=1.5)
    s.marker("end", 2.0)
    return s


# ---------------------------------------------------------------- the hand-off


def test_a_worker_engine_applies_the_parents_bake_and_never_bakes(monkeypatch):
    e = _scene().engine
    payload = e.worker_payload()
    import phasor.engine as engine_module

    def no_bake(*a, **k):
        raise AssertionError("a worker must not bake")

    monkeypatch.setattr(engine_module, "bake", no_bake)
    w = Engine.from_payload(payload, passes=None)
    assert w.layout == e.layout
    for t in (0.0, 0.7, 1.9):
        assert w.state_at(t).objects == e.state_at(t).objects


def test_a_bake_for_another_document_is_refused():
    payload = _scene().engine.worker_payload()
    other = _scene("something else").engine.worker_payload()
    with pytest.raises(ValueError, match="another document"):
        Engine.from_payload({**payload, "fingerprint": other["fingerprint"]}, passes=None)


def test_the_payload_crosses_a_process_boundary():
    import pickle

    payload = _scene().engine.worker_payload()
    assert pickle.loads(pickle.dumps(payload)) == payload


# ---------------------------------------------------------------- whole renders


def _png_bytes(folder: Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(folder.glob("frame_*.png"))]


@pytest.mark.gl
@needs_gl
@pytest.mark.parametrize("preset", ["preview", "final"])
def test_parallel_render_equals_serial_render(tmp_path, preset):
    from phasor.render.job import render_file
    from phasor.scene import parallel_frames

    e = _scene().engine
    kw = {"preset": preset, "fps": 12.0}
    serial = render_file(e, tmp_path / "serial", **kw)
    many = render_file(e, tmp_path / "many", frame_source=parallel_frames(e, 3, chunk=4), **kw)
    assert serial["frames"] == many["frames"] == 24
    a, b = _png_bytes(tmp_path / "serial"), _png_bytes(tmp_path / "many")
    assert len(a) == len(b) == 24
    assert all(x == y for x, y in zip(a, b, strict=True))


@pytest.mark.gl
@needs_gl
def test_a_parallel_video_has_every_frame(tmp_path):
    import av

    from phasor.render.job import render_file
    from phasor.scene import parallel_frames

    e = _scene().engine
    out = tmp_path / "v.mp4"
    r = render_file(e, out, preset="preview", fps=12.0, frame_source=parallel_frames(e, 2, chunk=5))
    with av.open(str(out)) as c:
        assert sum(1 for _ in c.decode(video=0)) == r["frames"] == 24


@pytest.mark.gl
@needs_gl
def test_a_failing_worker_stops_the_render_with_its_error(tmp_path):
    from phasor.render.job import render_file
    from phasor.scene import parallel_frames

    e = _scene().engine
    src = parallel_frames(e, 2, chunk=4, _fail_at=9)
    with pytest.raises(RuntimeError, match="frame 9"):
        render_file(e, tmp_path / "x", preset="preview", fps=12.0, frame_source=src)


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.mark.gl
@needs_gl
@pytest.mark.slow
def test_workers_stop_when_the_render_process_is_killed(tmp_path):
    """The desktop app cancels an export by killing its render process; on Windows that
    leaves child processes running unless they notice. Workers must stop by themselves."""
    pids = tmp_path / "pids.txt"
    script = textwrap.dedent(f"""
        from phasor.render.job import render_file
        from phasor.scene import parallel_frames
        from tests.test_workers import _scene
        e = _scene().engine
        def started(ps):
            open(r"{pids}", "w").write(" ".join(map(str, ps)))
        src = parallel_frames(e, 2, chunk=4, _on_start=started, _slow=0.5)
        render_file(e, r"{tmp_path / "out"}", preset="preview", fps=12.0, frame_source=src)
    """)
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=Path(__file__).parents[1])
    deadline = time.time() + 120
    while not pids.exists() or not pids.read_text().strip():
        assert time.time() < deadline and proc.poll() is None, "workers never started"
        time.sleep(0.2)
    workers = [int(p) for p in pids.read_text().split()]
    assert workers and all(_alive(p) for p in workers)
    proc.kill()
    proc.wait()
    deadline = time.time() + 30
    while any(_alive(p) for p in workers) and time.time() < deadline:
        time.sleep(0.2)
    assert not any(_alive(p) for p in workers)


def test_render_frame_is_the_one_frame_rule():
    """``render_frame`` decides blur or not for both the serial loop and the workers."""
    from phasor.render.job import render_frame

    calls = []

    class Fake:
        def frame(self, t, size, output):
            calls.append(("frame", t))
            return np.zeros((2, 2, 3), np.uint8)

        def frame_blurred(self, t, fps, *, shutter, samples, size, output):
            calls.append(("blur", t))
            return np.zeros((2, 2, 3), np.uint8)

    render_frame(Fake(), 0.5, 30.0, (2, 2), 0.5, 4)
    render_frame(Fake(), 0.5, 30.0, (2, 2), 0.0, 4)
    render_frame(Fake(), 0.5, 30.0, (2, 2), 0.5, 1)
    assert calls == [("blur", 0.5), ("frame", 0.5), ("frame", 0.5)]


@pytest.mark.gl
@needs_gl
def test_a_script_without_a_main_guard_runs_once(tmp_path):
    """Spawned workers re-run the parent's main script unless told not to; a scene script
    calling the workers without ``if __name__ == "__main__":`` must not run again in each."""
    marker, script = tmp_path / "runs.txt", tmp_path / "scene_script.py"
    script.write_text(
        textwrap.dedent(f"""
        import sys
        sys.path.insert(0, r"{Path(__file__).parents[1]}")
        open(r"{marker}", "a").write("ran" + chr(10))
        from phasor.render.job import render_file
        from phasor.scene import parallel_frames
        from tests.test_workers import _scene
        e = _scene().engine
        render_file(e, r"{tmp_path / "out"}", preset="preview", fps=12.0,
                    frame_source=parallel_frames(e, 2, chunk=6))
    """),
        encoding="utf-8",
    )
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=300, cwd=tmp_path)  # fmt: skip
    assert done.returncode == 0, done.stderr[-2000:]
    assert marker.read_text().splitlines() == ["ran"]
    assert len(list((tmp_path / "out").glob("frame_*.png"))) == 24
