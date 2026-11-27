"""Parallel render workers.

The parent ``Engine`` bakes once and hands ``worker_payload()`` to each worker, whose
``Engine.from_payload`` applies that bake and never bakes itself. Workers render every Nth
stretch of frames; ``scene.parallel_frames`` yields them in order, so a parallel render is
the serial render, byte for byte.
"""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------- state the scene script sets up

# A user's wishlist (scene reverb_la): a scene-local type draws programs from a module-level
# registry that the scene *script* fills. The document names only the type's module
# (``meta.modules``), so a worker that rebuilt the scene from the document loaded a fresh copy
# of that module with an empty registry: the type drew nothing, in every worker, silently.

PROGS_MODULE = """
from typing import ClassVar

import numpy as np

from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.values import Vec3

PROGRAMS = {}  # filled by the scene script, not by this module


@register
class Prog(ObjectBase):
    type: ClassVar[str] = "{name}"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        program: str = ""

    def draw(self, ctx, state):
        p = self.me(state).params
        for tri in PROGRAMS.get(p.program, []):
            pts = np.array(tri, dtype=float) + np.asarray(p.pos)
            ctx.triangles(pts, color=(1.0, 0.5, 0.2, p.opacity))
"""

SCENE_SCRIPT = """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import {module}

from phasor import Scene, Text

{module}.PROGRAMS["wedge"] = [[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0]],
                              [[0, 0, 0], [-1.5, 0, 0], [0, -1.5, 0]]]
s = Scene("{title}", look="none")
s.set("camera.height", 4)
s.add({module}.Prog(id="prog", program="wedge"))
s.add(Text(id="t", text="eq", pos=(0.0, 1.5)))
s.at(0).to("prog.pos", (0.5, 0.0, 0.0), duration=0.5)
s.marker("end", 0.5)
"""


@pytest.fixture
def registry_scene(tmp_path):
    """A scene folder: ``<module>.py`` (a type drawing from a registry) and ``scene.py`` (the
    script that fills the registry and builds the scene). Yields (script, type, module)."""
    from phasor import engine as engine_mod
    from phasor.core.objects import unregister

    uniq = "".join(ch for ch in tmp_path.name.lower() if ch.isalnum())[-10:]
    name, module = f"wk_prog_{uniq}", f"wk_progs_{uniq}"
    (tmp_path / f"{module}.py").write_text(PROGS_MODULE.replace("{name}", name), "utf-8")
    script = tmp_path / "scene.py"
    script.write_text(SCENE_SCRIPT.replace("{module}", module).replace("{title}", "reg_" + uniq),
                      "utf-8")  # fmt: skip
    yield script, name, module
    unregister(name)
    sys.modules.pop(module, None)
    while str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
    engine_mod._LOADED.clear()


def _as_in_a_new_process(name: str, module: str) -> None:
    """Forget what running the script left in this process, as a spawned worker starts."""
    from phasor import engine as engine_mod
    from phasor.core.objects import unregister

    unregister(name)
    sys.modules.pop(module, None)
    engine_mod._LOADED.clear()


def test_the_bug_a_worker_built_from_the_document_alone_draws_nothing(registry_scene):
    """What workers did before: ``Engine.from_payload`` loads the type's module from
    ``meta.modules``, a fresh copy whose registry the script never filled."""
    from phasor.core.objects import get_type
    from phasor.core.record import census
    from phasor.scene import load

    script, name, module = registry_scene
    parent = load(script, passes=None)
    here = census(parent.state_at(0.0), ["prog"])
    assert here["prog"][1] == 6  # two triangles: the registry is full here
    payload = parent.worker_payload()
    # The document as it was before the script named itself in meta.modules:
    old = json.loads(payload["document"])
    assert old["meta"]["modules"] == [f"{module}.py", "scene.py"]
    old["meta"]["modules"] = [f"{module}.py"]
    _as_in_a_new_process(name, module)
    from phasor.core import io

    bare = Engine(io.parse_document(old), base_dir=script.parent, allow_scripts=True,
                  passes=None)  # fmt: skip
    assert get_type(name) is not None  # the type loaded, its module ran (no missing_type) ...
    assert census(bare.state_at(0.0), ["prog"])["prog"][1] == 0  # ... and it draws nothing


def test_a_saved_script_scene_reopens_drawing_what_its_script_set_up(registry_scene):
    """The document names the script after the type's module (contract 3), so the saved ``.phasor.json`` opened with scripts allowed draws in one
    process too, not only in workers started from the script."""
    from phasor.core.record import census
    from phasor.scene import load

    script, name, module = registry_scene
    parent = load(script, passes=None)
    assert parent.document.meta.modules == (f"{module}.py", "scene.py")  # script last
    here = census(parent.state_at(0.0), ["prog"])
    saved = script.parent / "scene.phasor.json"
    parent.save(saved)
    _as_in_a_new_process(name, module)
    reopened = Engine.open(saved, passes=None, allow_scripts=True)
    assert census(reopened.state_at(0.0), ["prog"]) == here
    assert here["prog"][1] == 6


def test_a_scene_without_its_own_code_names_no_script(tmp_path):
    from phasor.scene import load

    script = tmp_path / "plain.py"
    script.write_text(
        "from phasor import Scene, Text\ns = Scene('plain')\ns.add(Text(id='t', text='x'))\n",
        "utf-8",
    )
    assert load(script, passes=None).document.meta.modules == ()


def test_a_worker_runs_the_scene_script_its_parent_loaded(registry_scene):
    from phasor.core.record import census
    from phasor.scene import _worker_engine, load

    script, name, module = registry_scene
    parent = load(script, passes=None)
    payload = parent.worker_payload()
    assert payload["script"] == str(script.resolve())
    here = census(parent.state_at(0.3), ["prog", "t"])
    _as_in_a_new_process(name, module)
    worker = _worker_engine(payload, passes=None)
    assert census(worker.state_at(0.3), ["prog", "t"]) == here
    # (the type is a new class in the worker, so its params compare by value)
    assert [(o.id, o.params.model_dump()) for o in worker.state_at(0.3).objects] == [
        (o.id, o.params.model_dump()) for o in parent.state_at(0.3).objects
    ]
    assert worker.layout == parent.layout  # the parent's bake, as before (ADR 0003)


def test_a_worker_keeps_the_parents_edits_not_the_scripts(registry_scene):
    """The script runs for what it sets up; the document is still the parent's (edits made
    after loading, in the app or over MCP, reach the workers)."""
    from phasor.scene import _worker_engine, load

    script, name, module = registry_scene
    parent = load(script, passes=None)
    parent.apply({"op": "set_param", "path": "t.text", "value": "edited"}).raise_for_errors()
    payload = parent.worker_payload()
    _as_in_a_new_process(name, module)
    worker = _worker_engine(payload, passes=None)
    assert worker.document.object("t").params["text"] == "edited"


def test_a_script_changed_since_loading_is_refused(registry_scene):
    from phasor.scene import _worker_engine, load

    script, _, _ = registry_scene
    parent = load(script, passes=None)
    payload = parent.worker_payload()
    script.write_text(script.read_text("utf-8") + "\n# edited\n", "utf-8")
    with pytest.raises(RuntimeError, match="changed since"):
        _worker_engine(payload, passes=None)


def test_a_forked_engine_keeps_its_script(registry_scene):
    from phasor.scene import load

    script, _, _ = registry_scene
    parent = load(script, passes=None)
    payload, forked = parent.worker_payload(), parent.fork().worker_payload()
    assert (forked["script"], forked["script_sha"]) == (payload["script"], payload["script_sha"])


def test_a_document_scene_sends_no_script():
    assert _scene().engine.worker_payload()["script"] is None


def test_a_script_without_scene_code_is_not_run_again(tmp_path):
    """Built-in types draw from the document alone; running the script in every worker
    would only add its load time (seconds for big scenes) to each worker's start."""
    from phasor.scene import load

    script = tmp_path / "plain.py"
    script.write_text("from phasor import Dot, Scene\ns = Scene('plain')\n"
                      "s.add(Dot(id='d'))\n", "utf-8")  # fmt: skip
    e = load(script, passes=None)
    assert e.script == script.resolve() and e.worker_payload()["script"] is None


def test_the_census_is_empty_without_scene_code():
    """Built-in types draw from the document alone: the check costs nothing for them."""
    from phasor.scene import _census_ids

    assert _census_ids(_scene().engine) == ()


def test_the_census_watches_scene_types(registry_scene):
    from phasor.scene import _census_ids, load

    script, _, _ = registry_scene
    assert _census_ids(load(script, passes=None)) == ("prog",)


def test_census_counts_what_each_object_draws(registry_scene):
    from phasor.core.record import census
    from phasor.scene import load

    script, _, _ = registry_scene
    e = load(script, passes=None)
    c = census(e.state_at(0.0), ["prog", "t", "nobody"])
    assert c["prog"][:2] == (("triangles", "triangles"), 6)
    assert c["t"][0] == ("text",) and "nobody" not in c


def test_a_worker_that_draws_differently_stops_the_render(registry_scene, tmp_path):
    """The safety net: a parent whose scene code kept state the workers cannot rebuild (here
    an older saved document, which does not name the script, opened in a process where the
    registry happens to be full) fails at
    the first frame, naming the object, instead of rendering for hours without it.
    Workers are spawned, as on Windows (Linux uses spawn here too); needs no OpenGL: the
    check comes before any frame is drawn."""
    from phasor.scene import load, parallel_frames

    script, _, _ = registry_scene
    doc = tmp_path / "saved.phasor.json"
    load(script, passes=None).save(doc)
    # A document saved before the script named itself in meta.modules:
    saved = json.loads(doc.read_text("utf-8"))
    saved["meta"]["modules"] = [m for m in saved["meta"]["modules"] if m != "scene.py"]
    doc.write_text(json.dumps(saved, sort_keys=True), "utf-8")
    e = Engine.open(doc, allow_scripts=True, passes=None)  # no script: only the document
    assert e.worker_payload()["script"] is None
    frames = parallel_frames(e, 2, chunk=2)([0.0, 0.1, 0.2, 0.3], (64, 36), 10.0, 0.0, 1)
    with pytest.raises(RuntimeError, match=r"(?s)worker.*prog.*draws nothing"):
        next(frames)
    frames.close()


@pytest.mark.gl
@needs_gl
def test_spawned_workers_draw_what_the_scene_script_set_up(registry_scene, tmp_path):
    """The bug end to end: ``phasor render scene.py --workers 2`` is ``--workers 1``."""
    from phasor.cli import main

    script, _, _ = registry_scene
    one, two = tmp_path / "one", tmp_path / "two"
    for out, n in ((one, "1"), (two, "2")):
        assert main(["render", str(script), "-o", str(out), "--preset", "preview",
                     "--fps", "8", "--workers", n, "--quiet"]) == 0  # fmt: skip
    a, b = _png_bytes(one), _png_bytes(two)
    assert len(a) == len(b) == 4
    assert a == b
