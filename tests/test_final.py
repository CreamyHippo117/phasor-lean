"""Final checks across the whole system: every object type,
every example, the document's round trips, determinism, purity, undo, atomicity and both
math engines. Each test names the contract or promise it guards."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

import phasor
from phasor import Scene
from phasor.core.diff import diff_documents
from phasor.core.objects import is_builtin, registered_types
from phasor.core.texconv import split_math, to_latex, to_mathtext
from tests.conftest import gl_available

ROOT = Path(__file__).parents[1]
gl = [pytest.mark.gl, pytest.mark.skipif(not gl_available(), reason="needs OpenGL")]

for _m in phasor._MODULES:  # register every domain's types
    getattr(phasor, _m)
TYPES = sorted(n for n, c in registered_types().items() if is_builtin(c))  # not test types
# Types with nothing to draw and so no style roles (ADR 0005): a named number.
DRAWLESS = {"value"}
# Types that draw another object's look and so have no style roles of their own.
BORROWED = {"hinged_copy"}
# The fewest parameters each type needs (the rest have defaults).
REQUIRED = {
    "domain_coloring": {"function": "1/(z**2+1)"},
    "network": {"generator": "tree", "branching": 2, "depth": 3},
    "image": {"path": "examples/assets/sunset.png"},  # relative: tests run from the repo root
    "epicycles": {"points": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]},
    "phasor_field": {"poles": [[0.9, 0.3], [0.5, -0.6]], "frame_ticks": 1.0},
    "equation": {"parts": ["$e^{i\\pi}$", "$+1=0$"]},
    "line": {"points": [[0, 0, 0], [1, 1, 0]]},
    "text": {"text": "hello $x^2$"},
    "wavefront": {"radius": 1.0},  # 0: not emitted yet
    "pressure_slice": {"time": 0.004},  # 0: the sound has not started
    "tracer": {"path": "guide", "progress": 0.5},  # rides a curve
    "callout": {"target": "guide"},  # rings an object
    "sound_rays": {"time": 0.5},  # in a room; 0: the sound has not left
    "wavefront3d": {"time": 0.5},
    "hinged_copy": {"source": "guide", "hinges": [[[1, 0, 0], [0, 1, 0], 180.0]]},  # a copy
    "face": {"points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
}
# Types that point at another object: the scene gets a curve called "guide" for them.
PARTNERED = {"tracer", "callout", "hinged_copy"}
# Sound in a 3D room: the scene gets a Room3D called "room" for them.
IN_ROOM = {"sound_rays", "wavefront3d"}
# The surface of a sweep: the scene gets a SweepLattice called "lattice", swept 2 units out.
SWEPT = {"sweep_front"}
# Examples that build a Scene called ``s``: not domain_coloring (a script that renders the
# kernel directly) or manim_reverb (a Manim script, run through the importer in its own test).
EXAMPLES = sorted(p.stem for p in (ROOT / "examples").glob("*.py")
                  if p.stem not in {"domain_coloring", "manim_reverb"})  # fmt: skip
# Slow to build (minutes for riemann_sqrt's labels through an orbit): checked by their own
# tests, and here only for lint.
SLOW = {"deep_zoom", "residue_theorem", "riemann_sqrt"}


def _one(name: str, camera: str, passes="auto"):
    s = Scene(f"one {name}", passes=passes, base_dir=ROOT)  # files (a picture) from the repo
    if camera == "orbit":
        s.use_camera("orbit")
    if name in PARTNERED:
        s.engine.apply({"op": "add_object", "object": {"id": "guide", "type": "line", "params": {
            "points": [[-1, -0.5, 0], [0, 0.5, 0], [1, -0.5, 0]]}}}, origin="script")  # fmt: skip
    if name in IN_ROOM:
        r = s.engine.apply({"op": "add_object", "object": {"id": "room", "type": "room3d",
                                                           "params": {}}}, origin="script")  # fmt: skip
        assert r.ok, [i.message for i in r.errors]
    if name in SWEPT:
        r = s.engine.apply({"op": "add_object", "object": {"id": "lattice", "type": "sweep_lattice",
                                                           "params": {"time": 2.0}}},
                           origin="script")  # fmt: skip
        assert r.ok, [i.message for i in r.errors]
    r = s.engine.apply({"op": "add_object", "object": {
        "id": "thing", "type": name, "params": REQUIRED.get(name, {})}}, origin="script")  # fmt: skip
    assert r.ok, (name, [i.message for i in r.errors])
    return s.engine


_example_cache: dict[str, object] = {}


def _example(name: str):
    if name not in _example_cache:
        _example_cache[name] = runpy.run_path(str(ROOT / "examples" / f"{name}.py"),
                                              run_name="x")["s"].engine  # fmt: skip
    return _example_cache[name]


def _examples(include_slow: bool = False):
    out = []
    for name in EXAMPLES:
        marks = [pytest.mark.slow] if name in SLOW else []
        if name in SLOW and not include_slow:
            continue
        out.append(pytest.param(name, marks=marks))
    return out


# ---------------------------------------------------------------- every type (contract 5)


def test_every_type_keeps_the_object_contract_and_has_docs():
    from phasor.core.objects import CONTRACT_MEMBERS

    api = (ROOT / "docs" / "ai" / "API.md").read_text(encoding="utf-8")
    for name in TYPES:
        cls = registered_types()[name]
        assert all(hasattr(cls, m) for m in CONTRACT_MEMBERS), name
        assert cls.Style or name in DRAWLESS | BORROWED, f"{name}: no style roles"
        assert cls.__name__ in api, f"{name} ({cls.__name__}) is not in docs/ai/API.md"


@pytest.mark.parametrize("camera", ["2d", "orbit"])
@pytest.mark.parametrize("name", TYPES)
def test_every_type_inspects_in_2d_and_3d_as_plain_data(name, camera):
    """Bounds are finite, the inspector's view is JSON, semantics name a kind, and the type
    itself adds no lint error (with no OpenGL, as on CI for Windows)."""
    e = _one(name, camera, passes=None)
    st = e.state_at(0.0)
    b = st["thing"].impl.world_bounds(st)
    assert np.all(np.isfinite(b.lo)) and np.all(np.isfinite(b.hi)), name
    seen = e.inspect(0.0, id="thing", detail="full")["objects"][0]
    json.dumps(seen)
    assert seen["semantic"].get("kind"), name
    errors = [i for i in e.lint() if i.severity == "error"]
    assert not errors, (name, [i.message for i in errors])


@pytest.mark.parametrize("camera", ["2d", "orbit"])
@pytest.mark.parametrize("name", TYPES)
def test_every_type_draws_something_in_2d_and_3d(name, camera, request):
    for m in gl:
        request.node.add_marker(m)
    if not gl_available():
        pytest.skip("needs OpenGL")
    if name in DRAWLESS:
        pytest.skip(f"{name} has nothing to draw")
    e = _one(name, camera)
    # Hairlines are a fraction of a pixel in a tiny frame and vanish,
    # so judge at a size where a "hairline" is a line.
    img = e.still(0.0, (960, 540))
    assert img.shape == (540, 960, 3)
    # The same scene without the object (but with its guide curve, if it has one).
    blank = e.fork()
    assert blank.apply({"op": "remove_object", "id": "thing"}, origin="script").ok
    blank = blank.still(0.0, (960, 540))
    assert np.abs(img.astype(int) - blank.astype(int)).max() > 8, f"{name} drew nothing"


# ---------------------------------------------------------------- every example


@pytest.mark.parametrize("name", _examples(include_slow=True))
def test_every_example_runs_and_has_no_lint_errors(name):
    """fourier_linalg failed here (a morph of an integer parameter) before this test existed."""
    e = _example(name)
    errors = [(i.code, i.message) for i in e.lint() if i.severity == "error"]
    assert not errors, errors


@pytest.mark.parametrize("name", _examples())
def test_every_example_survives_save_and_load(name, tmp_path):
    """Contract 2: the document is the truth; saving and loading it changes nothing."""
    e = _example(name)
    path = tmp_path / f"{name}.phasor.json"
    e.save(path)
    again = phasor.load(path)  # measured the same way as the original (OpenGL when there)
    d = diff_documents(e.document, again.document)
    assert d["same"], d["changes"][:3]
    top = list(json.loads(path.read_text(encoding="utf-8")))
    assert top == sorted(top)  # sorted keys, so a diff of two saves is readable
    for t in (0.0, e.duration / 2):
        a = e.inspect(t, detail="brief")
        b = again.inspect(t, detail="brief")
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.parametrize("name", _examples())
def test_every_example_round_trips_through_its_python_code(name, tmp_path):
    """The Code tab's promise: the generated script rebuilds the same document."""
    import shutil

    from tests.test_pycode import plain, rebuild

    e = _example(name)
    shutil.copytree(ROOT / "examples" / "assets", tmp_path / "assets")  # files beside a scene
    d = diff_documents(plain(e.document), plain(rebuild(e.document, tmp_path)))
    assert d["same"], d["changes"][:3]


def test_every_math_string_in_the_examples_works_in_both_dialects():
    """Typeset stays the default; LaTeX is the option; either form typesets on mathtext."""
    from phasor.core import text as tm

    found = 0
    for name in EXAMPLES:
        if name in SLOW:
            continue
        for o in _example(name).document.objects:
            for v in o.params.values():
                for s in [v] if isinstance(v, str) else (v if isinstance(v, list) else []):
                    if isinstance(s, str) and tm.is_math(s):
                        found += 1
                        tm.check_math(to_mathtext(to_latex(s)))
                        assert any(m for m, _ in split_math(to_latex(s)))
    assert found >= 1


# ---------------------------------------------------------------- time is pure (contract 3)


@pytest.mark.parametrize("name", ["prism_primes", "basics_3d", "heat_dihedral"])
def test_state_depends_only_on_the_document_and_the_time(name):
    """The same time asked in any order, or by a fresh engine, gives the same scene."""
    from phasor.engine import Engine

    e = _example(name)
    ts = [0.0, e.duration * 0.7, e.duration * 0.3, e.duration]
    first = {t: json.dumps(e.inspect(t, detail="brief"), sort_keys=True) for t in ts}
    fresh = Engine(e.document)  # measured the same way (OpenGL when there)
    for t in reversed(ts):
        assert json.dumps(fresh.inspect(t, detail="brief"), sort_keys=True) == first[t]


@pytest.mark.parametrize("name", ["prism_primes", "sketch_contour"])
def test_the_same_frame_renders_to_the_same_bytes(name, request):
    for m in gl:
        request.node.add_marker(m)
    if not gl_available():
        pytest.skip("needs OpenGL")
    from phasor.engine import Engine

    e = _example(name)
    t = e.duration / 2
    a = e.still(t, (240, 135))
    b = e.still(t, (240, 135))
    c = Engine(e.document).still(t, (240, 135))
    assert np.array_equal(a, b) and np.array_equal(a, c)


# ---------------------------------------------------------------- one write path (contract 1)


def _plain_scene():
    from phasor.engine import Engine

    s = Scene("edits", passes=None)
    s.add(phasor.Dot(id="a", pos=(0, 0)))
    doc = s.engine.document.model_dump(mode="json")
    doc["meta"]["source"] = "document"
    from phasor.core.schema import Document

    return Engine(Document.model_validate(doc), passes=None)


def test_undo_all_then_redo_all_restores_each_document_exactly():
    e = _plain_scene()
    docs = [e.document.model_dump(mode="json")]
    patches = [
        {"op": "add_object", "object": {"id": "b", "type": "dot", "params": {"pos": [1, 1, 0]}}},
        {"op": "set_param", "path": "a.r", "value": 0.3},
        {"op": "set_marker", "name": "go", "t": 1.0},
        {"op": "add_segment", "path": "b.pos",
         "segment": {"t0": {"marker": "go"}, "t1": 2.0, "to": [2, 0, 0]}},
        {"op": "set_style", "id": "a", "key": "color", "role": "highlight"},
        {"op": "remove_object", "id": "a"},
    ]  # fmt: skip
    for p in patches:
        assert e.apply(p).ok, p
        docs.append(e.document.model_dump(mode="json"))
    for k in range(len(patches), 0, -1):
        e.undo()
        assert e.document.model_dump(mode="json") == docs[k - 1]
    for k in range(1, len(patches) + 1):
        e.redo()
        assert e.document.model_dump(mode="json") == docs[k]
    assert "a" in e.document.retired_ids  # removed ids are never reused


def test_a_patch_with_one_bad_op_changes_nothing():
    e = _plain_scene()
    before = e.document.model_dump(mode="json")
    r = e.apply([
        {"op": "set_param", "path": "a.r", "value": 0.5},
        {"op": "add_object", "object": {"id": "b", "type": "dot"}},
        {"op": "set_param", "path": "a.r", "value": -1.0},  # invalid radius
    ])  # fmt: skip
    assert not r.ok and r.errors
    assert e.document.model_dump(mode="json") == before


def test_a_fork_never_touches_the_scene_its_history_or_its_file(tmp_path):
    from phasor.interactive import Player

    e = _example("julia_explorer")
    path = tmp_path / "j.phasor.json"
    e.save(path)
    saved, doc, version = (
        path.read_text(encoding="utf-8"),
        e.document.model_dump(mode="json"),
        e.version,
    )
    p = Player(e)
    key = p.controls[0].key
    p.set(key, p.controls[0].max)
    assert p.values[key] == p.controls[0].max
    assert e.document.model_dump(mode="json") == doc and path.read_text(encoding="utf-8") == saved
    assert e.version == version  # no new step in the scene's history


def test_a_tweak_to_a_script_scene_survives_running_the_script_again(tmp_path):
    """Contract 2: re-running a script never loses overrides; a dangling one is linted."""
    script = tmp_path / "scene.py"
    script.write_text(
        "from phasor import Scene, Dot\n"
        "s = Scene('tweaks', passes=None)\n"
        "s.add(Dot(id='p', pos=(0, 0)))\n",
        encoding="utf-8",
    )
    e = phasor.load(script, passes=None)
    assert e.apply({"op": "set_param", "path": "p.r", "value": 0.4}).ok  # a tweak
    e.save(tmp_path / "scene.phasor.json")  # the tweak goes to scene.overrides.json
    assert (tmp_path / "scene.overrides.json").exists()
    again = phasor.load(script, passes=None)
    assert again.document.object("p").params["r"] == 0.4
    script.write_text(
        script.read_text(encoding="utf-8").replace("id='p'", "id='q'"), encoding="utf-8"
    )
    gone = phasor.load(script, passes=None)
    assert "orphan_override" in {i.code for i in gone.lint()}
