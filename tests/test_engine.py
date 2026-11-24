"""Engine: the single write path, atomic patches, undo, overrides, files (contracts 1-3)."""

from __future__ import annotations

import json
import textwrap

import pytest

from phasor import Dot, Engine, Scene, Text, expr, load
from phasor.core import io
from phasor.core.issues import DocumentError

ADD_DOT = {"op": "add_object", "object": {"id": "dot", "type": "dot", "params": {"r": 0.2}}}


def _engine() -> Engine:
    e = Engine(title="t", passes=None)
    assert e.apply(ADD_DOT).ok
    return e


def test_apply_returns_result_and_bumps_version():
    e = Engine(title="t", passes=None)
    r = e.apply(ADD_DOT)
    assert r.ok and r.version == 1 == e.version and r.errors == ()
    assert e.state_at(0)["dot"].params.r == 0.2
    assert set(r.to_dict()) == {"ok", "version", "errors", "lint_delta"}


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"op": "explode"}, "patch_schema"),
        ({"op": "set_param", "path": "ghost.r", "value": 1}, "unknown_id"),
        ({"op": "set_param", "path": "dot.wobble", "value": 1}, "unknown_param"),
        ({"op": "set_param", "path": "dot.r", "value": -1}, "bad_param"),
        ({"op": "set_param", "path": "dot.r", "value": {"expr": "dot.r"}}, "cycle"),
        ({"op": "set_style", "id": "dot", "key": "color", "role": "#ff0000"}, "bad_role"),
        ({"op": "remove_object", "id": "ghost"}, "unknown_id"),
        ({"op": "add_object", "object": {"id": "dot", "type": "dot"}}, "schema"),
        ({"op": "add_object", "object": {"id": "x", "type": "hologram"}}, "unknown_type"),
        ({"op": "clear_track", "path": "dot.r"}, "unknown_track"),
        ({"op": "remove_marker", "name": "ghost"}, "unknown_marker"),
        ({"op": "set_meta", "field": "nope", "value": 1}, "unknown_meta"),
        ({"op": "set_meta", "field": "width", "value": -5}, "schema"),
        ({"op": "add_segment", "path": "dot.r",
          "segment": {"t0": {"marker": "ghost"}, "t1": 1, "to": 1}}, "unknown_marker"),
    ],
)  # fmt: skip
def test_invalid_patches_change_nothing(patch, code):
    e = _engine()
    before, version = io.dumps(e.document), e.version
    r = e.apply(patch)
    assert not r.ok and code in {i.code for i in r.errors}, r.errors
    assert io.dumps(e.document) == before and e.version == version
    assert e.undo() is not None and e.undo() is None  # only the original add is in history


def test_patches_are_atomic():
    e = _engine()
    before = io.dumps(e.document)
    r = e.apply([{"op": "set_param", "path": "dot.r", "value": 0.9},
                 {"op": "set_param", "path": "ghost.r", "value": 1}])  # fmt: skip
    assert not r.ok and io.dumps(e.document) == before


def test_undo_redo_restore_exact_documents():
    e = _engine()
    a = io.dumps(e.document)
    e.apply({"op": "set_param", "path": "dot.r", "value": 0.7})
    b = io.dumps(e.document)
    assert e.undo().ok and io.dumps(e.document) == a and e.state_at(0)["dot"].params.r == 0.2
    assert e.redo().ok and io.dumps(e.document) == b and e.state_at(0)["dot"].params.r == 0.7
    assert e.redo() is None
    e.undo()
    e.apply({"op": "set_param", "path": "dot.r", "value": 0.3})
    assert e.redo() is None  # a new edit discards the redo branch


def test_state_cache_is_per_version():
    e = _engine()
    s = e.state_at(1.0)
    assert e.state_at(1.0) is s
    e.apply({"op": "set_param", "path": "dot.r", "value": 0.5})
    assert e.state_at(1.0) is not s and e.state_at(1.0)["dot"].params.r == 0.5


def test_remove_object_takes_its_tracks_but_not_dangling_references():
    e = _engine()
    e.apply({"op": "add_segment", "path": "dot.r", "segment": {"t0": 0, "t1": 1, "to": 1.0}})
    e.apply({"op": "add_object", "object": {"id": "b", "type": "dot",
                                            "params": {"r": {"expr": "dot.r"}}}})  # fmt: skip
    r = e.apply({"op": "remove_object", "id": "dot"})
    assert not r.ok and r.errors[0].code == "unknown_ref"  # b still reads dot.r: refuse loudly
    e.apply({"op": "remove_object", "id": "b"})
    assert e.apply({"op": "remove_object", "id": "dot"}).ok
    assert e.document.timeline.tracks == {} and e.document.objects == ()


def test_markers_retime_tracks_and_set_duration():
    e = _engine()
    e.apply([{"op": "set_marker", "name": "go", "t": 2.0},
             {"op": "add_segment", "path": "dot.r",
              "segment": {"t0": {"marker": "go"}, "t1": {"marker": "go", "offset": 1}, "to": 1.0,
                          "ease": "linear"}}])  # fmt: skip
    assert e.duration == 3.0 and e.state_at(2.5)["dot"].params.r == pytest.approx(0.6)
    e.apply({"op": "set_marker", "name": "go", "t": 10.0})
    assert e.duration == 11.0 and e.state_at(10.5)["dot"].params.r == pytest.approx(0.6)
    assert e.state_at(2.5)["dot"].params.r == 0.2


# ---------------------------------------------------------------- overrides (contract 2)


def _scripted() -> Scene:
    s = Scene("demo", passes=None)
    s.add(Dot(id="dot", pos=(1, 2), r=0.2))
    s.add(Text(id="label", text="z", pos=expr("dot.pos + vec(0, 0.5, 0)")))
    return s


def test_builder_raises_at_the_bad_line():
    s = _scripted()
    with pytest.raises(DocumentError, match="unknown_param"):
        s.add(Dot(id="other", radius=3))
    with pytest.raises(DocumentError, match="duplicate id"):
        s.add(Dot(id="dot"))
    with pytest.raises(DocumentError, match="unknown_marker"):
        s.at("ghost").to("dot.r", 1.0)
    assert [o.id for o in s.engine.document.objects] == ["dot", "label"]


def test_tweaks_to_a_scripted_scene_go_to_overrides_not_the_base(tmp_path):
    s = _scripted()
    e = s.engine
    base = io.dumps(e.base)
    assert e.scripted
    assert e.apply([{"op": "set_param", "path": "dot.r", "value": 0.5},
                    {"op": "set_style", "id": "dot", "key": "color", "role": "danger"}]).ok  # fmt: skip
    assert io.dumps(e.base) == base
    assert e.overrides.params == {"dot.r": 0.5} and e.overrides.styles == {"dot.color": "danger"}
    assert e.state_at(0)["dot"].params.r == 0.5 and e.state_at(0)["dot"].roles["color"] == "danger"

    r = e.apply({"op": "remove_object", "id": "label"})
    assert not r.ok and r.errors[0].code == "scripted_scene" and "script" in r.errors[0].fix_hint
    r = e.apply({"op": "set_param", "path": "dot.r", "value": -1})
    assert not r.ok and e.overrides.params == {"dot.r": 0.5}

    path = tmp_path / "demo.phasor.json"
    e.save(path)
    saved = json.loads((tmp_path / "demo.overrides.json").read_text(encoding="utf-8"))
    assert saved["params"] == {"dot.r": 0.5}
    reopened = Engine.open(path, passes=None)
    assert reopened.state_at(0)["dot"].params.r == 0.5 and io.dumps(reopened.base) == base

    e.undo()
    e.save(path)
    assert not (tmp_path / "demo.overrides.json").exists()  # no stale tweaks left behind


def _write_script(path, radius_line: str) -> None:
    path.write_text(
        textwrap.dedent(f"""
        from phasor import Dot, Scene
        s = Scene("demo", passes=None)
        {radius_line}
        s.add(Dot(id="other", r=0.3))
    """),
        encoding="utf-8",
    )


def test_rerunning_the_script_keeps_tweaks_and_flags_orphans(tmp_path):
    script = tmp_path / "demo.py"
    _write_script(script, 's.add(Dot(id="dot", r=0.2))')
    e = load(script, passes=None)
    e.apply({"op": "set_param", "path": "dot.r", "value": 0.9})
    io.save(tmp_path / "demo.overrides.json", e.overrides)

    # The script changes; the tweak survives the re-run.
    _write_script(script, 's.add(Dot(id="dot", r=0.25, pos=(3, 0)))')
    e = load(script, passes=None)
    p = e.state_at(0)["dot"].params
    assert p.r == 0.9 and p.pos == (3.0, 0.0, 0.0) and e.lint() == []

    # The script drops the object; the override is reported, not silently lost.
    _write_script(script, "pass")
    e = load(script, passes=None)
    (issue,) = e.lint()
    assert issue.code == "orphan_override" and issue.ids == ("dot",) and issue.severity == "error"
    assert e.overrides.params == {"dot.r": 0.9}


def test_lint_delta_reports_orphans_appearing_and_resolving(tmp_path):
    from phasor.core.schema import Overrides

    s = Scene("demo", passes=None)
    e = Engine(s.engine.base, overrides=Overrides(params={"dot.r": 0.9}), passes=None)
    assert [i.code for i in e.lint()] == ["orphan_override"]
    r = e.apply({"op": "add_object", "object": {"id": "dot", "type": "dot"}}, origin="script")
    assert r.ok and [i.code for i in r.lint_resolved] == ["orphan_override"] and e.lint() == []
    assert e.state_at(0)["dot"].params.r == 0.9
    r = e.undo()
    assert [i.code for i in r.lint_added] == ["orphan_override"]


def test_plain_documents_are_edited_in_place(tmp_path):
    e = _engine()
    assert not e.scripted
    e.apply({"op": "set_param", "path": "dot.r", "value": 0.4})
    assert e.overrides.is_empty() and e.base.object("dot").params["r"] == 0.4
    path = tmp_path / "plain.phasor.json"
    e.save(path)
    assert not (tmp_path / "plain.overrides.json").exists()
    assert io.dumps(Engine.open(path, passes=None).document) == io.dumps(e.document)


def test_load_script_needs_exactly_one_scene(tmp_path):
    script = tmp_path / "none.py"
    script.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one Scene"):
        load(script)


def test_unknown_object_queries_name_the_known_ones():
    e = _engine()
    with pytest.raises(KeyError, match="dot"):
        e.where("ghost", 0)


def test_engine_falls_back_to_analytic_inspection_without_opengl(monkeypatch):
    """Tier C: no renderer can be created, and everything except pixels still works."""
    import phasor.render.passes as passes
    from phasor.render.glcontext import GLUnavailableError

    def no_gl():
        raise GLUnavailableError("no OpenGL here")

    monkeypatch.setattr(passes, "default_renderer", no_gl)
    s = Scene("demo")
    s.add(Dot(id="dot", pos=(1, 2), r=0.2))
    e = s.engine
    assert e.pass_runner() is None
    assert e.inspect(0)["occlusion"] == "analytic"
    assert e.where("dot", 0)["center"]["px"] == [1095.0, 270.0]
    assert e.hit_test(1095, 270, 0) == ["dot"]
    with pytest.raises(RuntimeError, match="phasor doctor"):
        e.still(0)


def test_removed_ids_are_retired_and_never_reused(tmp_path):
    e = Engine(title="t", passes=None)
    assert e.apply({"op": "add_object", "object": {"id": "pole", "type": "dot"}}).ok
    assert e.apply({"op": "remove_object", "id": "pole"}).ok
    assert e.document.retired_ids == ("pole",)
    again = e.apply({"op": "add_object", "object": {"id": "pole", "type": "dot"}})
    assert not again.ok and again.errors[0].code == "id_retired" and again.errors[0].fix_hint
    path = tmp_path / "t.phasor.json"
    e.save(path)
    assert Engine.open(path, passes=None).document.retired_ids == ("pole",)  # survives a save
    e.undo()  # undoing the removal brings the object back, and its id is live again
    assert e.document.object("pole") is not None and e.document.retired_ids == ()
