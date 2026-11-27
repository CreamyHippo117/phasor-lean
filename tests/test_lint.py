"""Lint rules, auto-fix, storyboards, and the exit demo
("a deliberately cluttered scene is flagged, then auto-fixed")."""

from __future__ import annotations

import itertools
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Line, Scene, Text, load
from phasor.cli import main
from phasor.inspect.lint import sample_times
from tests.conftest import gl_available

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "examples" / "cluttered.py"
GOLDEN = Path(__file__).parent / "golden"

# Every rule must give the same verdict from real pixels and from geometry alone (tier C).
MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


def _codes(engine):
    return sorted({i.code for i in engine.lint()})


def _one(engine, code):
    (issue,) = [i for i in engine.lint() if i.code == code]
    return issue


def _scene(passes):
    s = Scene("lint", passes=passes)
    s.set("camera.height", 8)
    s.add(Dot(id="pole", pos=(0, 0), r=0.3, style={"color": "highlight"}, tags=["key"]))
    return s


@pytest.mark.parametrize("passes", MODES)
def test_a_clean_scene_has_no_issues(passes):
    s = _scene(passes)
    s.add(Text(id="pole.label", text="pole", pos=(0, 0.8)))
    assert s.engine.lint() == []


@pytest.mark.parametrize("passes", MODES)
def test_overlap_between_a_label_and_an_object_offers_the_near_fix(passes):
    s = _scene(passes)
    s.add(Text(id="pole.label", text="pole", pos=(0.35, 0.1)))
    issue = _one(s.engine, "overlap")
    assert issue.severity == "warning" and issue.ids == ("pole", "pole.label")
    assert issue.times == (0.0,) and "t = 0" in issue.message
    assert issue.fix == ({"op": "add_constraint", "rule": "label(pole) near pole"},)
    assert "fix" in issue.to_dict() and issue.fix_hint


@pytest.mark.parametrize("passes", MODES)
def test_overlap_between_two_labels_and_between_key_objects(passes):
    s = _scene(passes)
    s.add(Text(id="one", text="first label", pos=(3, 2)))
    s.add(Text(id="two", text="second label", pos=(3.4, 2.1)))
    s.add(Dot(id="moon", pos=(0.4, 0), r=0.3, tags=["key"]))
    s.add(Dot(id="dust", pos=(-0.4, 0), r=0.3))  # not key and not a label: nobody minds
    pairs = sorted(i.ids for i in s.engine.lint() if i.code == "overlap")
    assert pairs == [("moon", "pole"), ("one", "two")]
    labels = next(i for i in s.engine.lint() if i.ids == ("one", "two"))
    assert labels.fix == () and "<owner>.label" in labels.fix_hint  # not a child label: a hint


@pytest.mark.parametrize("passes", MODES)
def test_a_thin_line_through_a_label_counts_but_its_bounding_box_does_not(passes):
    s = _scene(passes)
    s.add(Line(id="ramp", points=[(-6, -3), (6, 3)]))
    s.add(Text(id="far", text="far from the line", pos=(3, -1)))  # inside the line's box
    assert "overlap" not in _codes(s.engine)
    s.add(Text(id="on", text="on the line", pos=(-4, -2)))
    assert _one(s.engine, "overlap").ids == ("on", "ramp")


@pytest.mark.parametrize("passes", MODES)
def test_a_label_sitting_on_a_shape_is_judged_by_contrast_not_overlap(passes):
    s = _scene(passes)
    s.add(Dot(id="disc", pos=(3, 0), r=1.5, style={"color": "primary"}))
    s.add(Text(id="disc.label", text="disc", pos=(3, 0), style={"color": "primary"}))
    assert _codes(s.engine) == ["contrast"]
    issue = _one(s.engine, "contrast")
    assert "contrast 1.0" in issue.message and issue.ids == ("disc.label",)
    assert s.engine.apply({"op": "set_style", "id": "disc.label", "key": "color",
                           "role": "muted"}).ok  # fmt: skip
    assert s.engine.lint() == []  # dark gray on near-white reads fine


@pytest.mark.parametrize("passes", MODES)
def test_characters_the_font_cannot_draw_are_flagged(passes):
    s = _scene(passes)
    s.add(Text(id="formula", text="z² + π", pos=(3, 2)))
    issue = _one(s.engine, "missing_glyph")
    assert issue.ids == ("formula",) and "'²', 'π'" in issue.message and issue.fix_hint
    assert s.engine.apply({"op": "set_param", "path": "formula.text", "value": "z^2 + pi"}).ok
    assert s.engine.lint() == []


@pytest.mark.parametrize("passes", MODES)
def test_faint_text_fails_contrast(passes):
    s = _scene(passes)
    s.add(Text(id="note", text="whisper", pos=(3, 2), opacity=0.15))
    assert _codes(s.engine) == ["contrast"]


@pytest.mark.parametrize("passes", MODES)
def test_offscreen_offers_on_screen_and_the_fix_holds(passes):
    s = _scene(passes)
    s.add(Text(id="title", text="A long title", pos=(6.6, 3.9), style={"size": "title"}))
    issue = _one(s.engine, "offscreen")
    assert issue.fix == ({"op": "add_constraint", "rule": "on_screen(title)"},)
    assert s.engine.visible("title", 0)["on_screen_fraction"] < 1
    report = s.engine.autofix()
    assert report["applied"] == list(issue.fix) and report["remaining"] == []
    assert s.engine.visible("title", 0)["on_screen_fraction"] == 1.0
    _x0, y0, x1, _y1 = s.engine.where("title", 0)["bbox"]["norm"]
    assert x1 == pytest.approx(1 - 0.02 * 9 / 16, abs=2e-3) and y0 == pytest.approx(0.02, abs=2e-3)


@pytest.mark.parametrize("passes", MODES)
def test_tiny_orphan_and_clutter(passes):
    s = _scene(passes)
    s.add(Dot(id="speck", pos=(2, 2), r=0.01))
    s.add(Dot(id="ghost", pos=(-2, 2), opacity=0.0))
    s.add(Dot(id="lost", pos=(40, 0)))
    assert _one(s.engine, "tiny").ids == ("speck",)
    assert sorted(i.ids[0] for i in s.engine.lint() if i.code == "orphan") == ["ghost", "lost"]
    s.marker("burst", 1.0)
    for k, name in enumerate(["ash", "bay", "elm", "fir", "oak"]):
        s.add(Dot(id=name, pos=(-4 + k, -2), r=0.2))
        s.at("burst").to(f"{name}.r", 0.3, duration=0.5)
    issue = _one(s.engine, "clutter")
    assert issue.ids == ("ash", "bay", "elm", "fir", "oak") and "'burst'" in issue.message
    assert issue.times == (1.0,) and "5 objects" in issue.message


@pytest.mark.parametrize("passes", MODES)
def test_a_layout_that_renders_many_occlusion_buffers_is_flagged(passes, monkeypatch):
    from phasor.style.theme import load_theme

    monkeypatch.setitem(load_theme("midnight_glow").lint, "bake_renders", 40)
    s = _scene(passes)
    s.add(Text(id="pole.label", text="pole"))
    s.marker("end", 6.0)
    s.constrain("label(pole) near pole")
    issue = _one(s.engine, "slow_layout")
    assert issue.severity == "warning" and issue.ids == ("pole.label",) and not issue.fix
    assert issue.times == (0.0,) and "61 occlusion buffers" in issue.message
    assert "0-6 s" in issue.message and issue.fix_hint
    # Shown for two seconds only, it costs about twenty: below the limit.
    s.at(1.0).reveal("pole.label")
    s.at(3.0).exit("pole.label")
    assert "slow_layout" not in _codes(s.engine)
    assert s.engine.layout.renders < 40


@pytest.mark.parametrize("passes", MODES)
def test_guards_are_errors_when_they_do_not_hold(passes):
    s = _scene(passes)
    s.add(Dot(id="moon", pos=(0.4, 0), r=0.3))
    s.add(Dot(id="speck", pos=(3, 3), r=0.02))
    s.add(Text(id="note", text="whisper", pos=(3, 2), opacity=0.15))
    s.add(Line(id="edge", points=[(6, 0), (9, 0)]))
    s.constrain("no_overlap(pole, moon)")
    s.constrain("min_size_px(speck, 12) and readable(note)")
    s.constrain("on_screen(edge)")  # a line has no position to move, so it is only checked
    errors = [i for i in s.engine.lint() if i.code == "unsolved"]
    assert all(i.severity == "error" and i.times == (0.0,) for i in errors)
    assert sorted((i.ids, i.path) for i in errors) == [
        (("edge",), "constraints[on_screen(edge)]"),
        (("note",), "constraints[min_size_px(speck, 12) and readable(note)]"),
        (("pole", "moon"), "constraints[no_overlap(pole, moon)]"),
        (("speck",), "constraints[min_size_px(speck, 12) and readable(note)]"),
    ]
    assert s.engine.apply({"op": "set_param", "path": "moon.pos", "value": [2, 0]}).ok
    assert s.engine.apply({"op": "set_param", "path": "speck.r", "value": 0.2}).ok
    left = [i.ids for i in s.engine.lint() if i.code == "unsolved"]
    assert left == [("edge",), ("note",)]


def test_lint_samples_markers_ends_and_a_grid():
    s = _scene(None)
    s.marker("beat", 0.33)
    s.at(0.1).to("pole.r", 0.5, duration=1.07)
    assert sample_times(s.engine._snap.compiled, 0.5) == (0.0, 0.33, 0.5, 1.0, 1.17)


@pytest.mark.parametrize("passes", MODES)
def test_the_patch_result_carries_the_lint_delta(passes):
    s = _scene(passes)
    s.add(Text(id="pole.label", text="pole", pos=(0, 0.8)))
    r = s.engine.apply({"op": "set_param", "path": "pole.label.pos", "value": [0.35, 0.1]})
    assert [i.code for i in r.lint_added] == ["overlap"] and r.lint_resolved == ()
    assert r.to_dict()["lint_delta"]["added"][0]["fix"][0]["op"] == "add_constraint"
    back = s.engine.undo()
    assert [i.code for i in back.lint_resolved] == ["overlap"]


# ---------------------------------------------------------------- the exit demo


@pytest.mark.parametrize("passes", MODES)
def test_exit_demo_cluttered_scene_is_flagged_then_auto_fixed(passes, tmp_path):
    script = tmp_path / "cluttered.py"
    shutil.copy(DEMO, script)
    e = load(script, passes=passes)

    before = e.lint()
    assert sorted({i.code for i in before}) == ["offscreen", "overlap"]
    assert {i.ids for i in before if i.code == "overlap"} == {
        ("contour", "pole_c.label"), ("pole_a.label", "pole_b"), ("pole_b.label", "pole_c"),
    }  # fmt: skip
    assert all(i.fix and i.times for i in before)

    report = e.autofix()
    assert sorted(op["rule"] for op in report["applied"]) == [
        "label(pole_a) near pole_a", "label(pole_b) near pole_b", "label(pole_c) near pole_c",
        "on_screen(title)",
    ]  # fmt: skip
    assert report["remaining"] == [] and e.lint() == []

    # Ground truth, not just the linter's word: at 30 Hz nothing important touches.
    for t in np.arange(0, e.duration + 1e-9, 1 / 30):
        for obj in e.inspect(float(t))["objects"]:
            if obj["id"] == "contour":  # the line runs off the frame on purpose
                continue
            assert obj["on_screen_fraction"] == 1.0, (t, obj["id"])
            if obj["id"].endswith(".label"):
                assert not any(r["overlaps"] for r in obj["relations"]), (t, obj["id"])
    # Labels follow their poles now, and none flickers between slots.
    for slots in e.layout.slots.values():
        assert sum(a != b for a, b in itertools.pairwise(slots)) <= 1

    # The script was never edited: the fixes live in the overrides file and survive a reload.
    e.save(tmp_path / "cluttered.phasor.json")
    assert (
        len(
            json.loads((tmp_path / "cluttered.overrides.json").read_text(encoding="utf-8"))[
                "constraints"
            ]
        )
        == 4
    )
    assert load(script, passes=passes).lint() == []
    assert load(tmp_path / "cluttered.phasor.json", passes=passes).lint() == []


def test_cli_lint_reports_fixes_and_sets_the_exit_code(tmp_path, capsys):
    script = tmp_path / "cluttered.py"
    shutil.copy(DEMO, script)
    assert main(["lint", str(script)]) == 0  # warnings only
    out = capsys.readouterr().out
    assert "warning: overlap" in out and "hint:" in out and "0 error(s), 4 warning(s)" in out
    assert main(["lint", str(script), "--strict"]) == 1
    capsys.readouterr()
    assert main(["lint", str(script), "--fix", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["fixed"]) == 4 and report["issues"] == []
    assert (tmp_path / "cluttered.overrides.json").exists()
    assert main(["lint", str(script), "--strict"]) == 0  # the fix persisted

    doc = tmp_path / "bad.phasor.json"
    s = Scene("bad", passes=None)
    s.add(Dot(id="a", pos=(0, 0)))
    s.add(Dot(id="b", pos=(0.1, 0)))
    s.constrain("no_overlap(a, b)")
    s.save(doc)
    assert main(["lint", str(doc)]) == 1  # an unsolved constraint is an error
    assert "error: unsolved" in capsys.readouterr().out
    assert main(["lint", str(tmp_path / "missing.phasor.json")]) == 1


# ---------------------------------------------------------------- storyboards


def test_storyboard_times_cover_markers_and_ends():
    s = _scene(None)
    assert s.engine.storyboard_times() == [0.0]
    s.marker("beat", 1.0)
    s.at("beat").to("pole.r", 0.5, duration=3.0)
    assert s.engine.storyboard_times() == [0.0, 0.8, 1.0, 1.6, 2.4, 3.2, 4.0]
    assert s.engine.storyboard_times(count=2) == [0.0, 1.0, 4.0]


def test_storyboard_needs_opengl_and_says_so():
    with pytest.raises(RuntimeError, match="no OpenGL"):
        _scene(None).engine.storyboard()


@pytest.mark.gl
def test_storyboard_sheet_frames_and_marks():
    e = load(DEMO)
    plain = e.storyboard([0.0, 1.0, 2.5], size=(320, 180), columns=2)
    assert plain["image"].shape == (10 + 2 * (180 + 22 + 10), 10 + 2 * (320 + 10), 3)
    assert [f["t"] for f in plain["frames"]] == [0.0, 1.0, 2.5]
    assert [f["marker"] for f in plain["frames"]] == [None, "squeeze", None]
    assert any(line.startswith("pole_a [dot]") for line in plain["frames"][0]["objects"])
    marked = e.storyboard([0.0, 1.0, 2.5], size=(320, 180), columns=2, marks=True)["image"]
    changed = (plain["image"] != marked).any(axis=2)
    assert 0.005 < changed.mean() < 0.2  # boxes and tags were drawn, the picture is still there
    # A mark has the color derived from the object's id-pass index.
    from phasor.core.color import mark_color

    assert (marked.reshape(-1, 3) == mark_color(1)).all(axis=1).sum() > 50


@pytest.mark.gl
@pytest.mark.parametrize("name", ["before", "after"])
def test_golden_storyboards(name):
    """Storyboard snapshot for this milestone's feature (SSIM >= 0.99, per renderer class)."""
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    e = load(DEMO)
    if name == "after":
        e.autofix()
    klass = e.pass_runner().renderer_class()
    path = GOLDEN / klass / f"cluttered_{name}.png"
    sheet = e.storyboard([0.0, 1.75, 2.5], marks=True)["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {klass} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.99


def test_cli_storyboard(tmp_path, capsys):
    out = tmp_path / "board.png"
    code = main(["storyboard", str(DEMO), "-o", str(out), "--marks", "--t", "0", "--t", "2.5"])
    if not gl_available():
        assert code == 1 and "no OpenGL" in capsys.readouterr().err
        return
    assert code == 0 and out.exists() and "t=2.5" in capsys.readouterr().out
