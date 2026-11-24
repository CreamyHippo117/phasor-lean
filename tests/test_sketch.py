"""Sketch-driven authoring. Strokes are classified by fitting, become
proposals (patch ops) that depend on what is under them, and land through Engine.apply.

Exit demo: draw a contour over 1/(z^2+1); it integrates (pi around the pole at i) and
animates (traced from the playhead)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from phasor import Dot, Scene, cx
from phasor.sketch.propose import Sketch, accept, contour_integral, pretty, residue_integral
from phasor.sketch.strokes import classify, group
from phasor.workspace.tools import Workspace

RNG = np.random.default_rng(7)
T = np.linspace(0.0, 1.0, 80)


def loop(center, r, n=120, noise=1.2, ccw=True, wobble=0.04):
    """A hand-drawn loop in output pixels. ``ccw``: counterclockwise as seen on screen
    (row 0 at the top, so the pixel y runs the other way)."""
    a = np.linspace(0, 2 * math.pi, n)
    rr = r * (1 + wobble * np.sin(3 * a))
    sgn = -1.0 if ccw else 1.0
    p = np.stack([center[0] + rr * np.cos(a), center[1] + sgn * rr * np.sin(a)], axis=1)
    return p + RNG.normal(0, noise, p.shape)


SHAPES = {
    "dot": np.array([[50.0, 50.0], [52.0, 51.0], [51.0, 53.0]]),
    "line": np.stack([100 + 300 * T, 100 + 4 * np.sin(9 * T)], axis=1),
    "arrow": np.concatenate(
        [np.stack([100 + 300 * T, 200 + 0 * T], axis=1), [[385, 188], [372, 178]]]
    ),
    "circle": loop((300, 300), 80),
    "closed": np.stack(
        [
            300 + 100 * np.cos(2 * np.pi * T) * (1 + 0.5 * np.cos(4 * np.pi * T)),
            300 + 50 * np.sin(2 * np.pi * T),
        ],
        axis=1,
    ),
    "scribble": np.stack([300 + 40 * np.sin(40 * T) + 30 * T, 300 + 25 * T], axis=1),
    "curve": np.stack([100 + 300 * T, 300 + 120 * T**2], axis=1),
}


@pytest.mark.parametrize("kind", sorted(SHAPES))
def test_each_stroke_is_named_by_its_shape_at_any_size(kind):
    assert classify(SHAPES[kind]).kind == kind
    if kind != "dot":  # the same shape three times bigger is the same shape
        c = SHAPES[kind].mean(axis=0)
        assert classify((SHAPES[kind] - c) * 3 + c).kind == kind


def test_a_wavy_line_is_a_wave():
    sh = classify(np.stack([100 + 400 * T, 300 + 30 * np.sin(12 * T)], axis=1))
    assert sh.kind == "curve" and sh.data["wavy"]


def test_a_line_with_a_v_at_its_end_is_an_arrow():
    shaft = np.stack([100 + 300 * T, 100 + 0 * T], axis=1)
    head = np.array([[385.0, 88.0], [400.0, 100.0], [385.0, 112.0]])
    shapes = group([shaft, head])
    assert [s.kind for s in shapes] == ["arrow"] and shapes[0].strokes == (0, 1)


def _letters():
    """Five small strokes side by side, like a short word."""
    out = []
    for k in range(5):
        x = 600 + 22 * k
        out.append(np.stack([x + 8 * np.sin(5 * T), 400 + 18 * T], axis=1))
    return out


def test_small_strokes_close_together_are_handwriting():
    shapes = group(_letters())
    assert [s.kind for s in shapes] == ["handwriting"]
    assert shapes[0].strokes == (0, 1, 2, 3, 4)


# ---------------------------------------------------------------- the maths


def test_contour_integrals_numeric_and_exact_agree():
    z = 1j + 0.6 * np.exp(1j * np.linspace(0, 2 * np.pi, 400, endpoint=False))
    assert contour_integral("1/(z**2+1)", z) == pytest.approx(math.pi, abs=1e-3)
    exact, found = residue_integral("1/(z**2+1)", z)
    assert exact == pytest.approx(math.pi) and found[0][0] == pytest.approx(1j)
    both = 2.0 * np.exp(1j * np.linspace(0, 2 * np.pi, 400, endpoint=False))
    assert residue_integral("1/(z**2+1)", both)[0] == pytest.approx(0.0)
    assert residue_integral("1/z", both)[0] == pytest.approx(2j * math.pi)
    assert residue_integral("exp(z)", both) == (None, [])  # not rational: numeric only
    assert contour_integral("exp(z)", both) == pytest.approx(0.0, abs=1e-6)


def test_values_are_written_as_multiples_of_pi_when_they_are():
    assert pretty(math.pi) == r"\pi"
    assert pretty(-2j * math.pi) == r"-2\pi i"
    assert pretty(math.pi / 2) == r"\frac{\pi}{2}"
    assert pretty(0.0) == "0"
    assert pretty(1.234 + 2.1j) == "1.234 + 2.1i"


# ---------------------------------------------------------------- proposals over a scene


def _scene(tmp_path=None):
    s = Scene("sketch demo")
    s.add(cx.DomainColoring(id="plane", function="1/(z**2+1)", size=(8.0, 4.5)))
    s.add(Dot(id="star", pos=(-3.0, -1.0)))
    s.marker("draw", 0.5)
    s.marker("end", 3.0)
    if tmp_path is None:
        return s.engine
    path = tmp_path / "demo.phasor.json"
    _save_plain(s.engine, path)
    ws = Workspace(tmp_path)
    ws.scene_open(path.name)
    return ws


def _save_plain(engine, path):
    """Save as a plain scene (one an editor or AI may add objects to); a scene built by a
    script takes structure only from its script."""
    doc = engine.document.model_dump(mode="json")
    doc["meta"]["source"] = "document"
    path.write_text(json.dumps(doc), encoding="utf-8")


def _px(sk, x, y):
    return sk.to_pixels(np.array([[x, y, 0.0]]))[0]


def test_exit_demo_draw_a_contour_and_it_integrates_and_animates(tmp_path):
    ws = _scene(tmp_path)
    sk = Sketch(ws.engine, t=0.5)
    c = _px(sk, 0.0, 1.0)
    r = _px(sk, 0.6, 1.0)[0] - c[0]
    (p,) = sk.propose([loop(c, r)])
    assert p.kind == "contour"
    assert p.data["exact"] == pytest.approx(math.pi)
    assert p.data["integral"] == pytest.approx(math.pi, abs=0.01)  # along the drawn path
    assert p.data["latex"] == r"\pi" and "around i" in p.summary
    assert accept(ws, p)["ok"]
    e = ws.engine
    line = e.document.object("contour")
    assert line.type == "line" and line.params["points"][0] == line.params["points"][-1]
    assert e.state_at(0.4)["contour"].params.progress == 0.0  # hidden until the playhead
    assert e.state_at(3.0)["contour"].params.progress == 1.0  # traced
    label = e.inspect(3.0, id="contour.label")["objects"][0]
    assert label["semantic"]["latex"] == r"$\oint f\,dz = \pi$"
    assert label["on_screen_fraction"] > 0.99
    assert not [i for i in e.lint() if i.severity == "error"]
    json.dumps(p.to_dict())  # proposals are plain data


def test_drawing_the_other_way_round_or_around_both_poles():
    e = _scene()
    sk = Sketch(e)
    c, r = _px(sk, 0.0, 1.0), _px(sk, 0.6, 0.0)[0] - _px(sk, 0.0, 0.0)[0]
    (cw,) = sk.propose([loop(c, r, ccw=False)])
    assert cw.data["exact"] == pytest.approx(-math.pi)
    (both,) = sk.propose([loop(_px(sk, 0, 0), r * 3.0)])
    assert both.data["exact"] == pytest.approx(0.0, abs=1e-12)
    assert {complex(*q["pole"]) for q in both.to_dict()["data"]["enclosed"]} == {1j, -1j}
    (none,) = sk.propose([loop(_px(sk, 2.5, 0), r)])
    assert none.data["exact"] == 0 and "no poles" in none.summary
    assert sk.propose([loop(c, r)])[0].data["id"] == "contour_4"  # fresh ids, by meaning


def test_a_dot_near_a_pole_lands_on_it():
    e = _scene()
    sk = Sketch(e)
    near = _px(sk, 0.03, 0.97)
    (p,) = sk.propose([np.array([near, near + 1.0])])
    assert p.kind == "dot" and p.data["snapped"] == "pole"
    assert p.data["at"][:2] == pytest.approx([0.0, 1.0])


def test_paint_over_move_emphasize_and_cross_out(tmp_path):
    ws = _scene(tmp_path)
    sk = Sketch(ws.engine)
    star, there = _px(sk, -3.0, -1.0), _px(sk, -2.0, 1.5)
    back = (star - there) / np.linalg.norm(star - there)  # the hook turns back, 30 degrees off
    turn = np.array([[math.cos(0.5), -math.sin(0.5)], [math.sin(0.5), math.cos(0.5)]])
    arrow = np.concatenate(
        [np.linspace(star, there, 40), [there + 12 * turn @ back, there + 24 * turn @ back]]
    )
    (move,) = sk.propose([arrow])
    assert move.kind == "move" and move.data["id"] == "star"
    assert move.data["to"][:2] == pytest.approx([-2.0, 1.5], abs=0.02)
    # a loop around the star, outside the plane's rectangle: emphasize it
    sk2 = Sketch(ws.engine)
    ws.scene_edit({"op": "set_param", "path": "plane.size", "value": [4.0, 2.0]})
    sk2 = Sketch(ws.engine)
    (emph,) = sk2.propose([loop(_px(sk2, -3.0, -1.0), 40)])
    assert emph.kind == "emphasize" and emph.data["ids"] == ["star"]
    assert accept(ws, emph)["ok"]
    assert ws.engine.document.object("star").style["color"] == "highlight"
    s = _px(sk2, -3.0, -1.0)
    scribble = np.stack([s[0] - 20 + 40 * T + 15 * np.sin(40 * T), s[1] + 6 * np.cos(40 * T)], 1)
    (gone,) = sk2.propose([scribble])
    assert gone.kind == "remove" and gone.data["ids"] == ["star"]
    assert accept(ws, gone)["ok"] and ws.engine.document.object("star") is None
    ws.undo()
    assert ws.engine.document.object("star") is not None


def test_handwriting_is_a_picture_to_read_then_text(tmp_path):
    ws = _scene(tmp_path)
    sk = Sketch(ws.engine)
    (p,) = sk.propose(_letters())
    assert p.kind == "handwriting" and p.image.startswith(b"\x89PNG") and not p.ops
    with pytest.raises(ValueError):
        accept(ws, p)  # nothing to apply until someone reads it
    t = sk.transcribe(p, r"$\oint f\,dz$")
    assert accept(ws, t)["ok"]
    note = ws.engine.document.object("note")
    assert note.params["text"] == r"$\oint f\,dz$"


def test_the_sketch_command_lists_and_applies(tmp_path, capsys):
    from phasor.cli import main

    e = _scene()
    _save_plain(e, tmp_path / "demo.phasor.json")
    sk = Sketch(e)
    c, r = _px(sk, 0.0, 1.0), _px(sk, 0.6, 0.0)[0] - _px(sk, 0.0, 0.0)[0]
    strokes = [loop(c, r).tolist(), *[s.tolist() for s in _letters()]]
    (tmp_path / "s.json").write_text(json.dumps({"t": 0.5, "strokes": strokes}), encoding="utf-8")
    scene, stroke_file = str(tmp_path / "demo.phasor.json"), str(tmp_path / "s.json")
    assert main(["sketch", scene, stroke_file]) == 0
    rows = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [r_["kind"] for r_ in rows] == ["contour", "handwriting"]
    assert (tmp_path / "demo.handwriting1.png").exists()
    assert main(["sketch", scene, stroke_file, "--accept", "--text", "1=$x$"]) == 0
    saved = json.loads((tmp_path / "demo.phasor.json").read_text(encoding="utf-8"))
    ids = {o["id"] for o in saved["objects"]}
    assert {"contour", "contour.label", "note"} <= ids


def _example():
    import runpy
    from pathlib import Path

    return runpy.run_path(str(Path(__file__).parents[1] / "examples" / "sketch_contour.py"),
                          run_name="x")  # fmt: skip


def test_the_sketch_example_is_the_exit_demo():
    ns = _example()
    p, e = ns["proposal"], ns["s"].engine
    assert p.data["exact"] == pytest.approx(math.pi)
    assert p.data["integral"] == pytest.approx(math.pi, abs=0.01)
    assert e.lint() == []
    assert e.state_at(0.4)["contour"].params.progress == 0.0
    assert e.state_at(3.5)["contour"].params.progress == 1.0


@pytest.mark.gl
@pytest.mark.skipif(not __import__("tests.conftest").conftest.gl_available(),
                    reason="needs OpenGL")  # fmt: skip
def test_the_sketch_example_matches_its_golden():
    import os

    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = _example()["s"].engine
    path = GOLDEN / e.pass_runner().renderer_class() / "sketch_contour.png"
    sheet = e.storyboard([0.5, 1.6, 3.5], size=(480, 270))["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
