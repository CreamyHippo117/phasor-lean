"""Manim import. A Manim Community script runs unchanged against phasor's
stand-in ``manim`` module and becomes a phasor scene. Exit demo: an old Manim-style room
reverb scene (``examples/manim_reverb.py``) runs in phasor."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from phasor.importers import import_manim
from phasor.importers.manim import ManimImportError, color_role
from tests.conftest import gl_available

ROOT = Path(__file__).parents[1]


def script(tmp_path, body: str, name="s.py") -> Path:
    p = tmp_path / name
    p.write_text("from manim import *\n\n" + body, encoding="utf-8")
    return p


def state(r, t, oid):
    return r.scene.engine.state_at(t)[oid].params


# ---------------------------------------------------------------- the exit demo


def test_the_manim_reverb_scene_runs_unchanged_and_keeps_manims_clock():
    r = import_manim(ROOT / "examples" / "manim_reverb.py")
    assert r.name == "RoomReverb" and r.others == []
    # the sum of every play run_time and wait in the script
    expected = 1.5 + 1 + 1 + 3 * (1.2 + 0.3) + 1.5 + 2 + 1 + 1 + 1
    assert r.duration == pytest.approx(expected)
    doc = r.scene.engine.document
    kinds = [o.type for o in doc.objects]
    assert kinds.count("line") == 5 and kinds.count("dot") == 2 and kinds.count("text") == 4
    assert kinds.count("axes") == 1
    # the room draws itself over 1.5 s from t = 0, the waves expand one after another, and
    # the formula writes in at the end: nothing is on screen before Manim shows it
    assert state(r, 0.0, "rectangle_1").progress == 0.0
    assert state(r, 0.75, "rectangle_1").progress == pytest.approx(0.5, abs=0.15)
    assert state(r, 1.6, "rectangle_1").progress == 1.0
    assert state(r, 1.0, "dot_1").opacity == 0.0 and state(r, 3.5, "dot_1").opacity == 1.0
    assert state(r, 3.0, "circle_1").opacity == 0.0 and state(r, 4.5, "circle_1").opacity == 1.0
    assert state(r, 9.0, "mathtex_1").opacity == 0.0 and state(r, 14.0, "mathtex_1").opacity == 1.0
    ring = [
        np.linalg.norm(
            np.asarray(state(r, t, "circle_1").points)[0]
            - np.asarray(state(r, 4.0, "circle_1").points).mean(axis=0)
        )
        for t in (4.2, 5.0)
    ]
    assert ring[1] > ring[0]  # a growing circle
    assert r.warnings == ["axes with different x and y scales are drawn with the x scale"]
    errors = [i for i in r.scene.engine.lint() if i.severity == "error"]
    assert errors == []


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_manim_reverb_scene_matches_its_golden():
    import os

    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = import_manim(ROOT / "examples" / "manim_reverb.py").scene.engine
    path = GOLDEN / e.pass_runner().renderer_class() / "manim_reverb.png"
    sheet = e.storyboard([2.0, 5.5, 9.0, 14.0], size=(480, 270), columns=2)["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97


def test_the_command_saves_a_scene_that_loads_and_lints(tmp_path, capsys):
    from phasor.cli import main
    from phasor.core import io

    out = tmp_path / "reverb.phasor.json"
    assert (
        main(["manim", str(ROOT / "examples" / "manim_reverb.py"), "-o", str(out), "--code"]) == 0
    )
    text = capsys.readouterr().out
    assert "RoomReverb: 12 objects, 14.5 s" in text and "note: axes with different" in text
    doc = io.load_document(out)
    assert len(doc.objects) == 12 and doc.meta.title == "RoomReverb"
    assert (
        (tmp_path / "reverb.scene.py").read_text(encoding="utf-8").startswith("from phasor import ")
    )
    assert main(["manim", str(tmp_path / "missing.py")]) == 1


# ---------------------------------------------------------------- what a Manim script can do


def test_shapes_positions_and_layout_follow_manims_frame(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        a = Dot(UP * 2 + LEFT * 3)
        b = Square(side_length=2).shift(RIGHT * 4)
        c = Text("hello", font_size=30).next_to(b, UP, buff=0.5)
        d = Circle(radius=1).to_edge(DOWN, buff=0.5)
        self.add(a, b, c, d)
        self.wait(1)
""")  # fmt: skip
    r = import_manim(p)
    assert state(r, 0, "dot_1").pos == pytest.approx((-3.0, 2.0, 0.0))
    sq = np.asarray(state(r, 0, "square_1").points)
    assert sq[:, 0].min() == pytest.approx(3.0) and sq[:, 0].max() == pytest.approx(5.0)
    assert sq[0] == pytest.approx(sq[-1])  # a closed outline
    # the label is 0.5 above the square's top edge (its box bottom is there)
    t = state(r, 0, "text_1").pos
    assert t[0] == pytest.approx(4.0) and t[1] > 1.0 + 0.5
    ring = np.asarray(state(r, 0, "circle_1").points)
    assert ring[:, 1].min() == pytest.approx(-4.0 + 0.5)  # 0.5 from the bottom of the 8-high frame
    assert state(r, 0, "dot_1").opacity == 1.0  # added at t = 0: visible at once


def test_animate_transform_and_groups(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        d = Dot(LEFT * 2, color=RED)
        sq = Square(side_length=1)
        sq2 = Square(side_length=2).shift(RIGHT * 3)
        group = VGroup(Dot(UP), Dot(DOWN)).shift(RIGHT * 5)
        self.add(d, sq, group)
        self.play(d.animate.shift(RIGHT * 4), run_time=2)
        self.play(Transform(sq, sq2), run_time=1)
        self.play(FadeOut(group), run_time=1)
        self.play(FadeIn(Text("end")), run_time=1)
""")  # fmt: skip
    r = import_manim(p)
    assert state(r, 0, "dot_1").pos == pytest.approx((-2.0, 0.0, 0.0))
    assert state(r, 1.0, "dot_1").pos[0] == pytest.approx(0.0, abs=0.2)  # halfway, smoothed
    assert state(r, 2.5, "dot_1").pos == pytest.approx((2.0, 0.0, 0.0))
    # Transform morphs the first square into the second (same number of points) ...
    assert np.ptp(np.asarray(state(r, 3.0, "square_1").points)[:, 0]) == pytest.approx(2.0)
    assert "square_2" not in r.scene.engine.state_at(3.0)  # ... the target is never created
    # the VGroup is two objects, faded together
    assert state(r, 3.0, "dot_2").opacity == 1.0 and state(r, 4.0, "dot_3").opacity == 0.0
    assert state(r, 4.5, "text_1").opacity == pytest.approx(0.5, abs=0.2)


def test_colors_become_the_nearest_style_roles():
    colors = (
        "#FFFFFF",
        "#888888",
        "#222222",
        "#FC6255",
        "#FFFF00",
        "#58C4DD",
        "#83C167",
        "#9A72AC",
        None,
    )
    want = [
        "primary",
        "muted",
        "muted",
        "danger",
        "highlight",
        "secondary",
        "secondary",
        "secondary",
        "primary",
    ]
    assert [color_role(c) for c in colors] == want


def test_axes_plot_and_math_text(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        ax = Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], x_length=6, y_length=6)
        g = ax.plot(lambda x: x ** 2 / 2, x_range=[-2, 2], color=YELLOW)
        label = MathTex(r"y = \\frac{x^2}{2}").to_corner(UR)
        self.play(Create(ax), Create(g), Write(label))
""")  # fmt: skip
    r = import_manim(p)
    pts = np.asarray(state(r, 0, "path_1").points)
    assert pts[0] == pytest.approx((-2.0, 2.0, 0.0), abs=1e-6)  # c2p: 1 unit = 1 world unit here
    assert pts[:, 1].min() == pytest.approx(0.0, abs=1e-3)
    assert state(r, 0, "axes_1").scale == pytest.approx(1.0)
    m = r.scene.engine.inspect(1.0, id="mathtex_1")["objects"][0]["semantic"]
    assert m["math"] is True and r"\frac" in m["text"]


def test_three_d_scenes_use_the_orbit_camera(tmp_path):
    p = script(tmp_path, """
class S(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.5)
        self.add(Dot(OUT))
        self.wait(4)
""")  # fmt: skip
    r = import_manim(p)
    cam = r.scene.engine.document.cameras[0]
    assert cam.kind == "orbit" and cam.params["elevation"] == pytest.approx(30.0)
    a0 = r.scene.engine.state_at(0.0).camera.eye
    a4 = r.scene.engine.state_at(4.0).camera.eye
    assert not np.allclose(a0, a4)  # the camera turned while the scene waited


def test_what_phasor_cannot_express_is_said_not_hidden(tmp_path):
    p = script(tmp_path, """
class S(Scene):
    def construct(self):
        t = ValueTracker(1)
        d = always_redraw(lambda: Dot(RIGHT * t.get_value()))
        sq = Square(fill_opacity=0.5)
        self.add(d, sq)
        self.play(t.animate.set_value(3))
""")  # fmt: skip
    r = import_manim(p)
    w = " ".join(r.warnings)
    assert "always_redraw" in w and "filled shapes" in w and "ValueTracker" in w
    bad = script(tmp_path, """
class S(Scene):
    def construct(self):
        self.play(Circle())
""", "bad.py")  # fmt: skip
    with pytest.raises(ManimImportError, match="wrap it in an animation"):
        import_manim(bad)
    missing = script(tmp_path, """
class S(Scene):
    def construct(self):
        self.add(Dot().whatever())
""", "missing.py")  # fmt: skip
    with pytest.raises(ManimImportError, match="Dot.whatever is not supported"):
        import_manim(missing)
    unknown = script(tmp_path, """
class S(Scene):
    def construct(self):
        self.add(FancyNewMobject())
""", "unknown.py")  # fmt: skip
    with pytest.raises(NameError):
        import_manim(unknown)


def test_scene_choice_and_cleanup(tmp_path):
    p = script(tmp_path, """
class A(Scene):
    def construct(self):
        self.wait(1)

class B(Scene):
    def construct(self):
        self.wait(2)
""")  # fmt: skip
    assert import_manim(p).name == "A" and import_manim(p).others == ["B"]
    assert import_manim(p, "B").duration == 2.0
    with pytest.raises(ManimImportError, match="no Scene 'C'"):
        import_manim(p, "C")
    none = tmp_path / "n.py"
    none.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ManimImportError, match="defines no Manim Scene"):
        import_manim(none)
    assert "manim" not in sys.modules  # the stand-in is removed when the import is done
    assert math.isclose(import_manim(p).duration, 1.0)


# ---------------------------------------------------------------- opening one in the editor


def test_a_manim_script_opens_as_a_plain_scene_you_can_edit(tmp_path):
    from phasor.importers import looks_like_manim
    from phasor.workspace import Workspace

    shutil_copy(ROOT / "examples" / "manim_reverb.py", tmp_path / "reverb.py")
    assert looks_like_manim(tmp_path / "reverb.py") and not looks_like_manim(
        ROOT / "examples" / "basics_3d.py"
    )
    ws = Workspace(tmp_path, allow_scripts=True)
    r = ws.scene_open("reverb.py")
    assert r["imported_from"] == "manim" and "axes with different" in r["notes"][0]
    assert ws.path == tmp_path / "reverb.phasor.json"
    assert not ws.engine.scripted  # not locked to a script: structure edits are allowed
    edit = ws.scene_edit({"op": "add_object", "object": {"id": "extra", "type": "dot"}})
    assert edit["ok"]
    ws.scene_save()
    assert (tmp_path / "reverb.phasor.json").exists()
    locked = Workspace(tmp_path)  # a server that does not run scripts refuses, as for any .py
    with pytest.raises(Exception, match="does not run scripts"):
        locked.scene_open("reverb.py")


def test_the_editor_opens_a_manim_script_directly(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from phasor.gui.window import open_editor

    shutil_copy(ROOT / "examples" / "manim_reverb.py", tmp_path / "reverb.py")
    _app, win, server = open_editor(str(tmp_path / "reverb.py"), show=False)
    try:
        assert win.ed.engine.document.meta.title == "RoomReverb"
        assert len(win.ed.engine.document.objects) == 12
        assert win.timeline.data["duration"] == pytest.approx(14.5)
        assert win.ed.set_param("dot_1", "r", "0.12")["ok"]  # a plain scene: edits apply directly
    finally:
        win.timer.stop()
        server.stop()
        win.close()


def shutil_copy(src, dst):
    import shutil

    shutil.copy(src, dst)
