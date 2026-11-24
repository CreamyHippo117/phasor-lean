"""The aesthetic lint rules."""

from __future__ import annotations

import pytest

from phasor import Dot, Line, Scene, Text, cx
from tests.conftest import gl_available

MODES = [None, pytest.param("auto", marks=[pytest.mark.gl,
                                          pytest.mark.skipif(not gl_available(), reason="gl")])]  # fmt: skip


def _codes(e, **kw):
    return {i.code: i for i in e.lint(aesthetic=True, **kw) if i.code in AESTHETIC}


AESTHETIC = {"focal_point", "color_restraint", "glow_clip", "density", "motion_rhythm",
             "margin", "stroke_mix"}  # fmt: skip


def _base(passes, **kw):
    s = Scene("a", passes=passes, **kw)
    s.set("camera.height", 6)
    return s


def test_aesthetic_rules_are_off_unless_asked():
    s = _base(None)
    s.add(Dot(id="a", pos=(0, 0), tags=["key"]))
    assert not [i for i in s.engine.lint() if i.code in AESTHETIC]


@pytest.mark.parametrize("passes", MODES)
def test_focal_point_wants_exactly_one_highlight(passes):
    s = _base(passes)
    s.add(Dot(id="a", pos=(-1, 0), r=0.3, style={"color": "secondary"}))
    s.marker("beat", 1.0)
    assert "focal_point" in _codes(s.engine)  # nothing highlighted
    assert "nothing is highlighted" in _codes(s.engine)["focal_point"].message
    s.add(Dot(id="b", pos=(1, 0), r=0.3, style={"color": "highlight"}))
    assert "focal_point" not in _codes(s.engine)
    s.add(Dot(id="c", pos=(0, 1), r=0.3, style={"color": "highlight"}))
    issue = _codes(s.engine)["focal_point"]
    assert set(issue.ids) == {"b", "c"} and "at once" in issue.message


@pytest.mark.parametrize("passes", MODES)
def test_color_restraint_counts_hues_and_exempts_phase_coloring(passes):
    s = _base(passes)
    for k, color in enumerate(("highlight", "secondary", "danger", "primary")):
        s.add(Dot(id=f"d{k}", pos=(k - 2, 0), r=0.2, style={"color": color}))
    s.marker("beat", 1.0)
    assert "color_restraint" not in _codes(s.engine)  # 3 saturated hues (primary is gray)
    s.add(Dot(id="extra", pos=(0, 2), r=0.2, style={"color": "highlight"}))
    assert "color_restraint" not in _codes(s.engine)  # same hue as another: still 3
    s2 = _base(passes)
    s2.add(cx.DomainColoring(id="plane", function="z"))  # the full wheel: hue is phase
    s2.add(Dot(id="a", pos=(0, 0), style={"color": "highlight"}))
    s2.marker("beat", 1.0)
    assert "color_restraint" not in _codes(s2.engine)
    s3 = _base(passes)
    s3.add(cx.Fractal(id="f", preset="classic"))  # a ramp through three hues ...
    s3.add(cx.RiemannSurface(id="r", preset="soft_glass"))  # ... and another three-hue palette
    s3.marker("beat", 1.0)
    assert "color_restraint" in _codes(s3.engine)  # four hues together
    s4 = _base(passes)
    s4.add(cx.Fractal(id="f", preset="ocean"))  # a restrained two-tone ramp
    s4.add(Dot(id="a", pos=(0, 0), style={"color": "highlight"}))
    s4.marker("beat", 1.0)
    assert "color_restraint" not in _codes(s4.engine)


def test_every_theme_keeps_its_restrained_palettes_restrained():
    from phasor.inspect.aesthetic import _lut_hues
    from phasor.style.theme import available_themes, load_theme

    for name in available_themes():
        cm = load_theme(name).values["colormap"]
        for role in ("soft_phase", "ember", "ocean", "sequential"):
            assert len(_lut_hues(cm[role])) <= 3, (name, role)
        assert len(_lut_hues(cm["phase"])) >= 4  # the full wheel is the full wheel


@pytest.mark.parametrize("passes", MODES)
def test_density_flags_empty_and_crowded_frames(passes):
    s = _base(passes)
    s.add(Dot(id="tiny", pos=(0, 0), r=0.05, style={"color": "highlight"}))
    s.marker("beat", 1.0)
    issue = _codes(s.engine)["density"]
    assert "empty" in issue.message
    big = _base(passes)
    big.add(Line(id="wall", points=[(-20, 0), (20, 0)], style={"stroke": "bold"}))
    big.add(Dot(id="huge", pos=(0, 0), r=6.5, style={"color": "highlight"}))
    big.marker("beat", 1.0)
    assert "crowded" in _codes(big.engine)["density"].message
    ok = _base(passes)
    ok.add(Dot(id="fair", pos=(0, 0), r=1.2, style={"color": "highlight"}))
    ok.marker("beat", 1.0)
    assert "density" not in _codes(ok.engine)


@pytest.mark.parametrize("passes", MODES)
def test_margin_flags_and_fixes_content_near_the_edge(passes):
    s = _base(passes)
    s.add(Dot(id="edge", pos=(4.9, 0), r=0.3, style={"color": "highlight"}))
    s.add(Dot(id="mid", pos=(0, 0), r=1.0, style={"color": "secondary"}))
    s.marker("beat", 1.0)
    issue = _codes(s.engine)["margin"]
    assert (
        issue.ids == ("edge",) and issue.fix and issue.fix[0]["rule"].startswith("on_screen(edge,")
    )
    s.engine.autofix(aesthetic=True)
    assert "margin" not in _codes(s.engine)  # the fix moved it inside the safe area


@pytest.mark.parametrize("passes", MODES)
def test_stroke_mix_and_its_fix(passes):
    s = _base(passes)
    for k, w in enumerate(("hairline", "regular", "bold")):
        s.add(Line(id=f"l{k}", points=[(-2, k - 1), (2, k - 1)], style={"stroke": w}))
    s.add(Dot(id="d", pos=(0, 0), style={"color": "highlight"}))
    s.marker("beat", 1.0)
    issue = _codes(s.engine)["stroke_mix"]
    assert issue.fix and all(op["key"] == "stroke" for op in issue.fix)
    s.engine.autofix(aesthetic=True)
    assert "stroke_mix" not in _codes(s.engine)


def test_motion_rhythm_flags_unrelated_motions_but_not_a_gesture():
    s = _base(None)
    for name, x in (("a", -2), ("b", 0), ("c", 2)):
        s.add(Dot(id=name, pos=(x, 0), style={"color": "highlight"}))
        s.add(Text(id=f"{name}.label", text=name))
    s.marker("one", 1.0)
    s.at("one").reveal("a", "a.label", "b")  # one gesture (and a label that belongs to a)
    assert "motion_rhythm" not in _codes(s.engine)
    s.at("one").morph("c.pos", (2, 1))  # a different motion, unrelated, at the same moment
    issue = _codes(s.engine)["motion_rhythm"]
    assert "c" in issue.ids and issue.times
    s2 = _base(None)
    s2.add(Dot(id="a", pos=(0, 0)))
    s2.add(Dot(id="c", pos=(2, 0)))
    s2.marker("one", 1.0)
    s2.marker("two", 3.0)
    s2.at("one").reveal("a")
    s2.at("two").morph("c.pos", (2, 1))
    assert "motion_rhythm" not in _codes(s2.engine)  # apart in time: fine


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_glow_clip_and_its_fix():
    s = _base("auto")
    s.add(Dot(id="sun", pos=(0, 0), r=2.2, style={"color": "highlight", "glow": "strong"}))
    s.marker("beat", 1.0)
    issue = _codes(s.engine)["glow_clip"]
    assert issue.ids == ("sun",) and issue.fix[0]["role"] == "soft"
    s.engine.autofix(aesthetic=True)
    assert "glow_clip" not in _codes(s.engine)


def test_the_cli_and_mcp_expose_the_aesthetic_pass(tmp_path):
    from phasor.cli import main
    from phasor.workspace.tools import Workspace

    s = _base(None)
    s.add(Dot(id="a", pos=(0, 0), r=0.05))
    s.marker("beat", 1.0)
    path = tmp_path / "a.phasor.json"
    s.engine.save(path)
    ws = Workspace(tmp_path)
    ws.scene_open("a.phasor.json")
    assert not ws.lint()["issues"]
    assert {i["code"] for i in ws.lint(aesthetic=True)["issues"]} >= {"focal_point", "density"}
    assert main(["lint", str(path), "--aesthetic", "--strict"]) == 1
