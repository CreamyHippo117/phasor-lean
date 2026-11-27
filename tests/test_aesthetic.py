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
    s.add(Dot(id="z", pos=(0, -1), r=0.3, style={"color": "secondary"}))
    s.marker("beat", 1.0)
    assert "focal_point" in _codes(s.engine)  # nothing highlighted
    assert "nothing is highlighted" in _codes(s.engine)["focal_point"].message
    s.add(Dot(id="b", pos=(1, 0), r=0.3, style={"color": "highlight"}))
    assert "focal_point" not in _codes(s.engine)
    s.add(Dot(id="c", pos=(0, 1), r=0.3, style={"color": "highlight"}))
    issue = _codes(s.engine)["focal_point"]
    assert set(issue.ids) == {"b", "c"} and "at once" in issue.message


@pytest.mark.parametrize("passes", MODES)
def test_focal_point_is_a_lone_object_whether_or_not_it_is_highlighted(passes):
    s = _base(passes)
    s.add(Dot(id="a", pos=(0, 0), r=1.2, style={"color": "secondary"}))
    s.marker("beat", 1.0)
    assert "focal_point" not in _codes(s.engine)  # one animation: it is the focal point
    s.add(Text(id="a.label", text="a"))
    assert "focal_point" not in _codes(s.engine)  # labels do not count
    s.add(Dot(id="b", pos=(2, 0), r=0.5, style={"color": "secondary"}))
    assert "focal_point" in _codes(s.engine)  # two objects, no highlight: still reported


@pytest.fixture
def two_hues(monkeypatch):
    """A theme that allows two saturated flat-paint hues (the stock ones allow three or more)."""
    from phasor.style.theme import load_theme
    from phasor.time import compile as compile_

    def tight(name):
        theme = load_theme(name)
        theme.lint["hues"] = 2
        return theme

    monkeypatch.setattr(compile_, "load_theme", tight)


def _paint_dots(s, *colors):
    for k, color in enumerate(colors):
        s.add(Dot(id=f"d{k}", pos=(k - 2, 0), r=0.2, style={"color": color}))


@pytest.mark.parametrize("passes", MODES)
def test_color_restraint_counts_flat_paint_hues(passes, two_hues):
    s = _base(passes)
    _paint_dots(s, "highlight", "secondary", "primary")  # primary is gray
    s.marker("beat", 1.0)
    assert "color_restraint" not in _codes(s.engine)  # 2 saturated hues
    s.add(Dot(id="same", pos=(0, 2), r=0.2, style={"color": "highlight"}))
    assert "color_restraint" not in _codes(s.engine)  # same hue as another: still 2
    s.add(Dot(id="third", pos=(0, -2), r=0.2, style={"color": "danger"}))
    assert "3 saturated paint hues" in _codes(s.engine)["color_restraint"].message


@pytest.mark.parametrize("passes", MODES)
def test_color_restraint_ignores_everything_colored_through_a_colormap(passes, two_hues):
    s = _base(passes)
    s.add(cx.DomainColoring(id="plane", function="z"))  # the full wheel: hue is phase
    s.add(cx.Fractal(id="f", preset="classic"))  # ramps carry hue as a quantity too,
    s.add(cx.RiemannSurface(id="r", preset="soft_glass"))  # whatever their colormap role
    _paint_dots(s, "highlight", "secondary")
    s.marker("beat", 1.0)
    assert "color_restraint" not in _codes(s.engine)
    s.add(Dot(id="third", pos=(0, 2), r=0.2, style={"color": "danger"}))
    assert "color_restraint" in _codes(s.engine)  # flat paint still counts beside them


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
    alone = _base(passes)
    alone.add(Dot(id="huge", pos=(0, 0), r=6.5, style={"color": "highlight"}))
    alone.marker("beat", 1.0)
    assert "density" not in _codes(alone.engine)  # one animation may fill the whole frame
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
    s.add(Dot(id="b", pos=(1, 0), r=0.05))
    s.marker("beat", 1.0)
    path = tmp_path / "a.phasor.json"
    s.engine.save(path)
    ws = Workspace(tmp_path)
    ws.scene_open("a.phasor.json")
    assert not ws.lint()["issues"]
    assert {i["code"] for i in ws.lint(aesthetic=True)["issues"]} >= {"focal_point", "density"}
    assert main(["lint", str(path), "--aesthetic", "--strict"]) == 1
