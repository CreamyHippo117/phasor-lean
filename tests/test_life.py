"""Living motion, looks and presets."""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Dot, Line, Scene, cx
from phasor.core import life
from phasor.style.looks import LookError, available_looks, load_look
from phasor.style.theme import available_themes, load_theme
from tests.conftest import gl_available


def _scene(life_level, **kw):
    s = Scene("lv", passes=None, life=life_level, **kw)
    s.add(Dot(id="d", pos=(0, 1), r=0.2, style={"glow": "soft"}, tags=["key"]))
    s.add(Dot(id="fixed", pos=(1, 1), r=0.2, tags=["exact"]))
    s.add(Line(id="beam", points=[(0, 0), (2, 0)], style={"glow": "strong"}))
    s.add(cx.DomainColoring(id="plane", function="1/(z**2+1)"))
    return s.engine


# ---------------------------------------------------------------- noise and purity


def test_noise_is_smooth_bounded_and_deterministic():
    ts = np.linspace(0, 20, 400)
    a = np.array([life.noise("d", 0, "x", t, 4.0) for t in ts])
    b = np.array([life.noise("d", 0, "x", t, 4.0) for t in ts])
    assert (a == b).all() and np.abs(a).max() <= 1.0
    assert np.abs(np.diff(a)).max() < 0.2  # no jumps at 20 Hz
    c = np.array([life.noise("d", 1, "x", t, 4.0) for t in ts])
    assert not np.allclose(a, c)  # another seed, another wiggle
    assert not np.allclose(a, [life.noise("other", 0, "x", t, 4.0) for t in ts])


def test_life_off_is_exactly_the_old_still_scene():
    e = _scene("off")
    for t in (0.0, 1.7, 9.0):
        assert e.state_at(t)["d"].params.pos == (0.0, 1.0, 0.0)
        assert e.state_at(t)["plane"].params.hue == 0.0
        assert (
            e.state_at(t)["beam"].style["glow"]
            == load_theme("midnight_glow").values["glow"]["strong"]
        )


def test_state_at_stays_pure_with_life_on():
    e1, e2 = _scene("lively"), _scene("lively")
    for t in (0.0, 3.3, 7.1):
        assert e1.state_at(t)["d"].params.pos == e2.state_at(t)["d"].params.pos
        assert e1.state_at(t)["d"].params.pos == e1.state_at(t)["d"].params.pos


# ---------------------------------------------------------------- what moves, and how much


def test_points_drift_a_fraction_of_their_radius_and_exact_ones_do_not():
    e = _scene("lively")
    th = load_theme("midnight_glow").life
    moved = []
    for t in np.linspace(0, 12, 120):
        x, y, z = e.state_at(float(t))["d"].params.pos
        moved.append(np.hypot(x - 0, y - 1))
        assert e.state_at(float(t))["fixed"].params.pos == (1.0, 1.0, 0.0)
        assert z == 0.0
    assert max(moved) > 0.2 * th["point"] * 0.2  # it does wander ...
    assert max(moved) <= 2 * th["point"] * 0.2 * (
        1.0 + 1e-9
    )  # ... never farther than the theme says


def test_subtle_is_gentler_than_lively():
    amp = {}
    for level in ("subtle", "lively"):
        e = _scene(level)
        amp[level] = max(
            abs(e.state_at(float(t))["d"].params.pos[0]) for t in np.linspace(0, 12, 120)
        )
    assert 0 < amp["subtle"] < amp["lively"]


def test_glow_breathes_but_plain_paint_does_not():
    e = _scene("lively")
    glow = [e.state_at(float(t))["d"].style["glow"] for t in np.linspace(0, 10, 60)]
    base = load_theme("midnight_glow").values["glow"]["soft"]
    assert max(glow) > base > min(glow) and all(g > 1.0 for g in glow)  # swells and fades
    plain = Scene("p", passes=None, life="lively")
    plain.add(Dot(id="paint", pos=(0, 0)))
    for t in (0.0, 4.0):
        assert plain.engine.state_at(t)["paint"].style["glow"] == 1.0  # nothing emitted


def test_color_fields_turn_a_little_but_geometry_never_moves():
    e = _scene("lively")
    hues = [e.state_at(float(t))["plane"].params.hue for t in np.linspace(0, 20, 80)]
    assert 0 < max(abs(h) for h in hues) <= 0.04 + 1e-9
    poles = {tuple(np.round(e.state_at(t)["plane"].impl.features(e.state_at(t))["poles"], 6).ravel())
             for t in (0.0, 5.0, 11.0)}  # fmt: skip
    assert len(poles) == 1


# ---------------------------------------------------------------- the rest of the system sees it


def test_inspector_sees_the_wiggle_and_layout_follows_it():
    s = Scene("lab", passes=None, life="lively")
    s.add(Dot(id="d", pos=(0, 0), r=0.2, tags=["key"]))
    from phasor import Text

    s.add(Text(id="d.label", text="d"))
    s.constrain("label(d) near d")
    e = s.engine
    a = e.inspect(0.0)["objects"]
    b = e.inspect(2.0)["objects"]
    d0 = next(o for o in a if o["id"] == "d")["center"]["px"]
    d1 = next(o for o in b if o["id"] == "d")["center"]["px"]
    assert d0 != d1  # the inspector reports the true, wiggling position
    l0 = next(o for o in a if o["id"] == "d.label")["center"]["px"]
    l1 = next(o for o in b if o["id"] == "d.label")["center"]["px"]
    off0 = (l0[0] - d0[0], l0[1] - d0[1])
    off1 = (l1[0] - d1[0], l1[1] - d1[1])
    assert off0 == pytest.approx(off1, abs=3.0)  # the label stays glued to it
    assert e.lint() == []  # and a wiggle never creates problems by itself


def test_old_documents_stay_still_and_life_is_stored_in_the_document():
    s = Scene("stored", passes=None, look="calm")
    assert s.engine.document.meta.life == "subtle" and s.engine.document.meta.look == "calm"
    from phasor.core import io

    again = io.parse_document(__import__("json").loads(io.dumps(s.engine.document)))
    assert again.meta.life == "subtle"


# ---------------------------------------------------------------- looks and presets


def test_dark_is_the_default_and_looks_exist():
    assert Scene("x", passes=None).engine.document.meta.theme == "midnight_glow"
    assert load_theme("midnight_glow").background[0] < 0.05  # the house look is dark
    assert {"calm", "glow", "chalk"} <= set(available_looks())
    assert {"midnight_glow", "aurora", "paper"} <= set(available_themes())
    assert load_theme("paper").background[0] > 0.7  # the light theme is opt-in
    with pytest.raises(LookError, match="unknown look"):
        load_look("nope")


def test_a_look_fills_defaults_but_explicit_values_win():
    s = Scene("lk", passes=None, look="calm")
    s.add(cx.DomainColoring(id="a", function="z"))
    s.add(cx.DomainColoring(id="b", function="z", softness=0.1))
    s.add(Line(id="l", points=[(0, 0), (1, 0)], style={"glow": "none"}))
    objs = {o.id: o for o in s.engine.document.objects}
    assert objs["a"].params["softness"] == 0.8  # from the look
    assert objs["b"].params["softness"] == 0.1  # explicit wins
    assert objs["l"].style["glow"] == "none"  # explicit wins
    chalk = Scene("c", passes=None, look="chalk")
    assert chalk.engine.document.meta.theme == "paper" and chalk.engine.document.meta.life == "off"
    over = Scene("o", passes=None, look="chalk", theme="aurora", life="lively")
    assert (
        over.engine.document.meta.theme == "aurora" and over.engine.document.meta.life == "lively"
    )


def test_object_presets_and_precedence():
    s = Scene("pr", passes=None, look="calm")
    s.add(cx.DomainColoring(id="p", function="z", preset="soft_phase"))
    s.add(cx.DomainColoring(id="q", function="z", preset="soft_phase", softness=0.2))
    o = {x.id: x for x in s.engine.document.objects}
    assert o["p"].style["colormap"] == "soft_phase" and o["p"].params["softness"] == 0.8
    assert o["q"].params["softness"] == 0.2  # explicit beats the preset
    with pytest.raises(ValueError, match="no preset 'nope'"):
        cx.DomainColoring(id="z", function="z", preset="nope")
    assert "beam" in Line.Presets


def test_varied_palettes_exist_and_domain_coloring_keeps_its_rainbow_by_default():
    th = load_theme("midnight_glow")
    from phasor.style.theme import PALETTES

    assert set(th.values["colormap"]) == {"phase", "soft_phase", "sequential", "ember", "ocean",
                                          "magnitude", *PALETTES}  # fmt: skip
    assert cx.DomainColoring.Style["colormap"] == ("colormap", "phase")  # hue means phase
    wheel = np.array(th.values["colormap"]["phase"])
    soft = np.array(th.values["colormap"]["soft_phase"])
    assert wheel.shape == soft.shape and not np.allclose(wheel, soft)
    assert np.abs(soft[0] - soft[-1]).max() < 0.05  # cyclic: no seam


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_soft_domain_coloring_has_fewer_hard_edges_than_classic():
    def edges(preset):
        s = Scene("e", passes="auto")
        s.set("camera.height", 4.5)
        s.add(cx.DomainColoring(id="p", function="1/(z**2+1)", preset=preset))
        img = s.engine.still(0, (480, 270)).astype(float)
        return np.abs(np.diff(img, axis=1)).max(axis=-1)

    hard, soft = edges("classic"), edges("soft_phase")
    assert (hard > 25).sum() > 3 * (soft > 25).sum()


def test_only_types_that_ask_for_it_wander():
    """Boundary review M7-M9: life once guessed "a dot" from the parameters pos and r, so
    polygons, clock diagrams and acoustic sources drifted (and a source's physics jittered)."""
    from phasor import Dot, Scene, algebra

    s = Scene("lively", passes=None, life="lively")
    s.add(Dot(id="d", pos=(0, 0, 0), r=0.5))
    s.add(algebra.Polygon(id="sq", n=4, pos=(3, 0, 0)))
    s.add(algebra.ClockDiagram(id="clock", n=6, pos=(-3, 0, 0)))
    e = s.engine
    for t in (0.7, 1.9, 3.3):
        st = e.state_at(t)
        assert st["sq"].params.pos == (3.0, 0.0, 0.0)
        assert st["clock"].params.pos == (-3.0, 0.0, 0.0)
    assert len({e.state_at(t)["d"].params.pos for t in (0.7, 1.9, 3.3)}) == 3  # dots still live
