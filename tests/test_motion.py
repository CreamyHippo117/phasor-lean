"""The motion verbs and objects added after the roadmap: equation transforms, shape morphs,
complex maps, flow particles, tracers, emphasis (flash, circumscribe, wiggle), writing in,
live numbers, sweeps and formula blends. Each is checked through ``Engine.state_at`` and
the inspector, as an AI would, and timed by the theme."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor import (
    Counter,
    Dot,
    Equation,
    Line,
    Scene,
    Text,
    Tracer,
    cx,
    expr,
    fourier,
    geo,
)
from phasor.domains.basic import match_parts
from phasor.domains.cx_math import deform, grid_lines, split_runs
from phasor.domains.geometry_math import particles
from phasor.style.theme import load_theme
from phasor.time.verbs import VerbError, _matched_shapes
from tests.conftest import gl_available

NEW_VERBS = ("transform", "become", "write", "travel", "flash", "circumscribe", "wiggle", "sweep")


def sem(e, i, t=0.0):
    return e.inspect(t, id=i)["objects"][0]["semantic"]


def circle(n=40, r=1.0, c=(0.0, 0.0)):
    a = np.linspace(0, 2 * math.pi, n)
    return [(c[0] + r * math.cos(t), c[1] + r * math.sin(t), 0.0) for t in a]


SQUARE = [(1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0), (1, 1, 0)]


def test_every_theme_times_every_new_verb():
    for name in ("midnight_glow", "paper", "aurora", "prism", "prism_vivid"):
        m = load_theme(name).motion
        assert all(m[v]["duration"] > 0 for v in NEW_VERBS), name


# ---------------------------------------------------------------- equation transforms


def test_matching_parts_is_a_longest_common_subsequence():
    assert match_parts(["x", "=2"], ["x", "+1", "=2"]) == [(0, 0), (1, 2)]
    assert match_parts(["a", "b", "c"], ["c", "a", "b"]) == [(0, 1), (1, 2)]
    assert match_parts(["a"], ["b"]) == []


def test_an_equation_becomes_the_next():
    s = Scene("eq", passes=None)
    s.add(Equation(id="before", parts=("$x$", "$=2$"), pos=(0, 1, 0)))
    s.add(Equation(id="after", parts=("$x$", "$+1$", "$=3$"), colors=("highlight",)))
    s.marker("go", 1.0)
    s.marker("end", 4.0)
    s.at("go").transform("before", "after")
    e = s.engine
    st = lambda t: e.state_at(t)  # noqa: E731
    assert st(0.5)["before"].params.opacity == 1 and st(0.5)["after"].params.opacity == 0
    mid = sem(e, "after", 1.6)
    assert mid["transforming_from"] == ["$x$", "$=2$"] and mid["matched"] == 1
    assert 0 < mid["morph"] < 1
    assert st(1.6)["before"].params.opacity == 0  # the source is gone once it starts
    done = sem(e, "after", 3.0)
    assert "transforming_from" not in done and st(3.0)["after"].params.morph == 1.0
    # mid-transform, the bounds cover both places (layout and the camera see it all)
    b = st(1.6)["after"].impl.world_bounds(st(1.6))
    assert b.hi[1] > 1.0 > 0.0 > b.lo[1] - 0.5
    assert not [i for i in e.lint() if i.severity == "error"]


def test_transform_refuses_what_is_not_an_equation():
    s = Scene("eq", passes=None)
    s.add(Equation(id="a", parts=("$x$",)))
    s.add(Text(id="b", text="x"))
    with pytest.raises(VerbError, match="equation"):
        s.at(0.0).transform("a", "b")


# ---------------------------------------------------------------- shape morphs


def test_closed_shapes_are_matched_so_points_travel_little():
    a, b = np.array(circle(40)), np.array(SQUARE, dtype=float)
    pa, pb = _matched_shapes(a, b)
    assert len(pa) == len(pb) >= 64
    naive = np.square(pa - np.vstack([b[:1]] * len(pa))).sum()
    assert np.square(pa - pb).sum() < naive  # aligned, not all points racing to one corner
    assert np.allclose(pb[0], pb[-1])  # still closed


def test_a_circle_becomes_a_square():
    s = Scene("shape", passes=None)
    s.add(Line(id="circle", points=circle()))
    s.add(Line(id="square", points=SQUARE))
    s.marker("go", 1.0)
    s.at("go").become("circle", "square")
    e = s.engine
    d = load_theme("midnight_glow").motion["become"]["duration"]
    start, end = (
        e.state_at(1.0)["circle"].params.points,
        e.state_at(1.0 + d)["circle"].params.points,
    )
    # resampled along the 40-gon's sides: on the circle to within the polygon's sag
    assert np.allclose(np.linalg.norm(np.asarray(start)[:, :2], axis=1), 1.0, atol=0.01)
    assert np.abs(np.asarray(end)[:, :2]).max(axis=1) == pytest.approx(
        1.0, abs=1e-6
    )  # on the square
    assert e.state_at(0.5)["square"].params.opacity == 0  # the target waits
    assert e.state_at(1.0 + d)["square"].params.opacity == 1
    assert e.state_at(1.0 + d)["circle"].params.opacity == 0


def test_become_refuses_points_already_animated():
    s = Scene("shape", passes=None)
    s.add(Line(id="a", points=circle()))
    s.add(Line(id="b", points=SQUARE))
    s.at(0.0).morph("a.points", [(x * 2, y * 2, z) for x, y, z in circle()])
    s.marker("go", 2.0)
    with pytest.raises(VerbError, match="unanimated"):
        s.at("go").become("a", "b")


# ---------------------------------------------------------------- complex maps


def test_a_grid_deforms_from_the_identity_to_the_map():
    z = grid_lines(0j, (2.0, 2.0), (3, 3), 9)
    assert len(z) == 6 and all(len(line) == 9 for line in z)
    sq = lambda w: w**2  # noqa: E731
    line = z[0]
    assert np.allclose(deform(sq, line, 0.0), line)
    assert np.allclose(deform(sq, line, 1.0), line**2)
    assert np.allclose(deform(sq, line, 0.5), 0.5 * line + 0.5 * line**2)


def test_lines_break_at_poles():
    w = np.array([0, 1, 2, np.inf, 3, 4, 100, 5, 6], dtype=complex)
    assert [len(r) for r in split_runs(w, 20.0)] == [3, 2, 2]


def test_complex_map_reports_its_function_and_bends_with_amount():
    s = Scene("map", passes=None)
    s.add(cx.ComplexMap(id="m", function="z**2", extent=(2.0, 2.0), amount=0.0))
    s.marker("go", 0.5)
    s.at("go").morph("m.amount", 1.0)
    e = s.engine
    flat, bent = (e.state_at(t)["m"].impl.world_bounds(e.state_at(t)) for t in (0.0, 3.0))
    assert flat.hi[0] == pytest.approx(1.0, abs=0.05)
    assert bent.hi[1] == pytest.approx(2.0, abs=0.1)  # (1+i)^2 = 2i: the corner climbs
    m = sem(e, "m", 3.0)
    assert m["latex"] == "z^{2}" and m["amount"] == 1.0
    moved = Scene("moved", passes=None)  # drawn elsewhere, still the map of the same z
    moved.add(cx.ComplexMap(id="m", function="z**2", extent=(2.0, 2.0), pos=(5.0, 1.0, 0)))
    b = moved.engine.state_at(0.0)["m"].impl.world_bounds(moved.engine.state_at(0.0))
    assert b.hi[1] == pytest.approx(1.0 + 2.0, abs=0.1) and b.lo[0] > 3.0
    inv = Scene("inv", passes=None)
    inv.add(cx.ComplexMap(id="m", function="1/z", extent=(2.0, 2.0)))
    assert sem(inv.engine, "m")["pieces"] > sem(inv.engine, "m")["lines"]  # cut at the pole


# ---------------------------------------------------------------- flow particles


def test_particles_are_a_pure_function_of_time():
    paths = np.cumsum(np.ones((3, 20, 3)), axis=1)
    phase = np.array([0.0, 0.3, 0.6])
    a1, al1 = particles(paths, phase, 1.0, 3.0, 0.2)
    a2, al2 = particles(paths, phase, 1.0, 3.0, 0.2)
    b, _ = particles(paths, phase, 1.5, 3.0, 0.2)
    assert np.array_equal(a1, a2) and np.array_equal(al1, al2)
    assert not np.array_equal(a1, b)  # they move
    c, _ = particles(paths, phase, 4.0, 3.0, 0.2)
    assert np.allclose(c, a1)  # and are born again after their lifetime
    assert (al1[:, -1] >= al1[:, 0]).all()  # bright head, faint tail


def test_flow_particles_follow_the_field():
    s = Scene("flow", passes=None)
    s.add(geo.FlowParticles(id="p", fp="1", fq="0", fr="0", extent=(4.0, 4.0, 0.0), count=20))
    st = s.engine.state_at(0.0)
    paths, _, strength = st["p"].impl._paths("p", st["p"].params)
    assert np.allclose(np.diff(paths[..., 1], axis=1), 0.0)  # a field along x: no drift in y
    assert (np.diff(paths[..., 0], axis=1) > 0).all()  # moving right
    assert np.allclose(strength, 1.0)
    assert sem(s.engine, "p")["count"] == 20
    other = Scene("flow2", passes=None)  # another id: other random starts
    other.add(geo.FlowParticles(id="q", fp="1", fq="0", fr="0", extent=(4.0, 4.0, 0.0), count=20))
    st2 = other.engine.state_at(0.0)
    assert not np.allclose(st2["q"].impl._paths("q", st2["q"].params)[0], paths)


# ---------------------------------------------------------------- tracers


def test_a_tracer_travels_its_path_and_says_where_it_is():
    s = Scene("trace", passes=None)
    s.add(Line(id="path", points=[(0, 0, 0), (4, 0, 0)]))
    s.add(Tracer(id="tr", path="path"))
    s.marker("go", 1.0)
    s.at("go").travel("tr")
    e = s.engine
    d = load_theme("midnight_glow").motion["travel"]["duration"]
    assert sem(e, "tr", 0.5)["position"] == [0.0, 0.0, 0.0]
    assert sem(e, "tr", 1.0 + d / 2)["position"][0] == pytest.approx(2.0, abs=1e-6)
    assert sem(e, "tr", 1.0 + d)["position"] == [4.0, 0.0, 0.0]


@pytest.mark.parametrize("make", [
    lambda: fourier.PartialSum(id="curve", kind="square", terms=5),
    lambda: geo.ParametricCurve(id="curve"),
    lambda: __import__("phasor").calc.Graph(id="curve", function="1/x", xmin=-2, xmax=2),
])  # fmt: skip
def test_tracers_ride_every_kind_of_curve(make):
    s = Scene("ride", passes=None)
    s.add(make())
    s.add(Tracer(id="tr", path="curve", progress=0.5))
    assert "position" in sem(s.engine, "tr")


def test_a_tracer_on_nothing_says_so_and_cannot_travel():
    s = Scene("lost", passes=None)
    s.add(Dot(id="d"))
    s.add(Tracer(id="tr", path="d"))  # a dot offers no path
    assert sem(s.engine, "tr")["path_missing"]
    with pytest.raises(VerbError, match="not a curve"):
        s.at(0.0).travel("tr")


# ---------------------------------------------------------------- emphasis


def test_flash_swells_and_falls_back():
    s = Scene("flash", passes=None)
    s.add(Dot(id="d"))
    s.marker("go", 1.0)
    s.at("go").flash("d")
    e = s.engine
    d = load_theme("midnight_glow").motion["flash"]["duration"]
    vals = [e.state_at(1.0 + k * d / 10)["d"].params.flash for k in range(11)]
    assert vals[0] == 0.0 and max(vals) == pytest.approx(1.0) and vals[-1] == 0.0


def test_circumscribe_draws_a_ring_and_reuses_it():
    s = Scene("ring", passes=None)
    s.add(Dot(id="d", r=0.5))
    s.marker("a", 1.0)
    s.marker("b", 4.0)
    s.at("a").circumscribe("d")
    s.at("b").circumscribe("d", shape="box")  # same ring, drawn again
    e = s.engine
    d = load_theme("midnight_glow").motion["circumscribe"]["duration"]
    assert [o.id for o in e.document.objects].count("d.ring") == 1
    assert e.state_at(0.5)["d.ring"].params.progress == 0.0  # nothing before
    assert e.state_at(1.0 + 0.6 * d)["d.ring"].params.progress == pytest.approx(1.0)
    assert e.state_at(1.0 + d)["d.ring"].params.opacity == pytest.approx(0.0)
    assert e.state_at(4.0 + 0.3 * d)["d.ring"].params.opacity == 1.0  # back for the second
    ring = e.state_at(1.5)["d.ring"]
    b = ring.impl.world_bounds(e.state_at(1.5))
    assert b.hi[0] > 0.5 and b.lo[0] < -0.5  # around the dot, not on it


def test_wiggle_shakes_and_returns_to_rest():
    s = Scene("wig", passes=None)
    s.add(Dot(id="d", pos=(1.0, 2.0)))
    s.marker("go", 1.0)
    s.at("go").wiggle("d")
    e = s.engine
    d = load_theme("midnight_glow").motion["wiggle"]["duration"]
    xs = [e.state_at(1.0 + k * d / 8)["d"].params.pos[0] for k in range(9)]
    assert max(xs) > 1.0 > min(xs) and xs[-1] == pytest.approx(1.0)
    assert e.state_at(1.0 + d / 2)["d"].params.pos[1] == pytest.approx(2.0)  # sideways only


# ---------------------------------------------------------------- writing in, numbers, sweeps


def test_text_and_equations_write_in():
    s = Scene("write", passes=None)
    s.add(Text(id="t", text="hello"))
    s.add(Equation(id="e", parts=("$a$", "$=b$"), pos=(0, -1, 0)))
    s.marker("go", 1.0)
    s.at("go").write("t", "e")
    e = s.engine
    assert e.state_at(0.5)["t"].params.write == 0.0
    assert 0 < e.state_at(1.5)["t"].params.write < 1
    assert e.state_at(5.0)["e"].params.write == 1.0
    with pytest.raises(VerbError, match="text or an equation"):
        s.add(Dot(id="d"))
        s.at("go").write("d")


def test_a_counter_follows_what_it_measures():
    s = Scene("count", passes=None)
    s.add(Dot(id="d", pos=(0.0, 0.0)))
    s.add(Counter(id="x", value=expr("d.pos[0]"), prefix="x = ", digits=1))
    s.marker("go", 1.0)
    s.at("go").morph("d.pos", (3.0, 0.0))
    e = s.engine
    assert sem(e, "x", 0.0)["text"] == "$x = 0.0$"
    assert sem(e, "x", 5.0)["text"] == "$x = 3.0$"
    tiny = Counter.Params(value=-0.001, digits=2, math=False)
    assert Counter.text_of(tiny) == "0.00"  # no "-0.00"


def test_sweep_runs_a_number_evenly_with_a_live_readout():
    s = Scene("sweep", passes=None)
    s.add(fourier.PartialSum(id="w", kind="square", terms=3))
    s.marker("go", 1.0)
    s.at("go").sweep("w.terms", 40)
    e = s.engine
    d = load_theme("midnight_glow").motion["sweep"]["duration"]
    assert e.state_at(0.5)["w.readout"].params.opacity == 0.0  # appears with the sweep
    assert sem(e, "w.readout", 1.0 + d)["text"] == r"$\mathrm{terms} = 40$"
    terms = [e.state_at(1.0 + k * d / 10)["w"].params.terms for k in range(11)]
    assert terms == sorted(terms) and terms[0] == 3 and terms[-1] == 40
    assert all(isinstance(v, int) for v in terms)  # whole steps
    assert not [i for i in e.lint() if i.severity == "error"]
    with pytest.raises(VerbError, match="runs a number"):
        s.at("go").sweep("w.kind", 2)


# ---------------------------------------------------------------- formula blends


def test_a_surface_flows_into_another():
    s = Scene("blend", passes=None)
    s.use_camera("orbit")
    s.add(geo.ParametricSurface(id="s", fz="u**2 - v**2", to_fz="0", blend=0.0, res=(9, 9)))
    s.marker("go", 0.5)
    s.at("go").morph("s.blend", 1.0)
    e = s.engine
    h0 = e.state_at(0.0)["s"].impl.world_bounds(e.state_at(0.0))
    h1 = e.state_at(5.0)["s"].impl.world_bounds(e.state_at(5.0))
    assert h0.hi[2] == pytest.approx(4.0, abs=0.05) and h1.hi[2] == pytest.approx(0.0, abs=0.05)


def test_a_curve_flows_into_another():
    c = geo.ParametricCurve(id="c", fx="cos(t)", fy="sin(t)", to_fy="0", blend=1.0)
    s = Scene("blend", passes=None)
    s.add(c)
    pts = s.engine.state_at(0.0)["c"].impl.path_points(s.engine.state_at(0.0))
    assert np.allclose(pts[:, 1], 0.0)


# ---------------------------------------------------------------- the AI and the editor


def test_the_ai_tool_takes_the_new_verbs(tmp_path):
    from phasor.workspace import Workspace
    from phasor.workspace.tools import ToolError

    ws = Workspace(tmp_path)
    ws.scene_new("ai")
    ws.scene_edit([
        {"op": "add_object", "object": {"id": "a", "type": "equation", "params": {"parts": ["$x$"]}}},
        {"op": "add_object", "object": {"id": "b", "type": "equation", "params": {"parts": ["$x$", "$=1$"]}}},
        {"op": "add_object", "object": {"id": "d", "type": "dot"}},
        {"op": "set_marker", "name": "go", "t": 1.0},
    ])  # fmt: skip
    assert ws.animate("transform", "go", targets=["a", "b"])["end"] > 1.0
    assert ws.animate("circumscribe", "go", targets=["d"], value="box")["end"] > 1.0
    assert ws.animate("sweep", "go", path="d.r", value=0.4)["end"] > 1.0
    with pytest.raises(ToolError, match="two target ids"):
        ws.animate("become", "go", targets=["a"])


def test_verbs_give_the_same_scene_whatever_order_they_are_written_in():
    def build(order):
        s = Scene("order", passes=None)
        s.add(Dot(id="d", r=0.4))
        s.add(Text(id="t", text="hi"))
        s.marker("go", 1.0)
        steps = {"flash": lambda: s.at("go").flash("d"), "write": lambda: s.at("go").write("t"),
                 "ring": lambda: s.at("go").circumscribe("d")}  # fmt: skip
        for k in order:
            steps[k]()
        return s.engine

    a, b = build(["flash", "write", "ring"]), build(["ring", "write", "flash"])
    for t in (0.5, 1.3, 2.0):
        assert a.inspect(t, detail="brief") == b.inspect(t, detail="brief")


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_motion_example_matches_its_golden():
    import os
    import runpy
    from pathlib import Path

    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "motion_gallery.py"),
                       run_name="x")["s"].engine  # fmt: skip
    assert not [i for i in e.lint() if i.severity == "error"]
    times = sorted(e.document.timeline.markers.values())[1:5]
    sheet = e.storyboard([t + 0.6 for t in times], size=(480, 270))["image"]
    path = GOLDEN / e.pass_runner().renderer_class() / "motion_gallery.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97


# ---------------------------------------------------------------- narration and sliders


def _narrated(script: str, tmp_path):
    s = Scene("narrated", passes=None)
    s.engine.base_dir = tmp_path
    s.add(Equation(id="before", parts=("$x$", "$=2$")))
    s.add(Equation(id="after", parts=("$x$", "$+1$", "$=3$"), pos=(0, -1, 0)))
    s.add(geo.ParametricCurve(id="loop", fx="cos(t)", fy="sin(2*t)", pos=(3, 0, 0)))
    s.add(Tracer(id="comet", path="loop"))
    s.add(fourier.PartialSum(id="w", kind="square", terms=2, pos=(0, -3, 0)))
    s.add(Dot(id="pt", pos=(-3, 0)))
    s.narrate(script, voice="estimate")
    s.at("change").transform("before", "after")
    s.at("ride").travel("comet")
    s.at("more").sweep("w.terms", 20)
    s.at("look").flash("pt").circumscribe("pt").wiggle("pt")
    return s.engine


@pytest.mark.slow
def test_the_motion_verbs_follow_the_narration(tmp_path):
    """Every new verb is anchored to its marker: say it slower and they all move with it."""
    quick = "[[change]] It changes. [[ride]] It rides. [[more]] More. [[look]] Look."
    slow = ("[[change]] First, slowly and with great care, the equation changes. [[ride]] Then "
            "the comet rides around its loop. [[more]] Now many more terms. [[look]] Look here.")  # fmt: skip
    for script in (quick, slow):
        e = _narrated(script, tmp_path)
        m = e.document.timeline.markers
        st = e.state_at
        assert st(m["change"] - 0.01)["after"].params.opacity == 0
        assert st(m["change"] + 0.01)["after"].params.opacity == 1
        assert st(m["ride"] - 0.01)["comet"].params.progress == 0
        assert st(m["ride"] + 0.5)["comet"].params.progress > 0
        assert st(m["more"] - 0.01)["w"].params.terms == 2
        assert st(m["look"] - 0.01)["pt"].params.flash == 0 < st(m["look"] + 0.2)["pt"].params.flash
    assert m["look"] > 10  # the slow script really moved the markers
    strict = [i for i in e.lint(strict=True) if i.code == "raw_motion" or i.severity == "error"]
    assert not strict, [i.message for i in strict]  # verbs, anchored: what strict asks for


def test_sliders_drive_the_new_objects_and_a_counter_follows(tmp_path):
    from phasor.interactive import Player

    s = Scene("sliders", passes=None)
    s.add(cx.ComplexMap(id="m", function="z**2", amount=0.0, extent=(2.0, 2.0)))
    s.add(geo.ParametricCurve(id="loop", pos=(-4, 0, 0)))
    s.add(Tracer(id="comet", path="loop"))
    s.add(geo.FlowParticles(id="p", fp="-y", fq="x", extent=(3.0, 3.0, 0.0), pos=(4, 0, 0)))
    s.add(Counter(id="amt", value=expr("m.amount"), prefix="a = ", pos=(0, -2, 0)))
    s.control("m.amount", min=0, max=1, step=0.1)
    s.control("comet.progress", min=0, max=1, step=0.05)
    s.control("p.speed", min=0.2, max=3, step=0.2)
    e = s.engine
    pl = Player(e)
    pl.set("m.amount", 0.7)
    pl.set("comet.progress", 0.5)
    pl.set("p.speed", 2.0)
    st = pl.state_at(0.0)
    assert st["amt"].impl.semantics(st)["text"] == "$a = 0.70$"  # the readout follows the slider
    assert st["comet"].impl.semantics(st)["position"][0] == pytest.approx(-5.0, abs=1e-3)
    assert e.document.object("m").params["amount"] == 0.0  # the scene itself is untouched
    s.marker("go", 1.0)
    s.at("go").morph("m.amount", 1.0)  # an animation on a slider's parameter: lint says so
    assert "orphan_control" in {i.code for i in e.lint()}
