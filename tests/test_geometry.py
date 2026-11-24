"""3D basics (from reference pictures): axes with ticks, a parametric curve (a helix), a
parametric surface that maps an input square to an output shape (a half sphere), vectors and
flow lines of a field, color by output, and rotating around all of it."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from phasor import Scene, calc, expr, geo
from phasor.domains import geometry_math as gm
from tests.conftest import gl_available


def test_formulas_compile_and_hostile_ones_are_refused():
    f = gm.compile_fn("cos(t) * 2", ("t",))
    np.testing.assert_allclose(f(np.array([0.0, math.pi])), [2.0, -2.0], atol=1e-12)
    assert gm.compile_fn("3", ("u", "v"))(np.zeros((2, 3)), np.zeros((2, 3))).shape == (2, 3)
    for bad in ("__import__('os')", "t.real", "open('x')", "w + 1"):
        with pytest.raises(ValueError):
            gm.check(bad, ("t",))


def test_a_half_sphere_has_half_the_area_and_its_points_sit_on_the_sphere():
    fx = gm.compile_fn("2*sin(u)*cos(v)", ("u", "v"))
    fy = gm.compile_fn("2*sin(u)*sin(v)", ("u", "v"))
    fz = gm.compile_fn("2*cos(u)", ("u", "v"))
    pts, _U, _V = gm.surface_grid(fx, fy, fz, (0, math.pi / 2), (0, 2 * math.pi), (60, 120))
    assert np.abs(np.linalg.norm(pts, axis=-1) - 2.0).max() < 1e-9
    assert gm.area_of_grid(pts) == pytest.approx(2 * math.pi * 4, rel=2e-3)  # 2 pi r^2
    assert pts[..., 2].min() >= -1e-12  # the upper half


def test_a_helix_has_the_arc_length_the_formula_gives():
    t = np.linspace(0, 6 * math.pi, 4000)
    pts = np.stack([gm.compile_fn("cos(t)", ("t",))(t), gm.compile_fn("sin(t)", ("t",))(t),
                    gm.compile_fn("0.2*t", ("t",))(t)], axis=1)  # fmt: skip
    length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    assert length == pytest.approx(6 * math.pi * math.hypot(1, 0.2), rel=1e-4)


def test_ticks_skip_zero_and_stay_inside_the_range():
    np.testing.assert_allclose(gm.tick_values(-3, 3, 1), [-3, -2, -1, 1, 2, 3])
    np.testing.assert_allclose(gm.tick_values(0, 2, 0.5), [0.5, 1.0, 1.5, 2.0])
    with pytest.raises(ValueError):
        gm.tick_values(-1, 1, 0)


def test_color_by_output_normalizes_and_samples_a_ramp():
    v = np.array([[0.0, 5.0], [10.0, 2.5]])
    n = gm.normalize01(v)
    assert n.min() == 0 and n.max() == 1 and n[0, 1] == 0.5
    assert (gm.normalize01(np.ones(4)) == 0.5).all()  # a flat output: the middle color
    assert gm.normalize01(v, 0, 5)[1, 0] == 1.0  # fixed range clips
    lut = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=float)
    np.testing.assert_allclose(gm.sample_ramp(lut, np.array([0.0, 0.25, 1.0])),
                               [[0, 0, 0], [0.5, 0, 0], [1, 1, 1]])  # fmt: skip


def test_flow_lines_follow_the_field():
    spin = lambda p: np.stack([-p[:, 1], p[:, 0], np.zeros(len(p))], axis=1)  # noqa: E731
    seeds = np.array([[1.0, 0, 0], [2.0, 0, 0]])
    lines, speed = gm.flow_lines(spin, seeds, steps=400, h=0.05)
    r = np.linalg.norm(lines[..., :2], axis=-1)
    assert np.abs(r[0] - 1).max() < 1e-4 and np.abs(r[1] - 2).max() < 1e-4  # circles
    assert speed[1, 0] == pytest.approx(2.0)  # speed is the field's size (grows outward)
    arc = np.linalg.norm(np.diff(lines[1], axis=0), axis=1).sum()
    assert arc == pytest.approx(400 * 0.05, rel=1e-3)  # arc length = steps * h
    still = lambda p: np.zeros_like(p)  # noqa: E731
    stop, _ = gm.flow_lines(still, seeds, steps=5, h=0.1)
    assert (stop == seeds[:, None, :]).all()  # nowhere to go: the line stays
    both, _ = gm.flow_lines(lambda p: np.tile([1.0, 0, 0], (len(p), 1)), seeds[:1], 10, 0.1, True)
    assert both.shape[1] == 21 and both[0, 0, 0] == pytest.approx(0.0, abs=1e-12)


def test_divergence_and_curl_of_known_fields():
    assert gm.divergence("x", "y", "z") == 3
    assert gm.divergence("-y", "x", "0") == 0
    cx, cy, cz = gm.curl("-y", "x", "0")
    assert (cx, cy, cz) == (0, 0, 2)


# ---------------------------------------------------------------- the objects


def sem(e, oid, t=0.0):
    return e.inspect(t, id=oid)["objects"][0]["semantic"]


BALL = dict(fx="2*sin(u)*cos(v)", fy="2*sin(u)*sin(v)", fz="2*cos(u)",
            u_range=(0, math.pi / 2), v_range=(0, 2 * math.pi), res=(40, 80))  # fmt: skip


def test_axes_report_their_ticks_and_ends():
    s = Scene("a", passes=None)
    s.add(geo.Axes(id="ax", dims=3, x_range=(-3, 3, 1), z_range=(-1, 6, 2), pos=(1, 0, 0), scale=2))
    e = s.engine
    m = sem(e, "ax")
    assert m["dims"] == 3 and m["axes"]["x"]["ticks"] == 6 and m["axes"]["z"]["ticks"] == 3
    f = e.inspect(0, id="ax", detail="full")["objects"][0]["features"]
    assert f["x_end"]["world"][0] == pytest.approx([7.0, 0, 0]) and f["z_end"]["world"][
        0
    ] == pytest.approx([1, 0, 12.0])
    two = Scene("b", passes=None)
    two.add(geo.Axes(id="ax", dims=2))
    assert set(sem(two.engine, "ax")["axes"]) == {"x", "y"}
    with pytest.raises(Exception, match="min < max"):
        Scene("c", passes=None).add(geo.Axes(id="ax", x_range=(2, 1, 1)))


def test_a_curve_reports_its_length_and_color_range():
    s = Scene("c", passes=None)
    s.add(geo.ParametricCurve(id="h", fx="cos(t)", fy="sin(t)", fz="0.2*t", t_range=(0, 6 * math.pi),
                              samples=3000, color_by="z"))  # fmt: skip
    s.add(
        geo.ParametricCurve(
            id="ring", t_range=(0, 2 * math.pi), samples=2000, progress=expr("t / 4")
        )
    )
    e = s.engine
    h = sem(e, "h")
    assert h["length"] == pytest.approx(6 * math.pi * math.hypot(1, 0.2), rel=1e-3)
    assert h["color_by"] == "z" and h["color_range"][1] == pytest.approx(
        0.2 * 6 * math.pi, rel=1e-3
    )
    assert not h["closed"] and sem(e, "ring")["closed"]
    assert sem(e, "ring", 2.0)["drawn"] == pytest.approx(0.5)
    with pytest.raises(Exception, match="cannot use|cannot read|unknown name"):
        Scene("x", passes=None).add(geo.ParametricCurve(id="bad", fx="__import__('os')"))


def test_the_half_sphere_surface_and_its_map_from_the_flat_input():
    s = Scene("s", passes=None)
    s.add(geo.ParametricSurface(id="ball", color_by="height", with_map=True, domain_pos=(-6, 0, 0),
                                morph=expr("t / 2"), **BALL))  # fmt: skip
    e = s.engine
    m = sem(e, "ball", 2.0)
    assert m["area"] == pytest.approx(2 * math.pi * 4, rel=3e-3) and m["color_range"][
        0
    ] == pytest.approx(0, abs=1e-9)
    assert m["maps"]["morph"] == 1.0 and sem(e, "ball", 0.0)["maps"]["morph"] == 0.0
    obj = e.state_at(0.0)["ball"].impl
    state0, state2 = e.state_at(0.0), e.state_at(2.0)
    flat = obj._positions(state0["ball"].params)[0]
    done = obj._positions(state2["ball"].params)[0]
    assert np.allclose(flat[..., 2], 0) and np.allclose(
        flat[..., 0].mean(), -6, atol=1e-9
    )  # flat input
    assert np.abs(np.linalg.norm(done, axis=-1) - 2).max() < 1e-9  # the output: on the sphere
    mid = obj._positions(e.state_at(1.0)["ball"].params)[0]
    assert np.abs(mid - (flat + done) / 2).max() < 1e-9  # the map moves them smoothly


def test_color_by_formula_of_the_output():
    s = Scene("s", passes=None)
    s.add(geo.ParametricSurface(id="a", color_by="expr", color_expr="x**2 + y**2", **BALL))
    s.add(geo.ParametricSurface(id="b", color_by="radius", **BALL))
    e = s.engine
    assert sem(e, "a")["color_range"] == pytest.approx([0.0, 4.0], abs=1e-6)  # r^2 sin^2(u)
    assert sem(e, "b")["color_range"] == pytest.approx([2.0, 2.0], abs=1e-9)  # a sphere: constant
    with pytest.raises(Exception, match="cannot use|cannot read|unknown name"):
        Scene("x", passes=None).add(geo.ParametricSurface(id="c", color_expr="w + 1"))


def test_vector_fields_report_divergence_curl_and_flow():
    s = Scene("v", passes=None)
    s.add(geo.VectorField3D(id="f", fp="x", fq="y", fr="z", extent=(4, 4, 4), counts=(3, 3, 3)))
    s.add(geo.VectorField3D(id="spin", fp="-y", fq="x", fr="0"))
    s.add(geo.FlowLines(id="l", fp="-y", fq="x", fr="0", extent=(4, 0, 0), seeds=(3, 1, 1),
                        length=2.0, steps=200))  # fmt: skip
    e = s.engine
    assert sem(e, "f")["divergence"] == "3" and sem(e, "f")["arrows"] == 27
    assert sem(e, "spin")["curl"] == ["0", "0", "2"] and sem(e, "spin")["divergence"] == "0"
    flow = sem(e, "l")
    assert flow["lines"] == 3 and flow["mean_length"] == pytest.approx(
        2.0 * 2 / 3, rel=0.02
    )  # the centre seed is still


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_you_can_rotate_around_a_3d_object_and_it_stays_in_view():
    """A full orbit: every quarter turn shows a different picture, nothing leaves the frame."""
    s = Scene("orbit")
    s.use_camera("orbit")
    s.set("camera.target", (0, 0, 2.5))
    s.set("camera.distance", 14)
    s.set("camera.elevation", 20)
    s.set("camera.drift", 0.0)
    s.add(geo.Axes(id="axes"))
    s.add(geo.ParametricCurve(id="helix", fx="1.6*cos(t)", fy="1.6*sin(t)", fz="0.16*t",
                              t_range=(0, 37.7), samples=600, color_by="t"))  # fmt: skip
    s.marker("start", 0.0)
    s.marker("end", 4.0)
    s.at("start").orbit(360.0, until="end")
    e = s.engine
    frames = [e.still(t, (320, 180)) for t in (0.0, 1.0, 2.0, 3.0)]
    for a, b in itertools.pairwise(frames):
        assert np.abs(a.astype(int) - b.astype(int)).mean() > 0.2  # the view really turned
    for t in (0.0, 1.0, 2.0, 3.0, 4.0):
        h = e.inspect(t, id="helix")["objects"][0]
        assert h["on_screen_fraction"] > 0.98
    az = [e.state_at(t).camera.eye for t in (0.0, 4.0)]
    assert np.allclose(az[0], az[1], atol=1e-6)  # a full turn comes back to the start


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_arrows_colored_by_size_differ_in_color_and_the_surface_draws_at_full_map():
    s = Scene("c")
    s.set("camera.height", 8)
    s.add(geo.ParametricSurface(id="n", **{**BALL, "res": (20, 40)}))  # plain, no map
    plain = Scene("p")
    plain.set("camera.height", 6)
    plain.add(calc.VectorField(id="f", p="-y", q="x", color_by_size=True))
    img = plain.engine.still(0, (400, 300))
    lit = img.reshape(-1, 3)[img.reshape(-1, 3).max(1) > 60]
    assert len({tuple(c // 24) for c in lit}) > 4  # many different colors: size is shown
    assert s.engine.lint() == []


def _demo():
    import runpy
    from pathlib import Path

    return runpy.run_path(str(Path(__file__).parents[1] / "examples" / "basics_3d.py"),
                          run_name="x")["s"].engine  # fmt: skip


def test_the_3d_basics_example_runs_and_reads_back():
    e = _demo()
    assert (
        sem(e, "ball", 12.0)["maps"]["morph"] == 1.0 and sem(e, "ball", 0.0)["maps"]["morph"] == 0.0
    )
    assert sem(e, "helix", 12.0)["length"] > 40 and "drawn" not in sem(e, "helix", 12.0)
    assert sem(e, "flow")["lines"] == 16
    assert e.lint() == []


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_3d_basics_example_matches_its_golden():
    import os

    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = _demo()
    path = GOLDEN / e.pass_runner().renderer_class() / "basics_3d.png"
    sheet = e.storyboard([2.0, 8.0, 11.0], size=(480, 270))["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97


def test_a_formula_is_compiled_once_and_gives_the_same_numbers():
    """``compile_fn`` used to run SymPy's lambdify on every draw (7 s of loading
    motion_gallery). The compiled function is kept per (formula,
    names); its results do not change."""
    import numpy as np

    from phasor.domains import geometry_math as gm

    t = np.linspace(0.0, 6.0, 101)
    a = gm.compile_fn("cos(3*t) * exp(-t/4)", ("t",))
    assert gm.compile_fn("cos(3*t) * exp(-t/4)", ("t",)) is a
    assert np.array_equal(a(t), np.cos(3 * t) * np.exp(-t / 4))
    assert gm.compile_fn("2", ("t",))(t).shape == t.shape  # constants still broadcast
