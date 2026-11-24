"""Bifurcation diagrams: ``cloud_math.bifurcation`` and
``PointCloud(source="bifurcation")``, drawn with the ``splat`` kernel."""

from __future__ import annotations

import math
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, cloud
from phasor.domains import cloud as cloud_mod
from phasor.domains import cloud_math as clm
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden"
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _period(r: float, map: str = "logistic", n_iter: int = 4000, tol: float = 1e-4) -> int:
    """How many distinct values the settled orbit at ``r`` takes (clustered at ``tol``)."""
    pts, _ = clm.bifurcation(map, r, r, 1, n_iter, 256)
    x = np.sort(pts.imag)
    return 1 + int((np.diff(x) > tol).sum())


def _lyapunov(r: float, n_iter: int = 4000, n_keep: int = 2000, x0: float = 0.5) -> float:
    xs = clm.orbits("logistic", [r], n_iter, x0)[n_iter - n_keep + 1 :]
    return float(clm.lyapunov("logistic", [r], xs)[0])


# ---------------------------------------------------------------- the maths


def test_the_first_period_doubling_is_at_r_3():
    assert _period(2.9) == _period(2.97) == 1
    assert _period(3.03) == _period(3.1) == 2
    # the fixed point 1 - 1/r, before the doubling
    pts, _ = clm.bifurcation("logistic", 2.8, 2.8, 1, 4000, 10)
    np.testing.assert_allclose(pts.imag, 1 - 1 / 2.8, atol=1e-9)


def test_the_second_doubling_is_near_1_plus_sqrt_6():
    r2 = 1 + math.sqrt(6)  # 3.449...
    assert _period(r2 - 0.02) == 2
    assert _period(r2 + 0.02) == 4
    # the 2-cycle is the pair (r + 1 -+ sqrt((r - 3)(r + 1))) / (2 r)
    r = 3.2
    pts, _ = clm.bifurcation("logistic", r, r, 1, 4000, 2)
    want = (r + 1 + np.array([-1, 1]) * math.sqrt((r - 3) * (r + 1))) / (2 * r)
    np.testing.assert_allclose(np.sort(pts.imag), want, atol=1e-9)


def test_the_cascade_reaches_8_and_16_before_chaos_at_3_5699():
    assert _period(3.55, n_iter=8000) == 8
    assert _period(3.566, n_iter=20000, tol=1e-5) == 16
    for r in (3.5, 3.56):  # still a cycle: the Lyapunov exponent is negative
        assert _lyapunov(r) < 0
    for r in (3.6, 3.7, 3.9):  # chaos past the accumulation point: positive
        assert _lyapunov(r) > 0.05
    # the exact value at r = 4 (from 1/2 the orbit falls on 0 in two steps: start elsewhere)
    assert _lyapunov(4.0, x0=0.3) == pytest.approx(math.log(2), abs=0.03)


def test_the_window_of_period_three_opens_at_1_plus_sqrt_8():
    r3 = 1 + math.sqrt(8)  # 3.8284...
    assert _lyapunov(r3 - 0.01) > 0  # chaos just before it
    assert _period(3.835) == 3
    assert _period(r3 + 0.004) == 3
    assert _lyapunov(3.835) < 0
    assert _period(3.845) == 6  # and its own doubling


def test_points_match_direct_iteration():
    """Each kept point is an iterate of the map, run one r at a time in plain Python: the
    same bits where the orbit settles, and every point within rounding where it is chaotic
    only for a few steps (chaos grows a last-bit difference, so compare the first ones)."""
    n_iter, n_keep = 300, 50
    pts, value = clm.bifurcation("logistic", 2.9, 3.9, 11, n_iter, n_keep)
    assert len(pts) == 11 * n_keep and len(value) == len(pts)
    f, _ = clm.map_function("logistic")
    for k, r in enumerate(np.linspace(2.9, 3.9, 11)):
        x, seen = np.array([0.5]), []
        for _ in range(n_iter):
            x = f(np.array([r]), x)
            seen.append(float(x[0]))
        col = pts[k * n_keep : (k + 1) * n_keep]
        np.testing.assert_array_equal(col.real, r)
        np.testing.assert_array_equal(col.imag, seen[-n_keep:])  # the very same bits
    # and the plain formula, for the first steps (before chaos can grow a rounding error)
    xs = clm.orbits("logistic", [3.7], 10)
    x = 0.5
    for k in range(1, 11):
        x = 3.7 * x * (1 - x)
        assert xs[k, 0] == pytest.approx(x, rel=1e-12)


def test_bifurcation_is_deterministic_and_colours_cycles_apart_from_chaos():
    a = clm.bifurcation("logistic", 2.8, 4.0, 200, 400, 100)
    b = clm.bifurcation("logistic", 2.8, 4.0, 200, 400, 100)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    pts, v = a
    assert ((v >= 0) & (v < clm.VALUE_SPAN + 1e-12)).all()
    cyc = v[np.isclose(pts.real, pts.real[pts.real < 3.2].max())]
    chaos = v[pts.real > 3.95]
    assert cyc.max() < 0.5 * clm.VALUE_SPAN < chaos.min()


def test_a_formula_map_and_the_sine_map():
    a, _ = clm.bifurcation("r*x*(1 - x)", 3.0, 3.9, 20, 200, 30)
    b, _ = clm.bifurcation("logistic", 3.0, 3.9, 20, 200, 30)
    np.testing.assert_allclose(a, b, atol=1e-9)
    # the sine map r sin(pi x) doubles first where its fixed point's slope reaches -1
    # (r about 0.7199); before that one value, after it two
    assert _period(0.70, "sine") == 1 and _period(0.74, "sine") == 2
    # an orbit that escapes is dropped, not drawn at infinity
    pts, _ = clm.bifurcation("logistic", 4.2, 4.3, 5, 100, 10)
    assert len(pts) == 0


@pytest.mark.parametrize("bad", ["r*x*(1 - x) + i", "import os", "r * y", ""])
def test_a_bad_map_is_refused_with_a_readable_message(bad):
    with pytest.raises(ValueError) as err:
        Scene("b", passes=None).add(cloud.PointCloud(id="b", source="bifurcation", map=bad))
    assert "map" in str(err.value)


def test_bad_arguments():
    with pytest.raises(ValueError, match="n_keep"):
        clm.bifurcation(n_iter=10, n_keep=20)
    s = Scene("b", passes=None)
    with pytest.raises(ValueError):
        s.add(cloud.PointCloud(id="b", source="bifurcation", n_iter=10, n_keep=20))
    with pytest.raises(ValueError):
        s.add(cloud.PointCloud(id="c", source="bifurcation", r_range=(3.0, 2.0)))


def test_zoom_window_is_geometric_and_ends_on_both_windows():
    out, inn = (2.8, 4.0), (3.8, 3.86)
    assert clm.zoom_window(out, inn, 0.0) == out
    assert clm.zoom_window(out, inn, 1.0) == inn
    widths = [np.subtract(*clm.zoom_window(out, inn, s)[::-1]) for s in (0.0, 0.25, 0.5, 0.75)]
    ratios = np.array(widths[1:]) / np.array(widths[:-1])
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)  # the same factor each step
    a, b = clm.zoom_window(out, inn, 0.5)
    assert out[0] < a < b < out[1]


# ---------------------------------------------------------------- the object


def _engine(**kw):
    s = Scene("bif", passes=None)
    s.set("camera.height", 4.5)
    s.add(cloud.PointCloud(id="bif", source="bifurcation", size=(8.0, 4.5), **kw))
    return s.engine


def _diag(e, t=0.0, center=(4.0, 2.25), height=4.5, pw=480, ph=270):
    st = e.state_at(t)
    return cloud_mod._diagram("bif", st["bif"].params, t, center, height, pw, ph)


def test_the_cloud_is_the_diagram_in_the_rectangle():
    e = _engine(r_range=(2.8, 4.0), x_range=(0.0, 1.0))
    st = e.state_at(0.0)
    pts, v = st["bif"].impl.cloud(st)
    assert len(pts) > 100_000 and len(v) == len(pts)
    assert pts[:, 0].min() >= -4.0 and pts[:, 0].max() <= 4.0
    assert pts[:, 1].min() >= -2.25 and pts[:, 1].max() <= 2.25
    # each point is an iterate of its column: x = r x (1 - x) maps the kept set into itself
    r = 2.8 + (pts[:, 0] + 4.0) / 8.0 * 1.2
    x = (pts[:, 1] + 2.25) / 4.5
    left = r < 2.95  # a fixed point there: every point at 1 - 1/r
    np.testing.assert_allclose(x[left], 1 - 1 / r[left], atol=1e-6)
    sem = e.inspect(0.0, id="bif")["objects"][0]["semantic"]
    assert sem["source"] == "bifurcation" and sem["map"] == "logistic"
    assert sem["window"] == {"r": [2.8, 4.0], "x": [0.0, 1.0]}
    assert sem["points"] == sem["in_rectangle"] == len(pts)


def test_the_view_is_computed_with_columns_about_a_pixel_apart():
    e = _engine()
    whole = _diag(e)
    zoomed = _diag(e, center=(6.0, 2.25), height=0.45)  # a camera ten times closer
    for d, px in ((whole, 4.5 / 270), (zoomed, 0.45 / 270)):
        cols = np.unique(np.round(d.points[:, 0], 9))
        assert np.median(np.diff(cols)) / px <= 1.0 + 1e-6  # at most a pixel apart
        assert len(cols) >= 480 * 0.9
    # the close view sees only its part, and the columns there are new ones, not stretched
    assert zoomed.points[:, 0].min() > 6.0 - 0.45 * 480 / 270 / 2 - 0.05


def test_columns_keep_their_points_as_the_zoom_goes_on():
    """A column of r is on a lattice fixed in r: seen at two zooms, its points are the
    same bits, so a zoom never shimmers (a zoom in r alone, so x shows the same rows)."""
    e = _engine(r_focus=(3.3, 3.6))
    p = e.state_at(0.0)["bif"].params
    a = cloud_mod._diagram(
        "bif", p.model_copy(update={"focus": 0.50}), 0.0, (4.0, 2.25), 4.5, 480, 270
    )
    b = cloud_mod._diagram(
        "bif", p.model_copy(update={"focus": 0.55}), 0.0, (4.0, 2.25), 4.5, 480, 270
    )
    (r0a, r1a), _ = cloud_mod._window(p.model_copy(update={"focus": 0.50}))
    (r0b, r1b), _ = cloud_mod._window(p.model_copy(update={"focus": 0.55}))
    ra = np.round(r0a + a.points[:, 0] / 8.0 * (r1a - r0a), 12)
    rb = np.round(r0b + b.points[:, 0] / 8.0 * (r1b - r0b), 12)
    common = np.intersect1d(ra, rb)
    assert len(common) > 100
    for c in common[:: max(1, len(common) // 20)]:
        xa, xb = a.points[ra == c, 1], b.points[rb == c, 1]
        np.testing.assert_allclose(xa, xb, atol=1e-9)  # the same iterates, in order


def test_light_and_columns_change_continuously_across_a_lattice_level():
    e = _engine()
    p = e.state_at(0.0)["bif"].params
    # the camera height at which the lattice level steps up: LATTICE * px_per_r = 2^k
    pw = ph = 300
    w = p.size[0]
    lights, counts = [], []
    level_h = w / 1.2 * cloud_mod.LATTICE * ph / 2.0**3  # level_f = 3 exactly here
    for h in (level_h * 1.0005, level_h * 0.9995):
        d = cloud_mod._diagram("bif", p, 0.0, (4.0, 2.25), h, pw, ph)
        lights.append(d.light)
        counts.append(float(d.weights.sum()))
    assert lights[0] == pytest.approx(lights[1], rel=0.01)
    assert counts[0] == pytest.approx(counts[1], rel=0.02)


def test_twinkle_groups_follow_r_not_the_lattice_level():
    j = np.array([3, 6, 12, 5, 10])
    g_fine = cloud_mod._column_group(j * 2, 7)
    g = cloud_mod._column_group(j, 6)
    np.testing.assert_array_equal(g, g_fine)
    assert len(set(g.tolist())) > 1  # different r, (most likely) different groups


def test_sweep_draws_from_the_left_and_focus_zooms_the_window():
    e = _engine(sweep=0.0)
    assert len(_diag(e).points) == 0
    e = _engine(sweep=0.5)
    d = _diag(e)
    assert 0 < d.points[:, 0].max() < 4.0 + 8.0 * 0.05
    e = _engine(r_focus=(3.82, 3.86), x_focus=(0.4, 0.6), focus=1.0)
    sem = e.inspect(0.0, id="bif")["objects"][0]["semantic"]
    assert sem["window"] == {"r": [3.82, 3.86], "x": [0.4, 0.6]}
    d = _diag(e)
    assert len(d.points) > 10_000  # zoomed in on x, columns show more of their orbit


def test_a_tweened_focus_is_pure_and_the_points_are_deterministic():
    s = Scene("bif", passes=None)
    s.add(cloud.PointCloud(id="bif", source="bifurcation", r_focus=(3.4, 3.6), twinkle=0.5))
    s.at(0.0).to("bif.focus", 1.0, duration=2.0)
    e = s.engine
    a, b = _diag(e, 1.0), _diag(e, 1.0)
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.weights, b.weights)
    cloud_mod._orbit_cache.clear()
    cloud_mod._orbit_bytes[0] = 0
    c = _diag(e, 1.0)
    np.testing.assert_array_equal(a.points, c.points)
    w0 = e.inspect(0.5, id="bif")["objects"][0]["semantic"]["window"]["r"]
    w1 = e.inspect(1.5, id="bif")["objects"][0]["semantic"]["window"]["r"]
    assert w1[1] - w1[0] < w0[1] - w0[0]


def test_the_diagram_is_light_on_the_ground():
    s = Scene("bif", passes=None)
    s.set("camera.height", 4.5)
    s.add(cloud.PointCloud(id="bif", preset="prism_bifurcation", size=(8.0, 4.5)))
    st = s.engine.state_at(0.0)
    calls = []

    class Ctx:
        def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0):
            calls.append(pixels)

    st["bif"].impl.draw(Ctx(), st, "color")
    img = calls[0](160, 90, (4.0, 2.25), 4.5)
    assert img.shape == (90, 160, 4)
    a = img[..., 3]
    assert a.max() > 0.5  # the branches glow
    assert (a[:5, :30] == 0).all()  # nothing above the fixed point at the left


@needs_gl
@pytest.mark.slow
def test_bifurcation_golden_storyboard():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    e = runpy.run_path(str(ROOT / "examples" / "bifurcation.py"), run_name="x")["s"].engine
    assert [i for i in e.lint() if i.severity == "error"] == []
    sheet = e.storyboard([3.0, 6.0, 9.6, 20.0], size=(320, 180))["image"]
    klass = e.pass_runner().renderer_class()
    path = GOLDEN / klass / "bifurcation.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {klass} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.98


def test_orbits_can_skip_storing_the_transient():
    r = np.linspace(2.9, 3.95, 37)
    whole = clm.orbits("logistic", r, 500)
    np.testing.assert_array_equal(clm.orbits("logistic", r, 500, start=420), whole[420:])
    np.testing.assert_array_equal(clm.orbits("logistic", r, 500, start=0), whole)
    with pytest.raises(ValueError, match="start"):
        clm.orbits("logistic", r, 10, start=11)
