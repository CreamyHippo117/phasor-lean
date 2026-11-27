"""The N-dimensional echo lattice drawn as a star in 3D:
``domains/ndlattice_math.py`` and ``nd.StarLattice``."""

from __future__ import annotations

import itertools
import math
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, expr, nd
from phasor.domains import ndlattice_math as nlm
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _brute(delays, max_time, max_order=nlm.MAX_ORDER):
    """Every n with sum n_i d_i <= max_time, by trying them all."""
    top = [int(max_time // d) for d in delays]
    out = [n for n in itertools.product(*[range(t + 1) for t in top])
           if np.dot(n, delays) <= max_time + 1e-9 and sum(n) <= max_order]  # fmt: skip
    return {tuple(n) for n in out}


# ---------------------------------------------------------------- the maths


@pytest.mark.parametrize("k, n", [(1, 7), (2, 5), (3, 4), (5, 3), (16, 4), (32, 2)])
def test_unit_delays_count_the_simplex(k, n):
    """With every delay 1 the echoes by time n are the n in N^K with sum n_i <= n: C(n+K, K)."""
    lat = nlm.enumerate_lattice((1.0,) * k, float(n), max_points=nlm.MAX_POINTS)
    assert len(lat["arrival"]) == nlm.lattice_count(k, n) == math.comb(n + k, k)
    assert not lat["truncated"] and lat["complete_until"] == n
    # max_order cuts the same simplex whatever the time
    lat = nlm.enumerate_lattice(nlm.default_delays(k), 100.0, max_order=n - 1,
                                max_points=nlm.MAX_POINTS)  # fmt: skip
    assert len(lat["arrival"]) == math.comb(n - 1 + k, k)
    assert lat["order"].max() == n - 1


def test_real_delays_give_exactly_the_points_under_the_hyperplane():
    d = (1.0, 1.31, 0.77, 2.05)
    lat = nlm.enumerate_lattice(d, 5.5)
    got = {tuple(int(x) for x in n) for n in lat["index"]}
    assert got == _brute(d, 5.5)
    # arrival times are sum n_i d_i, sorted, and order is the total laps
    np.testing.assert_allclose(lat["arrival"], lat["index"] @ np.array(d), atol=1e-12)
    assert np.all(np.diff(lat["arrival"]) >= 0)
    np.testing.assert_array_equal(lat["order"], lat["index"].sum(axis=1))
    assert tuple(lat["index"][0]) == (0, 0, 0, 0) and lat["arrival"][0] == 0.0


def test_the_cap_drops_whole_arrival_times_never_a_ragged_sample():
    d = nlm.default_delays(6)
    full = nlm.enumerate_lattice(d, 6.0, max_points=nlm.MAX_POINTS)
    cut = nlm.enumerate_lattice(d, 6.0, max_points=300)
    assert cut["truncated"] and len(cut["arrival"]) <= 300
    t = cut["complete_until"]
    below = full["arrival"] < t - 1e-9
    assert len(cut["arrival"]) == int(below.sum())  # every echo before t, nothing after
    np.testing.assert_array_equal(cut["index"], full["index"][below])
    # far too many asked for: the time steps down until the count fits, still complete
    huge = nlm.enumerate_lattice(nlm.default_delays(16), 40.0, max_points=5000)
    assert huge["truncated"] and 0 < len(huge["arrival"]) <= 5000
    assert huge["arrival"].max() < huge["complete_until"] < 40.0


def test_the_count_grows_like_t_to_the_k():
    """The story: with K loops the echoes by time t grow like t^K / (K! prod d_i)."""
    for k, (t0, t1) in ((3, (20.0, 40.0)), (8, (8.0, 12.0))):
        d = nlm.default_delays(k)
        n0, n1 = (len(nlm.enumerate_lattice(d, t, max_points=nlm.MAX_POINTS)["arrival"])
                  for t in (t0, t1))  # fmt: skip
        slope = math.log(n1 / n0) / math.log(t1 / t0)
        # the count is about the volume of the simplex grown by half a cell, (t + c)^K /
        # (K! prod d_i) with c = sum d_i / 2: its log-log slope K t / (t + c) tends to K
        tm, c = math.sqrt(t0 * t1), sum(d) / 2
        assert slope == pytest.approx(k * tm / (tm + c), rel=0.08)
    # leading term for K = 3 at t = 40
    d = nlm.default_delays(3)
    n = len(nlm.enumerate_lattice(d, 40.0, max_points=nlm.MAX_POINTS)["arrival"])
    lead = 40.0**3 / (6 * np.prod(d))
    assert lead < n < 1.4 * lead


def test_star_directions_grid_first_then_spread_and_prefix_stable():
    u3 = nlm.star_directions(3)
    np.testing.assert_array_equal(u3, np.eye(3))  # three loops: the cubic grid
    u16 = nlm.star_directions(16)
    np.testing.assert_array_equal(u16[:3], u3)
    np.testing.assert_array_equal(nlm.star_directions(9), u16[:9])  # adding arms moves none
    np.testing.assert_allclose(np.linalg.norm(u16, axis=1), 1.0)
    cos = u16 @ u16.T - 2 * np.eye(16)
    assert np.degrees(np.arccos(cos.max())) > 25  # well spread: no two arms close
    c = np.ones(3) / math.sqrt(3)
    assert np.all(u16 @ c >= math.cos(nlm.CAP_RADIANS) - 1e-12)  # a fan, not a sphere
    sphere = nlm.star_directions(12, "sphere")
    assert sphere[:, 2].min() < -0.5 < 0.5 < sphere[:, 2].max()
    with pytest.raises(ValueError, match="fan"):
        nlm.star_directions(3, "cube")


def test_basis_echoes_project_to_their_arms_and_sums_add():
    d = (1.0, 2.0, 1.5, 1.2, 1.7)
    v = nlm.arm_vectors(nlm.star_directions(5), d, spacing=0.8)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 0.8 * np.array(d) / np.mean(d))
    flat = nlm.arm_vectors(nlm.star_directions(5), d, spacing=0.8, by_delay=False)
    np.testing.assert_allclose(np.linalg.norm(flat, axis=1), 0.8)
    eye = np.eye(5, dtype=np.int16)
    np.testing.assert_allclose(nlm.project(eye, v), v)
    np.testing.assert_allclose(nlm.project(eye, v, origin=(1, 2, 3)), v + np.array([1.0, 2.0, 3.0]))
    n = np.array([[2, 0, 1, 3, 0]], dtype=np.int16)
    np.testing.assert_allclose(nlm.project(n, v)[0], 2 * v[0] + v[2] + 3 * v[3])


def test_arms_fade_in_smoothly_and_their_echoes_slide_out():
    w = nlm.arm_weights(5, 3.25)
    np.testing.assert_allclose(w, [1, 1, 1, 0.25, 0])
    idx = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 2, 0], [0, 0, 0, 0, 1]], dtype=np.int16)
    np.testing.assert_allclose(nlm.point_weights(idx, w), [1.0, 0.25, 0.0])
    v = nlm.arm_vectors(nlm.star_directions(5), (1.0,) * 5)
    at0 = nlm.project(idx, v, nlm.arm_weights(5, 3.0))
    np.testing.assert_allclose(at0[1], at0[0])  # before arm 4 shows, its echoes sit on the others
    # continuous in ``arms``: a small step moves every echo a little
    a = nlm.project(idx, v, nlm.arm_weights(5, 3.5))
    b = nlm.project(idx, v, nlm.arm_weights(5, 3.5 + 1e-3))
    assert np.abs(a - b).max() < 1e-2
    np.testing.assert_allclose(nlm.project(idx, v, nlm.arm_weights(5, 5.0)), nlm.project(idx, v))


def test_flash_is_dark_before_bright_at_arrival_and_settles():
    tau = np.array([-1.0, 0.0, 0.05, 0.6, 50.0])
    light, burst = nlm.flash(tau, rise=0.05, decay=0.6, afterglow=0.3)
    assert light[0] == 0 and burst[0] == 0 and light[1] == 0  # rises from arrival
    assert light[2] == pytest.approx(1.0, abs=0.1) and burst[2] > 0.9
    assert light[4] == pytest.approx(0.3) and burst[4] == pytest.approx(0.0)
    fine = np.linspace(-0.1, 2, 4001)
    lf, _ = nlm.flash(fine)
    assert np.abs(np.diff(lf)).max() < 0.02  # continuous


def test_values_and_lattice_points_feed_a_generic_point_object():
    d = nlm.default_delays(4)
    out = nlm.lattice_points(d, 4.0, color_by="arm")
    m = len(out["arrival"])
    assert out["points"].shape == (m, 3) and out["values"].shape == (m,) and m > 10
    assert np.all((out["values"] >= 0) & (out["values"] < 1))
    np.testing.assert_allclose(out["weight"], 1.0)
    arr = nlm.values("arrival", out["index"], out["arrival"], 4.0)
    assert arr.max() <= 1.0 and arr[0] == 0.0
    gain = nlm.values("value", out["index"], out["arrival"], 4.0, arm_values=(-3, -1, -2, -6))
    np.testing.assert_allclose(gain, out["index"] @ np.array([-3, -1, -2, -6]))
    with pytest.raises(ValueError, match="arm values"):
        nlm.values("value", out["index"], out["arrival"], 4.0, arm_values=(1.0,))


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="positive"):
        nlm.enumerate_lattice((1.0, 0.0), 3.0)
    with pytest.raises(ValueError, match="delays"):
        nlm.enumerate_lattice((1.0,) * 33, 3.0)
    with pytest.raises(ValueError, match="loops"):
        nlm.default_delays(0)


# ---------------------------------------------------------------- the object


def _scene(**kw):
    s = Scene("nd", passes=None)
    s.use_camera("orbit", target=(1.5, 1.5, 1.5), distance=12)
    s.add(nd.StarLattice(id="echoes", **kw))
    return s


def test_star_lattice_semantics_count_the_heard_echoes():
    s = _scene(loops=3, delays=(1.0, 1.0, 1.0), max_time=4.0, time=expr("t"))
    e = s.engine
    sem = e.inspect(2.0, id="echoes")["objects"][0]["semantic"]
    assert sem["kind"] == "star_lattice" and sem["loops"] == 3 and sem["arms_shown"] == 3
    assert sem["points"] == math.comb(4 + 3, 3) and sem["complete_until"] == 4.0
    assert sem["heard"] == math.comb(2 + 3, 3)  # echoes with n1 + n2 + n3 <= 2
    assert not sem["truncated"]
    feats = e.state_at(2.0)["echoes"].impl.features(e.state_at(2.0))
    np.testing.assert_allclose(feats["tips"][:, :3].max(axis=0), [4.0, 4.0, 4.0])
    b = e.state_at(2.0)["echoes"].impl.world_bounds(e.state_at(2.0))
    assert b.hi[0] >= 4.0 and b.lo[0] <= 0.0


def test_delays_set_the_loop_count_and_mismatch_is_refused():
    s = _scene(delays=(1.0, 1.2, 1.4, 1.1, 1.3))
    sem = s.engine.inspect(0.0, id="echoes")["objects"][0]["semantic"]
    assert sem["loops"] == 5
    with pytest.raises(Exception, match="delays"):
        _scene(loops=3, delays=(1.0, 2.0))
    with pytest.raises(Exception, match="directions"):
        _scene(loops=2, directions=((1.0, 0.0, 0.0),))


def test_adding_arms_over_time_fills_the_frame_and_stays_pure():
    s = _scene(loops=16, max_time=6.0, arms=3.0)
    s.at(1.0).to("echoes.arms", 16.0, duration=6.0)
    e = s.engine
    early = e.inspect(0.5, id="echoes")["objects"][0]["semantic"]
    late = e.inspect(7.5, id="echoes")["objects"][0]["semantic"]
    assert early["arms_shown"] == 3 and late["arms_shown"] == 16
    assert late["shown"] > 50 * early["shown"]  # t^16 against t^3
    st = e.state_at(4.0)
    a = st["echoes"].impl.visible(st)
    b = e.state_at(4.0)["echoes"].impl.visible(e.state_at(4.0))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])  # the same t, the same echoes

    def step(dt):
        st2 = e.state_at(4.0 + dt)
        return np.abs(st2["echoes"].impl.visible(st2)["points"] - a["points"]).max()

    # continuous: a frame apart a small step (echoes many laps out move fastest), and a
    # quarter of the time a quarter of the step
    assert step(1 / 60) < 0.4 and step(1 / 240) < step(1 / 60) / 3


def test_draw_records_glowing_points_and_faint_arms():
    s = _scene(loops=4, max_time=3.0, time=2.0, style={"glow": "strong"})
    e = s.engine
    st = e.state_at(0.0)
    calls = []

    class Ctx:
        def triangles(self, positions, indices=None, *, color, glow=1.0, colors=None):
            calls.append((len(positions), glow, colors))

    st["echoes"].impl.draw(Ctx(), st, "color")
    assert len(calls) >= 2
    assert any(g > 1.0 for _, g, _ in calls)  # the points shine through the glow role
    sem = e.inspect(0.0, id="echoes")["objects"][0]["semantic"]
    assert 0 < sem["heard"] < sem["points"]


def test_star_lattice_is_registered_with_the_contract():
    from phasor.core.objects import CONTRACT_MEMBERS, registered_types

    cls = registered_types()["star_lattice"]
    assert all(hasattr(cls, m) for m in CONTRACT_MEMBERS) and cls.Style
    assert "hue" not in cls.Life  # colour means arrival time: life never turns it


@pytest.mark.gl
@needs_gl
@pytest.mark.slow
def test_star_lattice_golden_storyboard():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    e = runpy.run_path(str(ROOT / "examples" / "echo_lattice.py"), run_name="x")["s"].engine
    assert [i for i in e.lint() if i.severity == "error"] == []
    sheet = e.storyboard([1.5, 4.0, 8.0, 12.0], size=(320, 180))["image"]
    klass = e.pass_runner().renderer_class()
    path = GOLDEN / klass / "echo_lattice.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {klass} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97


@pytest.mark.gl
@needs_gl
@pytest.mark.slow
def test_star_lattice_vivid_keeps_the_first_look():
    """``flash_style="vivid"`` (preset ``vivid``) is the star's first look, kept as an option
    beside the shared flash: its golden is the storyboard from before the flash was shared
    (both are kept as options)."""
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    s = runpy.run_path(str(ROOT / "examples" / "echo_lattice.py"), run_name="x")["s"]
    s.set("echoes.flash_style", "vivid")
    e = s.engine
    sheet = e.storyboard([1.5, 4.0, 8.0, 12.0], size=(320, 180))["image"]
    klass = e.pass_runner().renderer_class()
    path = GOLDEN / klass / "echo_lattice_vivid.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {klass} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
