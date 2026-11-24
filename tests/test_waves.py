"""Interference and beamforming: the wave field against known results, the objects'
colours and semantics, the GPU kernel's determinism and speed, and the golden storyboard."""

from __future__ import annotations

import math
import runpy
import time
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, ac, expr
from phasor.domains import waves_math as wm
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
LAM = 1.0
K = 2 * math.pi / LAM


def _two(phase: float = 0.0, half: float = 1.0) -> np.ndarray:
    return wm.source_table([(-half, 0.0), (half, 0.0)], [1.0, 1.0], [0.0, phase],
                           [wm.ALWAYS, wm.ALWAYS])  # fmt: skip


def _intensity(x, y, table, k=K) -> np.ndarray:
    return np.abs(wm.amplitude(x, y, table, k)) ** 2


def _on_hyperbola(diff: float, y: float, half: float = 1.0) -> float:
    """The x > 0 where |x - s1| - |x - s2| = ``diff`` at height ``y`` (bisection)."""
    lo, hi = 0.0, 1e3
    for _ in range(200):
        x = (lo + hi) / 2
        d = math.hypot(x + half, y) - math.hypot(x - half, y)
        lo, hi = (x, hi) if d < diff else (lo, x)
    return (lo + hi) / 2


def test_a_single_source_spreads_as_one_over_root_r_and_runs_at_the_wavenumber():
    table = wm.source_table([(0.0, 0.0)], [2.0], [0.3], [wm.ALWAYS])
    r = np.array([0.5, 1.0, 4.0, 9.0])
    a = wm.amplitude(r, np.zeros_like(r), table, K)
    assert np.abs(a) == pytest.approx(2.0 / np.sqrt(r))
    assert np.angle(a) == pytest.approx(np.angle(np.exp(1j * (K * r + 0.3))))


def test_two_sources_in_phase_peak_on_the_perpendicular_bisector():
    table = _two()
    for y in (3.0, 6.0, 12.0):
        xs = np.linspace(-3.0, 3.0, 6001)
        i = _intensity(xs, np.full_like(xs, y), table)
        # the brightest fringe at this height is the one on the bisector x = 0
        assert abs(xs[int(np.argmax(i))]) < 0.01
        assert i[3000] == pytest.approx(4.0 / math.hypot(1.0, y), rel=1e-9)  # full sum


def test_the_nodal_lines_are_where_the_path_difference_is_half_a_wavelength():
    table = _two()
    for y in (4.0, 8.0, 16.0):
        bisector = _intensity(0.0, y, table)
        for m in (0, 1):  # half a wavelength, then one and a half
            x = _on_hyperbola((m + 0.5) * LAM, y)
            nodal = _intensity(x, y, table)
            # 1/sqrt(r) spreading leaves a little: |1/sqrt(r1) - 1/sqrt(r2)|^2, not zero
            r1, r2 = math.hypot(x + 1, y), math.hypot(x - 1, y)
            assert nodal == pytest.approx((r1**-0.5 - r2**-0.5) ** 2, abs=1e-12)
            assert nodal / bisector < 0.01
        bright = _intensity(_on_hyperbola(LAM, y), y, table)
        assert bright / bisector > 0.5  # a whole wavelength: bright again


def test_sources_in_antiphase_swap_the_fringes():
    table = _two(phase=math.pi)
    assert _intensity(0.0, 8.0, table) < 1e-20  # the bisector is now a nodal line
    x = _on_hyperbola(0.5 * LAM, 8.0)
    assert _intensity(x, 8.0, table) > 0.2


def test_a_source_switching_on_sends_out_a_ring():
    table = wm.source_table([(0.0, 0.0)], [1.0], [0.0], [2.0])  # its front is 2 away
    r = np.array([1.0, 1.9, 2.1, 3.0])
    a = np.abs(wm.amplitude(r, np.zeros_like(r), table, K, edge=0.0))
    assert a[0] > 0 and a[1] > 0 and a[2] == 0 and a[3] == 0


@pytest.mark.parametrize("steer", [-50.0, -20.0, 0.0, 15.0, 40.0])
def test_steering_puts_the_beam_peak_at_theta_within_a_beamwidth(steer):
    s = Scene("beam", passes=None)
    s.add(ac.PhasedArray(id="a", center=(0.0, 0.0), count=8, spacing=0.5, steer=steer))
    st = s.engine.state_at(0.0)
    impl, p = st["a"].impl, st["a"].params
    table = impl._table(p)
    sem = s.engine.inspect(0.0, id="a")["objects"][0]["semantic"]
    # the far field on an arc in front of the array (it is 3.5 wavelengths long)
    radius = 400.0
    ang = np.radians(np.linspace(-89.0, 89.0, 35601))
    a = wm.amplitude(radius * np.cos(ang), radius * np.sin(ang), table, K)
    peak = math.degrees(ang[int(np.argmax(np.abs(a)))])
    assert sem["beam_direction_deg"] == pytest.approx(steer % 360.0)
    assert abs(peak - steer) < sem["beamwidth_deg"] / 2
    assert abs(peak - steer) < 0.5  # in fact on it (the far field of a line array)


def test_phased_array_semantics_and_the_ring_layout():
    s = Scene("arr", passes=None)
    s.add(ac.PhasedArray(id="line", count=4, spacing=0.5, steer=30.0, center=(-3.0, 0.0)))
    s.add(ac.PhasedArray(id="ring", count=6, layout="ring", center=(2.0, 1.0)))
    line = s.engine.inspect(0.0, id="line")["objects"][0]["semantic"]
    assert line["count"] == 4 and len(line["sources"]) == 4
    ys = sorted(y for _, y in line["sources"])
    assert np.diff(ys) == pytest.approx([0.5] * 3)  # half a wavelength apart, along y
    # neighbouring elements differ in phase by k d sin(steer)
    step = np.diff(np.radians(line["phases_deg"]))
    assert np.allclose(np.angle(np.exp(1j * (step - K * 0.5 * math.sin(math.radians(30))))), 0)
    ring = s.engine.inspect(0.0, id="ring")["objects"][0]["semantic"]
    pts = np.asarray(ring["sources"])
    assert np.hypot(*(pts - (2.0, 1.0)).T) == pytest.approx([6 * 0.5 / (2 * math.pi)] * 6)
    assert "beamwidth_deg" not in ring


def test_bad_parameters_are_refused():
    for bad in (ac.Interference(id="a", sources=((0, 0), (1, 0)), phases=(0.0,)),
                ac.PhasedArray(id="b", count=3, amplitudes=(1.0, 1.0)),
                ac.Interference(id="c", size=(0.0, 1.0))):  # fmt: skip
        with pytest.raises(ValueError):
            Scene("bad", passes=None).add(bad)


def test_interference_reports_its_wave_and_the_sources_that_have_started():
    s = Scene("i", passes=None)
    s.add(ac.Interference(id="i", sources=((-1, 0), (1, 0)), speed=2.0, frequency=4.0,
                          onsets=(0.0, 1.0), time=expr("t")))  # fmt: skip
    s.marker("end", 2.0)
    at = s.engine.inspect(0.5, id="i")["objects"][0]["semantic"]
    assert at["wavelength"] == 0.5 and at["sources_on"] == 1 and at["backdrop"]
    assert s.engine.inspect(1.5, id="i")["objects"][0]["semantic"]["sources_on"] == 2


# ---------------------------------------------------------------- the pictures


def _pixels(obj, t=0.0, n=300):
    from tests.test_beauty import _field_pixels

    return _field_pixels(obj, t=t, size=(n, n))


def _at(px_shape, size, x, y):
    """The pixel (row, col) of world (x, y) for a square field of ``size`` around 0."""
    n = px_shape[0]
    return int((size / 2 - y) / size * n), int((x + size / 2) / size * n)


def test_the_cancelling_lines_stay_dark_and_crests_glow_warm_troughs_cool():
    size = 12.0
    obj = ac.Interference(id="w", sources=((-1, 0), (1, 0)), size=(size, size), time=0.0)
    for t in (0.0, 0.13, 0.41, 0.77):
        obj = ac.Interference(id="w", sources=((-1, 0), (1, 0)), size=(size, size), time=t)
        px, bg, doc = _pixels(obj)
        assert doc.params["isolines"] % 2 == 1  # no contour at zero: it would light the nodes
        for y in (3.5, 4.5):
            r, c = _at(px.shape, size, _on_hyperbola(0.5, y), y)
            assert px[r, c, 3] < 0.05  # no glow on the nodal line
            assert np.abs(px[r, c, :3] - bg).max() < 0.05
    # along the bisector the field swings: crests warm (red > blue), troughs cool
    px, bg, _ = _pixels(ac.Interference(id="w", sources=((-1, 0), (1, 0)), size=(size, size)))
    col = px[:, px.shape[1] // 2, :]
    ys = size / 2 - (np.arange(px.shape[0]) + 0.5) / px.shape[0] * size
    field = wm.amplitude(np.zeros_like(ys), ys, _two(), K).real  # time 0: Re A
    lit = col[:, 3] > 0.3
    crest, trough = lit & (field > 0), lit & (field < 0)
    assert crest.sum() > 5 and trough.sum() > 5
    warm = col[crest, 0] - col[crest, 2]
    cool = col[trough, 2] - col[trough, 0]
    assert np.median(warm) > 0.05 and np.median(cool) > 0.0


def test_the_intensity_picture_stands_still_and_the_field_does_not():
    size = 10.0

    def pic(t, mode):
        obj = ac.Interference(id="w", size=(size, size), time=t, mode=mode)
        return _pixels(obj, n=120)[0]

    assert np.allclose(pic(0.1, "intensity"), pic(0.6, "intensity"), atol=1e-5)
    assert not np.allclose(pic(0.1, "field"), pic(0.6, "field"), atol=1e-2)


def test_the_phased_array_draws_beacons_and_a_ray_along_the_beam():
    from phasor.render.passes import _Collector

    s = Scene("a", look="prism", passes=None)
    s.add(ac.PhasedArray(id="a", center=(-3.0, 0.0), count=4, steer=25.0))
    st = s.engine.state_at(0.0)
    c = _Collector()
    st["a"].impl.draw(c, st, "color")
    kinds = [k for k, _ in c.commands]
    assert "field" in kinds and len(kinds) >= 3  # the field, the beacons, the ray
    end = st["a"].impl._ray_end(st["a"].params)
    d = end - np.array([-3.0, 0.0])
    assert math.degrees(math.atan2(d[1], d[0])) == pytest.approx(25.0)


def test_the_wave_types_are_in_the_prism_looks():
    s = Scene("p", look="prism", passes=None)
    s.add(ac.Interference(id="i"))
    s.add(ac.PhasedArray(id="a"))
    for oid in ("i", "a"):
        o = s.engine.document.object(oid)
        assert o.params["hue_cycles"] > 0 and o.style["glow"] == "soft"


# ---------------------------------------------------------------- the GPU


@pytest.mark.gpu
def test_the_gpu_field_is_deterministic_and_fast_at_1080p():
    from phasor.render import gpu

    src = np.array([[-3.0, 0.5 * k - 1.75, 1.0, 0.7 * k, wm.ALWAYS if k else 4.0]
                    for k in range(8)])  # fmt: skip
    kw = {"sources": src, "center": (0.0, 0.0), "height": 4.5, "wavenumber": 9.0,
          "omega_t": 2.0, "edge": 0.3, "backend": "cuda"}  # fmt: skip
    first = gpu.render("wave_field", 1920, 1080, **kw)
    again = gpu.render("wave_field", 1920, 1080, **kw)
    assert np.array_equal(first, again)  # the same bits every run
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        gpu.render("wave_field", 1920, 1080, **kw)
        times.append(time.perf_counter() - t0)
    assert sorted(times)[2] < 0.05  # including the copy back to the host


@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.slow
def test_interference_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "interference.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([1.3, 4.7, 9.2, 13.9, 17.6, 25.3], size=(320, 180))["image"]
    _golden("interference", sheet, e.pass_runner().renderer_class())
