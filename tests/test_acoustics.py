"""Room acoustics, the physics of the reverb scene.

The reverb scene (a wavefront, image sources, rays, an impulse response, RT60, a
convolved clip) is checked here as numbers rather than as a film: the image-source method
against geometry and Sabine/Eyring theory, the FDTD field against the speed of sound and its
stability limit, convolution reverb against the convolution theorem.
"""

from __future__ import annotations

import numpy as np
import pytest

from phasor.core.simulation import SimRunner
from phasor.domains import ac_math as ac

SIZE = (8.0, 5.0, 3.0)
SRC = (2.0, 2.5, 1.5)
MIC = (6.0, 2.0, 1.5)


# ---------------------------------------------------------------- image sources


def test_order_zero_is_the_source_and_order_one_adds_the_six_walls():
    im0 = ac.image_sources(SIZE, SRC, 0, 0.2)
    assert len(im0["pos"]) == 1 and np.allclose(im0["pos"][0], SRC) and im0["gain"][0] == 1.0
    im1 = ac.image_sources(SIZE, SRC, 1, 0.2)
    assert len(im1["pos"]) == 7  # the source and one ghost behind each of the six walls
    ghosts = {tuple(np.round(p, 6)) for p in im1["pos"][1:]}
    assert ghosts == {
        (-2.0, 2.5, 1.5),
        (14.0, 2.5, 1.5),
        (2.0, -2.5, 1.5),
        (2.0, 7.5, 1.5),
        (2.0, 2.5, -1.5),
        (2.0, 2.5, 4.5),
    }  # mirrored across each wall
    assert np.allclose(im1["gain"][1:], np.sqrt(1 - 0.2))


def test_image_count_grows_and_gain_falls_with_reflections():
    counts = [len(ac.image_sources(SIZE, SRC, k, 0.3)["pos"]) for k in range(5)]
    assert counts == sorted(counts) and counts[2] == 25 and counts[3] == 63 and counts[4] > 100
    im = ac.image_sources(SIZE, SRC, 4, 0.3)
    assert np.allclose(im["gain"], np.sqrt(0.7) ** im["reflections"])
    with pytest.raises(ValueError, match="limit"):
        ac.image_sources(SIZE, SRC, 150, 0.3)


def test_direct_sound_arrives_at_distance_over_the_speed_of_sound():
    arr = ac.arrivals(SIZE, SRC, MIC, 3, 0.2)
    d = float(np.linalg.norm(np.subtract(SRC, MIC)))
    assert arr["time"][0] == pytest.approx(d / ac.C) and arr["reflections"][0] == 0
    assert arr["amplitude"][0] == pytest.approx(1 / d)  # 1/r spreading
    assert np.all(np.diff(arr["time"]) >= 0)  # sorted by arrival
    # first reflections arrive after the direct sound, each from a longer path
    assert (arr["distance"][1:7] > arr["distance"][0]).all()
    # the floor reflection has a path of sqrt(d_h^2 + (2 z)^2)  (mirror the source in the floor)
    floor = np.sqrt(4.0**2 + 0.5**2 + (1.5 + 1.5) ** 2)
    assert any(np.isclose(arr["distance"], floor, atol=1e-9))


def test_reflections_thicken_into_a_dense_tail():
    arr = ac.arrivals(SIZE, SRC, MIC, 12, 0.3)
    t = arr["time"]
    dens = [np.sum((t >= a) & (t < a + 0.01)) for a in (0.02, 0.05, 0.08)]
    assert dens[0] < dens[1] < dens[2]  # echo density grows with time (the tail)


# ---------------------------------------------------------------- RT60


@pytest.mark.parametrize(
    ("alpha", "order", "t1", "t2"), [(0.3, 60, 0.15, 0.35), (0.5, 32, 0.08, 0.2)]
)
def test_rt60_from_the_impulse_response_sits_where_room_theory_says(alpha, order, t1, t2):
    """A shoebox is not a diffuse field: energy travelling along an axis meets walls less
    often, so the late tail rings *longer* than Eyring's (diffuse) estimate. The honest check
    is therefore a bracket from theory, and the integral against a direct sum."""
    fs = 4000  # order is high enough that every image inside order * 3 m / c is included
    arr = ac.arrivals(SIZE, SRC, MIC, order, alpha)
    eyring, sabine = ac.eyring_rt60(SIZE, alpha), ac.sabine_rt60(SIZE, alpha)
    ir = ac.impulse_response(arr, fs, length=1.2)
    fit = ac.rt60_fit(ir, fs)
    assert 0.9 * eyring < fit["rt60"] < 2.0 * sabine
    assert sabine > eyring  # Sabine over-estimates when the walls absorb a lot
    assert fit["slope"] < 0 and fit["curve"][0] == pytest.approx(0.0, abs=1e-9)
    # the Schroeder curve is the energy still to come: check it against the arrivals directly
    inside = arr["time"] < 1.2 - 1.0 / fs  # what the sampled IR actually holds
    e, t_arr = arr["amplitude"][inside] ** 2, arr["time"][inside]
    left = {t: e[t_arr >= t].sum() for t in (t1, t2)}
    drop_direct = 10 * np.log10(left[t2] / left[t1])  # dB lost between t1 and t2
    drop_curve = fit["curve"][int(t2 * fs)] - fit["curve"][int(t1 * fs)]
    # Splitting each arrival over two samples spreads its energy a little (about a dB here),
    # so the dense tail agrees with the direct sum to within that.
    assert drop_curve == pytest.approx(drop_direct, abs=1.5)


def test_small_absorption_makes_sabine_and_eyring_agree():
    assert ac.eyring_rt60(SIZE, 0.01) == pytest.approx(ac.sabine_rt60(SIZE, 0.01), rel=0.01)
    assert ac.sabine_rt60(SIZE, 0.0) == float("inf")


def test_more_absorption_means_a_shorter_tail():
    fs = 4000
    t = []
    for alpha in (0.25, 0.5):
        arr = ac.arrivals(SIZE, SRC, MIC, 18, alpha)
        t.append(ac.rt60_fit(ac.impulse_response(arr, fs, 0.9), fs)["rt60"])
    assert t[1] < t[0]


def test_impulse_response_places_energy_between_samples():
    arr = {"time": np.array([0.00125]), "amplitude": np.array([1.0])}
    ir = ac.impulse_response(arr, 1000, 0.01)  # 1.25 samples: 0.75 on sample 1, 0.25 on sample 2
    assert (
        ir[1] == pytest.approx(0.75)
        and ir[2] == pytest.approx(0.25)
        and ir.sum() == pytest.approx(1)
    )


def test_materials_and_absorption():
    assert ac.absorption_of("plaster") == 0.04 and ac.absorption_of("plaster", 0.6) == 0.6
    assert ac.absorption_of("anything", 1.7) == 1.0  # clamped
    with pytest.raises(ValueError, match="unknown material"):
        ac.absorption_of("marshmallow")
    assert ac.room_stats((2, 3, 4)) == (24, 52)


# ---------------------------------------------------------------- convolution reverb


def test_convolution_reverb_follows_the_convolution_theorem_and_adds_a_tail():
    fs = 4000
    arr = ac.arrivals(SIZE, SRC, MIC, 14, 0.4)
    ir = ac.impulse_response(arr, fs, 0.5)
    g = np.random.default_rng(1)
    dry = g.normal(size=fs // 4)  # a quarter second of noise
    wet = ac.convolve(dry, ir)
    assert len(wet) == len(dry) + len(ir) - 1
    assert np.allclose(wet, np.convolve(dry, ir), atol=1e-9)  # same as the direct sum
    spec = np.fft.rfft(wet, 8192)
    assert np.allclose(spec, np.fft.rfft(dry, 8192) * np.fft.rfft(ir, 8192), atol=1e-6)
    tail = wet[len(dry) :]
    assert np.abs(tail).max() > 0.01 * np.abs(wet).max()  # the sound rings on after the clip ends


# ---------------------------------------------------------------- FDTD


def test_fdtd_refuses_an_unstable_step():
    ok = ac.RoomFDTD((4, 3, 2), (1, 1, 1), dx=0.25)
    assert ok.dt <= 0.25 / (ac.C * np.sqrt(3))
    with pytest.raises(ValueError, match="unstable"):
        ac.RoomFDTD((4, 3, 2), (1, 1, 1), dx=0.25, dt=0.25 / ac.C)  # fine in 1D, not in 3D
    edge = 0.25 / (ac.C * np.sqrt(3))
    assert ac.RoomFDTD((4, 3, 2), (1, 1, 1), dx=0.25, dt=edge).dt == edge  # exactly at the limit


def test_the_pressure_wave_travels_at_the_speed_of_sound(tmp_path):
    sim = ac.RoomFDTD((6, 4, 3), (1.0, 2.0, 1.5), dx=0.2, alpha=0.1, frequency=250.0)
    r = SimRunner(sim, every=50, cache_dir=tmp_path)
    near, far = (2.0, 2.0, 1.5), (5.0, 2.0, 1.5)  # 1 m and 4 m from the source
    t_end = 0.026
    times = np.arange(0.0, t_end, sim.dt * 2)
    sig = {
        name: [sim.at(r.state_at(t), pt) for t in times] for name, pt in (("n", near), ("f", far))
    }
    peak = lambda s: times[int(np.argmax(np.abs(s)))]  # noqa: E731
    delay = peak(sig["f"]) - peak(sig["n"])
    assert delay == pytest.approx(3.0 / ac.C, abs=0.0012)  # 3 m further: 8.7 ms later
    assert np.abs(sig["f"]).max() < np.abs(sig["n"]).max()  # and quieter (spreading)


def test_the_field_stays_bounded_and_is_a_pure_function_of_time(tmp_path):
    sim = ac.RoomFDTD((4, 3, 2), (1, 1, 1), dx=0.25)
    a = SimRunner(sim, every=20, cache_dir=tmp_path / "a")
    b = SimRunner(sim, every=20, cache_dir=tmp_path / "b")
    t = 400 * sim.dt
    pa = a.state_at(t)["p"]
    a.state_at(30 * sim.dt)  # scrub back, then forward again
    assert np.array_equal(pa, b.state_at(t)["p"]) and np.array_equal(pa, a.state_at(t)["p"])
    assert (
        np.isfinite(pa).all() and np.abs(pa).max() < 10 * np.abs(a.state_at(60 * sim.dt)["p"]).max()
    )
    late = np.abs(a.state_at(1500 * sim.dt)["p"]).max()  # decays through the damped walls
    assert late < np.abs(pa).max()
