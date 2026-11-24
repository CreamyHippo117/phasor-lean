"""Fourier numerics (known maths) and objects."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor.domains import fourier_math as fm

# ---------------------------------------------------------------- epicycles


def test_a_circle_needs_one_epicycle_and_a_square_needs_many():
    s = np.linspace(0, 1, 64, endpoint=False)
    circle = 2.0 * np.exp(2j * math.pi * s) + (1 + 1j)  # radius 2 around (1, 1)
    c = fm.epicycle_coefficients(circle, 2)
    assert (c[0][0], abs(c[0][1])) == (1, pytest.approx(2.0)) and c[1][0] == 0
    assert c[1][1] == pytest.approx(1 + 1j)  # k = 0 is the centre
    sq = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    sq = np.concatenate(
        [np.linspace(a, b, 16, endpoint=False) for a, b in zip(sq, np.roll(sq, -1), strict=True)]
    )
    errs = [
        np.abs(fm.epicycle_curve(fm.epicycle_coefficients(sq, n), len(sq)) - sq).max()
        for n in (3, 9, 33)
    ]
    assert errs[0] > errs[1] > errs[2]  # more circles, closer to the curve
    full = fm.epicycle_coefficients(sq, len(sq))
    assert np.abs(fm.epicycle_curve(full, len(sq)) - sq).max() < 1e-9  # all terms: exact


def test_the_chain_ends_at_the_curve_and_circles_are_sorted_by_size():
    g = np.random.default_rng(3)
    pts = g.normal(size=40) + 1j * g.normal(size=40)
    co = fm.epicycle_coefficients(pts, 40)
    radii = [abs(c) for _, c in co]
    assert radii == sorted(radii, reverse=True)
    chain = fm.epicycle_chain(co, 0.0)
    assert chain[0] == 0 and chain[-1] == pytest.approx(pts[0])  # s = 0 is the first sample
    k = 7
    assert fm.epicycle_chain(co, k / 40)[-1] == pytest.approx(pts[k])
    with pytest.raises(ValueError, match="at least 4"):
        fm.epicycle_coefficients([1 + 0j, 2 + 0j], 2)


# ---------------------------------------------------------------- series and Gibbs


def test_partial_sums_converge_to_their_wave_away_from_the_jumps():
    x = np.linspace(0.3, math.pi - 0.3, 50)
    for kind in fm.SERIES:
        err = [
            np.abs(fm.series_partial_sum(kind, n, x) - fm.series_target(kind, x)).max()
            for n in (3, 12, 60)
        ]
        assert err[0] > err[1] > err[2], kind
        assert err[2] < 0.05, kind
    assert fm.series_partial_sum("square", 1, np.array([math.pi / 2]))[0] == pytest.approx(
        4 / math.pi
    )
    with pytest.raises(ValueError, match="unknown series"):
        fm.series_partial_sum("noise", 3, 0.0)


def test_gibbs_overshoot_stays_near_nine_percent_however_many_terms():
    over = [fm.gibbs_overshoot("square", n) for n in (5, 20, 80, 320)]
    assert all(0.085 < o < 0.1 for o in over)  # the famous ~8.95% of the jump
    assert over[-1] == pytest.approx(0.0895, abs=0.002)
    assert fm.gibbs_overshoot("triangle", 20) == 0.0  # no jump, no overshoot
    # the overshoot moves toward the jump but does not shrink: its position is ~ pi / (2 n)
    x = np.linspace(1e-4, 0.5, 5000)
    pos = [x[np.argmax(fm.series_partial_sum("square", n, x))] for n in (20, 80)]
    assert pos[1] < pos[0] and pos[0] == pytest.approx(math.pi / 40, rel=0.1)


# ---------------------------------------------------------------- windows and leakage


def test_windows_and_the_cost_of_a_wrong_guess_about_the_frequency():
    n, fs = 1024, 1024.0
    on_bin, off_bin = [100.0], [100.5]
    assert fm.leakage(on_bin, [1.0], "rect", n, fs) < 1e-9  # a tone on a bin: no leakage
    assert fm.leakage(off_bin, [1.0], "rect", n, fs) > 1e-3  # between bins it smears
    assert (
        fm.leakage(off_bin, [1.0], "hann", n, fs) < fm.leakage(off_bin, [1.0], "rect", n, fs) / 20
    )
    assert fm.leakage(off_bin, [1.0], "blackman", n, fs) < fm.leakage(off_bin, [1.0], "hann", n, fs)
    sp = fm.spectrum(on_bin, [0.7], "hann", n, fs)
    assert sp["mag"][100] == pytest.approx(0.7, rel=1e-6)  # amplitude-corrected
    assert fm.window("rect", 8).tolist() == [1.0] * 8
    with pytest.raises(ValueError, match="unknown window"):
        fm.window("triangle-ish", 8)


def test_spectrogram_follows_a_chirp():
    fs = 4000.0
    x = fm.chirp(200.0, 1200.0, 1.0, fs)
    sg = fm.spectrogram(x, fs, nwin=256)
    peak = sg["freq"][np.argmax(sg["mag"], axis=0)]
    assert peak[2] < peak[len(peak) // 2] < peak[-3]  # the pitch rises
    assert peak[len(peak) // 2] == pytest.approx(700.0, abs=60.0)  # halfway: 700 Hz


def test_the_convolution_theorem():
    g = np.random.default_rng(0)
    a, b = g.normal(size=50), g.normal(size=30)
    c = fm.convolve(a, b)
    assert np.allclose(c, np.convolve(a, b), atol=1e-10)
    n = 128
    assert np.allclose(np.fft.rfft(c, n), np.fft.rfft(a, n) * np.fft.rfft(b, n), atol=1e-8)
