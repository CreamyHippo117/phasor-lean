"""Filtering: the design against known maths and scipy, the response, the
steady state of tones, the transfer function as text; the objects through the engine."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from phasor import Scene, cx, filt
from phasor.domains import filters_math as fm
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
FS = 8000.0


def _sorted(r):
    return sorted(np.round(r, 6), key=lambda c: (c.real, c.imag))


@pytest.mark.parametrize(
    ("kind", "order", "wn"),
    [
        ("lowpass", 4, 1000.0),
        ("highpass", 3, 1500.0),
        ("bandpass", 2, (800.0, 1600.0)),
        ("bandstop", 2, (900.0, 1400.0)),
    ],
)
def test_the_design_matches_scipy_butterworth(kind, order, wn):
    lo, hi = (wn, None) if np.isscalar(wn) else wn
    mine = fm.butterworth(kind, order, lo, FS, hi)
    z, p, k = signal.butter(order, wn, btype=kind, output="zpk", fs=FS)
    assert _sorted(mine.zeros) == pytest.approx(_sorted(z), abs=1e-6)
    assert _sorted(mine.poles) == pytest.approx(_sorted(p), abs=1e-6)
    assert mine.gain == pytest.approx(k, rel=1e-9)
    f = np.linspace(0, FS / 2, 200, endpoint=False)
    _, h = signal.freqz_zpk(z, p, k, worN=f, fs=FS)
    assert mine.response(f) == pytest.approx(h, abs=1e-9)


@pytest.mark.parametrize("order", [1, 2, 5])
def test_lowpass_magnitude_is_the_butterworth_formula(order):
    """|H|^2 = 1 / (1 + (tan(w/2) / tan(wc/2))^2n), exactly, for the bilinear design."""
    fc = 1200.0
    f = fm.butterworth("lowpass", order, fc, FS)
    fr = np.linspace(10, FS / 2 - 10, 300)
    ratio = np.tan(math.pi * fr / FS) / math.tan(math.pi * fc / FS)
    assert np.abs(f.response(fr)) ** 2 == pytest.approx(1 / (1 + ratio ** (2 * order)), rel=1e-9)
    assert abs(f.response(0.0)) == pytest.approx(1.0)  # unit gain at DC
    assert abs(f.response(fc)) == pytest.approx(1 / math.sqrt(2))  # half power at the cutoff
    assert abs(f.response(FS / 2)) == pytest.approx(0.0, abs=1e-12)  # zeros at z = -1


@pytest.mark.parametrize("order", [1, 2, 3, 6])
def test_lowpass_phase_at_the_cutoff_is_minus_n_quarter_pi(order):
    f = fm.butterworth("lowpass", order, 900.0, FS)
    assert np.angle(f.response(900.0)) == pytest.approx(
        (-order * math.pi / 4 + math.pi) % (2 * math.pi) - math.pi, abs=1e-9
    )


def test_highpass_blocks_dc_and_band_kinds_have_their_edges_and_notch():
    hp = fm.butterworth("highpass", 4, 2000.0, FS)
    assert abs(hp.response(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert abs(hp.response(FS / 2)) == pytest.approx(1.0)
    assert abs(hp.response(2000.0)) == pytest.approx(1 / math.sqrt(2))
    bp = fm.butterworth("bandpass", 3, 700.0, FS, 1500.0)
    assert abs(bp.response(700.0)) == pytest.approx(1 / math.sqrt(2))
    assert abs(bp.response(1500.0)) == pytest.approx(1 / math.sqrt(2))
    assert abs(bp.response(0.0)) == pytest.approx(0.0, abs=1e-12)
    bs = fm.butterworth("bandstop", 3, 700.0, FS, 1500.0)
    assert abs(bs.response(700.0)) == pytest.approx(1 / math.sqrt(2))
    assert abs(bs.response(1500.0)) == pytest.approx(1 / math.sqrt(2))
    assert abs(bs.response(0.0)) == pytest.approx(1.0)
    # the stop band is centred (in the pre-warped sense) where the response is exactly zero
    t1, t2 = (math.tan(math.pi * v / FS) for v in (700.0, 1500.0))
    centre = FS / math.pi * math.atan(math.sqrt(t1 * t2))
    assert abs(bs.response(centre)) == pytest.approx(0.0, abs=1e-9)


def test_notch_zeros_on_the_circle_poles_inside_and_unit_dc_gain():
    f = fm.notch(1000.0, 0.95, FS)
    w = 2 * math.pi * 1000.0 / FS
    assert sorted(np.angle(f.zeros)) == pytest.approx([-w, w])
    assert np.abs(f.zeros) == pytest.approx([1.0, 1.0])
    assert np.abs(f.poles) == pytest.approx([0.95, 0.95])
    assert abs(f.response(1000.0)) == pytest.approx(0.0, abs=1e-12)
    assert abs(f.response(0.0)) == pytest.approx(1.0)
    assert f.stable
    narrower = fm.notch(1000.0, 0.99, FS)
    assert abs(narrower.response(1100.0)) > abs(f.response(1100.0))  # closer to 1: narrower


def test_stability_is_the_poles_inside_the_unit_circle():
    assert fm.butterworth("lowpass", 8, 300.0, FS).stable
    assert not fm.custom([], [(1.1, 0.0)], 1.0, FS).stable


def test_bad_designs_are_refused():
    with pytest.raises(fm.FilterError, match="between 0 and 4000"):
        fm.butterworth("lowpass", 2, 5000.0, FS)
    with pytest.raises(fm.FilterError, match="cutoff2"):
        fm.butterworth("bandpass", 2, 1000.0, FS, 500.0)
    with pytest.raises(fm.FilterError, match="conjugate"):
        fm.custom([], [(0.5, 0.5)], 1.0, FS)
    with pytest.raises(fm.FilterError, match="at least as many poles"):
        fm.custom([(0.0, 0.0)], [], 1.0, FS)


def test_filtering_matches_scipy_and_the_difference_equation():
    f = fm.butterworth("lowpass", 3, 1000.0, FS)
    x = np.random.default_rng(1).standard_normal(400)
    b, a = signal.butter(3, 1000.0, fs=FS)
    assert f.apply(x) == pytest.approx(signal.lfilter(b, a, x), abs=1e-10)
    first = fm.custom([], [(0.5, 0.0)], 1.0, FS).apply([1.0, 0.0, 0.0, 0.0])
    assert first == pytest.approx([1.0, 0.5, 0.25, 0.125])  # y[n] = x[n] + 0.5 y[n-1]


def test_steady_state_tones_come_out_scaled_and_shifted_by_the_response():
    f = fm.butterworth("lowpass", 4, 1000.0, FS)
    for fr in (500.0, 1000.0, 2500.0):
        n = 4000
        t = np.arange(n) / FS
        y = f.apply(np.sin(2 * math.pi * fr * t))
        tail = slice(n // 2, None)  # after the transient
        h = f.response(fr)
        assert y[tail] == pytest.approx(abs(h) * np.sin(2 * math.pi * fr * t[tail] + np.angle(h)),
                                        abs=1e-6)  # fmt: skip
    x, y = fm.filtered_tones(f, (200.0, 3000.0), (1.0, 1.0), 4096)
    late = slice(2048, None)  # the 3 kHz tone is gone: what is left is the 200 Hz tone alone
    assert np.std(y[late]) == pytest.approx(abs(f.response(200.0)) / math.sqrt(2), rel=5e-3)
    assert np.std(x[late]) == pytest.approx(1.0, rel=1e-2)  # two unit tones: rms 1


def test_the_cut_band_loses_its_spectrum():
    f = fm.butterworth("lowpass", 6, 1000.0, FS)
    freqs = np.arange(250.0, 3800.0, 250.0)
    amps = np.ones_like(freqs)
    after = fm.tone_amplitudes_after(f, freqs, amps)
    assert after[freqs <= 500].min() > 0.99
    assert after[freqs >= 2500].max() < 0.01
    # the same from filtering a signal and reading its spectrum
    _, y = fm.filtered_tones(f, freqs, amps, 8192)
    spec = np.abs(np.fft.rfft(y[4096:] * np.hanning(4096))) / (np.hanning(4096).sum() / 2)
    k = np.round(freqs / (FS / 4096)).astype(int)
    assert spec[k] == pytest.approx(after, abs=0.03)


def test_curve_runs_break_at_jumps():
    x = np.arange(6.0)
    runs = fm.curve_runs(x, np.array([0.0, 0.1, 0.2, 3.3, 3.4, np.nan]), 2.0)
    assert [len(a) for a, _ in runs] == [3, 2]
    assert fm.minus_3db(fm.butterworth("lowpass", 4, 1000.0, FS)) == pytest.approx(
        [1000.0], abs=1.5
    )


# ---------------------------------------------------------------- the transfer function as text


@pytest.mark.parametrize("pos", [(0.0, 0.0), (-4.3, 1.3)])
def test_the_plane_function_is_H_at_the_shifted_point(pos):
    from phasor.render.gpu.codegen import ComplexFunction

    f = fm.butterworth("bandpass", 2, 800.0, FS, 1600.0)
    src = fm.transfer_source(f, pos, 2.0)
    g = ComplexFunction.parse(src).numpy()
    w = np.array([-3 + 1j, 0.3 - 0.2j, 1.1 + 2.5j])
    assert g(w) == pytest.approx(f.at((w - complex(*pos)) / 2.0), rel=1e-9)


def test_the_domain_coloring_finds_the_filters_zeros_and_poles_where_they_are():
    f = fm.butterworth("lowpass", 2, 1000.0, FS)
    plane = filt.transfer_plane("plane", kind="lowpass", order=2, cutoff=1000.0, fs=FS,
                                pos=(-1.0, 0.5, 0.0), scale=1.5)  # fmt: skip
    s = Scene("plane", passes=None)
    s.add(plane)
    sem = s.engine.inspect(0.0, id="plane")["objects"][0]["semantic"]
    back = lambda pts: np.array([(complex(x, y) - complex(-1.0, 0.5)) / 1.5 for x, y in pts])  # noqa: E731
    assert _sorted(back(sem["poles"])) == pytest.approx(_sorted(f.poles), abs=1e-4)
    assert back(sem["zeros"]) == pytest.approx(f.zeros, abs=2e-3)  # a double zero at -1


def test_a_repeated_zero_does_not_defeat_the_coloring_features():
    """A Butterworth low-pass has all its zeros at z = -1: the root finder must still answer."""
    s = Scene("repeat", passes=None)
    s.add(filt.transfer_plane("plane", kind="lowpass", order=6, cutoff=1000.0, fs=FS))
    sem = s.engine.inspect(0.0, id="plane")["objects"][0]["semantic"]
    assert len(sem["poles"]) == 6 and len(sem["zeros"]) == 6  # scattered a little: it is 6-fold
    assert np.abs(np.array(sem["zeros"]) / 2.0 - [-1.0, 0.0]).max() < 0.05


# ---------------------------------------------------------------- the objects


def _scene(**kw):
    s = Scene("filt_test", passes=None)
    spec = {"kind": "lowpass", "order": 4, "cutoff": 1000.0, "fs": FS, **kw}
    s.add(filt.PoleZero(id="pz", **spec))
    s.add(filt.FrequencyResponse(id="gain", plot="gain", **spec))
    s.add(filt.FrequencyResponse(id="phase", plot="phase", **spec))
    s.add(
        filt.FilteredSignal(id="wave", frequencies=(300.0, 2800.0), amplitudes=(1.0, 1.0), **spec)
    )
    s.add(filt.FilteredSpectrum(id="spec", frequencies=(300.0, 2800.0), amplitudes=(1.0, 1.0),
                                **spec))  # fmt: skip
    return s


def test_objects_report_what_they_find():
    objs = {o["id"]: o["semantic"] for o in _scene().engine.inspect(0.5)["objects"]}
    pz = objs["pz"]
    assert pz["kind"] == "pole_zero" and pz["stable"] is True
    assert len(pz["poles"]) == 4 and pz["zeros"] == [[-1.0, 0.0, 4]]
    g = objs["gain"]
    assert g["cutoffs_hz"] == [1000.0] and g["gain_db_at_cutoff"][0] == pytest.approx(
        -3.0103, abs=1e-3
    )
    assert objs["phase"]["phase_at_cutoff"][0] == pytest.approx(-math.pi, abs=1e-3)  # -n pi / 4
    assert objs["wave"]["kind"] == "filtered_signal"
    sp = objs["spec"]
    assert sp["kind"] == "filtered_spectrum" and 0.4 < sp["removed_energy_share"] < 0.6
    assert sp["tones"][0]["kept"] > 0.99 and sp["tones"][1]["kept"] < 0.01


def test_bad_filters_are_rejected_when_added():
    s = Scene("filt_bad", passes=None)
    with pytest.raises(Exception, match="between 0 and 4000"):
        s.add(filt.PoleZero(id="a", cutoff=9000.0))
    with pytest.raises(Exception, match="conjugate"):
        s.add(filt.PoleZero(id="b", kind="custom", poles=((0.5, 0.5),)))
    with pytest.raises(Exception, match="one value per frequency"):
        s.add(filt.FilteredSpectrum(id="c", frequencies=(100.0, 200.0), amplitudes=(1.0,)))


def test_the_scene_is_a_pure_function_of_time_and_the_fade_is_a_parameter():
    s = _scene()
    s.add(filt.FilteredSpectrum(id="half", fade=0.5))
    e = s.engine
    assert e.inspect(0.0)["objects"] == e.inspect(0.0)["objects"]
    sem = {o["id"]: o["semantic"] for o in e.inspect(0.0)["objects"]}
    assert sem["half"]["fade"] == 0.5 and sem["spec"]["fade"] == 0.0


@needs_gl
def test_the_objects_draw_and_glow_in_the_prism_look():
    def emit(**kw):
        s = Scene("filt_glow", **kw)
        s.add(filt.PoleZero(id="pz", probe=0.3))
        s.add(filt.FrequencyResponse(id="gain"))
        s.add(filt.FilteredSignal(id="wave", pos=(0, -3, 0)))
        s.add(filt.FilteredSpectrum(id="spec", pos=(3, 3, 0)))
        r = s.engine.pass_runner()
        return r.color_and_emission(s.engine.state_at(0.0), (160, 90))

    pic, light = emit()
    assert pic[..., :3].max() > 0.0 and light[..., :3].max() == 0.0  # plain: paint only
    again, _ = emit()
    assert np.array_equal(pic, again)
    pic, light = emit(look="prism")
    assert light[..., :3].max() > 0.2


@needs_gl
@pytest.mark.slow
def test_filtering_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "filtering.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([1.5, 5.0, 9.5], size=(320, 180))["image"]
    _golden("filtering", sheet, e.pass_runner().renderer_class())


def test_cx_is_the_coloring_the_plane_uses():
    assert isinstance(filt.transfer_plane("p"), cx.DomainColoring)
