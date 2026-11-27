"""Sampling and aliasing: the alias of a tone (frequency and phase) above
and below Nyquist and at whole multiples of the rate, the alias through every sample, the sinc
reconstruction, then the object through the engine, its glow and its golden storyboard."""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, fourier
from phasor.domains import fourier_math as fm
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _samples(f, fs, amp, phase, n=64):
    t = np.arange(-n, n + 1) / fs
    return t, amp * np.sin(2 * math.pi * f * t + phase)


@pytest.mark.parametrize("phase", [0.0, 0.4, -1.1, 2.9])
def test_below_nyquist_the_tone_is_its_own_alias(phase):
    fa, a, ph = fm.alias_tone(3.0, 10.0, 0.7, phase)
    assert (fa, a) == (3.0, 0.7) and ph == pytest.approx(phase)


def test_above_nyquist_the_tone_folds_and_its_phase_reflects():
    # 7 Hz at 10 Hz: the samples are those of 3 Hz running the other way
    fa, a, ph = fm.alias_tone(7.0, 10.0, 1.0, 0.3)
    assert fa == pytest.approx(3.0) and a == 1.0
    assert ph == pytest.approx(math.pi - 0.3)
    # 12 Hz at 10 Hz: one whole rate above 2 Hz, no reflection
    fa, _, ph = fm.alias_tone(12.0, 10.0, 1.0, 0.3)
    assert fa == pytest.approx(2.0) and ph == pytest.approx(0.3)
    # 19 Hz at 10 Hz: 9 Hz folds to 1 Hz, reflected
    fa, _, ph = fm.alias_tone(19.0, 10.0, 1.0, 0.3)
    assert fa == pytest.approx(1.0) and ph == pytest.approx(math.pi - 0.3)


@pytest.mark.parametrize("k", [1, 2, 5])
def test_a_whole_multiple_of_the_rate_aliases_to_a_constant(k):
    fa, a, ph = fm.alias_tone(k * 8.0, 8.0, 2.0, 0.5)
    assert fa == 0.0
    _, x = _samples(k * 8.0, 8.0, 2.0, 0.5)
    assert x == pytest.approx(np.full_like(x, a * math.sin(ph)), abs=1e-9)
    assert a * math.sin(ph) == pytest.approx(2.0 * math.sin(0.5))


def test_exactly_at_nyquist_the_alias_is_the_cosine():
    fa, a, ph = fm.alias_tone(5.0, 10.0, 1.0, 0.6)
    assert fa == 5.0 and ph == pytest.approx(math.pi / 2)
    assert a == pytest.approx(math.sin(0.6))
    assert fm.alias_tone(5.0, 10.0, 1.0, 0.0)[1] == 0.0  # sin through its zeros: all samples 0


def test_fold_is_signed_and_turns_the_short_way():
    assert fm.fold(3.0, 10.0) == pytest.approx(3.0)
    assert fm.fold(7.0, 10.0) == pytest.approx(-3.0)  # backwards
    assert fm.fold(10.0, 10.0) == pytest.approx(0.0)  # stands still
    assert fm.fold(11.0, 10.0) == pytest.approx(1.0)  # creeps forwards
    assert fm.fold(5.0, 10.0) == pytest.approx(-5.0)  # [-fs/2, fs/2)
    # |fold| is the alias frequency
    for f in (0.5, 3.3, 6.1, 9.9, 14.2, 27.0):
        assert abs(fm.fold(f, 10.0)) == pytest.approx(fm.alias_tone(f, 10.0)[0], abs=1e-12)


@pytest.mark.parametrize("f", [0.0, 1.3, 4.9, 5.0, 5.1, 7.0, 10.0, 13.7, 25.0, 31.4])
@pytest.mark.parametrize("phase", [0.0, 0.8, -2.0])
def test_the_alias_passes_through_every_sample(f, phase):
    fs = 10.0
    t, x = _samples(f, fs, 1.3, phase)
    fa, a, ph = fm.alias_tone(f, fs, 1.3, phase)
    assert 0 <= fa <= fs / 2
    assert a * np.sin(2 * math.pi * fa * t + ph) == pytest.approx(x, abs=1e-9)


def test_a_negative_frequency_is_the_mirrored_tone():
    t, x = _samples(-3.0, 10.0, 1.0, 0.4)
    fa, a, ph = fm.alias_tone(-3.0, 10.0, 1.0, 0.4)
    assert fa == pytest.approx(3.0)
    assert a * np.sin(2 * math.pi * fa * t + ph) == pytest.approx(x, abs=1e-9)


def test_a_sum_of_tones_aliases_tone_by_tone():
    f, a, ph, fs = (2.0, 9.0, 13.0), (1.0, 0.5, 0.3), (0.0, 1.0, -0.5), 10.0
    t = np.arange(40) / fs
    x = fm.tone_sum(f, a, ph, t)
    assert fm.alias_signal(f, a, ph, fs, t) == pytest.approx(x, abs=1e-9)
    assert [round(v[0], 9) for v in fm.aliases(f, a, ph, fs)] == [2.0, 1.0, 3.0]


def test_sample_times_cover_the_window():
    t = fm.sample_times(4.0, 0.1, 1.0)
    assert t == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_sinc_reconstruction_is_the_original_when_band_limited():
    fs, n = 10.0, 801
    t0 = -(n // 2) / fs
    ts = t0 + np.arange(n) / fs
    x = fm.tone_sum((1.7, 3.1), (1.0, 0.6), (0.3, -1.0), ts)
    t = np.linspace(-1.0, 1.0, 301)  # far from the ends of the sum
    rec = fm.sinc_reconstruct(x, fs, t, t0)
    assert rec == pytest.approx(fm.tone_sum((1.7, 3.1), (1.0, 0.6), (0.3, -1.0), t), abs=5e-3)
    assert fm.sinc_reconstruct(x, fs, ts, t0) == pytest.approx(x, abs=1e-12)  # exact at samples


def test_sinc_reconstruction_of_a_fast_tone_is_its_alias():
    fs, n = 10.0, 801
    t0 = -(n // 2) / fs
    ts = t0 + np.arange(n) / fs
    x = fm.tone_sum((7.0,), (1.0,), (0.3,), ts)
    t = np.linspace(-1.0, 1.0, 301)
    rec = fm.sinc_reconstruct(x, fs, t, t0)
    assert rec == pytest.approx(fm.alias_signal((7.0,), (1.0,), (0.3,), fs, t), abs=5e-3)
    assert np.abs(rec - fm.tone_sum((7.0,), (1.0,), (0.3,), t)).max() > 0.5  # not the original


# ---------------------------------------------------------------- the object


def _sem(fs, **kw):
    s = Scene("alias_sem", passes=None)
    s.add(fourier.Aliasing(id="a", fs=fs, **kw))
    return s.engine.inspect(0.0, id="a")["objects"][0]["semantic"]


def test_semantics_name_the_rate_the_tones_and_their_aliases():
    sem = _sem(12.0, frequencies=(5.0,))
    assert sem["kind"] == "aliasing" and sem["fs"] == 12.0 and sem["nyquist"] == 6.0
    assert sem["frequencies"] == [5.0] and sem["alias_frequencies"] == [5.0]
    assert sem["aliased"] is False and sem["turns_per_sample"] == [pytest.approx(5 / 12)]
    sem = _sem(7.0, frequencies=(5.0,))
    assert sem["aliased"] is True and sem["alias_frequencies"] == [2.0]
    assert sem["reversed"] == [True] and sem["turns_per_sample"] == [pytest.approx(-2 / 7)]
    sem = _sem(5.0, frequencies=(5.0,))
    assert sem["alias_frequencies"] == [0.0] and sem["turns_per_sample"] == [0.0]
    assert sem["samples"] == len(fm.sample_times(5.0, 0.0, 2.0))


def test_a_formula_is_reconstructed_by_sinc():
    sem = _sem(12.0, signal="sin(tau * 3 * t) + 0.5 * sin(tau * 9 * t)")
    assert sem["reconstruction"] == "sinc" and sem["frequencies"] is None
    assert sem["aliased"] is None  # a formula's frequencies are not known


def test_bad_parameters_are_rejected():
    s = Scene("alias_bad")
    with pytest.raises(Exception, match="one value per"):
        s.add(fourier.Aliasing(id="a", frequencies=(1.0, 2.0), amplitudes=(1.0,)))
    with pytest.raises(Exception, match="formula in t"):
        s.add(fourier.Aliasing(id="b", signal="a.fs * t"))
    with pytest.raises(Exception, match="too many samples"):
        s.add(fourier.Aliasing(id="c", fs=1e6, duration=10.0))


def test_the_alias_drawn_passes_through_the_samples_drawn():
    s = Scene("alias_feat", passes=None)
    a = fourier.Aliasing(id="a", fs=7.0, frequencies=(5.0,), phases=(0.4,))
    s.add(a)
    f = a.features(s.engine.state_at(0.0))
    assert len(f["samples"]) == _sem(7.0)["samples"]
    assert f["alias_at_samples"][:, 1] == pytest.approx(f["samples"][:, 1], abs=1e-9)


@needs_gl
def test_it_draws_and_glows_in_the_prism_look():
    def emit(**kw):
        s = Scene("alias_glow", **kw)
        s.add(fourier.Aliasing(id="a", fs=7.0))
        r = s.engine.pass_runner()
        return r.color_and_emission(s.engine.state_at(0.0), (320, 180))

    pic, light = emit()
    assert pic[..., :3].max() > 0.0 and light[..., :3].max() == 0.0  # plain: paint only
    again, _ = emit()
    assert np.array_equal(pic, again)
    pic, light = emit(look="prism")
    assert light[..., :3].max() > 0.2


@needs_gl
@pytest.mark.slow
def test_aliasing_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "aliasing.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([1.8, 6.0, 9.5], size=(320, 180))["image"]
    _golden("aliasing", sheet, e.pass_runner().renderer_class())
