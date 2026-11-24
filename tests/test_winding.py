import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, fourier
from phasor.domains import fourier_math as fm
from tests.test_beauty import _golden, needs_gl


def test_winding_mean_matches_normalized_dft():
    n, fs, tone = 128, 64, 4
    signal = fm.sinusoids((tone,), (1.0,), n, fs)
    bins = np.fft.rfftfreq(n, 1 / fs)
    mean = fm.winding_mean(signal, bins, fs)
    np.testing.assert_allclose(mean, np.fft.rfft(signal) / n, atol=1e-14)
    assert abs(mean[bins == tone].item()) == pytest.approx(0.5)
    assert np.max(np.abs(mean[bins != tone])) < 1e-14
    np.testing.assert_allclose(
        2 * abs(mean), fm.spectrum((tone,), (1,), "rect", n, fs)["mag"], atol=1e-14
    )


def test_off_bin_winding_and_seek_are_deterministic():
    scene = Scene("winding_test")
    obj = fourier.Winding(id="wind", freq=2.25)
    scene.add(obj)
    scene.marker("sweep", 0.5)
    scene.marker("end", 4.0)
    scene.at("sweep").sweep("wind.freq", 8.0, show=False)
    engine = scene.engine
    state = engine.state_at(1.3)
    first = obj.semantics(state)
    p = obj.me(state).params
    signal = fm.sinusoids(p.frequencies, p.amplitudes, p.samples, p.fs)
    expected = np.dot(signal, np.exp(-2j * np.pi * p.freq * np.arange(p.samples) / p.fs))
    expected /= p.samples
    assert first["centroid"] == pytest.approx((expected.real, expected.imag))
    engine.state_at(3.7)
    assert obj.semantics(engine.state_at(1.3)) == first
    assert (
        obj.features(state)["spectrum_tip"][0]
        < obj.features(engine.state_at(3.7))["spectrum_tip"][0]
    )


@needs_gl
@pytest.mark.slow
def test_fourier_winding_golden_storyboard():
    example = Path(__file__).resolve().parents[1] / "examples" / "fourier_winding.py"
    engine = runpy.run_path(str(example), run_name="x")["s"].engine
    assert engine.lint(strict=True) == []
    sheet = engine.storyboard([2.35, 6.3, 13.8], size=(320, 180))["image"]
    _golden("fourier_winding", sheet, engine.pass_runner().renderer_class())
