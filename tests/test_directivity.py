"""Polar patterns against analytic results and through the renderer."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, ac
from phasor.domains import ac_math
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]


def test_array_peak_and_first_null_at_steer():
    angle = np.linspace(-np.pi / 2, np.pi / 2, 4001)
    steer = 0.43
    elements = 10
    wavelength = 0.5
    spacing = wavelength / 2
    frequency = ac_math.C / wavelength
    gain = ac_math.array_directivity(angle, elements, spacing, frequency, steer)
    first_null = np.arcsin(np.sin(steer) + wavelength / (elements * spacing))
    assert abs(angle[gain.argmax()] - steer) < (first_null - steer) / 10
    assert gain.max() == pytest.approx(1.0, abs=1e-5)
    assert ac_math.array_directivity(first_null, elements, spacing, frequency, steer) < 1e-12


def test_array_grating_lobe_and_single_element():
    angles = np.array([0, np.pi / 6, -np.pi / 6])
    assert ac_math.array_directivity(angles, 8, 2.0, ac_math.C, 0) == pytest.approx(1)
    assert ac_math.array_directivity(angles, 1, 0.5, ac_math.C, 1) == pytest.approx(1)


def test_cardioid_endpoints():
    assert ac_math.cardioid_directivity(np.array([0, np.pi])) == pytest.approx([1, 0])


def test_piston_first_null():
    first_j1_zero = 3.8317059702075125
    wavelength = 0.5
    radius = 0.5
    ka = 2 * np.pi * radius / wavelength
    theta = np.arcsin(first_j1_zero / ka)
    angles = np.linspace(0, theta - 0.01, 256)
    assert np.min(ac_math.piston_directivity(angles, radius, ac_math.C / wavelength)) > 0.01
    assert ac_math.piston_directivity(theta, radius, ac_math.C / wavelength) < 1e-10
    assert ac_math.piston_directivity(0, radius, ac_math.C / wavelength) == pytest.approx(1)
    assert ac_math.piston_directivity(np.pi, radius, ac_math.C / wavelength) == 0


def test_inspector_reports_the_steered_axis():
    scene = Scene("directivity_test")
    scene.add(ac.Directivity(id="beam", steer=30, size=2, pos=(1, 2, 0)))
    engine = scene.engine
    beam = engine.inspect(0)["objects"][0]
    assert beam["semantic"]["kind"] == "array"
    assert beam["semantic"]["steer"] == 30


@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.slow
def test_directivity_golden_storyboard():
    from tests.test_beauty import _golden

    scene = runpy.run_path(str(ROOT / "examples" / "directivity.py"))["s"]
    engine = scene.engine
    assert engine.lint(strict=True) == []
    sheet = engine.storyboard([0.9, 3.65, 5.0, 7.75, 10.5], size=(320, 180))["image"]
    _golden("directivity", sheet, engine.pass_runner().renderer_class())
