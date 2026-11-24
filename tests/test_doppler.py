"""Doppler wavefront geometry and observed frequency, through maths and the engine."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, ac, expr
from phasor.domains import ac_math

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "doppler.py"


@pytest.mark.parametrize("mach", [0.0, 0.4, 0.95])
def test_observed_frequency_ahead_and_behind(mach):
    frequency = 12.0
    heard = ac_math.doppler_frequency(mach, frequency, np.array([1.0, -1.0]))
    assert heard == pytest.approx([frequency / (1 - mach), frequency / (1 + mach)])


@pytest.mark.parametrize("mach", [0.3, 0.8])
def test_ring_geometry_gives_the_observed_frequencies(mach):
    """Sound speed is 1: the spacing of successive crests on the axis is the heard period."""
    frequency = 5.0
    centers, radii = ac_math.doppler_wavefronts(mach, frequency, 2.37, 6)
    ahead = np.diff(centers + radii)  # leading edges, oldest first
    behind = np.diff(centers - radii)
    np.testing.assert_allclose(-1 / ahead, frequency / (1 - mach))
    np.testing.assert_allclose(1 / behind, frequency / (1 + mach))


def test_rings_expand_from_the_source_position_at_emission():
    centers, radii = ac_math.doppler_wavefronts(0.5, 4.0, 0.8, 2)
    assert centers == pytest.approx(np.array([0.25, 0.375]))
    assert radii == pytest.approx(np.array([0.30, 0.05]))
    assert not ac_math.doppler_wavefronts(0.5, 4.0, 0.0, 2)[0].size


def test_sonic_front_piles_up_at_the_source():
    centers, radii = ac_math.doppler_wavefronts(1.0, 8.0, 0.85, 16)
    np.testing.assert_allclose(centers + radii, 0.85)
    assert np.isinf(ac_math.doppler_frequency(1.0, 8.0, 1.0))


@pytest.mark.parametrize("mach", [1.01, 1.5, 3.0])
def test_supersonic_rings_are_tangent_to_the_mach_cone(mach):
    time = 0.87
    centers, radii = ac_math.doppler_wavefronts(mach, 8.0, time, 16)
    rays = ac_math.mach_cone_rays(mach)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0)
    np.testing.assert_allclose(np.arcsin(np.abs(rays[:, 1])), np.arcsin(1 / mach))
    assert np.all(rays[:, 0] < 0)
    offsets = centers - mach * time
    np.testing.assert_allclose(np.abs(offsets[:, None] * rays[:, 1]), radii[:, None] * np.ones(2))


def test_engine_doppler_is_deterministic_and_bounds_cover_the_fronts():
    scene = Scene("doppler_test")
    source = ac.Doppler(id="source", speed=1.5, frequency=8, time=expr("t"), size=2)
    scene.add(source)
    scene.marker("end", 1.0)
    engine = scene.engine
    before = engine.inspect(0.84, id="source")["objects"][0]["semantic"]
    engine.inspect(0.1, id="source")
    assert engine.inspect(0.84, id="source")["objects"][0]["semantic"] == before
    assert before["frequency_ahead"] is None
    assert before["frequency_behind"] == pytest.approx(8 / 2.5)
    bounds = source.world_bounds(engine.state_at(0.84))
    assert bounds.lo == pytest.approx((-1.68, -1.68, 0.0))
    assert bounds.hi == pytest.approx((2.52, 1.68, 0.0))


@pytest.mark.parametrize(
    ("time", "outgoing", "incoming"),
    [(3.0, "subsonic", "sonic"), (6.0, "sonic", "supersonic")],
)
def test_showcase_matches_source_tips_and_staggers_captions(time, outgoing, incoming):
    engine = runpy.run_path(str(EXAMPLE))["s"].engine
    for offset in (-0.2, 0.0, 0.2):
        state = engine.state_at(time + offset)
        first, second = state[outgoing], state[incoming]
        assert first.params.time == second.params.time
        assert first.params.opacity > 0 and second.params.opacity > 0
        assert first.params.opacity + second.params.opacity == pytest.approx(1.0)
        np.testing.assert_allclose(
            first.impl.features(state)["source"], second.impl.features(state)["source"]
        )
        captions = [state[f"{name}_caption"].params.opacity for name in (outgoing, incoming)]
        assert sum(alpha > 1e-9 for alpha in captions) == int(offset != 0)
        if offset != 0:
            assert captions[int(offset > 0)] > 0


@pytest.mark.gl
@pytest.mark.slow
def test_doppler_golden_storyboard():
    from tests.test_beauty import _golden

    engine = runpy.run_path(str(EXAMPLE))["s"].engine
    assert engine.lint(strict=True) == []
    sheet = engine.storyboard([1.7, 3.0, 4.7, 6.0, 8.4], size=(320, 180))["image"]
    _golden("doppler", sheet, engine.pass_runner().renderer_class())
