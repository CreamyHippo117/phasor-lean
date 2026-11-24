"""Fixed-step strange attractors, scene integration and prism storyboard."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, dyn
from phasor.core.simulation import SimRunner
from phasor.domains import dynamics_math as dm
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _path(sim, times):
    runner = SimRunner(sim, cache_dir=False)
    return np.array([runner.state_at(t)["point"] for t in times])


@pytest.mark.parametrize(("system", "bound"), [("lorenz", 60), ("rossler", 40), ("thomas", 9)])
def test_classic_attractors_remain_bounded(system, bound):
    points = _path(dm.AttractorSimulation(system=system), np.arange(0, 80, 0.1))
    assert np.max(np.linalg.norm(points, axis=1)) < bound
    assert np.all(np.ptp(points, axis=0) > 1)


def test_nearby_lorenz_starts_have_a_positive_finite_time_lyapunov_estimate():
    start = (1.0, 1.0, 1.0)
    nearby = (1.0 + 1e-7, 1.0, 1.0)
    times = np.linspace(0, 24, 241)
    a = _path(dm.AttractorSimulation(start=start), times)
    b = _path(dm.AttractorSimulation(start=nearby), times)
    separation = np.linalg.norm(b - a, axis=1)
    slope = np.polyfit(times, np.log(separation / separation[0]), 1)[0]
    assert slope > 0.3
    assert separation[-1] > 100 * separation[0]


def test_thomas_is_dissipative_and_does_not_conserve_squared_radius():
    sim = dm.AttractorSimulation(system="thomas")
    p = np.array([1.0, 2.0, 3.0])
    offsets = 1e-5 * np.eye(3)
    divergence = sum(
        (sim.velocity(p + dp)[i] - sim.velocity(p - dp)[i]) / 2e-5 for i, dp in enumerate(offsets)
    )
    assert divergence == pytest.approx(-3 * 0.208186)
    assert np.ptp(np.sum(_path(sim, np.arange(0, 60, 0.1)) ** 2, axis=1)) > 1


def test_seed_and_scrub_order():
    sim = dm.AttractorSimulation(seed=19)
    a = SimRunner(sim, cache_dir=False)
    b = SimRunner(dm.AttractorSimulation(seed=19), cache_dir=False)
    assert sim.params()["seed"] == 19
    a.state_at(12.0)
    np.testing.assert_array_equal(a.state_at(0.75)["point"], b.state_at(0.75)["point"])
    np.testing.assert_array_equal(a.state_at(12.0)["point"], b.state_at(12.0)["point"])
    other = SimRunner(dm.AttractorSimulation(seed=20), cache_dir=False)
    assert not np.array_equal(a.state_at(0)["point"], other.state_at(0)["point"])


def test_object_reports_full_bounds_and_progress():
    s = Scene("attractor_test")
    a = s.add(dyn.StrangeAttractor(id="orbit", progress=0.5, t_end=8, size=4, pos=(1, 2, 3)))
    e = s.engine
    state = e.state_at(0)
    info = e.inspect(0, id="orbit")["objects"][0]["semantic"]
    assert info["system"] == "lorenz"
    assert info["simulated_time"] == pytest.approx(4)
    bounds = a.world_bounds(state)
    assert np.max(np.array(bounds.hi) - bounds.lo) == pytest.approx(4)
    head = a.features(state)["head"]
    assert np.all(head >= bounds.lo) and np.all(head <= bounds.hi)


def test_showcase_curve_has_room_throughout_the_camera_orbit():
    scene = runpy.run_path(str(ROOT / "examples" / "strange_attractor.py"), run_name="x")
    engine = scene["s"].engine
    for time in np.linspace(0, engine.duration, 339):
        state = engine.state_at(time)
        obj = state["trajectory"]
        points = obj.impl.path_points(state)
        tip = state["moving_tip"]
        if tip.params.opacity > 0:
            points = np.vstack((points, tip.impl.world_bounds(state).corners()))
        screen = state.camera_for(obj).to_screen(points, 1, 1)
        assert ((screen > 0.08) & (screen < 0.92)).all(), f"tight framing at {time:.2f}s"


def test_showcase_tip_tracks_the_lobe_crossing_and_fades_for_the_hold():
    engine = runpy.run_path(str(ROOT / "examples" / "strange_attractor.py"), run_name="x")[
        "s"
    ].engine

    for time in (2.85, 3.83, 5.35, 7.5):
        state = engine.state_at(time)
        trace, tip = state["trajectory"], state["moving_tip"]
        assert tip.params.opacity == 1
        np.testing.assert_allclose(
            tip.impl.features(state)["tip"], trace.impl.features(state)["head"]
        )

    crossing = engine.state_at(3.83)
    assert crossing["trajectory"].params.progress * 30 == pytest.approx(14.34, abs=0.03)
    assert crossing.values["camera.azimuth"] == engine.state_at(0).values["camera.azimuth"]
    settled = engine.state_at(9.25)
    end = engine.state_at(engine.duration)
    for state in (settled, end):
        assert state["trajectory"].params.progress == 1
        assert state["moving_tip"].params.opacity == 0
    assert settled.values["camera.azimuth"] == end.values["camera.azimuth"]
    assert engine.duration - 9.25 >= 2


@needs_gl
@pytest.mark.slow
def test_strange_attractor_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "strange_attractor.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    state = e.state_at(3.83)
    _, emission = e.pass_runner().color_and_emission(state, (160, 90))
    assert emission[..., :3].max() > 0.2
    sheet = e.storyboard([0.0, 2.1, 3.83, 5.2, 7.2, 10.7], size=(320, 180))["image"]
    _golden("strange_attractor", sheet, e.pass_runner().renderer_class())
