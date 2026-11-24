"""Hot paths made faster must give the same bits as before."""

import numpy as np

from phasor.core.spatial import OrthoCameraParams, ortho_camera
from phasor.core.values import normalize


def _old_project(cam, points):
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    clip = np.c_[p, np.ones(len(p))] @ cam.view_proj.T
    return clip[:, :3] / clip[:, 3:4]


def test_project_and_clip_w_are_bit_identical_to_the_old_form():
    rng = np.random.default_rng(3)
    cam = ortho_camera(OrthoCameraParams(center=(0.3, -0.2, 0.0), height=7.0), 16 / 9)
    for n in (1, 2, 7, 500):
        pts = rng.normal(size=(n, 3)) * 5
        assert np.array_equal(cam.project(pts), _old_project(cam, pts))
        old_w = np.c_[pts, np.ones(n)] @ cam.view_proj[3]
        assert np.array_equal(cam.clip_w(pts), old_w)
    one = [1.0, 2.0, 3.0]  # a single (3,) point
    assert np.array_equal(cam.project(one), _old_project(cam, one))


def test_normalize_is_unchanged():
    assert normalize(2) == 2.0 and type(normalize(2)) is float
    assert normalize(True) is True
    assert normalize([1, (2, [3.5])]) == (1.0, (2.0, (3.5,)))
    assert normalize(np.arange(3)) == (0.0, 1.0, 2.0)
    assert normalize(np.float32(1.5)) == 1.5
    for bad in (float("nan"), float("inf"), None, {}, object()):
        try:
            normalize(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_coverage_fractions_are_unchanged():
    from phasor.layout.geometry import Box, Coverage

    rng = np.random.default_rng(5)
    mask = rng.random((90, 160)) > 0.6
    cov = Coverage(mask, 16 / 9)
    sat = np.zeros((91, 161), dtype=np.int64)
    sat[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    assert np.array_equal(cov._sat, sat)
    assert 0.0 <= cov.fraction(Box(0.2, 0.1, 1.1, 0.8)) <= 1.0


def test_a_render_probes_the_machine_once_per_process(monkeypatch):
    from phasor.render import doctor

    calls = []
    real = doctor.run_doctor

    def counting():
        calls.append(1)
        return real()

    monkeypatch.setattr(doctor, "run_doctor", counting)
    monkeypatch.setattr(doctor, "_REPORT", None)
    a, b = doctor.cached_report(), doctor.cached_report()
    assert a is b and len(calls) == 1
    assert doctor.run_doctor() is not a  # `phasor doctor` itself always probes fresh
