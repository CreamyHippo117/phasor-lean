"""Dynamical systems: trajectories, nullclines, fixed points and their
classification against known results; the objects through the engine."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, dyn
from phasor.domains import dynamics_math as dm
from phasor.domains import linalg_math as lm
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")

PENDULUM = ("y", "-sin(x) - 0.3 * y")


def S(f, g, c=None):
    return dm.System.parse(f, g, c)


@pytest.mark.parametrize(
    ("matrix", "kind", "stable"),
    [
        ([[0, 1], [-1, 0]], "centre", None),
        ([[1, 0], [0, -1]], "saddle", False),
        ([[-2, 0], [0, -3]], "stable node", True),
        ([[2, 0], [0, 3]], "unstable node", False),
        ([[-1, 1], [0, -1]], "degenerate", True),
        ([[-0.5, 2], [-2, -0.5]], "stable spiral", True),
        ([[0.5, 2], [-2, 0.5]], "unstable spiral", False),
        ([[0, 0], [0, -1]], "non-hyperbolic", None),
    ],
)
def test_classification_by_the_jacobian(matrix, kind, stable):
    c = dm.classify(np.array(matrix, dtype=float))
    assert (c["kind"], c["stable"]) == (kind, stable)


def test_trace_and_determinant_agree_with_the_eigenvalues():
    c = dm.classify(np.array([[3.0, 1.0], [2.0, 4.0]]))
    ev = sum(complex(*e) for e in c["eigenvalues"])
    assert c["trace"] == pytest.approx(ev.real)
    assert c["det"] == pytest.approx(np.prod([complex(*e) for e in c["eigenvalues"]]).real)


def test_pendulum_fixed_points_and_their_types():
    s = S(*PENDULUM)
    pts = dm.fixed_points(s, (-4, 4, -3, 3))
    assert pts[:, 0] == pytest.approx([-np.pi, 0.0, np.pi], abs=1e-6)
    kinds = [dm.classify(s.jacobian_at(*p))["kind"] for p in pts]
    assert kinds == ["saddle", "stable spiral", "saddle"]
    spiral = dm.classify(s.jacobian_at(0.0, 0.0))
    assert spiral["eigenvalues"][0][0] == pytest.approx(-0.15)  # -damping / 2


def test_constants_and_unknown_names():
    s = S("x * (1 - x)", "y * (a - x)", "a=2")
    pts = dm.fixed_points(s, (-3, 3, -3, 3))
    assert sorted(map(tuple, np.round(pts, 6))) == [(0.0, 0.0), (1.0, 0.0)]
    with pytest.raises(dm.FunctionError, match="unknown name"):
        S("x + q", "y")
    with pytest.raises(dm.FunctionError, match="name=number"):
        S("x", "y", "mu")


def test_trajectories_follow_the_exact_solution_and_stop_at_the_window():
    s = S("-x", "-2 * y")  # x = x0 e^-t, y = y0 e^-2t
    path = dm.trajectories(s, [[1.0, 2.0]], 2.0, 200, (-9, 9, -9, 9))[0]
    assert path[-1] == pytest.approx([np.exp(-2.0), 2.0 * np.exp(-4.0)], abs=1e-8)
    out = dm.trajectories(S("x", "y"), [[1.0, 1.0]], 5.0, 100, (-3, 3, -3, 3))[0]
    gone = np.isnan(out[:, 0])
    assert gone[-1] and not gone[0] and gone.argmax() > 0
    assert gone[gone.argmax() :].all()  # once out, always out
    back = dm.trajectories(s, [[np.exp(-2.0), 2.0 * np.exp(-4.0)]], 2.0, 200,
                           (-9, 9, -9, 9), direction=-1.0)[0]  # fmt: skip
    assert back[-1] == pytest.approx([1.0, 2.0], abs=1e-6)


def test_conserved_energy_of_a_centre():
    s = S("y", "-x")
    path = dm.trajectories(s, [[1.0, 0.0]], 20.0, 2000, (-9, 9, -9, 9))[0]
    assert np.hypot(path[:, 0], path[:, 1]) == pytest.approx(1.0, abs=1e-6)


def test_nullcline_segments_lie_on_the_zero_set():
    s = S("y - x**2", "x - y")
    seg = dm.nullcline_segments(s, "x", (-2, 2, -1, 4), 80).reshape(-1, 2)
    assert len(seg) > 50
    assert np.abs(seg[:, 1] - seg[:, 0] ** 2).max() < 0.02  # on y = x^2
    seg = dm.nullcline_segments(s, "y", (-2, 2, -1, 4), 80).reshape(-1, 2)
    assert np.abs(seg[:, 1] - seg[:, 0]).max() < 1e-9  # on y = x


def test_seeds_are_inside_the_window_and_deterministic():
    a = dm.hash_seeds(12, (8, 5), 3)
    assert len(a) == 12 and (np.abs(a[:, 0]) < 4).all() and (np.abs(a[:, 1]) < 2.5).all()
    assert np.array_equal(a, dm.hash_seeds(12, (8, 5), 3))


def _scene():
    s = Scene("dyn_test")
    s.add(dyn.PhasePortrait(id="flow", f=PENDULUM[0], g=PENDULUM[1], extent=(8, 5), n_seeds=6,
                            progress=0.5))  # fmt: skip
    s.add(dyn.FixedPoints(id="rest", f=PENDULUM[0], g=PENDULUM[1], extent=(8, 5)))
    return s


def test_objects_report_what_they_find():
    e = _scene().engine
    objs = {o["id"]: o["semantic"] for o in e.inspect(0.5)["objects"]}
    assert objs["flow"]["fixed_points"] == 3 and objs["flow"]["trajectories"] == 6
    pts = objs["rest"]["points"]
    assert [p["type"] for p in pts] == ["saddle", "stable spiral", "saddle"]
    assert pts[1]["stable"] is True


def test_bad_formulas_are_rejected_when_added():
    s = Scene("dyn_bad")
    with pytest.raises(Exception, match="cannot use|unknown"):
        s.add(dyn.PhasePortrait(id="p", f="__import__('os')", g="y"))
    with pytest.raises(Exception, match="unknown name"):
        s.add(dyn.PhasePortrait(id="q", f="x + w", g="y"))


@needs_gl
def test_a_portrait_draws_and_glows_in_the_prism_look():
    def emit(**kw):
        s = Scene("dyn_glow", **kw)
        s.add(dyn.PhasePortrait(id="flow", f=PENDULUM[0], g=PENDULUM[1], extent=(8, 5),
                                n_seeds=8))  # fmt: skip
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
def test_phase_portrait_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "phase_portrait.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([2.0, 4.5, 9.0], size=(320, 180))["image"]
    _golden("phase_portrait", sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- D5: linear systems


def test_expm_matches_closed_forms():
    a = np.pi / 3
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    np.testing.assert_allclose(lm.expm([[0, 1], [-1, 0]], a), rot, atol=1e-12)
    np.testing.assert_allclose(lm.expm([[2, 0], [0, -1]], 1.5), np.diag([np.exp(3), np.exp(-1.5)]))
    np.testing.assert_allclose(lm.expm([[0, 1], [0, 0]], 2.0), [[1, 2], [0, 1]])  # nilpotent
    assert lm.expm([[30, 0], [0, -30]], 1.0)[0, 0] == pytest.approx(np.exp(30), rel=1e-9)


def test_expm_is_a_flow_and_its_determinant_is_e_to_the_trace():
    A = np.array([[0.3, 1.2], [-0.7, -0.1]])
    np.testing.assert_allclose(lm.expm(A, 0.7) @ lm.expm(A, 0.4), lm.expm(A, 1.1), atol=1e-12)
    np.testing.assert_allclose(lm.expm(A, 0.0), np.eye(2))
    assert np.linalg.det(lm.expm(A, 2.0)) == pytest.approx(np.exp(np.trace(A) * 2.0))


def test_linear_paths_agree_with_the_integrator_and_eigenlines_are_invariant():
    A = [[1.0, 2.0], [2.0, 1.0]]
    big = (-99, 99, -99, 99)
    exact = dm.linear_paths(A, [[0.3, -0.2]], 1.0, 200, big)[0]
    rk = dm.trajectories(S("x + 2*y", "2*x + y"), [[0.3, -0.2]], 1.0, 200, big)[0]
    np.testing.assert_allclose(exact, rk, rtol=1e-8, atol=1e-8)
    on = dm.linear_paths(A, [[1.0, 1.0]], 1.0, 20, big)[0]  # eigenvalue 3
    assert on[:, 0] == pytest.approx(on[:, 1])
    assert on[-1, 0] == pytest.approx(np.exp(3.0))
    off = dm.linear_paths(A, [[1.0, -1.0]], 1.0, 20, big)[0]  # eigenvalue -1
    assert off[-1] == pytest.approx([np.exp(-1.0), -np.exp(-1.0)])


def test_linear_paths_stop_at_the_window_and_run_backward():
    out = dm.linear_paths([[1, 0], [0, 1]], [[1.0, 1.0]], 5.0, 100, (-3, 3, -3, 3))[0]
    gone = np.isnan(out[:, 0])
    assert gone[-1] and not gone[0] and gone[gone.argmax() :].all()
    back = dm.linear_paths([[-1, 0], [0, -2]], [[np.exp(-1.0), np.exp(-2.0)]], 1.0, 50,
                           (-9, 9, -9, 9), direction=-1.0)[0]  # fmt: skip
    assert back[-1] == pytest.approx([1.0, 1.0])


def test_transform_grid_shows_the_flow_of_a_system():
    from phasor import linalg

    A = ((1, 2), (2, 1))
    s = Scene("lin_grid", passes=None)
    s.add(linalg.TransformGrid(id="grid", flow=A, flow_time=0.5))
    s.add(linalg.TransformGrid(id="plain", matrix=A))
    sem = {o["id"]: o["semantic"] for o in s.engine.inspect(0.0)["objects"]}
    assert sem["grid"]["determinant"] == pytest.approx(np.exp(1.0), abs=1e-5)  # e^(trace t)
    assert sem["plain"]["determinant"] == pytest.approx(-3.0)  # ``matrix`` still means itself
    assert sem["grid"]["matrix"] == np.round(lm.expm(A, 0.5), 6).tolist()


def _lin(**kw):
    s = Scene("lin_test")
    s.add(dyn.LinearSystem(id="sys", extent=(8, 5), n_seeds=6, **kw))
    return s


def _sem(s):
    return s.engine.inspect(0.0, id="sys")["objects"][0]["semantic"]


@pytest.mark.parametrize(
    ("matrix", "kind"),
    [(((1, 2), (2, 1)), "saddle"), (((-2, 0), (0, -3)), "stable node"),
     (((0.2, 2), (-2, 0.2)), "unstable spiral"), (((0, 1), (-1, 0)), "centre")],
)  # fmt: skip
def test_linear_system_reports_its_type(matrix, kind):
    sem = _sem(_lin(matrix=matrix))
    assert sem["type"] == kind and sem["trajectories"] == 6


def test_linear_system_semantics_for_the_saddle():
    sem = _sem(_lin(matrix=((1, 2), (2, 1))))
    assert sorted(e[0] for e in sem["eigenvalues"]) == pytest.approx([-1.0, 3.0])
    assert sem["trace"] == 2.0 and sem["det"] == pytest.approx(-3.0)
    assert len(sem["eigenvectors"]) == 2


def test_linear_system_rejects_a_matrix_that_is_not_2x2():
    with pytest.raises(Exception, match="2 items|2x2"):
        Scene("lin_bad").add(dyn.LinearSystem(id="s", matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1))))


@needs_gl
def test_a_linear_system_draws_and_glows_in_the_prism_look():
    s = Scene("lin_glow", look="prism")
    s.add(dyn.LinearSystem(id="sys", extent=(8, 5), n_seeds=8))
    pic, light = s.engine.pass_runner().color_and_emission(s.engine.state_at(0.0), (160, 90))
    assert pic[..., :3].max() > 0.0 and light[..., :3].max() > 0.2


@needs_gl
@pytest.mark.slow
def test_linear_system_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "linear_system.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([2.0, 4.5, 9.0], size=(320, 180))["image"]
    _golden("linear_system", sheet, e.pass_runner().renderer_class())
