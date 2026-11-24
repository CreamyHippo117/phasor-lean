"""PDE solvers against their exact solutions (the heat equation is the exit demo)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor.core.simulation import SimRunner
from phasor.domains import pde_math as pm


def run(sim, t, tmp_path):
    return SimRunner(sim, every=200, cache_dir=tmp_path).state_at(t)


# ---------------------------------------------------------------- heat


def test_heat_1d_matches_the_exact_modes_and_decays_at_the_right_rate(tmp_path):
    sim = pm.Heat1D(k=0.5, length=2.0, n=100, modes=(1.0, 0.3))
    t = 0.4
    u = run(sim, t, tmp_path)["u"]
    assert np.abs(u - sim.exact(t)).max() < 2e-3
    # the m = 2 mode dies 4 times faster in the exponent than the fundamental
    k1, k2 = ((m * math.pi / 2.0) ** 2 * 0.5 for m in (1, 2))
    assert k2 == pytest.approx(4 * k1)

    def peak(tt):
        return np.abs(
            run(pm.Heat1D(k=0.5, length=2.0, n=100, modes=(1.0,)), tt, tmp_path)["u"]
        ).max()

    assert peak(1.0) / peak(0.0) == pytest.approx(math.exp(-k1 * 1.0), rel=0.01)
    assert u[0] == u[-1] == 0.0  # the ends are held


def test_heat_1d_smooths_a_bump_and_never_exceeds_the_start(tmp_path):
    sim = pm.Heat1D(k=1.0, n=100, modes=())
    u0, u1 = sim.initial()["u"], run(sim, 0.01, tmp_path)["u"]
    assert u1.max() < u0.max() and (u1 >= -1e-12).all()  # maximum principle: no new peaks
    assert (
        u1.sum() * sim.dx < u0.sum() * sim.dx + 1e-9
        and np.abs(np.diff(u1, 2)).max() < np.abs(np.diff(u0, 2)).max()
    )


def test_heat_2d_single_mode_decays_exactly_as_theory_says(tmp_path):
    sim = pm.Heat2D(k=0.3, size=(1.0, 0.5), n=40, mode=(2, 1))
    t = 0.05
    u = run(sim, t, tmp_path)["u"]
    assert np.abs(u - sim.exact(t)).max() < 3e-3
    lam = (2 * math.pi) ** 2 + (math.pi / 0.5) ** 2
    assert u.max() / sim.initial()["u"].max() == pytest.approx(math.exp(-0.3 * lam * t), rel=0.03)
    spot = pm.Heat2D(mode=(0, 0), n=32)
    a, b = spot.initial()["u"], run(spot, 0.02, tmp_path)["u"]
    assert b.max() < a.max() and spot.exact(0.0) is None


@pytest.mark.parametrize("cls", [pm.Heat1D, pm.Heat2D])
def test_an_unstable_heat_step_is_refused(cls):
    ok = cls(n=20)
    with pytest.raises(ValueError, match="unstable"):
        cls(n=20, dt=ok.dt * 3)
    assert cls(n=20, dt=ok.dt * 1.0).dt == ok.dt  # at the (80%) default it is fine


# ---------------------------------------------------------------- waves


def test_a_string_in_a_standing_wave_oscillates_at_its_frequency(tmp_path):
    sim = pm.Wave1D(c=2.0, length=1.0, n=200, mode=2)
    period = 2 * math.pi / (2.0 * 2 * math.pi)  # w = c m pi / L = 4 pi: T = 0.5
    for frac in (0.25, 0.5, 1.0):
        u = run(sim, frac * period, tmp_path)["u"]
        assert np.abs(u - sim.exact(frac * period)).max() < 0.03
    assert abs(run(sim, period / 4, tmp_path)["u"]).max() < 0.05  # a quarter period: flat
    plucked = pm.Wave1D(mode=0)
    assert plucked.exact(0.0) is None and plucked.initial()["u"].max() == pytest.approx(1.0)


def test_a_drum_mode_has_the_frequency_of_theory(tmp_path):
    sim = pm.Wave2D(c=1.0, size=(1.0, 1.0), n=48, mode=(1, 2))
    assert sim.omega() == pytest.approx(math.pi * math.sqrt(5))
    t = 2 * math.pi / sim.omega() / 2  # half a period: the shape is inverted
    u = run(sim, t, tmp_path)["u"]
    assert np.abs(u + sim._shape_fn()).max() < 0.06
    assert (u[0, :] == 0).all() and (u[:, -1] == 0).all()  # fixed edge
    with pytest.raises(ValueError, match="unstable"):
        pm.Wave2D(c=1.0, n=48, dt=1.0 / 48)


def test_a_plucked_string_returns_to_its_shape_after_one_period(tmp_path):
    sim = pm.Wave1D(c=1.0, length=1.0, n=200, mode=0)
    start = sim.initial()["u"]
    t_period = 2 * 1.0 / 1.0  # 2 L / c
    back = run(sim, t_period, tmp_path)["u"]
    assert np.abs(back - start).max() < 0.05
    half = run(sim, t_period / 2, tmp_path)["u"]
    assert np.abs(half + start[::-1]).max() < 0.08  # half a period: upside down and mirrored


# ---------------------------------------------------------------- advection


def test_advection_carries_the_pulse_at_speed_c_and_stays_stable(tmp_path):
    sim = pm.Advection1D(c=0.7, n=200)
    t = 0.6
    u = run(sim, t, tmp_path)["u"]
    assert np.abs(u - sim.exact(t)).max() < 0.06  # second order: a little dispersion
    peak_x = sim.x[np.argmax(u)]
    assert peak_x == pytest.approx((0.3 + 0.7 * t) % 1.0, abs=0.02)
    assert u.max() < 1.05 and u.sum() * sim.dx == pytest.approx(
        sim.initial()["u"].sum() * sim.dx, rel=1e-6
    )  # mass kept
    with pytest.raises(ValueError, match="unstable"):
        pm.Advection1D(c=2.0, n=100, dt=0.01)


# ---------------------------------------------------------------- reaction-diffusion


def test_gray_scott_grows_a_pattern_from_a_seed_and_is_deterministic(tmp_path):
    sim = pm.GrayScott(n=48, seed=3)
    early, late = run(sim, 5, tmp_path)["v"], run(sim, 600, tmp_path)["v"]
    assert late.std() > early.std() and late.std() > 0.1  # a pattern has grown
    assert late.min() >= 0 and late.max() <= 1.5  # and stays bounded
    again = SimRunner(pm.GrayScott(n=48, seed=3), every=100, cache_dir=tmp_path / "b").state_at(
        600
    )["v"]
    assert np.array_equal(late, again)  # same seed, same pattern
    other = SimRunner(pm.GrayScott(n=48, seed=4), every=100, cache_dir=tmp_path / "c").state_at(
        600
    )["v"]
    assert not np.array_equal(late, other)
    with pytest.raises(ValueError, match="unstable"):
        pm.GrayScott(dt=3.0)


# ---------------------------------------------------------------- Schrodinger


def test_schrodinger_keeps_its_norm_and_a_free_packet_spreads_as_theory_says(tmp_path):
    sim = pm.Schrodinger1D(length=40.0, n=512, x0=0.0, p0=0.0, width=1.0)
    s0, s1 = sim.initial(), run(sim, 3.0, tmp_path)
    norm = lambda s: float(sim.density(s).sum() * sim.dx)  # noqa: E731
    assert norm(s0) == pytest.approx(1.0) and norm(s1) == pytest.approx(1.0, abs=1e-9)  # unitary
    std = lambda s: math.sqrt(float((sim.density(s) * sim.x**2).sum() * sim.dx))  # noqa: E731
    assert std(s0) == pytest.approx(1.0, rel=1e-3)
    assert std(s1) == pytest.approx(1.0 * math.sqrt(1 + (3.0 / (2 * 1.0**2)) ** 2), rel=2e-3)


def test_a_moving_packet_travels_at_p_and_a_harmonic_well_returns_it(tmp_path):
    free = pm.Schrodinger1D(x0=-4.0, p0=2.0, n=512)
    s = run(free, 2.0, tmp_path)
    centre = float((free.density(s) * free.x).sum() * free.dx)
    assert centre == pytest.approx(-4.0 + 2.0 * 2.0, abs=0.05)  # x = x0 + p t
    well = pm.Schrodinger1D(potential="harmonic", x0=3.0, p0=0.0, n=256)
    back = run(well, 2 * math.pi, tmp_path)  # one classical period
    assert float((well.density(back) * well.x).sum() * well.dx) == pytest.approx(3.0, abs=0.1)
    with pytest.raises(ValueError, match="potential"):
        pm.Schrodinger1D(potential="lava")


# ---------------------------------------------------------------- Poisson / Laplace


def test_laplace_with_harmonic_boundary_values_reproduces_the_harmonic_function():
    n, h = 33, 1.0 / 32
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    exact = X**2 - Y**2  # harmonic: its laplacian is 0
    boundary = exact.copy()
    boundary[1:-1, 1:-1] = 0.0
    u = pm.solve_poisson(np.zeros((n, n)), h, boundary)
    assert np.abs(u - exact).max() < 1e-10  # the five-point stencil is exact for quadratics


def test_poisson_with_a_source_matches_the_known_solution():
    n, h = 41, 1.0 / 40
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
    f = -2 * math.pi**2 * exact
    u = pm.solve_poisson(f, h, np.zeros((n, n)))
    assert np.abs(u - exact).max() < 2e-3  # second order in h
    h2 = 1.0 / 80
    x2 = np.linspace(0, 1, 81)
    X2, Y2 = np.meshgrid(x2, x2, indexing="ij")
    e2 = np.sin(math.pi * X2) * np.sin(math.pi * Y2)
    u2 = pm.solve_poisson(-2 * math.pi**2 * e2, h2, np.zeros((81, 81)))
    assert np.abs(u2 - e2).max() < np.abs(u - exact).max() / 3.5  # halving h quarters the error
    with pytest.raises(ValueError, match="same shape"):
        pm.solve_poisson(np.zeros((5, 5)), 0.1, np.zeros((4, 4)))


# ---------------------------------------------------------------- the rest of the module table


def test_spectral_heat_is_exact_for_sine_modes_at_any_step():
    sim = pm.Heat1D(k=0.5, length=2.0, n=64, modes=(1.0, 0.3, 0.1), method="spectral", dt=0.05)
    s = sim.initial()
    for _ in range(40):
        s = sim.step(s, sim.dt)
    np.testing.assert_allclose(s["u"], sim.exact(40 * sim.dt), atol=1e-12)
    # explicit with that step would be unstable; spectral has no such limit
    with pytest.raises(ValueError, match="unstable"):
        pm.Heat1D(k=0.5, length=2.0, n=64, dt=0.05)


def test_insulated_ends_keep_the_heat_and_flatten_to_the_mean():
    for method in ("explicit", "spectral"):
        sim = pm.Heat1D(k=1.0, length=1.0, n=80, modes=(), boundary="neumann", method=method)
        s = sim.initial()
        total0 = np.trapezoid(s["u"], sim.x)
        steps = int(0.5 / sim.dt)
        for _ in range(steps):
            s = sim.step(s, sim.dt)
        assert np.trapezoid(s["u"], sim.x) == pytest.approx(total0, rel=1e-3), method
        assert np.ptp(s["u"]) < 1e-2  # nearly flat at the mean
    cos = pm.Heat1D(k=1.0, length=1.0, n=100, modes=(1.0,), boundary="neumann")
    s = cos.initial()
    for _ in range(200):
        s = cos.step(s, cos.dt)
    np.testing.assert_allclose(s["u"], cos.exact(200 * cos.dt), atol=3e-3)  # cos(pi x)


def test_mode_decomposition_recovers_the_modes_and_their_decay():
    sim = pm.Heat1D(k=0.5, length=2.0, n=200, modes=(1.0, 0.0, 0.4))
    b = pm.sine_modes(sim.x, sim.initial()["u"], 2.0, 5)
    np.testing.assert_allclose(b, [1.0, 0.0, 0.4, 0.0, 0.0], atol=1e-4)
    bump = pm.Heat1D(k=0.5, length=2.0, n=400, modes=())
    u = bump.initial()["u"]
    errs = [np.abs(pm.mode_sum(bump.x, pm.sine_modes(bump.x, u, 2.0, m), 2.0) - u).max()
            for m in (3, 9, 27)]  # fmt: skip
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-3  # partial sums converge
    a = pm.cosine_modes(bump.x, u, 2.0, 3)
    assert a[0] == pytest.approx(np.trapezoid(u, bump.x) / 2.0)  # a_0 is the mean


def test_heat3d_mode_decays_at_the_exact_rate_and_slices_match():
    sim = pm.Heat3D(k=0.2, size=1.0, n=16, mode=(1, 1, 1))
    s = sim.initial()
    steps = 60
    for _ in range(steps):
        s = sim.step(s, sim.dt)
    ex = sim.exact(steps * sim.dt)
    assert np.abs(s["u"] - ex).max() < 0.02 * np.abs(ex).max()
    sl = pm.slice3d(s["u"], "z", 0.5)
    assert sl.shape == (17, 17)
    np.testing.assert_allclose(sl, s["u"][:, :, 8])
    with pytest.raises(ValueError, match="unstable"):
        pm.Heat3D(k=1.0, size=1.0, n=16, dt=1.0)


def test_a_field_moves_between_solver_steps():
    """A solver step longer than a frame must not freeze the picture for several frames:
    between two steps the state is blended, and at a step it is that step exactly."""
    from phasor.domains import pde

    sim = pde._sim("wave2d", (("mode", (2, 3)), ("n", 64), ("size", (4.6, 4.6)), ("speed", 1.0)))
    t0 = 40 * sim.dt
    a, b = pde._state(sim, t0)["u"], pde._state(sim, t0 + sim.dt)["u"]
    mid = pde._state(sim, t0 + 0.25 * sim.dt)["u"]
    assert not np.array_equal(mid, a) and not np.array_equal(mid, b)
    assert np.allclose(mid, 0.75 * a + 0.25 * b)
    assert np.array_equal(pde._state(sim, t0)["u"], a)  # pure: same time, same state
