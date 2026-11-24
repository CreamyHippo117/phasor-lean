"""`Simulation` with fixed dt, a seed and cached checkpoints (contract 3)."""

from __future__ import annotations

import numpy as np
import pytest

from phasor.core.rng import rng
from phasor.core.simulation import SimRunner, runner_for, sim_key


class Walk:
    """A toy simulation: a seeded random walk plus a decaying oscillator."""

    def __init__(self, seed=0, dt=0.01, k=1.0):
        self.seed, self.dt, self.k = seed, dt, k

    def params(self):
        return {"seed": self.seed, "dt": self.dt, "k": self.k}

    def initial(self):
        g = rng("walk", self.seed)
        return {"x": g.normal(size=3), "v": np.zeros(3), "n": np.zeros(1)}

    def step(self, s, dt):
        g = rng("walk", self.seed + int(s["n"][0]) + 1)  # per-step stream: order independent
        v = s["v"] - self.k * s["x"] * dt - 0.1 * s["v"] * dt
        return {"x": s["x"] + v * dt + 1e-3 * g.normal(size=3), "v": v, "n": s["n"] + 1}


def direct(sim, steps):
    s = sim.initial()
    for _ in range(steps):
        s = sim.step(s, sim.dt)
    return s


def test_state_at_is_the_nearest_checkpoint_plus_remaining_steps(tmp_path):
    sim = Walk()
    r = SimRunner(sim, every=50, cache_dir=tmp_path)
    for t in (0.0, 0.37, 1.0, 2.5):
        want = direct(sim, round(t / sim.dt))
        got = r.state_at(t)
        assert all(np.array_equal(got[k], want[k]) for k in want), t
    assert r.state_at(0.37)["n"][0] == 37


def test_time_is_pure_whatever_order_you_ask_in(tmp_path):
    a = SimRunner(Walk(), every=40, cache_dir=tmp_path / "a")
    b = SimRunner(Walk(), every=40, cache_dir=tmp_path / "b")
    first = [a.state_at(t)["x"].copy() for t in (0.2, 1.5, 0.7, 3.0, 0.0)]
    second = [b.state_at(t)["x"].copy() for t in (3.0, 0.0, 0.7, 0.2, 1.5)]
    order = {0.2: 0, 1.5: 1, 0.7: 2, 3.0: 3, 0.0: 4}
    for t, i in order.items():
        j = [3.0, 0.0, 0.7, 0.2, 1.5].index(t)
        assert np.array_equal(first[i], second[j])


def test_scrubbing_backwards_restarts_from_a_checkpoint_not_from_zero(tmp_path):
    r = SimRunner(Walk(), every=100, cache_dir=tmp_path)
    r.state_at(5.0)  # 500 steps, checkpoints at 100..500
    before = r.steps_run
    r.state_at(4.2)  # 420: restart from checkpoint 400, 20 steps
    assert r.steps_run - before == 20
    r.state_at(4.2)
    assert r.steps_run - before == 20  # the same answer again costs nothing extra


def test_checkpoints_survive_on_disk_under_a_hash_of_the_parameters(tmp_path):
    sim = Walk(seed=3)
    SimRunner(sim, every=100, cache_dir=tmp_path).state_at(3.0)
    folder = tmp_path / sim_key(sim)
    assert sorted(p.name for p in folder.glob("*.npz")) == [
        "000100.npz",
        "000200.npz",
        "000300.npz",
    ]
    fresh = SimRunner(sim, every=100, cache_dir=tmp_path)  # a new process, in effect
    s = fresh.state_at(3.25)
    assert fresh.steps_run == 25  # read checkpoint 300 from disk, then 25 steps
    assert np.array_equal(s["x"], direct(sim, 325)["x"])
    other = Walk(seed=3, k=2.0)  # different physics: a different cache entry
    assert sim_key(other) != sim_key(sim)
    assert not (tmp_path / sim_key(other)).exists()


def test_a_damaged_checkpoint_is_recomputed_not_trusted(tmp_path):
    sim = Walk()
    SimRunner(sim, every=100, cache_dir=tmp_path).state_at(2.0)
    (tmp_path / sim_key(sim) / "000200.npz").write_bytes(b"not an archive")
    r = SimRunner(sim, every=100, cache_dir=tmp_path)
    assert np.array_equal(r.state_at(2.0)["x"], direct(sim, 200)["x"])


def test_results_are_read_only_and_dt_is_fixed(tmp_path):
    r = SimRunner(Walk(dt=0.02), every=10, cache_dir=tmp_path)
    s = r.state_at(0.1)
    assert s["n"][0] == 5 and not s["x"].flags.writeable
    with pytest.raises(ValueError):
        s["x"][0] = 1.0
    assert r.state_at(-3.0)["n"][0] == 0  # before the start: the initial state


def test_runner_for_shares_one_runner_per_simulation(tmp_path):
    a, b = runner_for(Walk(), cache_dir=tmp_path), runner_for(Walk(), cache_dir=tmp_path)
    assert a is b and runner_for(Walk(seed=9), cache_dir=tmp_path) is not a


def test_disk_cache_can_be_switched_off(tmp_path):
    r = SimRunner(Walk(), every=10, cache_dir=None)
    r.state_at(0.5)
    assert not list(tmp_path.iterdir())


def test_a_changed_solver_is_a_different_cache(monkeypatch):
    """Boundary review M7-M9: the key hashed parameters only, so after a solver bug fix the
    disk cache kept serving the old code's answers. It now hashes the solver's source too."""
    from phasor.core import simulation
    from phasor.domains import pde_math as pm

    sim = pm.Heat1D(n=20)
    before = sim_key(sim)
    monkeypatch.setattr(simulation, "_code", lambda cls: "patched solver")
    assert sim_key(sim) != before
