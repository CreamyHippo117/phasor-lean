"""Simulations (contract 3): fixed dt, a seed, cached checkpoints.

A ``Simulation`` has a fixed ``dt``, a ``params()`` dict (everything that decides its
answer, seed included), ``initial()`` and a pure ``step(state, dt)``; a state is a dict of
numpy arrays. ``SimRunner.state_at(t)`` is *the nearest earlier checkpoint plus the remaining
steps*, so it depends only on the simulation's parameters and ``t``, however you scrub.
Checkpoints are cached in memory and, by default, on disk under
``.phasor_cache/<hash of the parameters and the solver's source>/`` (override with
``PHASOR_CACHE``); a changed parameter or solver is a different hash, a damaged file is
recomputed rather than trusted.

Randomness inside a simulation must come from ``phasor.core.rng.rng`` (contract 3).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from phasor.core.timing import stage

State = dict[str, np.ndarray]
CACHE_VERSION = 1
DEFAULT_EVERY = 100  # steps between checkpoints
MEMORY = 6  # checkpoints kept in memory per runner


class Simulation(Protocol):
    dt: float

    def params(self) -> dict[str, Any]:
        """Everything that decides the result (seed included); must be JSON-friendly."""
        ...

    def initial(self) -> State: ...

    def step(self, state: State, dt: float) -> State:
        """One fixed step. Pure: returns new arrays, never changes ``state``."""
        ...


def _code(cls: type) -> str:
    """A hash of the source file that defines a simulation class: a fixed solver bug must
    not be answered from a cache written by the old code (boundary review M7-M9)."""
    try:
        path = inspect.getsourcefile(cls)
        if path:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except (OSError, TypeError):
        pass
    return "unknown"


def sim_key(sim: Simulation) -> str:
    """The cache name: a hash of the class, its module's source and every parameter."""
    blob = json.dumps(
        {"class": f"{type(sim).__module__}.{type(sim).__qualname__}", "version": CACHE_VERSION,
         "code": _code(type(sim)), "dt": sim.dt, "params": sim.params()},
        sort_keys=True, default=_plain,
    )  # fmt: skip
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _plain(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


def default_cache_dir() -> Path:
    return Path(os.environ.get("PHASOR_CACHE", ".phasor_cache"))


def _frozen(state: State) -> State:
    out = {}
    for k, v in state.items():
        a = np.array(v, copy=True)
        a.setflags(write=False)
        out[k] = a
    return out


class SimRunner:
    def __init__(
        self,
        sim: Simulation,
        *,
        every: int = DEFAULT_EVERY,
        cache_dir: str | Path | bool | None = "default",
    ) -> None:
        if sim.dt <= 0:
            raise ValueError("a simulation needs a positive dt")
        self.sim, self.every = sim, max(1, int(every))
        if cache_dir == "default":
            cache_dir = default_cache_dir()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.key = sim_key(sim)
        self.steps_run = 0  # steps actually computed by this runner (for tests and budgets)
        self._memory: OrderedDict[int, State] = OrderedDict()
        self._cursor: tuple[int, State] | None = None

    # ------------------------------------------------------------ checkpoints

    def _file(self, n: int) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / self.key / f"{n:06d}.npz"

    def _load(self, n: int) -> State | None:
        if n in self._memory:
            self._memory.move_to_end(n)
            return self._memory[n]
        path = self._file(n)
        if path is not None and path.exists():
            try:
                with np.load(path) as z:
                    state = {k: z[k] for k in z.files}
            except (OSError, ValueError, EOFError):
                return None  # damaged: recompute
            self._remember(n, state)
            return self._memory[n]
        return None

    def _remember(self, n: int, state: State) -> None:
        self._memory[n] = _frozen(state)
        self._memory.move_to_end(n)
        while len(self._memory) > MEMORY:
            self._memory.popitem(last=False)

    def _save(self, n: int, state: State) -> None:
        self._remember(n, state)
        path = self._file(n)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                # a name of its own: render workers may write the same checkpoint at once
                tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez(tmp, **state)
                tmp.replace(path)
            except OSError:
                pass  # a read-only disk costs speed, never correctness

    def _nearest(self, n: int) -> tuple[int, State]:
        """The latest checkpoint at or before step ``n`` that we can find."""
        best = max((k for k in self._memory if k <= n), default=0)
        if self.cache_dir is not None and (self.cache_dir / self.key).is_dir():
            on_disk = (int(p.stem) for p in (self.cache_dir / self.key).glob("*.npz")
                       if p.stem.isdigit() and int(p.stem) <= n)  # fmt: skip
            for k in sorted(on_disk, reverse=True):
                if k <= best:
                    break
                if self._load(k) is not None:
                    best = k
                    break
        if best and (state := self._load(best)) is not None:
            return best, state
        return 0, _frozen(self.sim.initial())

    # ------------------------------------------------------------ the one query

    def step_of(self, t: float) -> int:
        return max(0, round(float(t) / self.sim.dt))

    def state_at(self, t: float) -> State:
        """The state at time ``t`` (rounded to the nearest whole step). Read-only arrays."""
        n = self.step_of(t)
        start, state = self._nearest(n)
        if self._cursor is not None and start <= self._cursor[0] <= n:
            start, state = self._cursor  # carry on from the last answer (a playing timeline)
        if start == n:
            self._cursor = (n, state)
            return state
        cur = dict(state)
        with stage("simulation"):
            for k in range(start + 1, n + 1):
                cur = self.sim.step(cur, self.sim.dt)
                self.steps_run += 1
                if k % self.every == 0:
                    self._save(k, cur)
        frozen = _frozen(cur)
        self._cursor = (n, frozen)
        return frozen


_RUNNERS: OrderedDict[tuple, SimRunner] = OrderedDict()


def runner_for(
    sim: Simulation, *, every: int = DEFAULT_EVERY, cache_dir: str | Path | bool | None = "default"
) -> SimRunner:
    """One shared runner per (simulation parameters, cache folder), so frame after frame of a
    render continues the same run instead of starting over."""
    folder = default_cache_dir() if cache_dir == "default" else cache_dir
    key = (sim_key(sim), every, str(folder))
    if key not in _RUNNERS:
        _RUNNERS[key] = SimRunner(sim, every=every, cache_dir=cache_dir)
        while len(_RUNNERS) > 16:
            _RUNNERS.popitem(last=False)
    _RUNNERS.move_to_end(key)
    return _RUNNERS[key]
