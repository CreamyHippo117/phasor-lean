"""Where a frame's time goes.

Any layer marks a stretch of work with ``with stage("draw"): ...``. Nothing is recorded
unless a ``Stopwatch`` is running (``with Stopwatch() as sw:``), so the marks cost one
attribute check in normal use. Times are *exclusive*: a stage nested in another (the
simulation inside a draw) is counted once, under the inner name.

GPU work is asynchronous: a stage that launches CUDA kernels passes ``sync`` (a function
that waits for the card), or its time would show up in whichever stage waits next.

One measured render at a time. Work hops between threads (drawing runs on the GL thread
while the caller waits), so the stack of open stages is shared, not per thread.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

_active: Stopwatch | None = None


class Stopwatch:
    """Collects exclusive seconds per stage name while it is running."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.wall = 0.0
        self._stack: list[list[Any]] = []  # [name, started, seconds spent in children]
        self._lock = threading.Lock()
        self._t0 = 0.0

    def __enter__(self) -> Stopwatch:
        global _active
        if _active is not None:
            raise RuntimeError("a Stopwatch is already running")
        self._t0 = time.perf_counter()
        _active = self
        return self

    def __exit__(self, *exc: object) -> None:
        global _active
        self.wall = time.perf_counter() - self._t0
        _active = None

    def _open(self, name: str) -> None:
        with self._lock:
            self._stack.append([name, time.perf_counter(), 0.0])

    def _close(self) -> None:
        with self._lock:
            name, started, children = self._stack.pop()
            spent = time.perf_counter() - started
            self.seconds[name] = self.seconds.get(name, 0.0) + spent - children
            self.calls[name] = self.calls.get(name, 0) + 1
            if self._stack:
                self._stack[-1][2] += spent

    def other(self) -> float:
        """Wall time no stage claimed (loop overhead, Python glue)."""
        return max(self.wall - sum(self.seconds.values()), 0.0)


@contextmanager
def _measured(sw: Stopwatch, name: str, sync: Callable[[], None] | None) -> Iterator[None]:
    sw._open(name)
    try:
        yield
        if sync is not None:
            sync()
    finally:
        sw._close()


class _Nothing:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


_NOTHING = _Nothing()


def stage(name: str, sync: Callable[[], None] | None = None):
    """A context manager timing ``name`` when a Stopwatch runs; otherwise it does nothing
    (and ``sync`` is not called, so normal renders never wait for the card)."""
    sw = _active
    if sw is None:
        return _NOTHING
    return _measured(sw, name, sync)
