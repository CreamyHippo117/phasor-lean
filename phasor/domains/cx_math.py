"""Numerics for pictures of complex maps (no drawing): a grid pushed through z -> f(z)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def grid_lines(center: complex, extent: tuple[float, float], lines: tuple[int, int],
               samples: int) -> list[np.ndarray]:  # fmt: skip
    """The lines of a rectangular grid as complex polylines: ``lines[0]`` vertical lines and
    ``lines[1]`` horizontal ones, each with ``samples`` points, centred on ``center``."""
    w, h = extent
    xs = np.linspace(-w / 2, w / 2, lines[0])
    ys = np.linspace(-h / 2, h / 2, lines[1])
    tx = np.linspace(-w / 2, w / 2, samples)
    ty = np.linspace(-h / 2, h / 2, samples)
    out = [center + x + 1j * ty for x in xs]
    out += [center + tx + 1j * y for y in ys]
    return out


def deform(f: Callable[[np.ndarray], np.ndarray], z: np.ndarray, amount: float) -> np.ndarray:
    """``(1 - amount) z + amount f(z)``: the straight-line path from the identity to f."""
    if amount <= 0:
        return z
    with np.errstate(all="ignore"):
        w = np.asarray(f(z), dtype=np.complex128)
    return (1 - amount) * z + amount * w


def split_runs(w: np.ndarray, limit: float) -> list[np.ndarray]:
    """A complex polyline cut where it is not finite, leaves ``|w| <= limit``, or jumps
    farther than a quarter of ``limit`` between neighbours (a pole or a branch cut)."""
    ok = np.isfinite(w) & (np.abs(w) <= limit)
    runs: list[np.ndarray] = []
    cur: list[int] = []

    def flush() -> None:
        if len(cur) >= 2:
            runs.append(w[cur])
        cur.clear()

    for i in range(len(w)):
        if not ok[i]:
            flush()
            continue
        if cur and abs(w[i] - w[cur[-1]]) > limit / 4:
            flush()
        cur.append(i)
    flush()
    return runs
