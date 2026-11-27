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


FAMILY_STEP = 1e-5  # below this |a|, a family is extrapolated from a = step and 2 step


def family_at(f: Callable[..., np.ndarray], z: np.ndarray, a: float) -> np.ndarray:
    """``f(z, a)`` for a family of maps. Near ``a = 0`` it is extrapolated along a straight
    line from ``a = FAMILY_STEP`` and ``2 FAMILY_STEP`` (error about step^2), so a removable
    0/0 such as ``(exp(a z) - 1)/a`` gives its limit. Holomorphic in z wherever f is."""
    with np.errstate(all="ignore"):
        if abs(a) >= FAMILY_STEP:
            return np.asarray(f(z, a=a), dtype=np.complex128)
        h = FAMILY_STEP
        w1 = np.asarray(f(z, a=h), dtype=np.complex128)
        w2 = np.asarray(f(z, a=2 * h), dtype=np.complex128)
        return w1 + (a - h) / h * (w2 - w1)


def map_at(f: Callable[..., np.ndarray], z: np.ndarray, amount: float, path: str) -> np.ndarray:
    """The picture at ``amount``: ``"linear"`` is ``deform`` (a straight line from z to
    f(z)); ``"family"`` is ``family_at`` with ``a = amount``, a holomorphic map at every step."""
    if path == "family":
        return family_at(f, z, amount)
    return deform(f, z, amount)


def refine(g: Callable[[np.ndarray], np.ndarray], z: np.ndarray, w: np.ndarray, tol: float,
           limit: float, cap: int, rounds: int = 8) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """More points where the polyline ``w = g(z)`` strays from its curve: a midpoint whose
    image is farther than ``tol`` from its chord's middle goes in, round after round, until
    none do or there are ``cap`` points. Pieces ``split_runs`` would cut (not finite, past
    ``limit``, a jump of ``limit / 4``) are left alone."""
    for _ in range(rounds):
        if len(z) >= cap:
            break
        ok = np.isfinite(w) & (np.abs(w) <= limit)
        seg = ok[:-1] & ok[1:] & (np.abs(np.diff(w)) <= limit / 4)
        idx = np.flatnonzero(seg)
        if not len(idx):
            break
        zm = (z[idx] + z[idx + 1]) / 2
        with np.errstate(all="ignore"):
            wm = np.asarray(g(zm), dtype=np.complex128)
        far = np.abs(wm - (w[idx] + w[idx + 1]) / 2) > tol
        far &= np.isfinite(wm)
        if not far.any():
            break
        take = np.flatnonzero(far)[: cap - len(z)]
        z = np.insert(z, idx[take] + 1, zm[take])
        w = np.insert(w, idx[take] + 1, wm[take])
    return z, w
