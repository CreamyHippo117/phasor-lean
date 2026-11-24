"""Point sources for ``PointCloud`` (pure NumPy, no drawing).

Each source returns ``(points, value)``: points as complex numbers (x = Re, y = Im) and a
value in [0, 1) per point that the object turns into colour. Randomness only through the
generator handed in, which the object makes with ``rng(object_id, seed)`` (contract 3).
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sympy as sp
from numpy.random import Generator  # the type only; generators come from core.rng

from phasor.render.gpu.codegen import FunctionError, parse_expression

ROOT_CHUNK = 2048  # polynomials per batched eigenvalue solve (one thread's share)
# LAPACK lets go of the GIL, so the chunks solve side by side: a million complex roots
# (2^16 polynomials of degree 16) in about 0.5 s on 16 threads instead of 4.4 s on one
ROOT_THREADS = max(1, min(16, os.cpu_count() or 1))
ATTRACTORS = {  # kind -> the classic constants
    "clifford": (-1.4, 1.6, 1.0, 0.7),
    "dejong": (-2.0, -2.0, -1.2, 2.0),
    "lorenz": (10.0, 28.0, 8.0 / 3.0),
}
ORBITS = 2048  # orbits an attractor runs side by side
TRANSIENT = 64  # first iterates of each orbit, dropped (not yet on the attractor)
BUDDHA_BATCH = 1 << 16  # Buddhabrot: random c tried per pass


def polynomial_roots(
    degree: int,
    coefficients: tuple[float, ...] = (-1.0, 1.0),
    count: int | None = None,
    gen: Generator | None = None,
    twist: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The roots of polynomials of ``degree`` whose lower coefficients all come from
    ``coefficients``; the leading one is 1 if the set has 1, else its first nonzero entry
    (a sign flip of the whole polynomial changes no root). ``count``: how many polynomials,
    a random sample of them (with ``gen``); None or more than there are: all of them, in
    order. ``value`` is the polynomial's place in that list, so neighbours share a colour.

    ``twist`` turns the last entry of the set round the origin by that share of a full
    turn (b -> b e^(2 pi i twist)) in the lower coefficients; the leading one stays. The
    roots move continuously with it: the sample of polynomials does not depend on it, and
    the roots of polynomial k are always ``roots[k * degree : (k + 1) * degree]`` (in no
    particular order inside that run)."""
    if degree < 1:
        raise ValueError("degree must be at least 1")
    coeffs = np.asarray(coefficients, dtype=np.float64)
    if coeffs.ndim != 1 or len(coeffs) < 1:
        raise ValueError("coefficients must be a non-empty list of numbers")
    nonzero = coeffs[coeffs != 0]
    if not len(nonzero):
        raise ValueError("coefficients must include a nonzero number")
    lead = 1.0 if (coeffs == 1).any() else float(nonzero[0])
    k = len(coeffs)
    total = k**degree
    if count is None or count >= total:
        if total > 1 << 26:
            raise ValueError(f"{total} polynomials is too many; give a count to sample them")
        index = np.arange(total, dtype=np.int64)
    else:
        if gen is None:
            raise ValueError("sampling polynomials needs a random generator")
        if total > 1 << 62:
            raise ValueError(f"{total} polynomials is too many to sample from")
        index = np.sort(gen.choice(total, size=int(count), replace=False))
    if twist % 1.0:
        coeffs = coeffs.astype(np.complex128)
        coeffs[-1] *= np.exp(2j * np.pi * float(twist))

    def solve(start: int) -> np.ndarray:
        idx = index[start : start + ROOT_CHUNK]
        digits = np.empty((len(idx), degree), dtype=np.int64)  # base-k digits, low first
        rest = idx.copy()
        for d in range(degree):
            digits[:, d] = rest % k
            rest //= k
        low = coeffs[digits]  # (m, degree): a_0 .. a_{degree-1}
        companion = np.zeros((len(idx), degree, degree), dtype=coeffs.dtype)
        if degree > 1:
            companion[:, np.arange(1, degree), np.arange(degree - 1)] = 1.0
        companion[:, :, -1] = -low / lead
        return np.linalg.eigvals(companion).reshape(-1).astype(np.complex128)

    starts = range(0, len(index), ROOT_CHUNK)
    if len(starts) > 1 and ROOT_THREADS > 1:
        with ThreadPoolExecutor(ROOT_THREADS) as pool:
            roots = list(pool.map(solve, starts))
    else:
        roots = [solve(s) for s in starts]
    values = np.repeat(index.astype(np.float64) / total, degree)
    return np.concatenate(roots), values


def attractor(
    kind: str,
    n: int,
    gen: Generator,
    constants: tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points on a strange attractor, centred and scaled to fit the square
    [-1, 1] x [-1, 1]: ``clifford`` and ``dejong`` (2D maps), ``lorenz`` (the flow, seen
    from the side: x across, z up). Many orbits run side by side from random starts; their
    first iterates are dropped. ``value`` is how far each step jumped, as a rank (0-1), so
    the fast and slow parts of the attractor take different colours."""
    if kind not in ATTRACTORS:
        raise ValueError(f"kind must be one of {sorted(ATTRACTORS)}, got {kind!r}")
    c = tuple(constants) or ATTRACTORS[kind]
    if len(c) != len(ATTRACTORS[kind]):
        raise ValueError(f"{kind} takes {len(ATTRACTORS[kind])} constants, got {len(c)}")
    orbits = max(1, min(ORBITS, n))
    steps = -(-n // orbits)
    if kind == "lorenz":
        sigma, rho, beta = c
        p = gen.normal(0.0, 1.0, (orbits, 3)) + np.array([0.0, 0.0, 25.0])
        dt = 0.004

        def f(p):
            x, y, z = p[:, 0], p[:, 1], p[:, 2]
            return np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=1)

        def step(p):  # classic Runge-Kutta: a fixed dt
            k1 = f(p)
            k2 = f(p + 0.5 * dt * k1)
            k3 = f(p + 0.5 * dt * k2)
            k4 = f(p + dt * k3)
            return p + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        for _ in range(TRANSIENT * 8):
            p = step(p)
        out = np.empty((steps, orbits, 2))
        for s in range(steps):
            q = step(p)
            out[s] = q[:, [0, 2]]
            p = q
    else:
        a, b, cc, d = c
        x, y = gen.uniform(-1.0, 1.0, orbits), gen.uniform(-1.0, 1.0, orbits)

        def step(x, y):
            if kind == "clifford":
                return np.sin(a * y) + cc * np.cos(a * x), np.sin(b * x) + d * np.cos(b * y)
            return np.sin(a * y) - np.cos(b * x), np.sin(cc * x) - np.cos(d * y)

        for _ in range(TRANSIENT):
            x, y = step(x, y)
        out = np.empty((steps, orbits, 2))
        for s in range(steps):
            x, y = step(x, y)
            out[s, :, 0], out[s, :, 1] = x, y
    jump = np.zeros((steps, orbits))
    jump[1:] = np.hypot(*(out[1:] - out[:-1]).transpose(2, 0, 1))
    jump[0] = jump[1] if steps > 1 else 0.0
    pts = out.reshape(-1, 2)[:n]
    jump = jump.reshape(-1)[:n]
    lo, hi = pts.min(0), pts.max(0)
    mid, half = (lo + hi) / 2, max(float((hi - lo).max()) / 2, 1e-12)
    pts = (pts - mid) / half
    rank = np.empty(len(jump))
    rank[np.argsort(jump, kind="stable")] = np.arange(len(jump)) / max(len(jump), 1)
    return pts[:, 0] + 1j * pts[:, 1], rank


def buddhabrot(
    n: int,
    max_iter: int,
    gen: Generator,
    min_iter: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points of the Buddhabrot: the orbits z -> z^2 + c (from z = 0) of random c in
    the disc |c| < 2 that escape within ``max_iter`` steps (and not before ``min_iter``),
    every point of every such orbit. Points inside the main cardioid and the period-2 bulb
    never escape and are skipped at once. ``value`` is the orbit's escape time as
    log(time) / log(max_iter): short orbits and long ones take different colours, the
    "nebulabrot" colouring."""
    if max_iter < 2:
        raise ValueError("max_iter must be at least 2")
    pts: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    have = 0
    for _ in range(10_000):
        if have >= n:
            break
        r = 2.0 * np.sqrt(gen.random(BUDDHA_BATCH))
        c = r * np.exp(2j * np.pi * gen.random(BUDDHA_BATCH))
        q = (c.real - 0.25) ** 2 + c.imag**2
        inside = (q * (q + (c.real - 0.25)) <= 0.25 * c.imag**2) | (
            (c.real + 1) ** 2 + c.imag**2 <= 1 / 16
        )
        c = c[~inside]
        escaped = np.full(len(c), -1)
        # only the orbits still alive are stepped (most escape within a few steps), and
        # only their steps are kept, so memory follows the points made, not max_iter
        idx, z, ca = np.arange(len(c)), np.zeros_like(c), c
        steps_z: list[np.ndarray] = []
        steps_i: list[np.ndarray] = []
        for k in range(max_iter):
            z = z * z + ca
            steps_z.append(z)
            steps_i.append(idx)
            gone = z.real * z.real + z.imag * z.imag > 4.0
            if gone.any():
                escaped[idx[gone]] = k + 1
                stay = ~gone
                idx, z, ca = idx[stay], z[stay], ca[stay]
                if not len(idx):
                    break
        keep = escaped >= min_iter
        if not keep.any():
            continue
        zs, owner = np.concatenate(steps_z), np.concatenate(steps_i)
        mine = keep[owner]
        zs, owner = zs[mine], owner[mine]
        # steps were made in time order, orbits in index order within each: a stable sort
        # by orbit lays every orbit out whole, in order
        pts.append(zs[np.argsort(owner, kind="stable")])
        e = escaped[keep]
        vals.append(np.repeat(np.log(e) / np.log(max_iter), e))
        have += int(e.sum())
    if not pts:
        return np.zeros(0, dtype=np.complex128), np.zeros(0)
    return np.concatenate(pts)[:n], np.minimum(np.concatenate(vals)[:n], 0.999999)


def every_polynomial_count(degree: int, coefficients: tuple[float, ...]) -> int:
    """How many polynomials ``polynomial_roots`` enumerates (before any sampling)."""
    return len(coefficients) ** degree


# ---------------------------------------------------------------- bifurcation diagrams

MAPS = {  # named one-parameter maps x -> f(r, x)
    "logistic": "r*x*(1 - x)",
    "sine": "r*sin(pi*x)",
}
ESCAPE = 1e6  # an orbit past this size has left for infinity: its points are dropped
# colour: the Lyapunov exponent lambda of each r, squashed into [0, VALUE_SPAN) of the palette,
# so order (lambda < 0) and chaos (lambda > 0) take hues apart and never meet round the wheel
LYAPUNOV_SCALE = 0.25
VALUE_SPAN = 0.6


@functools.lru_cache(maxsize=32)
def map_function(map: str) -> tuple[Callable, Callable]:
    """``(f, df/dx)`` for a one-parameter map: a name from ``MAPS`` or a formula in ``r``
    and ``x`` (read by the whitelist parser, never evaluated as Python). Both take arrays
    ``(r, x)`` and broadcast them."""
    src = MAPS.get(map.strip(), map)
    expr = parse_expression(src, ("r", "x"))
    if expr.has(sp.I):
        raise FunctionError(f"map {map!r} is complex; a bifurcation map takes real r and x")
    r, x = sp.Symbol("r", real=True), sp.Symbol("x", real=True)

    def wrap(e: sp.Expr) -> Callable:
        fn = sp.lambdify((r, x), e, "numpy")

        def call(rr, xx):
            with np.errstate(all="ignore"):
                out = np.asarray(fn(rr, xx), dtype=np.float64)
            return np.broadcast_to(out, np.broadcast(rr, xx).shape)

        return call

    return wrap(expr), wrap(sp.diff(expr, x))


def orbits(map: str, r, n: int, x0: float = 0.5, start: int = 0) -> np.ndarray:
    """The orbit x_0 = ``x0``, x_(k+1) = f(r, x_k) for every value of ``r`` side by side:
    ``(n + 1 - start, len(r))``, row k is x_(start + k) (the iterates before ``start`` are
    run but not kept). An orbit that escapes (past ``ESCAPE``, or not a number) is nan from
    there on. Each column is computed on its own, element by element, so a column's orbit
    has the same bits whatever other columns run beside it."""
    if not 0 <= start <= n:
        raise ValueError(f"start must be 0 to n ({n}), got {start}")
    f, _ = map_function(map)
    r = np.ascontiguousarray(r, dtype=np.float64).reshape(-1)
    out = np.empty((n + 1 - start, len(r)))
    x = np.full(len(r), float(x0))
    if start == 0:
        out[0] = x
    for k in range(1, n + 1):
        x = f(r, x)
        with np.errstate(invalid="ignore"):
            x = np.where(np.abs(x) <= ESCAPE, x, np.nan)
        if k >= start:
            out[k - start] = x
    return out


def lyapunov(map: str, r, xs: np.ndarray) -> np.ndarray:
    """The Lyapunov exponent of each column: the mean of log |f'(r, x)| over the iterates
    ``xs`` (rows, one column per r). Negative where the orbit settles on a cycle, positive
    in chaos (log 2 for the logistic map at r = 4); nan where the orbit escaped."""
    _, df = map_function(map)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    with np.errstate(all="ignore"):
        d = np.abs(df(r[None, :], xs))
        return np.log(np.maximum(d, 1e-12)).mean(axis=0)


def lyapunov_value(lam: np.ndarray) -> np.ndarray:
    """A colour value in [0, ``VALUE_SPAN``) from a Lyapunov exponent: order low, chaos high,
    the edge of chaos (lambda = 0) in the middle."""
    return VALUE_SPAN * 0.5 * (1.0 + np.tanh(np.asarray(lam, dtype=np.float64) / LYAPUNOV_SCALE))


def bifurcation(
    map: str = "logistic",
    r0: float = 2.8,
    r1: float = 4.0,
    n_r: int = 1000,
    n_iter: int = 600,
    n_keep: int = 200,
    x0: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """The bifurcation diagram of x -> f(r, x) (``map``: ``logistic`` r x (1 - x), ``sine``
    r sin(pi x), or a formula in r and x): for ``n_r`` values of r from ``r0`` to ``r1`` the
    orbit of ``x0`` runs ``n_iter`` steps and its last ``n_keep`` iterates are kept. Returns
    ``(points, value)``: points r + i x (Re is r, Im is x), column by column in iterate order,
    escaped orbits dropped; value is the column's Lyapunov exponent as a colour
    (``lyapunov_value``), so cycles and chaos take different hues."""
    if not 1 <= n_keep <= n_iter:
        raise ValueError(f"n_keep must be 1 to n_iter ({n_iter}), got {n_keep}")
    if n_r < 1:
        raise ValueError("n_r must be at least 1")
    r = np.linspace(float(r0), float(r1), int(n_r))
    xs = orbits(map, r, n_iter, x0, start=n_iter - n_keep + 1)  # the last n_keep iterates
    value = np.broadcast_to(lyapunov_value(lyapunov(map, r, xs)), xs.shape)
    pts = (r[None, :] + 1j * xs).T.reshape(-1)
    value = value.T.reshape(-1)
    ok = np.isfinite(pts) & np.isfinite(value)
    return pts[ok], np.minimum(value[ok], VALUE_SPAN)


def zoom_window(outer: tuple[float, float], inner: tuple[float, float], s: float):
    """The window ``s`` of the way (0-1) on a zoom from ``outer`` to ``inner``: its width
    changes geometrically (the same zoom factor every step, so it never rushes at the end)
    and the point that both windows put at the same place stays there."""
    a0, b0 = _pair(outer)
    a1, b1 = _pair(inner)
    if s <= 0:
        return a0, b0
    if s >= 1:
        return a1, b1
    w0, w1 = b0 - a0, b1 - a1
    if abs(w0 - w1) <= 1e-12 * max(abs(w0), abs(w1)):  # the same size: slide across
        a = a0 + (a1 - a0) * s
        return a, a + w0
    p = (a1 * b0 - a0 * b1) / (w0 - w1)  # the fixed point of the map outer -> inner
    u = (p - a0) / w0  # its share across both windows
    w = w0 * (w1 / w0) ** s
    return p - u * w, p - u * w + w


def _pair(pair) -> tuple[float, float]:
    lo, hi = pair
    return float(lo), float(hi)
