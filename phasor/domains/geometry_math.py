"""Numerics for curves, surfaces and fields in 3D (no drawing): parametric curves and
surfaces from text formulas, coloring by what they output, and flow lines of a vector field.

Formulas are text read by the whitelist parser (``codegen.parse_expression``) and compiled
to NumPy with SymPy's lambdify; they are never evaluated as Python.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import numpy as np
import sympy as sp

from phasor.render.gpu.codegen import FunctionError, parse_expression

__all__ = [
    "FunctionError",
    "area_of_grid",
    "compile_fn",
    "curl",
    "divergence",
    "flow_lines",
    "normalize01",
    "particles",
    "sample_ramp",
    "surface_grid",
    "tick_values",
]


@functools.lru_cache(maxsize=256)
def _expr(source: str, names: tuple[str, ...]) -> sp.Expr:
    return parse_expression(source, names)


@functools.lru_cache(maxsize=256)
def compile_fn(source: str, names: tuple[str, ...]) -> Callable[..., np.ndarray]:
    """``f(*arrays)`` for a formula in ``names``; scalars broadcast to the input shape.
    Compiled once per (formula, names): SymPy's lambdify is slow, and draws ask again and
    again for the same few formulas."""
    expr = _expr(source, names)
    syms = [sp.Symbol(n, real=True) for n in names]
    fn = sp.lambdify(syms, expr, "numpy")

    def call(*args):
        shape = np.broadcast(*[np.asarray(a) for a in args]).shape
        with np.errstate(all="ignore"):
            out = np.asarray(fn(*[np.asarray(a, dtype=np.float64) for a in args]), dtype=np.float64)
        return np.broadcast_to(out, shape).astype(np.float64)

    return call


def check(source: str, names: tuple[str, ...]) -> str:
    """Validate a formula for a pydantic validator (raises ``ValueError`` naming the problem)."""
    try:
        _expr(source, names)
    except FunctionError as e:
        raise ValueError(str(e)) from None
    return source


def normalize01(v: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Values scaled to 0..1 between ``lo`` and ``hi`` (default: their own range; a constant
    field maps to 0.5)."""
    v = np.asarray(v, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.full(v.shape, 0.5)
    a = float(finite.min()) if lo is None else lo
    b = float(finite.max()) if hi is None else hi
    if b - a < 1e-12:
        return np.full(v.shape, 0.5)
    return np.clip(np.nan_to_num((v - a) / (b - a), nan=0.5), 0.0, 1.0)


def sample_ramp(lut: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Colors (…, 3) from a ramp at positions ``u`` in 0..1 (linear between entries)."""
    lut = np.asarray(lut, dtype=np.float64)
    g = np.clip(np.asarray(u, dtype=np.float64), 0, 1) * (len(lut) - 1)
    k0 = np.floor(g).astype(int)
    k1 = np.minimum(k0 + 1, len(lut) - 1)
    f = (g - k0)[..., None]
    return lut[k0] * (1 - f) + lut[k1] * f


def tick_values(lo: float, hi: float, step: float) -> np.ndarray:
    """Multiples of ``step`` in [lo, hi], zero left out (the axes cross there)."""
    if step <= 0:
        raise ValueError("a tick step must be positive")
    k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    return np.array([k * step for k in range(k0, k1 + 1) if k != 0], dtype=np.float64)


# ---------------------------------------------------------------- surfaces


def surface_grid(fx, fy, fz, u_range, v_range, res):
    """Points (nu, nv, 3) of a parametric surface and the (u, v) grids they came from."""
    u = np.linspace(u_range[0], u_range[1], res[0])
    v = np.linspace(v_range[0], v_range[1], res[1])
    U, V = np.meshgrid(u, v, indexing="ij")
    return np.stack([fx(U, V), fy(U, V), fz(U, V)], axis=-1), U, V


def area_of_grid(points: np.ndarray) -> float:
    """Area of the quad mesh through a (nu, nv, 3) grid (two triangles per cell)."""
    p = points
    a, b, c, d = p[:-1, :-1], p[1:, :-1], p[:-1, 1:], p[1:, 1:]
    t1 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)
    t2 = 0.5 * np.linalg.norm(np.cross(b - d, c - d), axis=-1)
    return float(np.nansum(t1 + t2))


# ---------------------------------------------------------------- vector fields


def _field_syms() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    return tuple(sp.Symbol(n, real=True) for n in "xyz")  # type: ignore[return-value]


def divergence(p: str, q: str, r: str) -> sp.Expr:
    x, y, z = _field_syms()
    P, Q, R = (_expr(s, ("x", "y", "z")) for s in (p, q, r))
    return sp.simplify(sp.diff(P, x) + sp.diff(Q, y) + sp.diff(R, z))


def curl(p: str, q: str, r: str) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    x, y, z = _field_syms()
    P, Q, R = (_expr(s, ("x", "y", "z")) for s in (p, q, r))
    return (sp.simplify(sp.diff(R, y) - sp.diff(Q, z)),
            sp.simplify(sp.diff(P, z) - sp.diff(R, x)),
            sp.simplify(sp.diff(Q, x) - sp.diff(P, y)))  # fmt: skip


def flow_lines(field, seeds: np.ndarray, steps: int, h: float, both: bool = False):
    """Flow lines of ``field(points (N, 3)) -> (N, 3)`` from ``seeds`` (N, 3) by RK4 with
    step ``h`` (arc length per step: the field is normalized), ``steps`` per direction.
    Returns (lines (N, M, 3), speed (N, M)): where the field vanishes a line stops (its
    remaining points repeat the last one). ``both`` also integrates backward."""

    def unit(p):
        v = np.asarray(field(p), dtype=np.float64)
        s = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.where(s > 1e-9, v / np.maximum(s, 1e-12), 0.0), s[..., 0]

    def run(direction: float):
        p = seeds.astype(np.float64).copy()
        pts, spd = [p.copy()], [unit(p)[1]]
        for _ in range(steps):
            k1, s1 = unit(p)
            k2, _ = unit(p + 0.5 * h * direction * k1)
            k3, _ = unit(p + 0.5 * h * direction * k2)
            k4, _ = unit(p + h * direction * k3)
            p = p + h * direction * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            pts.append(p.copy())
            spd.append(s1)
        return np.stack(pts, axis=1), np.stack(spd, axis=1)

    fwd, sf = run(1.0)
    if not both:
        return fwd, sf
    back, sb = run(-1.0)
    return np.concatenate([back[:, :0:-1], fwd], axis=1), np.concatenate([sb[:, :0:-1], sf], axis=1)


def particles(paths: np.ndarray, phase: np.ndarray, t: float, lifetime: float,
              tail: float) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Where particles are at time ``t`` along precomputed ``paths`` (N, M, 3): each lives
    ``lifetime`` seconds, born again at its start, ``phase`` (N,) apart. Returns (segments
    (N, K, 3), alpha (N, K)) of each particle's tail, ``tail`` (a share of its life) long:
    faint at the end, bright at the head, fading in after birth and out before death.
    A pure function of t."""
    n, m, _ = paths.shape
    age = np.mod(t / lifetime + phase, 1.0)  # (N,) share of life lived
    k = 8  # points per tail
    share = age[:, None] - tail * np.linspace(1.0, 0.0, k)[None, :]  # (N, K), oldest first
    share = np.clip(share, 0.0, None)
    idx = share * (m - 1)
    i0 = np.clip(np.floor(idx).astype(int), 0, m - 2)
    f = (idx - i0)[..., None]
    rows = np.arange(n)[:, None]
    seg = paths[rows, i0] * (1 - f) + paths[rows, i0 + 1] * f
    life = np.minimum(age / 0.15, (1 - age) / 0.15).clip(0.0, 1.0)  # fade in and out
    alpha = np.linspace(0.0, 1.0, k)[None, :] ** 1.5 * life[:, None]
    return seg, alpha
