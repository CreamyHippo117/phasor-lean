"""Calculus numerics: Riemann sums, Taylor
polynomials, epsilon-delta, vector fields with divergence and curl.

Functions are text (``"sin(x) / x"``, ``"x**2 - 1"``), read with the whitelist parser in
``render/gpu/codegen`` and never evaluated as Python.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy import integrate

from phasor.render.gpu.codegen import FunctionError, parse_expression

RULES = ("left", "right", "mid", "trapezoid", "upper", "lower")


@dataclass(frozen=True)
class RealFunction:
    """A function of one real variable ``x``."""

    source: str
    expr: sp.Expr

    @classmethod
    def parse(cls, source: str) -> RealFunction:
        return cls(source, parse_expression(source, ("x",)))

    @property
    def x(self) -> sp.Symbol:
        return sp.Symbol("x", real=True)

    def __call__(self, x) -> np.ndarray:
        """Values at real ``x``; NaN where the function is not real or not defined."""
        fn = sp.lambdify(self.x, self.expr, "numpy")
        with np.errstate(all="ignore"):
            w = np.asarray(fn(np.asarray(x, dtype=np.complex128)))
        w = np.broadcast_to(w, np.shape(x)).astype(np.complex128)
        return np.where(np.abs(w.imag) < 1e-12 * (1 + np.abs(w.real)), w.real, np.nan)

    def derivative(self, k: int = 1) -> RealFunction:
        return RealFunction(f"d{k}({self.source})", sp.diff(self.expr, self.x, k))

    def latex(self) -> str:
        return sp.latex(self.expr)


# ---------------------------------------------------------------- Riemann sums


def riemann_sum(f: RealFunction, a: float, b: float, n: int, rule: str = "left") -> dict:
    """The sum for ``n`` equal pieces of [a, b]: ``value`` and the ``rects`` as rows
    ``(x0, x1, height_at_left, height_at_right, drawn_height)`` (a trapezoid's two heights
    are the two sides). ``upper``/``lower`` take each piece's maximum/minimum."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; rules are {RULES}")
    if n < 1 or b <= a:
        raise ValueError("a sum needs n >= 1 pieces and b > a")
    xs = np.linspace(a, b, n + 1)
    h = (b - a) / n
    fl, fr = f(xs[:-1]), f(xs[1:])
    mid = f((xs[:-1] + xs[1:]) / 2)
    if rule == "left":
        drawn = fl
    elif rule == "right":
        drawn = fr
    elif rule == "mid":
        drawn = mid
    elif rule == "trapezoid":
        drawn = (fl + fr) / 2
    else:
        fine = np.array([f(np.linspace(xs[i], xs[i + 1], 41)) for i in range(n)])
        drawn = np.nanmax(fine, axis=1) if rule == "upper" else np.nanmin(fine, axis=1)
    return {
        "value": float(np.nansum(drawn) * h),
        "rects": np.stack([xs[:-1], xs[1:], fl, fr, drawn], axis=1),
    }


def exact_integral(f: RealFunction, a: float, b: float) -> float:
    value, _ = integrate.quad(lambda t: float(f(np.array(t))), a, b, limit=200)
    return float(value)


# ---------------------------------------------------------------- Taylor


def taylor_coefficients(f: RealFunction, center: float, order: int) -> list[float]:
    """``f^(k)(center) / k!`` for k = 0..order (exact derivatives, evaluated numerically)."""
    out = []
    for k in range(order + 1):
        d = sp.diff(f.expr, f.x, k)
        value = complex(sp.N(d.subs(f.x, center), 30))
        if abs(value.imag) > 1e-9 or not math.isfinite(value.real):
            raise ValueError(f"{f.source!r} has no finite real derivative of order {k} at {center}")
        out.append(value.real / math.factorial(k))
    return out


def taylor_polynomial(coeffs: list[float], center: float) -> Callable[[np.ndarray], np.ndarray]:
    def poly(x):
        t = np.asarray(x, dtype=np.float64) - center
        return sum((c * t**k for k, c in enumerate(coeffs)), np.zeros_like(t))

    return poly


# ---------------------------------------------------------------- epsilon-delta


def delta_for_epsilon(
    f: RealFunction, a: float, limit: float, epsilon: float, search: float = 4.0
) -> float:
    """The largest delta (up to ``search``) such that |f(x) - limit| < epsilon whenever
    0 < |x - a| < delta, found by scanning outward (the function may be undefined at ``a``)."""
    d = np.linspace(1e-6, search, 20001)
    left, right = f(a - d), f(a + d)
    ok = (np.abs(left - limit) < epsilon) & (np.abs(right - limit) < epsilon)
    bad = np.nonzero(~ok)[0]
    return (
        float(d[bad[0] - 1]) if len(bad) and bad[0] > 0 else (float(d[-1]) if not len(bad) else 0.0)
    )


def limit_at(f: RealFunction, a: float) -> float | None:
    """The two-sided limit of f at a, symbolically (SymPy), or None when the sides disagree
    or it is infinite."""
    x = f.x
    point = sp.nsimplify(a)
    left = sp.limit(f.expr, x, point, dir="-")
    right = sp.limit(f.expr, x, point, dir="+")
    if left != right or not left.is_finite:
        return None
    return float(left)


# ---------------------------------------------------------------- vector fields


@dataclass(frozen=True)
class VectorField:
    """A planar field F = (P(x, y), Q(x, y))."""

    p_source: str
    q_source: str
    p: sp.Expr
    q: sp.Expr

    @classmethod
    def parse(cls, p: str, q: str) -> VectorField:
        return cls(p, q, parse_expression(p, ("x", "y")), parse_expression(q, ("x", "y")))

    @property
    def _sym(self):
        return sp.Symbol("x", real=True), sp.Symbol("y", real=True)

    def __call__(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        X, Y = self._sym
        out = []
        for e in (self.p, self.q):
            fn = sp.lambdify((X, Y), e, "numpy")
            with np.errstate(all="ignore"):
                v = np.asarray(fn(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))
            out.append(np.broadcast_to(v, np.shape(x)).astype(np.float64))
        return out[0], out[1]

    def divergence(self) -> sp.Expr:
        X, Y = self._sym
        return sp.diff(self.p, X) + sp.diff(self.q, Y)

    def curl(self) -> sp.Expr:
        """The scalar curl dQ/dx - dP/dy (the z component)."""
        X, Y = self._sym
        return sp.diff(self.q, X) - sp.diff(self.p, Y)

    def evaluate(self, expr: sp.Expr, x: float, y: float) -> float:
        X, Y = self._sym
        return float(sp.N(expr.subs({X: x, Y: y})))

    def circulation(self, radius: float, center=(0.0, 0.0), n: int = 2000) -> float:
        """The line integral of F around a circle (counterclockwise): Green's theorem says it
        equals the integral of the curl over the disc."""
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        x, y = center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)
        p, q = self(x, y)
        dx, dy = -radius * np.sin(t), radius * np.cos(t)
        return float(np.sum(p * dx + q * dy) * (2 * math.pi / n))


def loop_integral(
    f: VectorField, center, radius: float, measure: str, *, upto: float = 1.0, n: int = 4096
) -> float:
    """The line integral of F around the circle, counterclockwise, over the first ``upto``
    share of the turn: ``circulation`` = integral of F . T ds (Green: = integral of the curl
    over the disc), ``flux`` = integral of F . n ds with n outward (= integral of the
    divergence). Midpoint rule in the angle, spectrally accurate for a whole turn."""
    if measure not in ("flux", "circulation"):
        raise ValueError(f"measure is flux or circulation, got {measure!r}")
    span = 2 * math.pi * float(np.clip(upto, 0.0, 1.0))
    if span == 0:
        return 0.0
    k = max(8, int(n * span / (2 * math.pi)))
    t = (np.arange(k) + 0.5) * span / k
    c, s = np.cos(t), np.sin(t)
    p, q = f(center[0] + radius * c, center[1] + radius * s)
    along = -p * s + q * c if measure == "circulation" else p * c + q * s
    return float(np.sum(along) * radius * span / k)


def disc_integral(f: VectorField, expr: sp.Expr, center, radius: float, n: int = 64) -> float:
    """The integral of a scalar expression in x, y over a disc (Gauss-Legendre in r, the
    trapezoid rule around), for checking Green's and the divergence theorem."""
    X, Y = f._sym
    fn = sp.lambdify((X, Y), expr, "numpy")
    r, w = np.polynomial.legendre.leggauss(n)
    r, w = (r + 1) * radius / 2, w * radius / 2
    t = np.arange(2 * n) * math.pi / n
    R, T = np.meshgrid(r, t, indexing="ij")
    vals = np.broadcast_to(fn(center[0] + R * np.cos(T), center[1] + R * np.sin(T)), R.shape)
    return float(np.sum(vals * R * w[:, None]) * math.pi / n)


__all__ = [
    "RULES",
    "FunctionError",
    "RealFunction",
    "VectorField",
    "delta_for_epsilon",
    "disc_integral",
    "exact_integral",
    "limit_at",
    "loop_integral",
    "riemann_sum",
    "taylor_coefficients",
    "taylor_polynomial",
]  # fmt: skip
