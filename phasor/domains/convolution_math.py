"""Convolution numerics: a signal is a function of ``t`` that is zero outside
a support, either a named shape or a formula (read with the whitelist parser, never run as
Python). The convolution (f * g)(tau) = integral of f(s) g(tau - s) ds is summed on a fine grid,
so the picture and the numbers agree. Pure NumPy and SymPy, no drawing."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import sympy as sp

from phasor.render.gpu.codegen import FunctionError, parse_expression

__all__ = ["NAMED", "FunctionError", "Signal", "convolution", "edge_samples", "product_at",
           "shift_range"]  # fmt: skip

GRID = 600  # points over f's support in every sum

# name -> (formula in t, support). The formula is only read inside its support.
NAMED: dict[str, tuple[str, tuple[float, float]]] = {
    "box": ("1", (-1.0, 1.0)),
    "triangle": ("1 - abs(t)", (-1.0, 1.0)),
    "decay": ("exp(-1.5 * t)", (0.0, 3.0)),
    "ramp": ("t / 2", (0.0, 2.0)),
    "gauss": ("exp(-2 * t**2)", (-2.0, 2.0)),
    "wave": ("sin(3 * t)", (-1.0, 1.0)),
}  # fmt: skip


@functools.lru_cache(maxsize=128)
def _compiled(expr: sp.Expr):
    return sp.lambdify(sp.Symbol("t", real=True), expr, "numpy")


def edge_samples(support: tuple[float, float], lo: float, hi: float, n: int) -> np.ndarray:
    """``n`` times over lo..hi plus each end of ``support`` twice (just outside, and at the
    edge), so a signal that jumps is drawn with its vertical sides."""
    eps = (hi - lo) * 1e-4
    ends = [support[0] - eps, support[0], support[1], support[1] + eps]
    t = np.concatenate([np.linspace(lo, hi, n), [e for e in ends if lo <= e <= hi]])
    return np.unique(t)


@dataclass(frozen=True)
class Signal:
    """A function of ``t`` that is zero outside ``support``."""

    source: str
    support: tuple[float, float]
    expr: sp.Expr

    @classmethod
    def parse(cls, spec: str, support: tuple[float, float] | None = None) -> Signal:
        """``spec`` is a name in ``NAMED`` or a formula in ``t``; ``support`` overrides the
        named shape's, and a formula without one gets -1..1."""
        spec = spec.strip()
        src, sup = NAMED.get(spec, (spec, (-1.0, 1.0)))
        lo, hi = (float(v) for v in (support or sup))
        if not lo < hi:
            raise FunctionError(f"the support ({lo}, {hi}) must run from low to high")
        e = parse_expression(src, ("t",))
        if e.free_symbols - {sp.Symbol("t", real=True)}:
            raise FunctionError(f"{spec!r} may use only t")
        return cls(spec, (lo, hi), e)

    def __call__(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        fn = _compiled(self.expr)
        with np.errstate(all="ignore"):
            v = np.broadcast_to(np.asarray(fn(t), dtype=np.float64), t.shape)
        inside = (t >= self.support[0]) & (t <= self.support[1])
        return np.where(inside & np.isfinite(v), v, 0.0)


def shift_range(f: Signal, g: Signal) -> tuple[float, float]:
    """The shifts tau at which f and g can overlap: the output lives here."""
    return (f.support[0] + g.support[0], f.support[1] + g.support[1])


def product_at(f: Signal, g: Signal, tau: float) -> dict[str, np.ndarray | float]:
    """What the picture shows at shift ``tau``: the grid ``s`` over f's support, f(s), the
    flipped and shifted kernel g(tau - s), their product, and its area (the output)."""
    s = np.linspace(f.support[0], f.support[1], GRID)
    fs, gs = f(s), g(float(tau) - s)
    prod = fs * gs
    return {"s": s, "f": fs, "g": gs, "product": prod, "area": float(np.trapezoid(prod, s))}


def convolution(f: Signal, g: Signal, samples: int = 241) -> tuple[np.ndarray, np.ndarray]:
    """``(tau, (f * g)(tau))`` over the whole range of shifts, by the same sum as
    ``product_at``."""
    lo, hi = shift_range(f, g)
    tau = np.linspace(lo, hi, samples)
    s = np.linspace(f.support[0], f.support[1], GRID)
    prod = f(s)[None, :] * g(tau[:, None] - s[None, :])
    return tau, np.trapezoid(prod, s, axis=1)
