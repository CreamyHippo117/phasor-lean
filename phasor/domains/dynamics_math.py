"""Dynamical-systems numerics: a planar system x' = f(x, y), y' = g(x, y) from formulas,
trajectories (RK4), nullclines, fixed points (Newton) and their classification from the
Jacobian's eigenvalues. Pure NumPy and SymPy, no drawing.

Formulas are read with the whitelist parser (never evaluated as Python). Named constants
(``constants={"mu": 1.5}``) may appear in them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import sympy as sp

from phasor.core.rng import rng
from phasor.core.simulation import State
from phasor.domains.linalg_math import expm
from phasor.render.gpu.codegen import FunctionError, parse_expression

__all__ = [
    "KINDS",
    "AttractorSimulation",
    "FunctionError",
    "System",
    "classify",
    "fixed_points",
    "linear_paths",
    "nullcline_segments",
    "trajectories",
]  # fmt: skip

KINDS = ("stable node", "unstable node", "saddle", "stable spiral", "unstable spiral", "centre",
         "degenerate", "non-hyperbolic")  # fmt: skip
_EPS = 1e-9


@dataclass(frozen=True)
class System:
    """x' = f(x, y), y' = g(x, y)."""

    f_source: str
    g_source: str
    f: sp.Expr
    g: sp.Expr

    @classmethod
    def parse(cls, f: str, g: str, constants: dict[str, float] | str | None = None) -> System:
        consts = parse_constants(constants) if isinstance(constants, str) else dict(constants or {})
        names = ("x", "y", *consts)
        out = []
        for src in (f, g):
            e = parse_expression(src, names)
            e = e.subs({sp.Symbol(k, real=True): float(v) for k, v in consts.items()})
            left = e.free_symbols - set(_xy())
            if left:
                raise FunctionError(f"{src!r} uses {sorted(map(str, left))}: not x or y")
            out.append(e)
        return cls(f, g, out[0], out[1])

    def _fn(self):
        x, y = _xy()
        fns = [sp.lambdify((x, y), e, "numpy") for e in (self.f, self.g)]

        def call(px, py):
            px = np.asarray(px, dtype=np.float64)
            py = np.asarray(py, dtype=np.float64)
            with np.errstate(all="ignore"):
                return tuple(np.broadcast_to(np.asarray(fn(px, py), dtype=np.float64), px.shape)
                             for fn in fns)  # fmt: skip

        return call

    def __call__(self, x, y):
        return self._fn()(x, y)

    def jacobian_at(self, x: float, y: float) -> np.ndarray:
        sx, sy = _xy()
        jac = sp.Matrix([self.f, self.g]).jacobian([sx, sy])
        return np.array(sp.N(jac.subs({sx: x, sy: y})), dtype=np.float64)

    def jacobian(self):
        """A function ``(x, y) -> 2x2 matrix`` (evaluated, vectorised over scalars)."""
        sx, sy = _xy()
        jac = sp.Matrix([self.f, self.g]).jacobian([sx, sy])
        fn = sp.lambdify((sx, sy), jac, "numpy")
        return lambda x, y: np.array(fn(float(x), float(y)), dtype=np.float64)


def parse_constants(text: str) -> dict[str, float]:
    """``"mu=1.5, k=2"`` as ``{"mu": 1.5, "k": 2.0}`` (commas or semicolons between)."""
    out: dict[str, float] = {}
    for part in text.replace(";", ",").split(","):
        if not part.strip():
            continue
        name, eq, value = part.partition("=")
        name = name.strip()
        if not eq or not name.isidentifier() or name in ("x", "y"):
            raise FunctionError(f"cannot read the constant {part.strip()!r}: write name=number")
        try:
            out[name] = float(value)
        except ValueError:
            raise FunctionError(
                f"cannot read the constant {part.strip()!r}: write name=number"
            ) from None
    return out


def _xy():
    return sp.Symbol("x", real=True), sp.Symbol("y", real=True)


# ------------------------------------------------------------------ trajectories


def trajectories(system: System, seeds, t_end: float, steps: int = 400,
                 bounds=(-1e3, 1e3, -1e3, 1e3), direction: float = 1.0) -> np.ndarray:  # fmt: skip
    """RK4 paths from each seed: an array (S, steps + 1, 2). A path that leaves ``bounds``
    (or blows up) stays NaN from there on, so a drawing can cut it. ``direction=-1`` runs
    time backwards (toward where the flow came from)."""
    fn = system._fn()
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 2)
    h = direction * t_end / steps
    out = np.full((len(seeds), steps + 1, 2), np.nan)
    p = seeds.copy()
    out[:, 0] = p
    x0, x1, y0, y1 = bounds

    def vel(q):
        a, b = fn(q[:, 0], q[:, 1])
        return np.stack([a, b], axis=1)

    alive = np.ones(len(seeds), dtype=bool)
    for k in range(steps):
        q = _rk4(vel, p, h)
        alive &= np.isfinite(q).all(axis=1)
        alive &= (q[:, 0] >= x0) & (q[:, 0] <= x1) & (q[:, 1] >= y0) & (q[:, 1] <= y1)
        p = np.where(alive[:, None], q, 0.0)  # a dead path keeps finite arithmetic, drawn NaN
        out[alive, k + 1] = p[alive]
        if not alive.any():
            break
    return out


# ------------------------------------------------------------------ nullclines


def nullcline_segments(system: System, which: str, bounds, n: int = 120) -> np.ndarray:
    """Line segments (M, 2, 2) tracing f = 0 (``which="x"``: where x does not change) or
    g = 0 (``"y"``) inside ``bounds = (x0, x1, y0, y1)``: marching squares on an n x n grid,
    crossings placed by linear interpolation."""
    x0, x1, y0, y1 = bounds
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    a, b = system(X, Y)
    z = a if which == "x" else b
    z = np.where(np.isfinite(z), z, np.nan)
    segs = []
    # cell corners: 00 (i,j), 10 (i+1,j), 11 (i+1,j+1), 01 (i,j+1)
    for i in range(n - 1):
        for j in range(n - 1):
            v = (z[i, j], z[i + 1, j], z[i + 1, j + 1], z[i, j + 1])
            if not np.isfinite(v).all():
                continue
            pts_xy = ((xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]),
                      (xs[i], ys[j + 1]))  # fmt: skip
            cross = []
            for e in range(4):
                va, vb = v[e], v[(e + 1) % 4]
                if (va < 0) != (vb < 0):
                    t = va / (va - vb)
                    pa, pb = pts_xy[e], pts_xy[(e + 1) % 4]
                    cross.append((pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])))
            if len(cross) == 2:
                segs.append(cross)
            elif len(cross) == 4:  # a saddle cell: join by the centre value's sign
                segs.append([cross[0], cross[1]])
                segs.append([cross[2], cross[3]])
    return np.array(segs, dtype=np.float64).reshape(-1, 2, 2)


# ------------------------------------------------------------------ fixed points


def fixed_points(system: System, bounds, grid: int = 9, tol: float = 1e-9) -> np.ndarray:
    """Points inside ``bounds`` where f = g = 0, by Newton's method from a grid of starts,
    sorted by x then y. Returns (K, 2)."""
    x0, x1, y0, y1 = bounds
    jac = system.jacobian()
    fn = system._fn()
    found: list[np.ndarray] = []
    pad_x, pad_y = (x1 - x0) * 1e-6, (y1 - y0) * 1e-6
    for sx in np.linspace(x0, x1, grid):
        for sy in np.linspace(y0, y1, grid):
            p = np.array([sx, sy], dtype=np.float64)
            ok = False
            for _ in range(60):
                a, b = fn(p[0], p[1])
                r = np.array([float(a), float(b)])
                if not np.isfinite(r).all():
                    break
                if np.abs(r).max() < tol:
                    ok = True
                    break
                try:
                    step = np.linalg.solve(jac(*p), r)
                except np.linalg.LinAlgError:
                    break
                if not np.isfinite(step).all():
                    break
                n = float(np.linalg.norm(step))
                cap = max(x1 - x0, y1 - y0)
                if n > cap:  # a long jump: damp it
                    step *= cap / n
                p = p - step
            if not ok:
                continue
            if not (x0 - pad_x <= p[0] <= x1 + pad_x and y0 - pad_y <= p[1] <= y1 + pad_y):
                continue
            if not any(np.linalg.norm(p - q) < 1e-6 * max(1.0, np.linalg.norm(q)) for q in found):
                found.append(p)
    pts = np.array(found, dtype=np.float64).reshape(-1, 2)
    if len(pts):
        pts = pts[np.lexsort((np.round(pts[:, 1], 7), np.round(pts[:, 0], 7)))]
    return pts


def classify(J) -> dict:
    """The type of a fixed point from its Jacobian: ``kind`` (one of ``KINDS``), ``stable``
    (all real parts negative: True; some positive: False; centre: None), the eigenvalues as
    (re, im) pairs, trace, determinant, and real eigen-directions (unit vectors) with their
    eigenvalues."""
    J = np.asarray(J, dtype=np.float64)
    tr, det = float(np.trace(J)), float(np.linalg.det(J))
    vals, vecs = np.linalg.eig(J)
    scale = max(float(np.abs(J).max()), 1.0)
    tol = 1e-7 * scale
    re, im = vals.real, vals.imag
    if np.abs(vals).min() < tol:
        kind, stable = "non-hyperbolic", None
    elif abs(im[0]) > tol:  # a complex pair
        if abs(re[0]) <= tol:
            kind, stable = "centre", None
        else:
            kind = "stable spiral" if re[0] < 0 else "unstable spiral"
            stable = bool(re[0] < 0)
    elif re[0] * re[1] < 0:
        kind, stable = "saddle", False
    else:
        stable = bool(re.max() < 0)
        defective = abs(re[0] - re[1]) < tol and np.linalg.matrix_rank(J - re[0] * np.eye(2),
                                                                      tol=tol) > 0  # fmt: skip
        kind = "degenerate" if defective else ("stable node" if stable else "unstable node")
    real_dirs = []
    for k in range(2):
        if abs(im[k]) <= tol:
            v = vecs[:, k].real
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            if v[0] < -1e-12 or (abs(v[0]) <= 1e-12 and v[1] < 0):
                v = -v
            real_dirs.append((float(re[k]), v))
    if kind == "degenerate" and len(real_dirs) == 2:  # one direction, found twice
        real_dirs = real_dirs[:1]
    return {
        "kind": kind,
        "stable": stable,
        "eigenvalues": [(float(a), float(b)) for a, b in zip(re, im, strict=True)],
        "trace": tr,
        "det": det,
        "directions": real_dirs,
    }


def eigenvalue_label(c: dict) -> str:
    """A short label: ``saddle (1.5, -2)`` or ``stable spiral (-0.5 +/- 1.2i)``: the
    eigenvalues, in plain characters every font draws."""
    (a, b), (c2, _) = c["eigenvalues"]
    if abs(b) > 1e-9:
        return f"{c['kind']} ({a:.3g} +/- {abs(b):.3g}i)"
    return f"{c['kind']} ({a:.3g}, {c2:.3g})"


def hash_seeds(n: int, extent, seed: int = 0) -> np.ndarray:
    """``n`` start points spread evenly over the window (a grid with a small fixed wobble)."""
    w, h = extent
    cols = max(1, round(math.sqrt(n * w / h)))
    rows = max(1, math.ceil(n / cols))
    gx = (np.arange(cols) + 0.5) / cols
    gy = (np.arange(rows) + 0.5) / rows
    X, Y = np.meshgrid(gx, gy)
    pts = np.c_[X.ravel(), Y.ravel()][:n]
    # a small, fixed wobble (golden-ratio sequence, shifted by ``seed``): no two seeds line up,
    # and no random numbers are needed
    k = np.arange(len(pts)) + 1 + seed
    wobble = np.c_[(k * 0.6180339887) % 1.0, (k * 0.7548776662) % 1.0] - 0.5
    pts = pts + wobble * np.array([0.6 / cols, 0.6 / rows])
    return (pts - 0.5) * np.array([w, h])


# ---------------------------------------------------------------- linear systems (D5)
# (the matrix exponential, ``expm``, lives in linalg_math)


def linear_paths(A, seeds, t_end: float, steps: int, bounds, direction: float = 1.0) -> np.ndarray:
    """Exact trajectories ``e^(tA) x0`` of x' = A x, shape (seeds, steps + 1, 2); a point
    outside ``bounds`` (x0, x1, y0, y1), and every later one, is NaN."""
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 2)
    ts = np.linspace(0.0, t_end * direction, steps + 1)
    paths = np.stack([seeds @ expm(A, t).T for t in ts], axis=1)
    x0, x1, y0, y1 = bounds
    out = (paths[..., 0] < x0) | (paths[..., 0] > x1) | (paths[..., 1] < y0) | (paths[..., 1] > y1)
    out = np.logical_or.accumulate(out, axis=1)
    paths[out] = np.nan
    return paths


def _rk4(velocity, point: np.ndarray, dt: float) -> np.ndarray:
    k1 = velocity(point)
    k2 = velocity(point + dt / 2 * k1)
    k3 = velocity(point + dt / 2 * k2)
    k4 = velocity(point + dt * k3)
    return point + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


AttractorSystem = Literal["lorenz", "rossler", "thomas"]


@dataclass(frozen=True)
class AttractorSimulation:
    """Fixed-step RK4 for the Lorenz, Rössler and Thomas differential equations."""

    system: AttractorSystem = "lorenz"
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8 / 3
    seed: int = 0
    object_id: str = "attractor"
    start: tuple[float, float, float] | None = None

    dt = 0.01

    def params(self) -> dict:
        return asdict(self)

    def initial(self) -> State:
        point = (
            rng(self.object_id, self.seed).uniform(0.5, 1.5, size=3)
            if self.start is None
            else np.asarray(self.start, dtype=np.float64)
        )
        return {"point": point}

    def velocity(self, point: np.ndarray) -> np.ndarray:
        x, y, z = point
        match self.system:
            case "lorenz":
                return np.array([self.sigma * (y - x), x * (self.rho - z) - y,
                                 x * y - self.beta * z])  # fmt: skip
            case "rossler":
                return np.array([-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)])
            case "thomas":
                return np.sin(point[[1, 2, 0]]) - 0.208186 * point
        raise ValueError(f"unknown attractor system: {self.system}")

    def step(self, state: State, dt: float) -> State:
        with np.errstate(over="ignore", invalid="ignore"):
            point = _rk4(self.velocity, state["point"], dt)
        # Custom parameters can make fixed-step integration escape numerically.
        if not np.isfinite(point).all():
            point = np.full(3, np.nan)
        return {"point": point}
