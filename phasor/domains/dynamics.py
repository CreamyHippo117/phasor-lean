"""Dynamical systems: phase portraits and attractors.

    from phasor import dyn

    s.add(dyn.PhasePortrait(id="flow", f="y", g="-sin(x) - 0.3 * y", extent=(9, 6)))
    s.add(dyn.FixedPoints(id="rest", f="y", g="-sin(x) - 0.3 * y", extent=(9, 6)))

``PhasePortrait`` draws the flow as small arrows, the nullclines (where x' = 0 and where
y' = 0), and trajectories traced from seeds, coloured along time; ``progress`` 0 -> 1 traces
them. ``FixedPoints`` finds the points where the system rests, classifies each from its
Jacobian's eigenvalues (saddle, node, spiral, centre; stable or not), draws it as a beacon
coloured by that, with its eigen-directions as rays and the eigenvalues as a label.
Both read the same formulas, with named ``constants`` allowed (``{"mu": 1.5}``); ``pos`` is
the window's centre in the world, one unit of (x, y) is one world unit. Numerics are in
``dynamics_math``.

``LinearSystem`` (D5) is the linear case x' = A x: exact trajectories e^(tA) x0 and the type
of the origin. Its modes and the matrix exponential acting on a grid are the existing
``linalg.EigenLines`` and ``linalg.TransformGrid(flow=A, flow_time=...)``, drawn beside it.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.simulation import SimRunner
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import dynamics_math as dm
from phasor.domains import linalg_math as lm
from phasor.domains.strokes import (
    disc_batch,
    field_arrows,
    polyline,
    segment_batch,
    with_alpha,
)

_CACHE: dict = {}
_LIMIT = 64


def _memo(key, make):
    if key not in _CACHE:
        if len(_CACHE) >= _LIMIT:
            _CACHE.clear()
        _CACHE[key] = make()
    return _CACHE[key]


class SystemParams(ParamsBase):
    f: str = "y"  # x' =
    g: str = "-x"  # y' =
    constants: str = ""  # named numbers the formulas may use: "mu=1.5, k=2"
    pos: Vec3 = (0.0, 0.0, 0.0)  # the window's centre
    extent: tuple[float, float] = (8.0, 5.0)  # the window's width and height (x, y units)

    @model_validator(mode="after")
    def _formulas(self):
        try:
            dm.System.parse(self.f, self.g, self.constants)
        except dm.FunctionError as e:
            raise ValueError(str(e)) from None
        return self


def _system(p) -> dm.System:
    key = ("sys", p.f, p.g, p.constants)
    return _memo(key, lambda: dm.System.parse(p.f, p.g, p.constants))


def _window(p) -> tuple[float, float, float, float]:
    return (-p.extent[0] / 2, p.extent[0] / 2, -p.extent[1] / 2, p.extent[1] / 2)


def _bounds(p) -> Bounds:
    c = np.asarray(p.pos, dtype=np.float64)
    half = np.array([p.extent[0] / 2, p.extent[1] / 2, 0.0])
    return Bounds(tuple(c - half), tuple(c + half))


def _lut_at(lut, u):
    """Colors (N, 3) from a ramp at positions ``u`` (0..1)."""
    from phasor.domains.geometry_math import sample_ramp

    return sample_ramp(np.asarray(lut, dtype=np.float64), u)


# ---------------------------------------------------------------- PhasePortrait


@register
class PhasePortrait(ObjectBase):
    """The picture of a planar system x' = f(x, y), y' = g(x, y): the flow as small arrows,
    the nullclines, and trajectories from seeds drawn as they run (``progress``), colour
    running along time. Seeds are given (``seeds``), or ``n_seeds`` are spread over the
    window. Presets: ``plain`` (trajectories and nullclines), ``flow`` (also the arrow field)."""

    type: ClassVar[str] = "phase_portrait"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "other": ("emphasis", "secondary"),
        "field": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "spectrum"),
    }
    Presets: ClassVar = {
        "plain": {"params": {"flow": False}, "style": {}},
        "flow": {"params": {"flow": True}, "style": {}},
    }  # fmt: skip

    class Params(SystemParams):
        seeds: tuple[tuple[float, float], ...] = ()
        n_seeds: int = Field(12, ge=0, le=200)  # used when ``seeds`` is empty
        t_end: float = Field(8.0, gt=0)  # how long each trajectory runs
        steps: int = Field(320, ge=8, le=4000)
        progress: float = Field(1.0, ge=0, le=1)  # the share of each trajectory drawn
        backward: bool = False  # also run each seed back in time
        flow: bool = False  # the small-arrow field
        flow_spacing: float = Field(0.5, gt=0.15)
        nullclines: bool = True
        hue_cycles: float = Field(1.0, ge=0)  # colormap cycles along each trajectory's time
        seed: int = 0

        @field_validator("seeds")
        @classmethod
        def _seeds(cls, v):
            if len(v) > 200:
                raise ValueError("at most 200 seeds")
            return v

    def _starts(self, p) -> np.ndarray:
        if p.seeds:
            return np.asarray(p.seeds, dtype=np.float64).reshape(-1, 2)
        return dm.hash_seeds(p.n_seeds, p.extent, p.seed) if p.n_seeds else np.zeros((0, 2))

    def _paths(self, p, direction: float) -> np.ndarray:
        x0, x1, y0, y1 = _window(p)
        pad = 0.02 * max(p.extent)
        b = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
        starts = self._starts(p)
        key = ("paths", p.f, p.g, p.constants, starts.tobytes(),
               p.t_end, p.steps, b, direction)  # fmt: skip
        return _memo(key, lambda: dm.trajectories(_system(p), starts, p.t_end, p.steps, b,
                                                  direction))  # fmt: skip

    def _nulls(self, p, which: str) -> np.ndarray:
        w = _window(p)
        key = ("null", p.f, p.g, p.constants, which, w)
        return _memo(key, lambda: dm.nullcline_segments(_system(p), which, w, 100))

    def world_bounds(self, state: SceneState) -> Bounds:
        return _bounds(self.me(state).params)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return {"center": np.asarray(p.pos, dtype=np.float64),
                "seeds": np.c_[self._starts(p), np.zeros(len(self._starts(p)))]}  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        fixed = dm.fixed_points(_system(p), _window(p))
        return {
            "kind": "phase_portrait",
            "f": p.f,
            "g": p.g,
            "trajectories": len(self._starts(p)),
            "fixed_points": len(fixed),
            "progress": round(p.progress, 6),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        o = np.asarray(p.pos, dtype=np.float64)
        w = me.style["stroke"] * float(cam.units_per_height(o))
        glow = me.style["glow"]
        sysm = _system(p)
        if p.flow:
            xs = np.arange(-p.extent[0] / 2 + p.flow_spacing / 2, p.extent[0] / 2, p.flow_spacing)
            ys = np.arange(-p.extent[1] / 2 + p.flow_spacing / 2, p.extent[1] / 2, p.flow_spacing)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            U, V = sysm(X, Y)
            pts = o + np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
            vec = np.c_[U.ravel(), V.ravel(), np.zeros(X.size)]
            field_arrows(ctx, cam, pts, vec, p.flow_spacing, width=w * 0.5,
                         colors=with_alpha(me.style["field"], p.opacity * 0.55), glow=1.0,
                         length_by="soft")  # fmt: skip
        if p.nullclines:
            for which, role in (("x", "other"), ("y", "color")):
                seg = self._nulls(p, which)
                if len(seg):
                    a = o + np.c_[seg[:, 0, 0], seg[:, 0, 1], np.zeros(len(seg))]
                    b = o + np.c_[seg[:, 1, 0], seg[:, 1, 1], np.zeros(len(seg))]
                    segment_batch(ctx, cam, a, b, me.style["stroke"] * 0.7,
                                  colors=with_alpha(me.style[role], p.opacity * 0.3),
                                  glow=1.0)  # fmt: skip
        if p.progress <= 0 or not len(self._starts(p)):
            return
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        runs = [(self._paths(p, 1.0), 1.0)]
        if p.backward:
            runs.append((self._paths(p, -1.0), -1.0))
        for paths, _ in runs:
            n = paths.shape[1]
            upto = max(2, round(p.progress * (n - 1)) + 1)
            for path in paths:
                part = path[:upto]
                ok = np.isfinite(part).all(axis=1)
                if ok.sum() < 2:
                    continue
                idx = np.flatnonzero(ok)
                for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                    if len(run) < 2:
                        continue
                    pts = o + np.c_[part[run], np.zeros(len(run))]
                    u = (run / (n - 1)) * p.hue_cycles % 1.0
                    rgb = _lut_at(np.vstack([lut, lut[:1]]), u)
                    cols = np.c_[rgb, np.full(len(run), p.opacity)]
                    polyline(ctx, cam, pts, w, color=with_alpha(me.style["color"], p.opacity),
                             glow=glow, colors=cols)  # fmt: skip


# ---------------------------------------------------------------- FixedPoints


@register
class FixedPoints(ObjectBase):
    """Where x' = y' = 0: each fixed point as a beacon coloured by its type (stable, unstable,
    saddle, centre), its real eigen-directions as rays (toward the point where the motion
    comes in, away where it goes out) and a label with the eigenvalues. ``reveal`` 0 -> 1
    shows them one after another."""

    type: ClassVar[str] = "fixed_points"
    Style: ClassVar = {
        "stable": ("emphasis", "secondary"),
        "unstable": ("emphasis", "danger"),
        "saddle": ("emphasis", "highlight"),
        "centre": ("emphasis", "primary"),
        "text": ("emphasis", "primary"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(SystemParams):
        rays: bool = True  # the eigen-directions
        labels: bool = True
        ray_length: float = Field(1.1, gt=0)
        reveal: float = Field(1.0, ge=0, le=1)  # how many are shown (in order along x)
        grid: int = Field(9, ge=2, le=30)  # Newton start points per side

    def _points(self, p) -> list[dict]:
        key = ("fp", p.f, p.g, p.constants, _window(p), p.grid)

        def make():
            s = _system(p)
            out = []
            for pt in dm.fixed_points(s, _window(p), p.grid):
                c = dm.classify(s.jacobian_at(*pt))
                c["at"] = (float(pt[0]), float(pt[1]))
                out.append(c)
            return out

        return _memo(key, make)

    def _shown(self, p) -> list[dict]:
        pts = self._points(p)
        n = len(pts)
        k = int(np.floor(p.reveal * n + 1e-9))
        return pts[:k]

    def world_bounds(self, state: SceneState) -> Bounds:
        return _bounds(self.me(state).params)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        pts = [np.r_[c["at"], 0.0] + np.asarray(p.pos) for c in self._points(p)]
        return {"center": np.asarray(p.pos, dtype=np.float64),
                "points": np.array(pts, dtype=np.float64).reshape(-1, 3)}  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {
            "kind": "fixed_points",
            "count": len(self._points(p)),
            "points": [
                {
                    "at": [round(c["at"][0], 6), round(c["at"][1], 6)],
                    "type": c["kind"],
                    "stable": c["stable"],
                    "eigenvalues": [[round(a, 6), round(b, 6)] for a, b in c["eigenvalues"]],
                    "trace": round(c["trace"], 6),
                    "det": round(c["det"], 6),
                }
                for c in self._points(p)
            ],
        }

    @staticmethod
    def _role(c: dict) -> str:
        return {"saddle": "saddle", "centre": "centre"}.get(
            c["kind"], "stable" if c["stable"] else "unstable")  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        o = np.asarray(p.pos, dtype=np.float64)
        unit = float(cam.units_per_height(o))
        w = me.style["stroke"] * unit
        glow = me.style["glow"]
        size = me.style["size"] * unit
        for c in self._shown(p):
            at = o + np.r_[c["at"], 0.0]
            col = with_alpha(me.style[self._role(c)], p.opacity)
            if p.rays:
                for lam, v in c["directions"]:
                    d = np.r_[v, 0.0] * p.ray_length
                    rc = with_alpha(me.style["stable" if lam < 0 else "unstable"], p.opacity * 0.8)
                    polyline(ctx, cam, [at - d, at + d], w * 0.7, color=rc, glow=glow)
            disc_batch(ctx, cam, at[None], unit * 0.016, colors=col, glow=max(glow, 1.0) * 1.2)
            if p.labels:
                ctx.text(dm.eigenvalue_label(c), at + cam.up * unit * 0.07, size,
                         color=with_alpha(me.style["text"], p.opacity), glow=1.0)  # fmt: skip


# ---------------------------------------------------------------- LinearSystem


@register
class LinearSystem(ObjectBase):
    """The linear system x' = A x: trajectories from seeds (exact, e^(tA) x0), the eigenvalue
    label and the type of the origin in the semantics. Draw its modes with ``linalg.EigenLines``
    (the real eigenvectors are invariant lines: a point on one stays on it, scaled by
    e^(lambda t)) and the matrix exponential acting on a grid with
    ``linalg.TransformGrid(flow=A, flow_time=...)``. ``progress`` 0 -> 1 traces the
    trajectories."""

    type: ClassVar[str] = "linear_system"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "text": ("emphasis", "primary"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "spectrum"),
    }

    class Params(ParamsBase):
        matrix: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 2.0), (2.0, 1.0))
        pos: Vec3 = (0.0, 0.0, 0.0)  # the window's centre (the origin of the system)
        extent: tuple[float, float] = (8.0, 5.0)  # the window's width and height
        seeds: tuple[tuple[float, float], ...] = ()
        n_seeds: int = Field(12, ge=0, le=200)  # used when ``seeds`` is empty
        t_end: float = Field(4.0, gt=0)
        steps: int = Field(240, ge=8, le=4000)
        progress: float = Field(1.0, ge=0, le=1)
        backward: bool = True  # also run each seed back in time (the whole curve)
        labels: bool = True  # the type and eigenvalues, as a label
        hue_cycles: float = Field(1.0, ge=0)
        seed: int = 0

        @field_validator("matrix")
        @classmethod
        def _m(cls, v):
            if lm.matrix(v).shape != (2, 2):
                raise ValueError("linear_system takes a 2x2 matrix")
            return v

        @field_validator("seeds")
        @classmethod
        def _seeds(cls, v):
            if len(v) > 200:
                raise ValueError("at most 200 seeds")
            return v

    def _class(self, p) -> dict:
        key = ("lin", p.matrix)
        return _memo(key, lambda: dm.classify(np.asarray(p.matrix, dtype=np.float64)))

    def _starts(self, p) -> np.ndarray:
        if p.seeds:
            return np.asarray(p.seeds, dtype=np.float64).reshape(-1, 2)
        return dm.hash_seeds(p.n_seeds, p.extent, p.seed) if p.n_seeds else np.zeros((0, 2))

    def _paths(self, p, direction: float) -> np.ndarray:
        x0, x1, y0, y1 = _window(p)
        pad = 0.02 * max(p.extent)
        b = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
        starts = self._starts(p)
        key = ("lpaths", p.matrix, starts.tobytes(), p.t_end, p.steps, b, direction)
        return _memo(key, lambda: dm.linear_paths(p.matrix, starts, p.t_end, p.steps, b, direction))

    def world_bounds(self, state: SceneState) -> Bounds:
        return _bounds(self.me(state).params)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return {"center": np.asarray(p.pos, dtype=np.float64),
                "seeds": np.c_[self._starts(p), np.zeros(len(self._starts(p)))]}  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        c = self._class(p)
        return {
            "kind": "linear_system",
            "matrix": [[round(v, 6) for v in row] for row in p.matrix],
            "type": c["kind"],
            "stable": c["stable"],
            "eigenvalues": [[round(a, 6), round(b, 6)] for a, b in c["eigenvalues"]],
            "eigenvectors": [
                [round(float(v[0]), 6), round(float(v[1]), 6)] for _, v in c["directions"]
            ],
            "trace": round(c["trace"], 6),
            "det": round(c["det"], 6),
            "trajectories": len(self._starts(p)),
            "progress": round(p.progress, 6),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        o = np.asarray(p.pos, dtype=np.float64)
        unit = float(cam.units_per_height(o))
        w = me.style["stroke"] * unit
        glow = me.style["glow"]
        if p.progress > 0 and len(self._starts(p)):
            lut = np.asarray(me.style["colormap"], dtype=np.float64)
            runs = [self._paths(p, 1.0)] + ([self._paths(p, -1.0)] if p.backward else [])
            for paths in runs:
                n = paths.shape[1]
                upto = max(2, round(p.progress * (n - 1)) + 1)
                for path in paths:
                    part = path[:upto]
                    idx = np.flatnonzero(np.isfinite(part).all(axis=1))
                    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                        if len(run) < 2:
                            continue
                        pts = o + np.c_[part[run], np.zeros(len(run))]
                        u = (run / (n - 1)) * p.hue_cycles % 1.0
                        rgb = _lut_at(np.vstack([lut, lut[:1]]), u)
                        cols = np.c_[rgb, np.full(len(run), p.opacity)]
                        polyline(ctx, cam, pts, w, color=with_alpha(me.style["color"], p.opacity),
                                 glow=glow, colors=cols)  # fmt: skip
        if p.labels:
            at = o + np.array([-p.extent[0] / 2 + 0.2, p.extent[1] / 2 - 0.35, 0.0])
            col = with_alpha(me.style["text"], p.opacity)
            ctx.text(
                dm.eigenvalue_label(self._class(p)),
                at,
                me.style["size"] * unit,
                anchor="left",
                glow=1.0,
                color=col,
            )


@register
class StrangeAttractor(ObjectBase):
    """A 3D Lorenz, Rössler or Thomas trajectory, traced in simulation time."""

    type: ClassVar[str] = "strange_attractor"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "spectrum"),
    }

    class Params(ParamsBase):
        system: dm.AttractorSystem = "lorenz"
        sigma: float = Field(10.0, gt=0, allow_inf_nan=False)
        rho: float = Field(28.0, allow_inf_nan=False)
        beta: float = Field(8 / 3, gt=0, allow_inf_nan=False)
        seed: int = 0
        t_end: float = Field(36.0, ge=dm.AttractorSimulation.dt, le=120.0)
        progress: float = Field(1.0, ge=0, le=1)
        pos: Vec3 = (0.0, 0.0, 0.0)
        size: float = Field(6.0, gt=0, allow_inf_nan=False)

    def _path(self, p) -> np.ndarray:
        sim = dm.AttractorSimulation(
            system=p.system, sigma=p.sigma, rho=p.rho, beta=p.beta, seed=p.seed, object_id=self.id
        )
        steps = round(p.t_end / sim.dt)
        key = ("attractor", sim, steps)

        def make():
            runner = SimRunner(sim, cache_dir=False)
            points = []
            for i in range(steps + 1):
                point = runner.state_at(i * sim.dt)["point"]
                if not np.isfinite(point).all():
                    break
                points.append(point)
            path = np.asarray(points)
            # Scale before computing the span so extreme finite states cannot overflow it.
            path /= max(1.0, float(np.abs(path).max()))
            low, high = path.min(axis=0), path.max(axis=0)
            span = float(np.max(high - low))
            return (path - (low + high) / 2) / span if span > 0 else np.zeros_like(path)

        return _memo(key, make)

    def _visible(self, p) -> tuple[np.ndarray, float]:
        path = self._path(p)
        step = min(p.progress * round(p.t_end / dm.AttractorSimulation.dt), len(path) - 1)
        whole = int(step)
        visible = path[: whole + 1]
        if step > whole:
            head = path[whole] + (step - whole) * (path[whole + 1] - path[whole])
            visible = np.vstack((visible, head))
        return np.asarray(p.pos) + p.size * visible, step * dm.AttractorSimulation.dt

    def path_points(self, state: SceneState) -> np.ndarray:
        """The visible trajectory, so a Tracer at progress 1 follows its head."""
        points, _ = self._visible(self.me(state).params)
        return points

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds.of_points(np.asarray(p.pos) + p.size * self._path(p))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        visible, _ = self._visible(p)
        return {"center": np.asarray(p.pos), "head": visible[-1]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        _, time = self._visible(p)
        return {
            "kind": "strange_attractor",
            "system": p.system,
            "seed": p.seed,
            "progress": p.progress,
            "simulated_time": time,
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        points, time = self._visible(p)
        if len(points) < 2:
            return
        cam = state.camera_for(me)
        width = me.style["stroke"] * cam.units_per_height(points)
        times = np.minimum(np.arange(len(points)) * dm.AttractorSimulation.dt, time)
        lut = me.style["colormap"]
        rgb = _lut_at(lut, times / p.t_end)
        colors = np.c_[rgb, np.full(len(points), p.opacity)]
        polyline(ctx, cam, points, width, color=with_alpha(me.style["color"], p.opacity),
                 glow=me.style["glow"], colors=colors)  # fmt: skip
