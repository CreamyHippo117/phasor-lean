"""Calculus objects.

    from phasor import calc

    s.add(calc.RiemannSum(id="area", function="x**2", a=0, b=3, n=expr("2 + 2 * floor(t)")))
    s.add(calc.Taylor(id="approx", function="sin(x)", center=0, order=5))
    s.add(calc.VectorField(id="flow", p="-y", q="x"))

The graph lives in the plane with x along the world x axis and y = f(x) up; ``pos`` moves the
origin. Numerics are in ``calc_math``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import calc_math as cm
from phasor.domains.strokes import polyline, with_alpha

_FN_CACHE: dict[str, cm.RealFunction] = {}


def _fn(source: str) -> cm.RealFunction:
    if source not in _FN_CACHE:
        _FN_CACHE[source] = cm.RealFunction.parse(source)
    return _FN_CACHE[source]


def _check_function(v: str) -> str:
    try:
        _fn(v.strip())
    except cm.FunctionError as e:
        raise ValueError(str(e)) from None
    return v


def _curve(f: cm.RealFunction, x0: float, x1: float, pos, samples=400, ymax=20.0):
    """The graph as separate runs: breaks at NaN, infinity and huge values (a pole)."""
    xs = np.linspace(x0, x1, samples)
    ys = f(xs)
    ok = np.isfinite(ys) & (np.abs(ys) < ymax)
    runs, start = [], None
    for i, good in enumerate([*ok, False]):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start >= 2:
                runs.append(
                    np.stack(
                        [xs[start:i] + pos[0], ys[start:i] + pos[1], np.full(i - start, pos[2])],
                        axis=1,
                    )
                )
            start = None
    return runs


# ---------------------------------------------------------------- Graph


@register
class Graph(ObjectBase):
    """The graph of y = f(x), with breaks where the function is undefined or has a pole."""

    type: ClassVar[str] = "graph"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {"glowing": {"style": {"glow": "soft"}}}

    class Params(ParamsBase):
        function: str = "sin(x)"
        xmin: float = -5.0
        xmax: float = 5.0
        pos: Vec3 = (0.0, 0.0, 0.0)

        @field_validator("function")
        @classmethod
        def _f(cls, v):
            return _check_function(v)

    def _runs(self, p):
        return _curve(_fn(p.function.strip()), p.xmin, p.xmax, p.pos)

    def path_points(self, state: SceneState) -> np.ndarray | None:
        """The longest unbroken run of the graph (what a ``Tracer`` travels along)."""
        runs = self._runs(self.me(state).params)
        return max(runs, key=len) if runs else None

    def world_bounds(self, state: SceneState) -> Bounds:
        runs = self._runs(self.me(state).params)
        return Bounds.of_points(np.vstack(runs) if runs else [self.me(state).params.pos], pad=0.05)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        runs = self._runs(self.me(state).params)
        return {"curve": np.vstack(runs)} if runs else {}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _fn(p.function.strip())
        return {
            "kind": "graph",
            "function": p.function,
            "latex": f.latex(),
            "breaks": max(len(self._runs(p)) - 1, 0),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        cam = state.camera_for(me)
        w = me.style["stroke"] * float(cam.units_per_height())
        for run in self._runs(me.params):
            polyline(
                ctx,
                cam,
                run,
                w,
                color=with_alpha(me.style["color"], me.params.opacity),
                glow=me.style["glow"],
            )


# ---------------------------------------------------------------- RiemannSum


@register
class RiemannSum(ObjectBase):
    """The area under y = f(x) on [a, b] approximated by ``n`` pieces (``rule``: left, right,
    mid, trapezoid, upper, lower): the rectangles, the curve, and in the semantics the sum, the
    exact integral and the error. Animate ``n`` upward to watch it converge."""

    type: ClassVar[str] = "riemann_sum"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "curve": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        function: str = "x**2"
        a: float = 0.0
        b: float = 3.0
        n: float = Field(6.0, ge=1, le=400)  # a number (so it can be animated); rounded
        rule: Literal["left", "right", "mid", "trapezoid", "upper", "lower"] = "left"
        pos: Vec3 = (0.0, 0.0, 0.0)

        @field_validator("function")
        @classmethod
        def _f(cls, v):
            return _check_function(v)

        @field_validator("b")
        @classmethod
        def _ab(cls, v, info):
            if v <= info.data.get("a", 0.0):
                raise ValueError("b must be greater than a")
            return v

    def _sum(self, p):
        return cm.riemann_sum(_fn(p.function.strip()), p.a, p.b, max(1, round(p.n)), p.rule)

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        r = self._sum(p)["rects"]
        y = np.nan_to_num(r[:, 4])
        top, bot = max(float(y.max()), 0.0), min(float(y.min()), 0.0)
        return Bounds(
            (p.a + p.pos[0], bot + p.pos[1], p.pos[2]), (p.b + p.pos[0], top + p.pos[1], p.pos[2])
        )

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        r = self._sum(p)["rects"]
        return {
            "tops": np.stack(
                [
                    (r[:, 0] + r[:, 1]) / 2 + p.pos[0],
                    np.nan_to_num(r[:, 4]) + p.pos[1],
                    np.full(len(r), p.pos[2]),
                ],
                axis=1,
            )
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _fn(p.function.strip())
        s = self._sum(p)
        exact = cm.exact_integral(f, p.a, p.b)
        return {
            "kind": "riemann_sum",
            "rule": p.rule,
            "pieces": len(s["rects"]),
            "sum": round(s["value"], 6),
            "exact": round(exact, 6),
            "error": round(abs(s["value"] - exact), 6),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        w = me.style["stroke"] * float(cam.units_per_height())
        o = np.asarray(p.pos, dtype=np.float64)
        fill = with_alpha(me.style["color"], p.opacity * 0.35)
        edge = with_alpha(me.style["color"], p.opacity)
        for x0, x1, fl, fr, h in self._sum(p)["rects"]:
            if not math.isfinite(h):
                continue
            if p.rule == "trapezoid":
                quad = np.array([[x0, 0, 0], [x1, 0, 0], [x1, fr, 0], [x0, fl, 0]]) + o
            else:
                quad = np.array([[x0, 0, 0], [x1, 0, 0], [x1, h, 0], [x0, h, 0]]) + o
            ctx.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]), color=fill)
            polyline(ctx, cam, np.vstack([quad, quad[:1]]), w * 0.6, color=edge)
        for run in _curve(_fn(p.function.strip()), p.a, p.b, o):
            polyline(
                ctx,
                cam,
                run,
                w * 1.4,
                color=with_alpha(me.style["curve"], p.opacity),
                glow=me.style["glow"],
            )


# ---------------------------------------------------------------- Taylor


@register
class Taylor(ObjectBase):
    """f(x) and its Taylor polynomial of degree ``order`` about ``center``: more terms hug the
    curve over a wider window. ``order`` is a number so it can be animated."""

    type: ClassVar[str] = "taylor"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "poly": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        function: str = "sin(x)"
        center: float = 0.0
        order: float = Field(3.0, ge=0, le=30)
        xmin: float = -6.0
        xmax: float = 6.0
        ymax: float = Field(3.0, gt=0)  # the polynomial is clipped above this height
        pos: Vec3 = (0.0, 0.0, 0.0)

        @field_validator("function")
        @classmethod
        def _f(cls, v):
            return _check_function(v)

    def _coeffs(self, p):
        return cm.taylor_coefficients(_fn(p.function.strip()), p.center, round(p.order))

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds(
            (p.xmin + p.pos[0], -p.ymax + p.pos[1], p.pos[2]),
            (p.xmax + p.pos[0], p.ymax + p.pos[1], p.pos[2]),
        )

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return {
            "center": np.array(
                [
                    p.center + p.pos[0],
                    float(_fn(p.function.strip())(np.array(p.center))) + p.pos[1],
                    p.pos[2],
                ]
            )
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _fn(p.function.strip())
        c = self._coeffs(p)
        xs = np.linspace(max(p.xmin, p.center - 1.0), min(p.xmax, p.center + 1.0), 81)
        err = float(np.nanmax(np.abs(cm.taylor_polynomial(c, p.center)(xs) - f(xs))))
        return {
            "kind": "taylor",
            "function": p.function,
            "order": len(c) - 1,
            "center": p.center,
            "coefficients": [round(v, 8) for v in c],
            "max_error_within_1": round(err, 8),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        w = me.style["stroke"] * float(cam.units_per_height())
        f = _fn(p.function.strip())
        for run in _curve(f, p.xmin, p.xmax, p.pos, ymax=p.ymax):
            polyline(ctx, cam, run, w, color=with_alpha(me.style["color"], p.opacity))
        poly = cm.taylor_polynomial(self._coeffs(p), p.center)
        xs = np.linspace(p.xmin, p.xmax, 400)
        ys = poly(xs)
        keep = np.abs(ys) <= p.ymax
        pts = np.stack([xs + p.pos[0], ys + p.pos[1], np.full(len(xs), p.pos[2])], axis=1)
        start = None
        for i, good in enumerate([*keep, False]):
            if good and start is None:
                start = i
            elif not good and start is not None:
                if i - start >= 2:
                    polyline(
                        ctx,
                        cam,
                        pts[start:i],
                        w * 1.3,
                        color=with_alpha(me.style["poly"], p.opacity),
                        glow=me.style["glow"],
                    )
                start = None


# ---------------------------------------------------------------- VectorField


@register
class VectorField(ObjectBase):
    """A planar field F = (P(x, y), Q(x, y)) as a grid of arrows (length scaled to the grid
    spacing). Divergence and curl are in the semantics.

    Presets: ``dense`` (many small thin arrows, lengths eased so weak ones still show, color
    saying how strong: the ``magnitude`` colormap), ``bold`` (few thick glowing arrows)."""

    type: ClassVar[str] = "vector_field"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "sequential"),
    }
    Presets: ClassVar = {
        "dense": {"params": {"spacing": 0.32, "length_by": "soft", "color_by_size": True},
                  "style": {"colormap": "magnitude", "stroke": "hairline"}},
        "bold": {"params": {"spacing": 1.0}, "style": {"stroke": "regular", "glow": "soft"}},
    }  # fmt: skip

    class Params(ParamsBase):
        p: str = "-y"
        q: str = "x"
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        extent: tuple[float, float] = (8.0, 5.0)
        spacing: float = Field(0.8, gt=0.1)
        color_by_size: bool = False  # color each arrow by its size through the colormap
        length_by: Literal["size", "soft"] = "size"  # "soft": weak arrows still show

        @field_validator("p", "q")
        @classmethod
        def _pq(cls, v):
            try:
                cm.parse_expression(v, ("x", "y"))
            except cm.FunctionError as e:
                raise ValueError(str(e)) from None
            return v

    def _field(self, p):
        return cm.VectorField.parse(p.p, p.q)

    def _samples(self, p):
        xs = np.arange(-p.extent[0] / 2, p.extent[0] / 2 + 1e-9, p.spacing)
        ys = np.arange(-p.extent[1] / 2, p.extent[1] / 2 + 1e-9, p.spacing)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return X, Y

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c = np.asarray(p.pos, dtype=np.float64)
        return Bounds(
            tuple(c - np.array([p.extent[0] / 2, p.extent[1] / 2, 0.0])),
            tuple(c + np.array([p.extent[0] / 2, p.extent[1] / 2, 0.0])),
        )

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        import sympy as sp

        p = self.me(state).params
        f = self._field(p)
        return {
            "kind": "vector_field",
            "p": p.p,
            "q": p.q,
            "divergence": sp.sstr(f.divergence()),
            "curl": sp.sstr(f.curl()),
            "arrows": int(self._samples(p)[0].size),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        from phasor.core.color import ramp
        from phasor.domains.geometry_math import normalize01, sample_ramp
        from phasor.domains.strokes import field_arrows

        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        X, Y = self._samples(p)
        U, V = self._field(p)(X, Y)
        o = np.asarray(p.pos, dtype=np.float64)
        pts = o + np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
        vec = np.c_[np.ravel(U), np.ravel(V), np.zeros(X.size)]
        w = me.style["stroke"] * float(cam.units_per_height(o)) * 2.5
        if p.color_by_size:
            mag = np.linalg.norm(vec, axis=1)
            rgb = sample_ramp(ramp(me.style["colormap"]), normalize01(mag, 0.0, None))
            cols = np.c_[rgb, np.full(len(rgb), p.opacity)]
        else:
            cols = with_alpha(me.style["color"], p.opacity)
        field_arrows(ctx, cam, pts, vec, p.spacing, width=w, colors=cols,
                     glow=me.style["glow"], length_by=p.length_by)  # fmt: skip


# ---------------------------------------------------------------- Limit (epsilon-delta)


@register
class Limit(ObjectBase):
    """The epsilon-delta picture of lim f(x) as x -> a: the graph, a band of half-height
    ``epsilon`` around the limit L, and the widest band of half-width delta around a whose
    graph stays inside it. Shrink ``epsilon`` and delta shrinks with it. The point (a, L) is
    drawn hollow: f need not be defined there."""

    type: ClassVar[str] = "limit"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "band": ("emphasis", "highlight"),
        "delta": ("emphasis", "primary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        function: str = "sin(x)/x"
        a: float = 0.0
        epsilon: float = Field(0.3, gt=0)
        xmin: float = -5.0
        xmax: float = 5.0
        pos: Vec3 = (0.0, 0.0, 0.0)
        bands: float = Field(0.14, ge=0, le=1)  # opacity of the epsilon and delta bands

        @field_validator("function")
        @classmethod
        def _f(cls, v):
            return _check_function(v)

        @model_validator(mode="after")
        def _has_limit(self):
            if cm.limit_at(_fn(self.function.strip()), self.a) is None:
                raise ValueError(f"{self.function} has no finite limit at x = {self.a:g}")
            return self

    def _numbers(self, p):
        f = _fn(p.function.strip())
        L = cm.limit_at(f, p.a)
        assert L is not None
        delta = cm.delta_for_epsilon(f, p.a, L, p.epsilon, search=(p.xmax - p.xmin) / 2)
        return f, L, delta

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        f, L, _ = self._numbers(p)
        runs = _curve(f, p.xmin, p.xmax, p.pos)
        pts = [
            *runs,
            np.array(
                [
                    [p.xmin + p.pos[0], L + p.pos[1] + p.epsilon, p.pos[2]],
                    [p.xmax + p.pos[0], L + p.pos[1] - p.epsilon, p.pos[2]],
                ]
            ),
        ]
        return Bounds.of_points(np.vstack(pts), pad=0.05)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        _, L, _ = self._numbers(p)
        return {"point": np.array([p.a + p.pos[0], L + p.pos[1], p.pos[2]])}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f, L, delta = self._numbers(p)
        x = np.concatenate(
            [p.a - np.linspace(1e-6, delta, 400), p.a + np.linspace(1e-6, delta, 400)]
        )
        worst = float(np.nanmax(np.abs(f(x) - L))) if delta > 0 else 0.0
        return {
            "kind": "limit",
            "function": p.function,
            "a": p.a,
            "limit": round(L, 9),
            "epsilon": round(p.epsilon, 9),
            "delta": round(delta, 6),
            "holds": bool(delta > 0 and worst < p.epsilon),
            "worst_inside": round(worst, 9),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        from phasor.domains.strokes import circle

        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        f, L, delta = self._numbers(p)
        ox, oy, oz = p.pos
        w = me.style["stroke"] * float(cam.units_per_height())
        y0, y1 = oy + L - p.epsilon, oy + L + p.epsilon
        x0, x1 = ox + p.xmin, ox + p.xmax
        bb = Bounds.of_points(np.vstack(_curve(f, p.xmin, p.xmax, p.pos) or [np.array([p.pos])]))
        top, bottom = max(bb.hi[1], y1), min(bb.lo[1], y0)

        def rect(xa, xb, ya, yb, color):
            quad = np.array([[xa, ya, oz], [xb, ya, oz], [xb, yb, oz], [xa, yb, oz]])
            ctx.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]), color=color)

        if p.bands > 0:
            rect(x0, x1, y0, y1, with_alpha(me.style["band"], p.opacity * p.bands))
            if delta > 0:
                rect(ox + p.a - delta, ox + p.a + delta, bottom, top,
                     with_alpha(me.style["delta"], p.opacity * p.bands * 0.6))  # fmt: skip
        for run in _curve(f, p.xmin, p.xmax, p.pos):
            polyline(ctx, cam, run, w, color=with_alpha(me.style["color"], p.opacity),
                     glow=me.style["glow"])  # fmt: skip
        hole = np.array([ox + p.a, oy + L, oz])
        r = 0.1 * float(cam.units_per_height(hole)) / 4
        circle(ctx, cam, hole, r, w, color=with_alpha(me.style["band"], p.opacity))


# ---------------------------------------------------------------- LoopIntegral (flux, Green)


@register
class LoopIntegral(ObjectBase):
    """A circle laid over a vector field, and a line integral around it, traced as
    ``progress`` goes 0 -> 1. ``measure="flux"``: arrows show F . n, what crosses the curve
    (outward bright, inward dim); the total equals the integral of the divergence inside.
    ``measure="circulation"``: arrows show F . T, what flows along it; the total equals the
    integral of the curl inside (Green's theorem). Field coordinates are relative to ``pos``,
    as for ``VectorField``, so the two line up when they share ``pos``."""

    type: ClassVar[str] = "loop_integral"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "positive": ("emphasis", "highlight"),
        "negative": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
    }

    class Params(ParamsBase):
        p: str = "x"
        q: str = "y"
        measure: Literal["flux", "circulation"] = "flux"
        center: tuple[float, float] = (0.0, 0.0)  # in field coordinates
        radius: float = Field(1.5, gt=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # where the field's origin is
        progress: float = Field(1.0, ge=0, le=1)
        arrows: int = Field(16, ge=0, le=128)

        @field_validator("p", "q")
        @classmethod
        def _pq(cls, v):
            try:
                cm.parse_expression(v, ("x", "y"))
            except cm.FunctionError as e:
                raise ValueError(str(e)) from None
            return v

    def _ring(self, p, t):
        o = np.asarray(p.pos, dtype=np.float64)
        return np.stack(
            [
                o[0] + p.center[0] + p.radius * np.cos(t),
                o[1] + p.center[1] + p.radius * np.sin(t),
                np.full_like(t, o[2]),
            ],
            1,
        )

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds.of_points(self._ring(p, np.linspace(0, 2 * np.pi, 64)), pad=0.6)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        o = np.asarray(p.pos, dtype=np.float64)
        return {
            "center": o + np.array([*p.center, 0.0]),
            "curve": self._ring(p, np.linspace(0, 2 * np.pi, 64)),
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        import sympy as sp

        p = self.me(state).params
        f = cm.VectorField.parse(p.p, p.q)
        inside = f.divergence() if p.measure == "flux" else f.curl()
        total = cm.loop_integral(f, p.center, p.radius, p.measure)
        area = cm.disc_integral(f, inside, p.center, p.radius)
        return {
            "kind": "loop_integral",
            "measure": p.measure,
            "so_far": round(cm.loop_integral(f, p.center, p.radius, p.measure, upto=p.progress), 6),
            "total": round(total, 6),
            "inside": sp.sstr(inside),
            "area_integral": round(area, 6),
            "theorem": (
                "divergence theorem: flux = integral of div F"
                if p.measure == "flux"
                else "Green: circulation = integral of curl F"
            ),
            "theorem_holds": bool(abs(total - area) < 1e-6 * max(1.0, abs(area))),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        from phasor.domains.strokes import arrow

        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        f = cm.VectorField.parse(p.p, p.q)
        o = np.asarray(p.pos, dtype=np.float64)
        w = me.style["stroke"] * float(cam.units_per_height(o))
        full = self._ring(p, np.linspace(0, 2 * np.pi, 129))
        polyline(ctx, cam, full, w * 0.5, color=with_alpha(me.style["negative"], p.opacity * 0.5))
        if p.progress > 0:
            done = self._ring(
                p, np.linspace(0, 2 * np.pi * p.progress, max(2, int(128 * p.progress) + 2))
            )
            polyline(ctx, cam, done, w, color=with_alpha(me.style["color"], p.opacity),
                     glow=me.style["glow"])  # fmt: skip
        if p.arrows == 0:
            return
        t = (np.arange(p.arrows) + 0.5) / p.arrows * 2 * np.pi
        t = t[t <= 2 * np.pi * p.progress + 1e-9]
        if not len(t):
            return
        c, s = np.cos(t), np.sin(t)
        P, Q = f(p.center[0] + p.radius * c, p.center[1] + p.radius * s)
        normal = p.measure == "flux"
        comp = P * c + Q * s if normal else -P * s + Q * c
        top = float(np.nanmax(np.abs(comp))) if np.isfinite(comp).any() else 0.0
        if top <= 0:
            return
        base = self._ring(p, t)
        dirs = np.stack([c, s], 1) if normal else np.stack([-s, c], 1)
        for b, d, v in zip(base, dirs, comp, strict=True):
            if not np.isfinite(v) or abs(v) < 1e-9 * top:
                continue
            tip = b + np.array([*d, 0.0]) * (v / top) * p.radius * 0.45
            role = "positive" if v > 0 else "negative"
            arrow(ctx, cam, b, tip, w * 0.8, with_alpha(me.style[role], p.opacity))
