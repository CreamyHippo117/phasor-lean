"""Linear algebra objects.

    from phasor import linalg

    s.add(linalg.TransformGrid(id="grid", matrix=((2, 1), (1, 2))))
    s.add(linalg.EigenLines(id="eig", matrix=((2, 1), (1, 2))))
    s.add(linalg.Matrix(id="m", matrix=expr("grid.matrix"), pos=(-5, 3)))   # live entries
    s.add(linalg.Arrow(id="v", start=(0, 0), end=(1, 2)))

A ``matrix`` is a parameter like any other, so it can be animated (``s.at("m").morph("grid.matrix",
...)``) and read by other objects with an expression, which is what keeps a matrix's printed
entries linked to the picture.
"""

from __future__ import annotations

import itertools
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import linalg_math as lm
from phasor.domains.strokes import arrow, polyline, with_alpha

Matrix3 = tuple[tuple[float, ...], ...]


def _check_matrix(v):
    lm.matrix(v)
    return v


# ---------------------------------------------------------------- TransformGrid


@register
class TransformGrid(ObjectBase):
    """What a matrix does to space: the grid it carries along, the unit square it turns into a
    parallelogram (its area is the determinant), and the images of the basis vectors. A 3x3
    matrix does the same to the unit cube."""

    type: ClassVar[str] = "transform_grid"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "grid": ("emphasis", "muted"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),
        "i": ("emphasis", "highlight"),
        "j": ("emphasis", "danger"),
    }

    class Params(ParamsBase):
        matrix: Matrix3 = ((1.0, 0.0), (0.0, 1.0))
        pos: Vec3 = (0.0, 0.0, 0.0)  # where the origin of the grid sits
        extent: int = Field(4, ge=1, le=20)  # lines either side of the axes
        spacing: float = Field(1.0, gt=0)
        square: float = Field(0.35, ge=0, le=1)  # opacity of the filled unit square/cube
        basis: float = Field(1.0, ge=0, le=1)  # opacity of the basis arrows
        # A system x' = flow x: when set, the grid shows its flow e^(flow * flow_time) instead
        # of ``matrix`` (animate ``flow_time`` for the grid being carried along, exactly).
        flow: Matrix3 = ()  # empty: off
        flow_time: float = Field(0.0, ge=0, le=50)

        @field_validator("matrix", "flow")
        @classmethod
        def _m(cls, v):
            return v if not v else _check_matrix(v)

    def _m(self, state):
        p = self.me(state).params
        if p.flow:
            return lm.expm(p.flow, p.flow_time)
        return lm.matrix(p.matrix)

    def _map(self, state, pts):
        p = self.me(state).params
        m = self._m(state)
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        if m.shape[0] == 2:
            out = pts[:, :2] @ m.T
            return np.c_[out, np.zeros(len(out))] + np.asarray(p.pos)
        return pts @ m.T + np.asarray(p.pos)

    def _extent_points(self, state):
        p = self.me(state).params
        n = p.extent * p.spacing
        m = self._m(state)
        corners = np.array(list(itertools.product((-n, n), repeat=m.shape[0])))
        if m.shape[0] == 2:
            corners = np.c_[corners, np.zeros(len(corners))]
        return self._map(state, corners)

    def world_bounds(self, state: SceneState) -> Bounds:
        return Bounds.of_points(self._extent_points(state))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        m = self._m(state)
        d = m.shape[0]
        basis = np.eye(d)
        pts = self._map(state, np.vstack([np.zeros(d), basis]))
        return {"origin": pts[0], "basis": pts[1:]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        m = self._m(state)
        ev = lm.eigen(m)["values"]
        return {
            "kind": "transform_grid",
            **({"backdrop": True} if self.me(state).params.flow else {}),  # a flow is a field
            "dimension": int(m.shape[0]),
            "matrix": m.round(6).tolist(),
            "determinant": round(lm.determinant(m), 6),
            "flips_orientation": bool(lm.determinant(m) < 0),
            "rank": lm.rank(m),
            "eigenvalues": [[round(float(v.real), 6), round(float(v.imag), 6)] for v in ev],
            "singular_values": [round(float(s), 6) for s in lm.svd(m)["S"]],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        m = self._m(state)
        d = m.shape[0]
        n, sp = p.extent, p.spacing
        w = me.style["stroke"] * float(cam.units_per_height())
        grid_c = with_alpha(me.style["grid"], p.opacity * 0.8)
        axis_c = with_alpha(me.style["color"], p.opacity)
        span = n * sp
        for i in range(-n, n + 1):  # lines of constant x and of constant y (z = 0 in 3D)
            for axis in range(2):
                a, b = np.zeros(d), np.zeros(d)
                a[axis], b[axis] = i * sp, i * sp
                a[1 - axis], b[1 - axis] = -span, span
                if d == 3:
                    pass
                pts = self._map(state, [a, b])
                polyline(
                    ctx,
                    cam,
                    pts,
                    w * (2.2 if i == 0 else 1.0),
                    color=axis_c if i == 0 else grid_c,
                    glow=me.style["glow"],
                )
        if p.square > 0:
            if d == 2:
                quad = self._map(state, [[0, 0], [1, 0], [1, 1], [0, 1]])
                ctx.triangles(
                    quad,
                    np.array([[0, 1, 2], [0, 2, 3]]),
                    color=with_alpha(me.style["i"], p.opacity * p.square * 0.5),
                )
                polyline(
                    ctx,
                    cam,
                    np.vstack([quad, quad[:1]]),
                    w * 1.6,
                    color=with_alpha(me.style["i"], p.opacity),
                )
            else:
                cube = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float)
                img = self._map(state, cube)
                for a, b in itertools.combinations(range(8), 2):
                    if np.abs(cube[a] - cube[b]).sum() == 1:
                        polyline(
                            ctx,
                            cam,
                            [img[a], img[b]],
                            w * 1.4,
                            color=with_alpha(me.style["i"], p.opacity * max(p.square, 0.4)),
                        )
        if p.basis > 0:
            o = self._map(state, np.zeros(d))[0]
            for k, role in enumerate(("i", "j", "color")[:d]):
                tip = self._map(state, np.eye(d)[k])[0]
                arrow(ctx, cam, o, tip, w * 3, with_alpha(me.style[role], p.opacity * p.basis))


# ---------------------------------------------------------------- Arrow


@register
class Arrow(ObjectBase):
    """A vector: an arrow from ``start`` to ``end``."""

    type: ClassVar[str] = "arrow"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {"glowing": {"style": {"glow": "soft"}}}

    class Params(ParamsBase):
        start: Vec3 = (0.0, 0.0, 0.0)
        end: Vec3 = (1.0, 0.0, 0.0)

    def _pts(self, state):
        p = self.me(state).params
        return np.asarray(p.start, dtype=np.float64), np.asarray(p.end, dtype=np.float64)

    def world_bounds(self, state: SceneState) -> Bounds:
        a, b = self._pts(state)
        return Bounds.of_points([a, b], pad=0.05)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        a, b = self._pts(state)
        return {"start": a, "tip": b}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        a, b = self._pts(state)
        return {
            "kind": "vector",
            "components": (b - a).round(6).tolist(),
            "length": round(float(np.linalg.norm(b - a)), 6),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        cam = state.camera_for(me)
        a, b = self._pts(state)
        w = me.style["stroke"] * float(cam.units_per_height((a + b) / 2))
        arrow(ctx, cam, a, b, w, with_alpha(me.style["color"], me.params.opacity), me.style["glow"])


# ---------------------------------------------------------------- EigenLines


@register
class EigenLines(ObjectBase):
    """The directions a 2x2 matrix only stretches: a line through ``pos`` along each real
    eigenvector, the stretch factor in the semantics. A rotation has none."""

    type: ClassVar[str] = "eigen_lines"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "other": ("emphasis", "secondary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        matrix: Matrix3 = ((2.0, 1.0), (1.0, 2.0))
        pos: Vec3 = (0.0, 0.0, 0.0)
        length: float = Field(4.0, gt=0)  # half-length of each line

        @field_validator("matrix")
        @classmethod
        def _m(cls, v):
            m = lm.matrix(v)
            if m.shape != (2, 2):
                raise ValueError("eigen_lines draws a 2x2 matrix")
            return v

    def _dirs(self, state):
        p = self.me(state).params
        e = lm.eigen(p.matrix)
        out = []
        for val, vec in zip(e["values"], e["vectors"].T, strict=True):
            if abs(val.imag) < 1e-9 and np.abs(vec.imag).max() < 1e-9:
                out.append((float(val.real), vec.real))
        return out

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c = np.asarray(p.pos, dtype=np.float64)
        dirs = self._dirs(state)
        pts = [c]
        for _, v in dirs:
            pts += [c + p.length * np.r_[v, 0.0], c - p.length * np.r_[v, 0.0]]
        return Bounds.of_points(pts, pad=0.05)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        c = np.asarray(self.me(state).params.pos, dtype=np.float64)
        v = [np.r_[d, 0.0] for _, d in self._dirs(state)]
        return {"center": c, "directions": np.array(v).reshape(-1, 3)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        e = lm.eigen(p.matrix)
        return {
            "kind": "eigen_lines",
            "real_directions": len(self._dirs(state)),
            "eigenvalues": [
                [round(float(v.real), 6), round(float(v.imag), 6)] for v in e["values"]
            ],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        c = np.asarray(p.pos, dtype=np.float64)
        w = me.style["stroke"] * float(cam.units_per_height())
        for k, (_, v) in enumerate(self._dirs(state)):
            d = np.r_[v, 0.0] * p.length
            role = "color" if k == 0 else "other"
            polyline(
                ctx,
                cam,
                [c - d, c + d],
                w,
                color=with_alpha(me.style[role], p.opacity),
                glow=me.style["glow"],
            )


# ---------------------------------------------------------------- Matrix


@register
class Matrix(ObjectBase):
    """A matrix printed with its brackets, entries live: give ``matrix`` an expression that
    reads another object's matrix and the numbers follow it."""

    type: ClassVar[str] = "matrix"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "size": ("text", "math"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),  # the prism look: brackets and entries shine
    }

    class Params(ParamsBase):
        matrix: Matrix3 = ((1.0, 0.0), (0.0, 1.0))
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        digits: int = Field(2, ge=0, le=6)

        @field_validator("matrix")
        @classmethod
        def _m(cls, v):
            return _check_matrix(v)

    def _layout(self, state):
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        h = me.style["size"] * float(cam.units_per_height(np.asarray(p.pos)))
        m = lm.matrix(p.matrix)
        texts = [
            [f"{(0.0 if abs(v) < 5 * 10 ** -(p.digits + 1) else v):.{p.digits}f}" for v in row]
            for row in m
        ]  # no "-0.00"
        cw = max(len(t) for row in texts for t in row) * h * 0.55 + h * 0.5
        rh = h * 1.35
        n = m.shape[0]
        c = np.asarray(p.pos, dtype=np.float64)
        w, hh = n * cw, n * rh
        return cam, h, texts, cw, rh, c, w, hh

    def world_bounds(self, state: SceneState) -> Bounds:
        cam, _, _, _, _, c, w, hh = self._layout(state)
        a = c - cam.right * (w / 2 + 0.2) - cam.up * (hh / 2)
        b = c + cam.right * (w / 2 + 0.2) + cam.up * (hh / 2)
        return Bounds.of_points([a, b])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        m = lm.matrix(self.me(state).params.matrix)
        return {
            "kind": "matrix",
            "rows": int(m.shape[0]),
            "entries": m.round(6).tolist(),
            "determinant": round(lm.determinant(m), 6),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        cam, h, texts, cw, rh, c, w, hh = self._layout(state)
        color = with_alpha(me.style["color"], me.params.opacity)
        gl = float(me.style.get("glow", 1.0))
        n = len(texts)
        for i, row in enumerate(texts):
            for j, t in enumerate(row):
                pos = c + cam.right * ((j - (n - 1) / 2) * cw) - cam.up * ((i - (n - 1) / 2) * rh)
                ctx.text(t, pos, h, color=color, anchor="center", glow=gl)
        width = me.style["stroke"] * float(cam.units_per_height(c)) * 2
        for sgn in (-1, 1):
            x = c + cam.right * (sgn * (w / 2 + 0.1))
            top, bot = x + cam.up * (hh / 2), x - cam.up * (hh / 2)
            tick = cam.right * (-sgn * 0.12)
            polyline(ctx, cam, [top + tick, top, bot, bot + tick], width, color=color, glow=gl)
