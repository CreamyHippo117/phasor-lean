"""Curves, surfaces and fields in space: the basics a math
video needs. Axes with ticks; a curve or surface from formulas (a helix, a sphere), colored by
what it outputs; a surface that can *map* a flat input square onto its output shape; vector
fields drawn as arrows or as flow lines. Give the scene an orbit camera to turn around them
(``s.use_camera("orbit")``, the ``orbit`` verb, or drag in the editor).

    s.add(geo.Axes(id="axes"))
    s.add(geo.ParametricCurve(id="helix", fx="cos(t)", fy="sin(t)", fz="0.2*t",
                              t_range=(0, 37.7)))
    s.add(geo.ParametricSurface(id="ball", fx="2*sin(u)*cos(v)", fy="2*sin(u)*sin(v)",
                                fz="2*cos(u)", u_range=(0, 1.57), v_range=(0, 6.28),
                                color_by="height", with_map=True, domain_pos=(-6, 0, 0),
                                morph=expr("t / 4")))

Numerics (formulas, flow lines, coloring) are in ``geometry_math``.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.color import ramp
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import geometry_math as gm
from phasor.domains.strokes import field_arrows, polyline, polylines, with_alpha

T = ("t",)
UV = ("u", "v")
XYZ = ("x", "y", "z")
COLOR_BY_CURVE = Literal["none", "t", "speed", "x", "y", "z"]
COLOR_BY_SURFACE = Literal["none", "height", "u", "v", "radius", "expr"]


def _vec(p) -> np.ndarray:
    return np.asarray(p, dtype=np.float64)


# ---------------------------------------------------------------- Axes


@register
class Axes(ObjectBase):
    """Coordinate axes with tick marks (2D: x and y; 3D: x, y and z), each running from its
    minimum to its maximum through the origin. Ranges are (min, max, tick step) in axis units;
    ``scale`` is world units per axis unit. ``labels`` ("x", "y", "z") are drawn at the ends."""

    type: ClassVar[str] = "axes"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "stroke": ("stroke", "hairline"),
        "size": ("text", "annotation"),
    }

    class Params(ParamsBase):
        dims: Literal[2, 3] = 3
        x_range: tuple[float, float, float] = (-3.0, 3.0, 1.0)
        y_range: tuple[float, float, float] = (-3.0, 3.0, 1.0)
        z_range: tuple[float, float, float] = (-1.0, 6.0, 1.0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # where the axes cross
        scale: float = Field(1.0, gt=0)
        tick: float = Field(0.12, gt=0)  # tick length in world units
        labels: tuple[str, ...] = ()

        @model_validator(mode="after")
        def _ranges(self):
            for name in ("x_range", "y_range", "z_range"):
                lo, hi, step = getattr(self, name)
                if hi <= lo or step <= 0:
                    raise ValueError(f"{name} must be (min, max, step) with min < max, step > 0")
            return self

    def _axes(self, p) -> list[tuple[np.ndarray, tuple[float, float, float], np.ndarray]]:
        """(direction, range, tick direction) per axis."""
        e = np.eye(3)
        out = [(e[0], p.x_range, e[1]), (e[1], p.y_range, e[0])]
        if p.dims == 3:
            out.append((e[2], p.z_range, e[0]))
        return out

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        o = _vec(p.pos)
        pts = [o + d * lo * p.scale for d, (lo, _, _), _ in self._axes(p)]
        pts += [o + d * hi * p.scale for d, (_, hi, _), _ in self._axes(p)]
        return Bounds.of_points(pts, pad=p.tick)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        o = _vec(p.pos)
        out = {"origin": o}
        for name, (d, (_, hi, _), _) in zip("xyz", self._axes(p), strict=False):
            out[f"{name}_end"] = o + d * hi * p.scale
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        axes = {
            n: {"range": [r[0], r[1]], "step": r[2], "ticks": len(gm.tick_values(*r))}
            for n, (_, r, _) in zip("xyz", self._axes(p), strict=False)
        }
        return {"kind": "axes", "dims": p.dims, "axes": axes, "scale": p.scale}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        o = _vec(p.pos)
        col = with_alpha(me.style["color"], p.opacity)
        size = me.style["size"] * float(cam.units_per_height(o))
        for i, (d, (lo, hi, step), td) in enumerate(self._axes(p)):
            a, b = o + d * lo * p.scale, o + d * hi * p.scale
            w = me.style["stroke"] * float(cam.units_per_height((a + b) / 2))
            polyline(ctx, cam, [a, b], w, color=col)
            for v in gm.tick_values(lo, hi, step):
                c = o + d * v * p.scale
                polyline(ctx, cam, [c - td * p.tick / 2, c + td * p.tick / 2], w, color=col)
            if i < len(p.labels):
                ctx.text(p.labels[i], b + d * size * 0.9, size, color=col)


# ---------------------------------------------------------------- ParametricCurve


@register
class ParametricCurve(ObjectBase):
    """A curve (x(t), y(t), z(t)) from formulas in ``t``, optionally colored by what it
    outputs: its parameter, its speed, or a coordinate (``color_by``, through the
    ``colormap``). ``progress`` draws the first share of it (the trace verb)."""

    type: ClassVar[str] = "parametric_curve"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "sequential"),
    }

    class Params(ParamsBase):
        fx: str = "cos(t)"
        fy: str = "sin(t)"
        fz: str = "0"
        t_range: tuple[float, float] = (0.0, 6.2832)
        samples: int = Field(400, ge=2, le=20000)
        pos: Vec3 = (0.0, 0.0, 0.0)
        scale: float = Field(1.0, gt=0)
        progress: float = Field(1.0, ge=0, le=1)
        color_by: COLOR_BY_CURVE = "none"
        hue: float = 0.0  # shifts the colors along the ramp (a share of it, wrapping)
        # Flowing into another curve: formulas to blend toward ("" keeps that coordinate's
        # own), and how far (0: this curve, 1: the other). Animate ``blend`` with ``morph``.
        to_fx: str = ""
        to_fy: str = ""
        to_fz: str = ""
        blend: float = Field(0.0, ge=0, le=1)

        @field_validator("fx", "fy", "fz")
        @classmethod
        def _f(cls, v):
            return gm.check(v, T)

        @field_validator("to_fx", "to_fy", "to_fz")
        @classmethod
        def _to(cls, v):
            return gm.check(v, T) if v else v

        @model_validator(mode="after")
        def _range(self):
            if self.t_range[1] <= self.t_range[0]:
                raise ValueError("t_range must be (min, max) with min < max")
            return self

    def _curve(self, p):
        t = np.linspace(p.t_range[0], p.t_range[1], p.samples)
        pts = np.stack([gm.compile_fn(f, T)(t) for f in (p.fx, p.fy, p.fz)], axis=1)
        if p.blend > 0 and (p.to_fx or p.to_fy or p.to_fz):
            pairs = ((p.fx, p.to_fx), (p.fy, p.to_fy), (p.fz, p.to_fz))
            to = np.stack([gm.compile_fn(g or f, T)(t) for f, g in pairs], axis=1)
            pts = pts * (1 - p.blend) + to * p.blend
        return t, _vec(p.pos) + p.scale * pts

    def path_points(self, state: SceneState) -> np.ndarray:
        """The whole curve (what a ``Tracer`` travels along)."""
        return self._curve(self.me(state).params)[1]

    def _drawn(self, p, t, pts):
        n = max(2, round(p.progress * (len(t) - 1)) + 1)
        return t[:n], pts[:n]

    def _quantity(self, p, t, pts) -> np.ndarray:
        if p.color_by == "t":
            return t
        if p.color_by == "speed":
            v = np.linalg.norm(np.gradient(pts, t, axis=0), axis=1)
            return v
        return pts[:, "xyz".index(p.color_by)]

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds.of_points(self._curve(p)[1], pad=0.02)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        _, pts = self._curve(p)
        return {"start": pts[0], "end": pts[-1], "curve": pts}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        t, pts = self._curve(p)
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        out: dict[str, Any] = {
            "kind": "parametric_curve",
            "formula": [p.fx, p.fy, p.fz],
            "t_range": list(p.t_range),
            "length": round(length, 6),
            "closed": bool(np.linalg.norm(pts[0] - pts[-1]) < 1e-6 * max(length, 1)),
            "start": [round(float(v), 6) for v in pts[0]],
            "end": [round(float(v), 6) for v in pts[-1]],
        }
        if p.color_by != "none":
            q = self._quantity(p, t, pts)
            out["color_by"] = p.color_by
            out["color_range"] = [round(float(q.min()), 6), round(float(q.max()), 6)]
        if p.progress < 1:
            out["drawn"] = round(p.progress, 4)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        if p.progress <= 0:
            return
        cam = state.camera_for(me)
        t, pts = self._curve(p)
        full_q = None if p.color_by == "none" else gm.normalize01(self._quantity(p, t, pts))
        t, pts = self._drawn(p, t, pts)
        colors = None
        if full_q is not None:
            u = (full_q[: len(pts)] + p.hue) % 1.0 if p.hue else full_q[: len(pts)]
            rgb = gm.sample_ramp(ramp(me.style["colormap"]), u)
            colors = np.c_[rgb, np.ones(len(pts))]
        width = me.style["stroke"] * cam.units_per_height(pts)
        polyline(ctx, cam, pts, width, color=with_alpha(me.style["color"], p.opacity),
                 glow=me.style["glow"], colors=colors)  # fmt: skip


# ---------------------------------------------------------------- ParametricSurface


@register
class ParametricSurface(ObjectBase):
    """A surface (x(u, v), y(u, v), z(u, v)) from formulas: a sphere, a torus, a ribbon.
    Colored by an output (``color_by``: height, u, v, distance from the center, or any formula
    of u, v, x, y, z in ``color_expr``) through the ``colormap``.

    **Mapping from input to output space.** With ``with_map=True``, the flat (u, v) input
    rectangle is drawn at ``domain_pos`` as a faint grid, and ``morph`` (0..1) carries the
    surface from that flat input to its true shape: animate ``morph`` to show the map
    happening. At ``morph = 1`` it is the output shape; the ghost grid stays behind."""

    type: ClassVar[str] = "parametric_surface"
    Style: ClassVar = {"colormap": ("colormap", "sequential"), "material": ("surface", "solid")}
    Presets: ClassVar = {
        "glass": {"style": {"material": "glass"}},
        "grid": {"style": {"material": "wire"}},
    }

    class Params(ParamsBase):
        fx: str = "u"
        fy: str = "v"
        fz: str = "0"
        u_range: tuple[float, float] = (-2.0, 2.0)
        v_range: tuple[float, float] = (-2.0, 2.0)
        res: tuple[int, int] = (40, 40)
        pos: Vec3 = (0.0, 0.0, 0.0)
        scale: float = Field(1.0, gt=0)
        color_by: COLOR_BY_SURFACE = "none"
        color_expr: str = "z"  # when color_by is "expr": a formula in u, v, x, y, z
        with_map: bool = False  # show the flat input rectangle and let ``morph`` carry it over
        domain_pos: Vec3 = (-4.0, 0.0, 0.0)  # where the flat input rectangle sits
        domain_size: tuple[float, float] = (3.0, 3.0)  # the input rectangle's size in the world
        morph: float = Field(1.0, ge=0, le=1)  # 0: flat input; 1: the output shape
        # Flowing into another surface (z = x^2 - y^2 into z = sin x cos y): formulas to blend
        # toward ("" keeps that coordinate's own) and how far. Animate ``blend`` with ``morph``.
        to_fx: str = ""
        to_fy: str = ""
        to_fz: str = ""
        blend: float = Field(0.0, ge=0, le=1)

        @field_validator("fx", "fy", "fz")
        @classmethod
        def _f(cls, v):
            return gm.check(v, UV)

        @field_validator("to_fx", "to_fy", "to_fz")
        @classmethod
        def _to(cls, v):
            return gm.check(v, UV) if v else v

        @field_validator("color_expr")
        @classmethod
        def _c(cls, v):
            return gm.check(v, ("u", "v", "x", "y", "z"))

        @model_validator(mode="after")
        def _ranges(self):
            for name in ("u_range", "v_range"):
                lo, hi = getattr(self, name)
                if hi <= lo:
                    raise ValueError(f"{name} must be (min, max) with min < max")
            if min(self.res) < 2 or max(self.res) > 400:
                raise ValueError("res is (nu, nv), each between 2 and 400")
            return self

    def _grid(self, p):
        fx, fy, fz = (gm.compile_fn(f, UV) for f in (p.fx, p.fy, p.fz))
        pts, U, V = gm.surface_grid(fx, fy, fz, p.u_range, p.v_range, p.res)
        if p.blend > 0 and (p.to_fx or p.to_fy or p.to_fz):
            gx, gy, gz = (gm.compile_fn(g or f, UV) for f, g in
                          ((p.fx, p.to_fx), (p.fy, p.to_fy), (p.fz, p.to_fz)))  # fmt: skip
            to, _, _ = gm.surface_grid(gx, gy, gz, p.u_range, p.v_range, p.res)
            pts = pts * (1 - p.blend) + to * p.blend
        return _vec(p.pos) + p.scale * pts, U, V

    def _flat(self, p, U, V) -> np.ndarray:
        """The input rectangle as a grid of points where ``domain_pos`` puts it."""
        su = (U - p.u_range[0]) / (p.u_range[1] - p.u_range[0]) - 0.5
        sv = (V - p.v_range[0]) / (p.v_range[1] - p.v_range[0]) - 0.5
        d = _vec(p.domain_pos)
        return np.stack([d[0] + su * p.domain_size[0], d[1] + sv * p.domain_size[1],
                         np.full_like(su, d[2])], axis=-1)  # fmt: skip

    def _positions(self, p):
        pts, U, V = self._grid(p)
        if not p.with_map:
            return pts, U, V, pts
        flat = self._flat(p, U, V)
        return flat * (1 - p.morph) + pts * p.morph, U, V, pts

    def _quantity(self, p, pts, U, V) -> np.ndarray:
        rel = (pts - _vec(p.pos)) / p.scale
        if p.color_by == "height":
            return rel[..., 2]
        if p.color_by == "u":
            return U
        if p.color_by == "v":
            return V
        if p.color_by == "radius":
            return np.linalg.norm(rel, axis=-1)
        return gm.compile_fn(p.color_expr, ("u", "v", "x", "y", "z"))(
            U, V, rel[..., 0], rel[..., 1], rel[..., 2]
        )

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        pos = self._positions(p)[0]
        pts = [pos.reshape(-1, 3)]
        if p.with_map:
            pts.append(self._flat(p, *self._grid(p)[1:]).reshape(-1, 3))
        return Bounds.of_points(np.vstack(pts), pad=0.02)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        pos = self._positions(p)[0]
        top = np.unravel_index(int(np.argmax(pos[..., 2])), pos.shape[:2])
        return {"center": pos.reshape(-1, 3).mean(axis=0), "top": pos[top],
                "corners": pos[[0, 0, -1, -1], [0, -1, 0, -1]]}  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        pts, U, V = self._grid(p)
        out: dict[str, Any] = {
            "kind": "parametric_surface",
            "formula": [p.fx, p.fy, p.fz],
            "u_range": list(p.u_range),
            "v_range": list(p.v_range),
            "area": round(gm.area_of_grid(pts), 6),
            "material": self.me(state).roles["material"],
        }
        if p.color_by != "none":
            q = self._quantity(p, pts, U, V)
            out["color_by"] = p.color_by if p.color_by != "expr" else p.color_expr
            out["color_range"] = [round(float(np.nanmin(q)), 6), round(float(np.nanmax(q)), 6)]
        if p.with_map:
            out["maps"] = {"from": "the flat (u, v) rectangle", "morph": round(p.morph, 4)}
        return out

    def _mesh(self, pos: np.ndarray):
        nu, nv = pos.shape[:2]
        du, dv = np.gradient(pos, axis=0), np.gradient(pos, axis=1)
        n = np.cross(du, dv)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        k = (np.arange(nu - 1)[:, None] * nv + np.arange(nv - 1)[None, :]).ravel()
        tris = np.vstack([np.c_[k, k + nv, k + 1], np.c_[k + 1, k + nv, k + nv + 1]])
        uv = np.stack(np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij"), -1)
        return n.reshape(-1, 3), tris, uv.reshape(-1, 2)

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        pos, U, V, true_pts = self._positions(p)
        normals, tris, uv = self._mesh(pos)
        if p.color_by == "none":
            colors = np.tile([*me.style["colormap"][len(me.style["colormap"]) // 2], 1.0],
                             (pos.shape[0] * pos.shape[1], 1))  # fmt: skip
        else:
            u = gm.normalize01(self._quantity(p, true_pts, U, V))
            rgb = gm.sample_ramp(ramp(me.style["colormap"]), u)
            colors = np.c_[rgb.reshape(-1, 3), np.ones(rgb.size // 3)]
        ctx.surface(pos.reshape(-1, 3), tris, normals=normals, colors=colors, uv=uv,
                    material=me.roles["material"],
                    opacity=p.opacity * me.style["material"])  # fmt: skip
        if p.with_map and p.morph > 0:  # the input stays behind, as a faint grid
            flat = self._flat(p, U, V)
            fn, ftris, fuv = self._mesh(flat)
            ctx.surface(flat.reshape(-1, 3), ftris, normals=fn, colors=colors, uv=fuv,
                        material="wire", opacity=0.35 * p.opacity)  # fmt: skip


# ---------------------------------------------------------------- vector fields


def _field(p):
    fns = [gm.compile_fn(s, XYZ) for s in (p.fp, p.fq, p.fr)]

    def at(pts: np.ndarray) -> np.ndarray:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return np.stack([f(x, y, z) for f in fns], axis=1)

    return at


class _FieldParams(ParamsBase):
    fp: str = "-y"
    fq: str = "x"
    fr: str = "0"
    pos: Vec3 = (0.0, 0.0, 0.0)  # center of the box the field is drawn in
    extent: tuple[float, float, float] = (4.0, 4.0, 0.0)  # its size (0 deep: a plane)

    @field_validator("fp", "fq", "fr")
    @classmethod
    def _f(cls, v):
        return gm.check(v, XYZ)


def _grid_points(center, extent, counts) -> np.ndarray:
    axes = [np.linspace(-e / 2, e / 2, n) if n > 1 else np.zeros(1)
            for e, n in zip(extent, counts, strict=True)]  # fmt: skip
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return _vec(center) + g


@register
class VectorField3D(ObjectBase):
    """A vector field F(x, y, z) = (fp, fq, fr) as arrows on a grid in a box, each colored by
    its size (the ``colormap``), lengths scaled to the grid spacing. A box with no depth
    (``extent`` z = 0) is a plane of arrows. Semantics: divergence, curl, strongest arrow.

    ``spacing`` > 0 places arrows that far apart instead of ``counts`` per axis. Presets:
    ``dense`` (many small thin arrows, lengths eased so weak ones still show, color saying
    how strong), ``bold`` (few thick glowing arrows)."""

    type: ClassVar[str] = "vector_field_3d"
    Style: ClassVar = {
        "colormap": ("colormap", "sequential"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {
        "dense": {"params": {"spacing": 0.45, "length_by": "soft"},
                  "style": {"colormap": "magnitude", "stroke": "hairline"}},
        "bold": {"params": {"counts": (5, 5, 1)},
                 "style": {"stroke": "regular", "glow": "soft"}},
    }  # fmt: skip

    class Params(_FieldParams):
        counts: tuple[int, int, int] = (7, 7, 1)  # arrows along x, y, z
        spacing: float = Field(0.0, ge=0)  # > 0: arrows this far apart (instead of counts)
        color_by_size: bool = True
        length_by: Literal["size", "soft"] = "size"  # "soft": weak arrows still show

    def _counts(self, p) -> tuple[int, int, int]:
        if p.spacing <= 0:
            return p.counts
        return tuple(max(1, int(e / p.spacing + 1e-9) + 1) if e > 0 else 1
                     for e in p.extent)  # type: ignore[return-value]  # fmt: skip

    def _spacing(self, p) -> float:
        if p.spacing > 0:
            return p.spacing
        gaps = [e / (n - 1) for e, n in zip(p.extent, p.counts, strict=True) if e > 0 and n > 1]
        return min(gaps) if gaps else 1.0

    def _arrows(self, p):
        pts = _grid_points(p.pos, p.extent, self._counts(p))
        v = _field(p)(pts - _vec(p.pos))
        return pts, v

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c, e = _vec(p.pos), _vec(p.extent)
        return Bounds.of_points([c - e / 2, c + e / 2], pad=0.05)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": _vec(self.me(state).params.pos)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        import sympy as sp

        p = self.me(state).params
        _, v = self._arrows(p)
        mag = np.linalg.norm(v, axis=1)
        cx, cy, cz = gm.curl(p.fp, p.fq, p.fr)
        return {"kind": "vector_field_3d", "field": [p.fp, p.fq, p.fr],
                "divergence": sp.sstr(gm.divergence(p.fp, p.fq, p.fr)),
                "curl": [sp.sstr(cx), sp.sstr(cy), sp.sstr(cz)],
                "arrows": len(mag), "strongest": round(float(np.nanmax(mag)), 6)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        pts, v = self._arrows(p)
        mag = np.linalg.norm(v, axis=1)
        w = me.style["stroke"] * float(cam.units_per_height(_vec(p.pos))) * 2.5
        if p.color_by_size:
            rgb = gm.sample_ramp(ramp(me.style["colormap"]), gm.normalize01(mag, 0.0, None))
            cols = np.c_[rgb, np.full(len(rgb), p.opacity)]
        else:
            # One color: a bright point on the ramp (its first entry is the background).
            rgb = gm.sample_ramp(ramp(me.style["colormap"]), np.array([0.7]))[0]
            cols = (*rgb, p.opacity)
        field_arrows(ctx, cam, pts, v, self._spacing(p), width=w, colors=cols,
                     glow=me.style["glow"], length_by=p.length_by)  # fmt: skip


@register
class FlowLines(ObjectBase):
    """Flow lines of a vector field: where a particle dropped at each seed would travel. Lines
    are colored by the field's size along them. ``progress`` grows them from the seeds (the
    flow arriving). Seeds sit on a grid in the box; ``both`` runs them backward as well."""

    type: ClassVar[str] = "flow_lines"
    Style: ClassVar = {
        "colormap": ("colormap", "ocean"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),
    }

    class Params(_FieldParams):
        seeds: tuple[int, int, int] = (5, 5, 1)
        steps: int = Field(120, ge=2, le=4000)
        length: float = Field(3.0, gt=0)  # arc length of each line (per direction)
        both: bool = False
        progress: float = Field(1.0, ge=0, le=1)

    def _lines(self, p):
        seeds = _grid_points(p.pos, p.extent, p.seeds)
        field = _field(p)
        return gm.flow_lines(lambda q: field(q - _vec(p.pos)), seeds, p.steps,
                             p.length / p.steps, p.both)  # fmt: skip

    def world_bounds(self, state: SceneState) -> Bounds:
        lines, _ = self._lines(self.me(state).params)
        return Bounds.of_points(lines.reshape(-1, 3), pad=0.05)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return {"seeds": _grid_points(p.pos, p.extent, p.seeds)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        lines, speed = self._lines(p)
        arc = np.linalg.norm(np.diff(lines, axis=1), axis=2).sum(axis=1)
        return {"kind": "flow_lines", "field": [p.fp, p.fq, p.fr], "lines": len(lines),
                "mean_length": round(float(arc.mean()), 6),
                "strongest": round(float(np.nanmax(speed)), 6)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        if p.progress <= 0:
            return
        cam = state.camera_for(me)
        lines, speed = self._lines(p)
        top = float(np.nanmax(speed)) if np.isfinite(speed).any() else 1.0
        lut = ramp(me.style["colormap"])
        n = max(2, round(p.progress * (lines.shape[1] - 1)) + 1)
        for ln, sp_ in zip(lines, speed, strict=True):
            pts = ln[:n]
            if np.linalg.norm(pts[-1] - pts[0]) < 1e-9:
                continue  # the field vanishes here: nothing to draw
            rgb = gm.sample_ramp(lut, gm.normalize01(sp_[:n], 0.0, top))
            width = me.style["stroke"] * cam.units_per_height(pts)
            polyline(ctx, cam, pts, width, color=with_alpha((1.0, 1.0, 1.0, 1.0), p.opacity),
                     glow=me.style["glow"], colors=np.c_[rgb, np.ones(len(pts))])  # fmt: skip


@register
class FlowParticles(ObjectBase):
    """Particles streaming along a vector field with fading tails: the field as motion.
    ``count`` particles start at random points in the box (``rng(id, seed)``), travel
    ``speed`` world units a second along the field's flow lines for ``lifetime`` seconds,
    then start again: a pure function of time, so every frame is reproducible. Colored by
    the field's strength (the ``magnitude`` colormap)."""

    type: ClassVar[str] = "flow_particles"
    Style: ClassVar = {
        "colormap": ("colormap", "magnitude"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
    }
    Presets: ClassVar = {
        "streaks": {"params": {"tail": 0.45, "count": 300}, "style": {"stroke": "hairline"}},
        "sparks": {"params": {"tail": 0.08, "count": 400}, "style": {"glow": "strong"}},
    }

    class Params(_FieldParams):
        count: int = Field(160, ge=1, le=5000)
        speed: float = Field(1.0, gt=0)  # world units a second
        lifetime: float = Field(3.0, gt=0)  # seconds before a particle starts again
        tail: float = Field(0.2, ge=0, le=1)  # its tail, as a share of its life's path
        seed: int = 0

    def _paths(self, object_id: str, p):
        return _particle_paths(object_id, p)

    def world_bounds(self, state: SceneState) -> Bounds:
        me = self.me(state)
        paths, _, _ = self._paths(me.id, me.params)
        return Bounds.of_points(paths.reshape(-1, 3), pad=0.05)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": _vec(self.me(state).params.pos)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        me = self.me(state)
        p = me.params
        _, _, strength = self._paths(me.id, p)
        return {"kind": "flow_particles", "field": [p.fp, p.fq, p.fr], "count": p.count,
                "speed": p.speed, "lifetime": p.lifetime,
                "strongest": round(float(np.nanmax(strength)), 6)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        if p.opacity <= 0:
            return
        cam = state.camera_for(me)
        paths, phase, strength = self._paths(me.id, p)
        seg, alpha = gm.particles(paths, phase, state.t, p.lifetime, max(p.tail, 0.02))
        top = float(np.nanmax(strength)) if np.isfinite(strength).any() else 1.0
        rgb = gm.sample_ramp(ramp(me.style["colormap"]), gm.normalize01(strength, 0.0, top))
        # Skip the invisible and the still; draw the rest as one batch of strokes, in order.
        keep = (alpha.max(axis=1) > 0.01) & (np.linalg.norm(seg[:, -1] - seg[:, 0], axis=1) >= 1e-9)
        seg, alpha, rgb = seg[keep], alpha[keep], rgb[keep]
        if not len(seg):
            return
        count, k = seg.shape[:2]
        cols = np.empty((count, k, 4))
        cols[..., :3], cols[..., 3] = rgb[:, None, :], alpha * p.opacity
        width = me.style["stroke"] * np.asarray(cam.units_per_height(seg.reshape(-1, 3)))
        # Per-point colors carry the hue; of ``color`` the shader uses only the opacity.
        polylines(ctx, cam, seg, width.reshape(count, k), color=(1.0, 1.0, 1.0, p.opacity),
                  glow=me.style["glow"], colors=cols)  # fmt: skip


@functools.lru_cache(maxsize=32)
def _particle_paths(object_id: str, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(paths (N, M, 3), birth phases (N,), field strength along each (N,)) for a
    FlowParticles' parameters: random starts from ``rng(id, seed)``, flow lines by RK4."""
    from phasor.core.rng import rng

    g = rng(object_id, p.seed)
    e = _vec(p.extent)
    starts = _vec(p.pos) + (g.random((p.count, 3)) - 0.5) * e
    phase = g.random(p.count)
    field = _field(p)
    steps = 48
    paths, speed = gm.flow_lines(lambda q: field(q - _vec(p.pos)), starts, steps,
                                 p.speed * p.lifetime / steps)  # fmt: skip
    return paths, phase, np.nanmean(speed, axis=1)
