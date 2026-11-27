"""Nets of convex polyhedra and geodesics on them (``net_math`` has the maths).

    from phasor import net

    s.add(net.Polyhedron(id="cube", size=(2, 2, 2), root="top",
                         geodesic=[(0, 0, 0), (2, 2, 2)], unfold=0.0))
    s.at("open").morph("cube.unfold", 1.0)     # the faces swing open into a flat net

A ``Polyhedron`` is a composite: its faces are parts (``cube.face_top``, ...), each a
``Face`` carried through the chain of hinge turns from it up the net's tree to the root
face, every turn ``unfold`` of the way. With ``geodesic`` (two points on the surface) the
shortest path between them is the part ``cube.geodesic`` (a ``Line``, its ends the dots
``cube.geodesic_start`` and ``cube.geodesic_end``): bent over the edges of the solid,
straight on the net.
"""

from __future__ import annotations

import functools
import itertools
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, model_validator

from phasor.core import unfold_math
from phasor.core.ids import is_valid_name
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import net_math
from phasor.domains.basic import Dot, Line
from phasor.domains.strokes import polyline, with_alpha

LIFT = 0.004  # the path rides this share of the solid's size above the faces


# ---------------------------------------------------------------- Face


@register
class Face(ObjectBase):
    """A flat convex polygon in 3D (its ``points`` in order round it): a lit, see-through
    fill and an outline. The faces of a ``Polyhedron`` are these."""

    type: ClassVar[str] = "face"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
        "material": ("surface", "glass"),
    }

    class Params(ParamsBase):
        points: tuple[Vec3, ...] = Field(min_length=3)
        fill: float = Field(0.35, ge=0, le=1)  # share of the material's opacity

    def _pts(self, state: SceneState) -> np.ndarray:
        return np.asarray(self.me(state).params.points, dtype=np.float64)

    def _normal(self, pts: np.ndarray) -> np.ndarray:
        n = np.cross(pts - pts.mean(axis=0), np.roll(pts, -1, axis=0) - pts.mean(axis=0)).sum(0)
        size = np.linalg.norm(n)
        return n / size if size > 0 else np.array([0.0, 0.0, 1.0])

    def world_bounds(self, state: SceneState) -> Bounds:
        return Bounds.of_points(self._pts(state))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self._pts(state)
        return {"center": p.mean(axis=0), "vertices": p}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self._pts(state)
        c = p.mean(axis=0)
        area = float(np.linalg.norm(np.cross(p - c, np.roll(p, -1, axis=0) - c).sum(0)) / 2)
        return {"kind": "face", "sides": len(p), "area": round(area, 6),
                "center": [round(float(v), 6) for v in c],
                "normal": [round(float(v), 6) for v in self._normal(p)],
                "backdrop": True}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        pts = self._pts(state)
        cam = state.camera_for(me)
        color = me.style["color"]
        if p.fill > 0:
            k = np.arange(1, len(pts) - 1)
            ctx.surface(pts, np.c_[np.zeros_like(k), k, k + 1],
                        normals=np.tile(self._normal(pts), (len(pts), 1)),
                        colors=np.tile(np.asarray(color), (len(pts), 1)),
                        material=me.roles["material"],
                        opacity=p.opacity * p.fill * float(me.style["material"]))  # fmt: skip
        ring = np.vstack([pts, pts[:1]])
        polyline(ctx, cam, ring, me.style["stroke"] * cam.units_per_height(ring),
                 color=with_alpha(color, p.opacity), glow=me.style["glow"])  # fmt: skip


# ---------------------------------------------------------------- Polyhedron


@functools.lru_cache(maxsize=64)
def _solid(size, origin, vertices, faces, names) -> net_math.Solid:
    if not faces:
        return net_math.box(size, origin)
    return net_math.solid(vertices, faces, names)


@functools.lru_cache(maxsize=256)
def _geodesic(s: net_math.Solid, a, b) -> net_math.Geodesic:
    return net_math.geodesic(s, a, b)


@functools.lru_cache(maxsize=256)
def _layout(s: net_math.Solid, root: str, tree, run) -> dict[int, int]:
    r = s.index(root)
    if not tree:
        return net_math.spanning_tree(s, r, run)
    parent = {s.index(f): s.index(g) for f, g in tree}
    net_math.check_tree(s, r, parent)
    for f, g in itertools.pairwise(run):
        if parent.get(f) != g and parent.get(g) != f:
            raise ValueError(f"the tree cuts the geodesic between {s.names[f]!r} and "
                             f"{s.names[g]!r}; keep them joined or leave tree out")  # fmt: skip
    return parent


@register
class Polyhedron(ObjectBase):
    """A convex polyhedron that opens out into a flat net (``unfold`` 0: the closed solid,
    1: the net), its faces hinged on shared edges; optionally the shortest path on its
    surface between two points (``geodesic``), which straightens as the net opens.

    The solid: a box (``size``, corner at ``origin``; faces ``bottom``, ``top``, ``south``,
    ``north``, ``west``, ``east``), or any convex polyhedron (``vertices``, ``faces`` as
    vertex indices, ``names`` one per face, default ``f0``, ``f1``, ...). The net: ``root``
    stays put (default ``bottom``, or the first face); ``tree`` maps a face to the face it
    hangs on (default: joined breadth first from the geodesic's faces, or the root, so the
    geodesic lies straight on the net)."""

    type: ClassVar[str] = "polyhedron"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "path": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "soft"),
        "material": ("surface", "glass"),
    }

    def __init__(self, id: str, **kw: Any) -> None:
        if isinstance(kw.get("tree"), dict):  # {face: the face it hangs on}, stored as pairs
            kw["tree"] = [[f, g] for f, g in sorted(kw["tree"].items())]
        super().__init__(id, **kw)

    class Params(ParamsBase):
        size: Vec3 = (2.0, 2.0, 2.0)
        origin: Vec3 = (0.0, 0.0, 0.0)
        vertices: tuple[Vec3, ...] = ()
        faces: tuple[tuple[int, ...], ...] = ()
        names: tuple[str, ...] = ()
        root: str = ""  # the face that stays put; empty: bottom (a box) or the first face
        tree: tuple[tuple[str, str], ...] = ()  # (face, the face it hangs on); empty: auto
        unfold: float = Field(0.0, ge=0, le=1)  # 0: the solid, 1: the flat net
        geodesic: tuple[Vec3, ...] = ()  # two points on the surface, or none
        trace: float = Field(1.0, ge=0, le=1)  # share of the geodesic drawn
        fill: float = Field(0.35, ge=0, le=1)

        @model_validator(mode="after")
        def _check(self):
            if any(not is_valid_name(n) for n in self.names):
                raise ValueError("face names are lower-case names like 'top' or 'f3'")
            if bool(self.vertices) != bool(self.faces):
                raise ValueError("a polyhedron needs both vertices and faces (or a box size)")
            if self.geodesic and len(self.geodesic) != 2:
                raise ValueError("a geodesic runs between two points")
            s = self.solid()
            self.layout(s)
            if self.geodesic:
                self.path(s)
            return self

        def solid(self) -> net_math.Solid:
            return _solid(tuple(self.size), tuple(self.origin), tuple(self.vertices),
                          tuple(self.faces), tuple(self.names))  # fmt: skip

        def root_name(self, s: net_math.Solid) -> str:
            return self.root or ("bottom" if "bottom" in s.names else s.names[0])

        def path(self, s: net_math.Solid) -> net_math.Geodesic | None:
            if not self.geodesic:
                return None
            return _geodesic(s, tuple(self.geodesic[0]), tuple(self.geodesic[1]))

        def layout(self, s: net_math.Solid) -> dict[int, int]:
            g = self.path(s)
            return _layout(s, self.root_name(s), tuple(self.tree), g.faces if g else ())

    def matrices(self, state: SceneState) -> np.ndarray:
        """(F, 4, 4): each face's place now, from the solid."""
        p = self.me(state).params
        s = p.solid()
        return net_math.net_matrices(s, s.index(p.root_name(s)), p.layout(s), p.unfold)

    def _path_points(self, s, g, m) -> np.ndarray:
        """The geodesic's points moved with their faces, lifted a hair off the surface."""
        lift = LIFT * s.size
        out = []
        last = len(g.points) - 1
        for k, q in enumerate(g.points):
            on = [g.faces[min(k, len(g.faces) - 1)]] if k in (0, last) else g.faces[k - 1 : k + 1]
            pos = unfold_math.apply(m[on[0]], q)
            n = sum(m[f][:3, :3] @ s.normals[f] for f in on)
            out.append(pos + lift * n / np.linalg.norm(n))
        return np.asarray(out)

    def build(self, state: SceneState) -> list[ObjectBase]:
        me = self.me(state)
        p = me.params
        s = p.solid()
        m = self.matrices(state)
        face_style = {k: me.roles[k] for k in ("color", "stroke", "glow", "material")}
        parts: list[ObjectBase] = [
            Face(id=f"face_{name}", fill=p.fill, style=face_style,
                 points=[tuple(x) for x in unfold_math.apply(m[k], s.v[list(f)])])
            for k, (f, name) in enumerate(zip(s.faces, s.names, strict=True))
        ]  # fmt: skip
        g = p.path(s)
        if g is not None:
            pts = self._path_points(s, g, m)
            look = {"color": me.roles["path"], "glow": me.roles["glow"]}
            parts.append(Line(id="geodesic", points=[tuple(x) for x in pts], progress=p.trace,
                              style={**look, "stroke": "bold"}))  # fmt: skip
            r = 0.035 * s.size
            parts.append(Dot(id="geodesic_start", pos=tuple(pts[0]), r=r, style=look))
            parts.append(Dot(id="geodesic_end", pos=tuple(pts[-1]), r=r, style=look))
        return parts

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        s = p.solid()
        parent = p.layout(s)
        out: dict[str, Any] = {
            "kind": "polyhedron", "faces": len(s.faces), "root": p.root_name(s),
            "unfold": round(p.unfold, 6),
            "shape": "solid" if p.unfold <= 0 else "net" if p.unfold >= 1 else "opening",
            "hinges": {s.names[f]: {"onto": s.names[g],
                                    "angle": round(net_math.hinge(s, f, g).degrees * p.unfold, 4)}
                       for f, g in sorted(parent.items())},
        }  # fmt: skip
        g = p.path(s)
        if g is not None:
            m = self.matrices(state)
            out["geodesic"] = {
                "length": round(g.length, 6), "faces": [s.names[f] for f in g.faces],
                "straight": net_math.straight_on_net(self._path_points(s, g, m), 1e-6),
            }  # fmt: skip
        return out
