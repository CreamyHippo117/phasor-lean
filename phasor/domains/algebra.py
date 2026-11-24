"""Abstract algebra objects.

    from phasor import algebra

    s.add(algebra.Polygon(id="square", n=4, rotation=expr("90 * floor(t)")))  # D4 on a square
    s.add(algebra.CayleyTable(id="table", group="D4", pos=(4, 0, 0)))
    s.add(algebra.ClockDiagram(id="clock", n=12, step=4))  # Z12 and a subgroup

The numerics (the groups themselves) are in ``algebra_math``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import algebra_math as am
from phasor.domains.strokes import arrow, circle, disc, polyline, with_alpha


def group_named(name: str) -> am.Group:
    """``"D4"``, ``"Z6"``, ``"S3"``: dihedral, cyclic, symmetric."""
    kind, digits = name[:1].upper(), name[1:]
    if not digits.isdigit() or kind not in "DZS":
        raise ValueError(f"unknown group {name!r}; write D<n>, Z<n> or S<n>, e.g. 'D4'")
    n = int(digits)
    g = {"D": am.dihedral, "Z": am.cyclic, "S": am.symmetric}[kind](n)
    if g.order > 24:
        raise ValueError(f"{name} has {g.order} elements; at most 24 are drawn")
    return g


# ---------------------------------------------------------------- Polygon


@register
class Polygon(ObjectBase):
    """A regular n-gon with its vertices numbered, to show a dihedral group acting on it:
    ``rotation`` (degrees) turns it and ``flip`` (0 to 1) mirrors it over the vertical axis (a
    smooth animation passes through edge-on). When it lands on a symmetry (a multiple of
    360/n, flip 0 or 1) the semantics name the group element."""

    type: ClassVar[str] = "polygon"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "accent": ("emphasis", "highlight"),
        "text": ("emphasis", "primary"),
        "size": ("text", "label"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {"glowing": {"style": {"glow": "soft"}}}

    class Params(ParamsBase):
        n: int = Field(4, ge=3, le=12)
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(1.5, gt=0)  # circumradius
        rotation: float = 0.0  # degrees, counterclockwise
        flip: float = Field(0.0, ge=0, le=1)
        fill: float = Field(0.25, ge=0, le=1)

    def _verts(self, p) -> np.ndarray:
        """Screen-plane vertex positions (n, 3): vertex 1 starts at the top."""
        k = np.arange(p.n)
        ang = math.pi / 2 + 2 * math.pi * k / p.n  # vertex 1 at the top, counterclockwise
        base = np.stack([np.cos(ang), np.sin(ang)], axis=1) * p.r
        t = math.radians(p.rotation)
        rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        mirror = np.array([[math.cos(math.pi * p.flip), 0.0], [0.0, 1.0]])
        xy = base @ (mirror @ rot).T  # rotate first, then mirror (r then s, as in algebra_math)
        return np.c_[xy + np.asarray(p.pos[:2]), np.full(p.n, p.pos[2])]

    def _element(self, p):
        step = 360.0 / p.n
        k = p.rotation / step
        near = abs(k - round(k)) < 1e-6
        if near and p.flip in (0.0, 1.0):
            return (round(k) % p.n) + (p.n if p.flip == 1.0 else 0)
        return None

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds.of_points(self._verts(p), pad=0.05 * p.r)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        v = self._verts(self.me(state).params)
        return {"vertices": v, "first_vertex": v[0], "center": v.mean(axis=0)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        el = self._element(p)
        group = am.dihedral(p.n)
        return {
            "kind": "polygon",
            "n": p.n,
            "group": group.name,
            "rotation": round(p.rotation, 6),
            "flipped": bool(p.flip > 0.5),
            "element": None if el is None else group.labels[el],
            "element_index": el,
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        v = self._verts(p)
        c = np.asarray(p.pos, dtype=np.float64)
        base = with_alpha(me.style["color"], p.opacity)
        w = me.style["stroke"] * float(cam.units_per_height(c))
        if p.fill > 0:
            pos = np.vstack([c[None, :], v])
            k = np.arange(p.n)
            ctx.triangles(
                pos,
                np.c_[np.zeros_like(k), 1 + k, 1 + (k + 1) % p.n],
                color=with_alpha(me.style["color"], p.opacity * p.fill),
                glow=1.0,
            )
        polyline(ctx, cam, np.vstack([v, v[:1]]), w, color=base, glow=me.style["glow"])
        size = me.style["size"] * float(cam.units_per_height(c))
        for k in range(p.n):
            col = with_alpha(me.style["accent"] if k == 0 else me.style["color"], p.opacity)
            disc(ctx, cam, v[k], size * 0.32, color=col, glow=me.style["glow"])
            out = v[k] + (v[k] - c) / max(np.linalg.norm(v[k] - c), 1e-9) * size * 1.0
            ctx.text(str(k + 1), out, size, color=with_alpha(me.style["text"], p.opacity))


# ---------------------------------------------------------------- CayleyTable


@register
class CayleyTable(ObjectBase):
    """The multiplication table of a small group (``D4``, ``Z6``, ``S3``): every cell colored
    by the element it holds, so the Latin-square structure shows."""

    type: ClassVar[str] = "cayley_table"
    Style: ClassVar = {
        "colormap": ("colormap", "phase"),
        "text": ("emphasis", "primary"),
        "size": ("text", "annotation"),
        "glow": ("glow", "none"),  # the prism look: the cells shine
    }
    Presets: ClassVar = {"soft": {"style": {"colormap": "soft_phase"}}}

    class Params(ParamsBase):
        group: str = "D4"
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        cell: float = Field(0.7, gt=0)
        reveal: float = Field(1.0, ge=0, le=1)  # share of the rows drawn (animate to fill it in)
        # Cosets: a subgroup by its generators ("r", "r2", "s"); the rows and columns are put
        # in coset order and ``cosets`` (0..1) recolors every cell by its coset, so a normal
        # subgroup shows the quotient group's table as blocks.
        subgroup: str = ""
        cosets: float = Field(0.0, ge=0, le=1)

        @field_validator("group")
        @classmethod
        def _g(cls, v):
            group_named(v)
            return v

        @model_validator(mode="after")
        def _sub(self):
            _gen_list(group_named(self.group), self.subgroup)
            return self

    def _g(self, p) -> am.Group:
        return group_named(p.group)

    def _cosets(self, p, g: am.Group):
        """(subgroup, cosets with the subgroup first, element order for rows and columns)."""
        if not p.subgroup.strip():
            return None, [], list(range(g.order))
        h = g.generated(_gen_list(g, p.subgroup))
        cos = sorted(g.left_cosets(h), key=lambda c: (0 not in c, min(c)))
        return h, cos, [x for c in cos for x in sorted(c)]

    def _origin(self, p, n):
        c = np.asarray(p.pos, dtype=np.float64)
        return c + np.array([-n * p.cell / 2, n * p.cell / 2, 0.0])  # top-left corner

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        n = self._g(p).order
        o = self._origin(p, n)
        return Bounds.of_points(
            [o - np.array([p.cell, 0, 0]), o + np.array([n * p.cell, -(n + 1) * p.cell, 0])]
        )

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        g = self._g(p)
        t = np.array(g.table)
        latin = all(
            sorted(t[k]) == list(range(g.order)) and sorted(t[:, k]) == list(range(g.order))
            for k in range(g.order)
        )
        return {
            "kind": "cayley_table",
            "group": g.name,
            "order": g.order,
            "abelian": g.is_abelian(),
            "latin_square": latin,
            "identity": g.labels[0],
            "elements": list(g.labels),
            "subgroup_orders": sorted({len(h) for h in g.subgroups()}),
            **self._coset_semantics(p, g),
        }

    def _coset_semantics(self, p, g: am.Group) -> dict[str, Any]:
        h, cos, _ = self._cosets(p, g)
        if h is None:
            return {}
        out: dict[str, Any] = {
            "subgroup": sorted(g.labels[x] for x in h),
            "index": len(cos),
            "cosets": [[g.labels[x] for x in sorted(c)] for c in cos],
            "normal": g.is_normal(h),
        }
        if out["normal"]:
            q = am.quotient(g, h)
            out["quotient"] = {"order": q.order, "abelian": q.is_abelian(),
                               "table": [list(r) for r in q.table]}  # fmt: skip
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        g = self._g(p)
        n = g.order
        cam = state.camera_for(me)
        o = self._origin(p, n)
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        rows = max(0, min(n, math.ceil(p.reveal * n - 1e-9)))
        size = me.style["size"] * float(cam.units_per_height(o))
        widest = max(len(x) for x in g.labels)
        size = min(size, 0.9 * p.cell / (0.5 * widest))  # every label fits inside its cell

        def center(i, j):  # row i (a), column j (b); row/column 0 of the grid are the headers
            return o + cam.right * ((j + 0.5) * p.cell) - cam.up * ((i + 0.5) * p.cell)

        def color(k):
            return (*lut[int(k / n * len(lut)) % len(lut)], p.opacity)

        h, cos, order = self._cosets(p, g)
        owner = {x: k for k, c in enumerate(cos) for x in c}

        def cell_color(k):
            rgb = np.asarray(color(k)[:3])
            if h is not None and p.cosets > 0:
                cc = lut[int((owner[k] + 0.5) / len(cos) * len(lut)) % len(lut)]
                rgb = rgb * (1 - p.cosets) + cc * p.cosets
            return (*rgb, p.opacity)

        text = with_alpha(me.style["text"], p.opacity)
        for j, b in enumerate(order):  # column headers (the second factor)
            ctx.text(g.labels[b], center(-1, j), size, color=text)
        for i, a in enumerate(order[:rows]):
            ctx.text(g.labels[a], center(i, -1), size, color=text)
            for j, b in enumerate(order):
                k = g.table[a][b]
                c = center(i, j)
                e = p.cell * 0.46
                rx, uy = cam.right * e, cam.up * e
                quad = np.array([c - rx - uy, c + rx - uy, c + rx + uy, c - rx + uy])
                ctx.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]), color=cell_color(k),
                              glow=float(me.style.get("glow", 1.0)))  # fmt: skip
                ctx.text(g.labels[k], c, size, color=(0.02, 0.02, 0.04, p.opacity))


# ---------------------------------------------------------------- ClockDiagram


@register
class ClockDiagram(ObjectBase):
    """The cyclic group Z_n as a clock: n points on a ring; the subgroup generated by ``step`` is
    highlighted and its cosets get one color each (``step`` = 4 on a 12-clock shows the
    four-hour hand's orbit and its three cosets)."""

    type: ClassVar[str] = "clock_diagram"
    Style: ClassVar = {
        "color": ("emphasis", "muted"),
        "accent": ("emphasis", "highlight"),
        "colormap": ("colormap", "phase"),
        "stroke": ("stroke", "hairline"),
        "size": ("text", "annotation"),
        "text": ("emphasis", "primary"),
        "glow": ("glow", "none"),  # the prism look: circle, points and orbit shine
    }

    class Params(ParamsBase):
        n: int = Field(12, ge=2, le=36)
        step: int = Field(1, ge=1)
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(2.0, gt=0)
        cosets: float = Field(0.0, ge=0, le=1)  # fade in the coset coloring

    def _pts(self, p):
        a = math.pi / 2 - 2 * math.pi * np.arange(p.n) / p.n  # 0 at the top, clockwise
        return np.stack([np.cos(a), np.sin(a), np.zeros(p.n)], axis=1) * p.r + np.asarray(p.pos)

    def _sub(self, p) -> frozenset[int]:
        return am.cyclic(p.n).generated([p.step % p.n])

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds.of_points(self._pts(p), pad=0.25)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        pts = self._pts(p)
        return {"points": pts, "orbit": pts[sorted(self._sub(p))]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        g = am.cyclic(p.n)
        sub = self._sub(p)
        return {
            "kind": "clock_diagram",
            "n": p.n,
            "generator": p.step % p.n,
            "subgroup": sorted(sub),
            "subgroup_order": len(sub),
            "index": p.n // len(sub),
            "generator_order": g.element_order(p.step % p.n),
            "cosets": [sorted(c) for c in g.left_cosets(sub)],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        g = am.cyclic(p.n)
        sub = self._sub(p)
        pts = self._pts(p)
        w = me.style["stroke"] * float(cam.units_per_height(np.asarray(p.pos)))
        gl = float(me.style.get("glow", 1.0))
        circle(ctx, cam, p.pos, p.r, w, color=with_alpha(me.style["color"], p.opacity * 0.6),
               glow=gl)  # fmt: skip
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        cosets = g.left_cosets(sub)
        owner = {x: i for i, c in enumerate(cosets) for x in c}
        size = me.style["size"] * float(cam.units_per_height(np.asarray(p.pos)))
        for k in range(p.n):
            in_sub = k in sub
            base = with_alpha(me.style["accent"] if in_sub else me.style["color"], p.opacity)
            if p.cosets > 0 and not in_sub:
                rgb = lut[int(owner[k] / max(len(cosets), 1) * len(lut)) % len(lut)]
                base = (*rgb, p.opacity * p.cosets)
            disc(ctx, cam, pts[k], size * (0.4 if in_sub else 0.28), color=base, glow=gl)
            out = pts[k] + (pts[k] - np.asarray(p.pos)) / p.r * size * 1.1
            ctx.text(str(k), out, size, color=with_alpha(me.style["text"], p.opacity))
        orbit = [pts[x] for x in sorted(sub)]
        if len(orbit) >= 2:
            polyline(
                ctx,
                cam,
                np.vstack([orbit, orbit[:1]]),
                w * 2,
                color=with_alpha(me.style["accent"], p.opacity * 0.8),
                glow=gl,
            )


# ---------------------------------------------------------------- shared drawing


def _gen_list(g: am.Group, text: str) -> list[int]:
    """Element labels separated by spaces (``"r s"``) -> element numbers."""
    names = text.split()
    unknown = [x for x in names if x not in g.labels]
    if unknown:
        raise ValueError(f"{g.name} has no element {unknown[0]!r}; elements are {list(g.labels)}")
    return [g.labels.index(x) for x in names]


def _levels(sizes: list[int]) -> list[int]:
    """Height of each lattice node: the number of prime factors of its size (so a step up
    a covering edge of prime index is one level)."""

    def omega(n: int) -> int:
        k, p = 0, 2
        while n > 1:
            while n % p == 0:
                n, k = n // p, k + 1
            p += 1
        return k

    return [omega(s) for s in sizes]


def _hasse_positions(levels: list[int], width: float, height: float, origin) -> np.ndarray:
    """Nodes spread evenly along their level; level 0 at the bottom of a width x height box
    centred on ``origin``."""
    top = max(max(levels), 1)
    out = np.zeros((len(levels), 3))
    for lv in set(levels):
        idx = [i for i, x in enumerate(levels) if x == lv]
        for j, i in enumerate(idx):
            x = (j + 0.5) / len(idx) - 0.5
            out[i] = (x * width, (lv / top - 0.5) * height, 0.0)
    return out + np.asarray(origin, dtype=np.float64)


def _node_graph(ctx, cam, me, pos, labels, edges, *, accents=(), edge_alpha=0.7):
    """Lines for ``edges``, then a disc and a label per node (``accents`` highlighted)."""
    p = me.params
    c = pos.mean(axis=0)
    w = me.style["stroke"] * float(cam.units_per_height(c))
    size = me.style["size"] * float(cam.units_per_height(c))
    for a, b in edges:
        polyline(ctx, cam, [pos[a], pos[b]], w,
                 color=with_alpha(me.style["color"], p.opacity * edge_alpha))  # fmt: skip
    # On a crowded level the outermost labels go sideways (outward) so the middle ones,
    # above their nodes, have room.
    rows: dict[float, list[int]] = {}
    for i, x in enumerate(pos):
        rows.setdefault(round(float(x[1]), 6), []).append(i)
    side: dict[int, int] = {}
    for idx in rows.values():
        if len(idx) >= 3:
            order = sorted(idx, key=lambda k: pos[k][0])
            side[order[0]], side[order[-1]] = -1, 1
    text_col = with_alpha(me.style["text"], p.opacity)
    for i, (x, text) in enumerate(zip(pos, labels, strict=True)):
        role = "accent" if i in accents else "color"
        disc(ctx, cam, x, size * 0.3, color=with_alpha(me.style[role], p.opacity))
        if i in side:
            at = x + cam.right * side[i] * size * 0.6 - cam.up * size * 0.05
            ctx.text(text, at, size, color=text_col, anchor="left" if side[i] > 0 else "right")
        else:
            ctx.text(text, x + cam.up * size * 0.95, size, color=text_col)


_GRAPH_STYLE: dict[str, tuple[str, str]] = {
    "color": ("emphasis", "muted"),
    "accent": ("emphasis", "highlight"),
    "text": ("emphasis", "primary"),
    "colormap": ("colormap", "phase"),
    "stroke": ("stroke", "regular"),
    "size": ("text", "annotation"),
    "glow": ("glow", "none"),
}


# ---------------------------------------------------------------- CayleyGraph


@register
class CayleyGraph(ObjectBase):
    """The Cayley graph of a small group: a node per element, an arrow x -> x g for each
    generator g, one color per generator. Dihedral groups put rotations on the outer ring and
    reflections on the inner one, so ``r`` turns the rings and ``s`` jumps between them."""

    type: ClassVar[str] = "cayley_graph"
    Style: ClassVar = _GRAPH_STYLE

    class Params(ParamsBase):
        group: str = "D4"
        generators: str = "r s"  # element labels, separated by spaces
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(2.0, gt=0)
        draw: float = Field(1.0, ge=0, le=1)  # share of each arrow drawn (animate to grow)

        @field_validator("group")
        @classmethod
        def _g(cls, v):
            group_named(v)
            return v

        @model_validator(mode="after")
        def _gens(self):
            g = group_named(self.group)
            am.cayley_edges(g, _gen_list(g, self.generators))
            return self

    def _setup(self, p):
        g = group_named(p.group)
        gens = _gen_list(g, p.generators)
        return g, gens, am.cayley_edges(g, gens)

    def _pos(self, p, g) -> np.ndarray:
        n = g.order
        if g.name.startswith("D"):
            m = n // 2
            k = np.arange(n)
            ang = math.pi / 2 + 2 * math.pi * (k % m) / m
            rad = np.where(k < m, p.r, p.r * 0.55)
        else:
            ang = math.pi / 2 + 2 * math.pi * np.arange(n) / n
            rad = np.full(n, p.r)
        xy = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], axis=1)
        return np.c_[xy, np.zeros(n)] + np.asarray(p.pos, dtype=np.float64)

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        return Bounds.of_points(self._pos(p, group_named(p.group)), pad=0.3)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        pts = self._pos(p, group_named(p.group))
        return {"nodes": pts, "identity": pts[0], "center": np.asarray(p.pos)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        g, gens, edges = self._setup(p)
        return {
            "kind": "cayley_graph",
            "group": g.name,
            "order": g.order,
            "generators": [g.labels[a] for a in gens],
            "edges": len(edges),
            "connected": True,  # cayley_edges refuses generators that do not generate
            "relations": {g.labels[a]: g.element_order(a) for a in gens},
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        g, gens, edges = self._setup(p)
        pts = self._pos(p, g)
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        c = np.asarray(p.pos, dtype=np.float64)
        w = me.style["stroke"] * float(cam.units_per_height(c))
        size = me.style["size"] * float(cam.units_per_height(c))
        for gi, s in enumerate(gens):
            rgb = lut[int((gi + 0.5) / len(gens) * len(lut)) % len(lut)]
            col = (*rgb, p.opacity)
            for x, _, y in (e for e in edges if e[1] == s):
                if g.element_order(s) == 2 and y < x:
                    continue  # an involution: x -> y and y -> x are one undirected edge
                a, b = pts[x], pts[y]
                d = b - a
                length = float(np.linalg.norm(d))
                if length < 1e-9:
                    continue
                a2 = a + d / length * size * 0.45
                b2 = a2 + (b - d / length * size * 0.45 - a2) * p.draw
                if g.element_order(s) == 2:
                    polyline(ctx, cam, [a2, b2], w, color=col, glow=me.style["glow"])
                else:
                    arrow(ctx, cam, a2, b2, w, col, me.style["glow"])
        text = with_alpha(me.style["text"], p.opacity)
        inner = g.name.startswith("D")
        for k, x in enumerate(pts):
            role = "accent" if k == 0 else "color"
            disc(ctx, cam, x, size * 0.32, color=with_alpha(me.style[role], p.opacity))
            way = -1.0 if inner and k >= g.order // 2 else 1.0  # reflections: label inward
            out = x + way * (x - c) / max(float(np.linalg.norm(x - c)), 1e-9) * size * 0.9
            ctx.text(g.labels[k], out, size, color=text)


# ---------------------------------------------------------------- Homomorphism


@register
class Homomorphism(ObjectBase):
    """A homomorphism of clocks, Z_n -> Z_m, k -> a k mod m, as arrows from one ring to the
    other. The kernel (what lands on 0) is highlighted; ``progress`` draws the arrows."""

    type: ClassVar[str] = "homomorphism"
    Style: ClassVar = _GRAPH_STYLE

    class Params(ParamsBase):
        n: int = Field(12, ge=2, le=24)
        m: int = Field(4, ge=1, le=24)
        a: int = Field(1, ge=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # midpoint between the two rings
        r: float = Field(1.4, gt=0)
        gap: float = Field(5.0, gt=0)  # distance between ring centres
        progress: float = Field(1.0, ge=0, le=1)

        @model_validator(mode="after")
        def _is_hom(self):
            am.clock_hom(self.n, self.m, self.a % self.m)
            return self

    def _hom(self, p) -> am.Hom:
        return am.clock_hom(p.n, p.m, p.a % p.m)

    def _rings(self, p):
        c = np.asarray(p.pos, dtype=np.float64)

        def ring(n, center):
            ang = math.pi / 2 - 2 * math.pi * np.arange(n) / n
            return np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=1) * p.r + center

        left, right = c - np.array([p.gap / 2, 0, 0]), c + np.array([p.gap / 2, 0, 0])
        return ring(p.n, left), ring(p.m, right), left, right

    def world_bounds(self, state: SceneState) -> Bounds:
        a, b, *_ = self._rings(self.me(state).params)
        return Bounds.of_points(np.vstack([a, b]), pad=0.4)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        a, b, left, right = self._rings(self.me(state).params)
        return {"source": a, "target": b, "source_center": left, "target_center": right}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        h = self._hom(p)
        return {
            "kind": "homomorphism",
            "map": f"Z{p.n} -> Z{p.m}, k -> {p.a % p.m}k",
            "images": list(h.images),
            "kernel": sorted(h.kernel),
            "image": sorted(h.image),
            "injective": len(h.kernel) == 1,
            "surjective": len(h.image) == p.m,
            "first_isomorphism": f"Z{p.n} / ker has {p.n // len(h.kernel)} elements = |image|",
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        h = self._hom(p)
        src, dst, left, right = self._rings(p)
        c = np.asarray(p.pos, dtype=np.float64)
        w = me.style["stroke"] * float(cam.units_per_height(c))
        size = me.style["size"] * float(cam.units_per_height(c))
        for pts, center in ((src, left), (dst, right)):
            circle(ctx, cam, center, p.r, w * 0.5,
                   color=with_alpha(me.style["color"], p.opacity * 0.5))  # fmt: skip
            for k, x in enumerate(pts):
                in_ker = pts is src and k in h.kernel
                role = "accent" if in_ker or (pts is dst and k == 0) else "color"
                disc(ctx, cam, x, size * (0.38 if in_ker else 0.28),
                     color=with_alpha(me.style[role], p.opacity))  # fmt: skip
                out = x + (x - center) / p.r * size * 1.0
                ctx.text(str(k), out, size, color=with_alpha(me.style["text"], p.opacity))
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        for k in range(p.n):
            u = np.clip(p.progress * p.n - k, 0, 1)  # one arrow after another
            if u <= 0:
                continue
            a, b = src[k], dst[h.images[k]]
            rgb = lut[int(h.images[k] / p.m * len(lut)) % len(lut)]
            arrow(ctx, cam, a, a + (b - a) * u, w * 0.6, (*rgb, p.opacity * 0.75))


# ---------------------------------------------------------------- SubgroupLattice


@register
class SubgroupLattice(ObjectBase):
    """Every subgroup of a small group, as a Hasse diagram: bigger subgroups higher, a line
    where one contains the next with nothing in between. Normal subgroups are highlighted."""

    type: ClassVar[str] = "subgroup_lattice"
    Style: ClassVar = _GRAPH_STYLE

    class Params(ParamsBase):
        group: str = "D4"
        pos: Vec3 = (0.0, 0.0, 0.0)
        width: float = Field(6.0, gt=0)
        height: float = Field(4.0, gt=0)

        @field_validator("group")
        @classmethod
        def _g(cls, v):
            group_named(v)
            return v

    def _setup(self, p):
        g = group_named(p.group)
        subs, edges = am.subgroup_lattice(g)
        pos = _hasse_positions(_levels([len(s) for s in subs]), p.width, p.height, p.pos)
        return g, subs, edges, pos

    @staticmethod
    def _name(g: am.Group, h: frozenset[int]) -> str:
        if len(h) == 1:
            return "{e}"
        if len(h) == g.order:
            return g.name
        gens = next(([a] for a in sorted(h) if g.generated([a]) == h), None)
        if gens is None:  # needs two generators: name it by the first pair that works
            gens = next([a, b] for a in sorted(h) for b in sorted(h) if g.generated([a, b]) == h)
        return "<" + ",".join(g.labels[a] for a in gens) + ">"

    def world_bounds(self, state: SceneState) -> Bounds:
        return Bounds.of_points(self._setup(self.me(state).params)[3], pad=0.4)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        pos = self._setup(self.me(state).params)[3]
        return {"nodes": pos, "bottom": pos[0], "top": pos[-1]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        g, subs, edges, _ = self._setup(self.me(state).params)
        names = [self._name(g, h) for h in subs]
        return {
            "kind": "subgroup_lattice",
            "group": g.name,
            "subgroups": [
                {"name": n, "order": len(h), "normal": g.is_normal(h)}
                for n, h in zip(names, subs, strict=True)
            ],
            "covers": [[names[a], names[b]] for a, b in edges],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        g, subs, edges, pos = self._setup(me.params)
        normal = {i for i, h in enumerate(subs) if g.is_normal(h)}
        _node_graph(ctx, state.camera_for(me), me, pos, [self._name(g, h) for h in subs],
                    edges, accents=normal)  # fmt: skip


# ---------------------------------------------------------------- GaloisDiagram


@register
class GaloisDiagram(ObjectBase):
    """The Galois correspondence for L = Q(sqrt p1, ..., sqrt pk): the lattice of fields
    between Q and L (left, Q at the bottom) beside the lattice of subgroups of Gal(L/Q)
    (right, upside down: the whole group at the bottom). Each field sits level with the
    subgroup that fixes it, so inclusion turns over from one side to the other."""

    type: ClassVar[str] = "galois_diagram"
    Style: ClassVar = _GRAPH_STYLE

    class Params(ParamsBase):
        primes: tuple[int, ...] = (2, 3)
        pos: Vec3 = (0.0, 0.0, 0.0)
        width: float = Field(5.0, gt=0)  # of each lattice
        height: float = Field(3.6, gt=0)
        gap: float = Field(4.0, ge=0)  # room for the labels between the two lattices

        @field_validator("primes")
        @classmethod
        def _p(cls, v):
            am.galois_sqrt(v)
            return v

    def _setup(self, p):
        gal = am.galois_sqrt(p.primes)
        # level = log2 of the field degree; the subgroup sits level with its fixed field
        levels = [f.degree.bit_length() - 1 for f in gal.fields]
        c = np.asarray(p.pos, dtype=np.float64)
        off = np.array([(p.width + p.gap) / 2, 0, 0])
        fpos = _hasse_positions(levels, p.width, p.height, c - off)
        gpos = _hasse_positions(levels, p.width, p.height, c + off)
        subs, edges = am.subgroup_lattice(gal.group)
        assert tuple(subs) == gal.subgroups  # both come from Group.subgroups(), in its order
        return gal, fpos, gpos, edges

    def world_bounds(self, state: SceneState) -> Bounds:
        _, fpos, gpos, _ = self._setup(self.me(state).params)
        return Bounds.of_points(np.vstack([fpos, gpos]), pad=0.5)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        _, fpos, gpos, _ = self._setup(self.me(state).params)
        return {"fields": fpos, "subgroups": gpos}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        gal, *_ = self._setup(self.me(state).params)
        g = gal.group
        return {
            "kind": "galois_diagram",
            "field": gal.fields[0].name,
            "degree": gal.degree,
            "group": "(Z2)^" + str(len(gal.primes)),
            "pairs": [
                {
                    "field": f.name,
                    "degree": f.degree,
                    "fixed_by": [g.labels[s] for s in sorted(h)],
                    "subgroup_order": len(h),
                }
                for f, h in zip(gal.fields, gal.subgroups, strict=True)
            ],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        cam = state.camera_for(me)
        gal, fpos, gpos, edges = self._setup(me.params)
        g = gal.group
        # A subgroup lattice edge (small inside big) is a field edge the other way round.
        _node_graph(ctx, cam, me, fpos, [_ascii(f.name) for f in gal.fields], edges)
        names = [_galois_name(g, h) for h in gal.subgroups]
        _node_graph(ctx, cam, me, gpos, names, edges, accents={0})


def _ascii(text: str) -> str:
    """The text font has no square-root or sigma glyph: draw sqrt and sig instead."""
    return text.replace("\u221a", "sqrt").replace("\u03c3(", "sig(")


def _galois_name(g: am.Group, h: frozenset[int]) -> str:
    """A subgroup by a generating set: {e}, <sig(2)>, <sig(2),sig(3)> or Gal."""
    if len(h) == 1:
        return "{e}"
    if len(h) == g.order:
        return "Gal"
    gens = (
        next([a] for a in sorted(h) if a and g.generated([a]) == h)
        if len(h) == 2
        else [x for x in sorted(h) if x and bin(x).count("1") == 1]
    )
    if g.generated(gens) != h:
        gens = sorted(h)[1:]
    return _ascii("<" + ",".join(g.labels[a] for a in gens) + ">")


# ---------------------------------------------------------------- Permutation


@register
class Permutation(ObjectBase):
    """Tokens 1..n in a row; as ``progress`` goes 0 -> 1 the token in slot i travels along
    an arc to slot perm[i]. Tokens are colored by the cycle they belong to."""

    type: ClassVar[str] = "permutation"
    Style: ClassVar = _GRAPH_STYLE

    class Params(ParamsBase):
        perm: tuple[int, ...] = (1, 2, 0, 4, 3)  # slot i goes to slot perm[i] (from 0)
        pos: Vec3 = (0.0, 0.0, 0.0)
        spacing: float = Field(0.9, gt=0)
        progress: float = Field(0.0, ge=0, le=1)

        @field_validator("perm")
        @classmethod
        def _p(cls, v):
            am.cycles(v)
            if len(v) > 16:
                raise ValueError("at most 16 tokens")
            return v

    def _slots(self, p) -> np.ndarray:
        n = len(p.perm)
        x = (np.arange(n) - (n - 1) / 2) * p.spacing
        return np.c_[x, np.zeros(n), np.zeros(n)] + np.asarray(p.pos, dtype=np.float64)

    def _where(self, p) -> np.ndarray:
        """Token positions now: arcs, upward for rightward moves, downward for leftward."""
        slots = self._slots(p)
        u = 0.5 - 0.5 * math.cos(math.pi * p.progress)  # ease
        out = slots.copy()
        for i, j in enumerate(p.perm):
            a, b = slots[i], slots[j]
            mid = (a + b) / 2
            half = (b[0] - a[0]) / 2
            ang = math.pi * (1 - u)
            out[i] = mid + np.array([math.cos(ang) * half, math.sin(ang) * abs(half) * 0.6
                                     * (1 if half > 0 else -1), 0.0])  # fmt: skip
        return out

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        slots = self._slots(p)
        h = (len(p.perm) * p.spacing) * 0.35
        up = np.array([0.0, h, 0.0])
        return Bounds.of_points(np.vstack([slots - up, slots + up]), pad=0.3)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return {"slots": self._slots(p), "tokens": self._where(p)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        cyc = am.cycles(p.perm)
        return {
            "kind": "permutation",
            "perm": [x + 1 for x in p.perm],
            "cycles": "".join("(" + " ".join(str(x + 1) for x in c) + ")" for c in cyc) or "e",
            "sign": am.sign(p.perm),
            "parity": "even" if am.sign(p.perm) == 1 else "odd",
            "order": am.perm_order(p.perm),
            "done": p.progress >= 1.0,
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        slots, now = self._slots(p), self._where(p)
        c = np.asarray(p.pos, dtype=np.float64)
        size = me.style["size"] * float(cam.units_per_height(c)) * 1.4
        w = me.style["stroke"] * float(cam.units_per_height(c))
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        cyc = am.cycles(p.perm)
        owner = {x: k for k, cy in enumerate(cyc) for x in cy}
        for x in slots:
            circle(ctx, cam, x, size * 0.5, w * 0.5,
                   color=with_alpha(me.style["color"], p.opacity * 0.6))  # fmt: skip
        for i, x in enumerate(now):
            if i in owner:
                rgb = lut[int((owner[i] + 0.5) / len(cyc) * len(lut)) % len(lut)]
                col = (*rgb, p.opacity)
            else:
                col = with_alpha(me.style["color"], p.opacity)
            disc(ctx, cam, x, size * 0.42, color=col, glow=me.style["glow"])
            ctx.text(str(i + 1), x, size * 0.8, color=(0.02, 0.02, 0.04, p.opacity))
