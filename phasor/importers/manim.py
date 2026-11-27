"""Manim import: a Manim Community script runs as it is, and phasor
records the scene it builds.

``import_manim("scene.py")`` runs the file with a small stand-in for the ``manim`` module on
the import path: ``from manim import *`` gives the usual names (``Scene``, ``Circle``,
``Create``, ``FadeIn``, ``Text``, ``MathTex``, ``Axes`` ...), and ``construct`` is called. Every
``add`` and ``play`` becomes objects and transitions on a phasor ``Scene`` at the right time:
what ``self.play(Create(circle), run_time=2)`` does for two seconds is a track on the
circle's ``progress`` here. The result is an ordinary phasor scene (save it, lint it, edit it
in the editor, render it), with the Manim frame (8 units tall, y up, origin in the middle).

What carries over, and what does not, is listed in ``docs/ai/API.md`` ("Manim import"):
anything phasor cannot express (updaters, ``always_redraw``, fills, arbitrary colors, font
sizes) becomes an entry in ``ManimImport.warnings`` and the closest thing phasor has: colors
become the nearest style *role* (phasor scenes use roles, never raw colors), fills are
outlines. A name phasor's stand-in lacks raises an error that says so.
"""

from __future__ import annotations

import colorsys
import copy
import math
import re
import runpy
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phasor.core import text as text_metrics
from phasor.core.schema import Document
from phasor.scene import Scene as PhasorScene

FRAME_HEIGHT = 8.0
FRAME_WIDTH = FRAME_HEIGHT * 16 / 9
PI, TAU = math.pi, 2 * math.pi
DEGREES = PI / 180

ORIGIN = np.array([0.0, 0.0, 0.0])
UP, DOWN = np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])
LEFT, RIGHT = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
OUT, IN = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
UL, UR, DL, DR = UP + LEFT, UP + RIGHT, DOWN + LEFT, DOWN + RIGHT

# Manim's named colors (hex), so ``color=RED`` reads the same; mapped to roles on the way in.
_COLORS = {
    "WHITE": "#FFFFFF", "BLACK": "#000000", "GRAY": "#888888", "GREY": "#888888",
    "LIGHT_GRAY": "#BBBBBB", "LIGHT_GREY": "#BBBBBB", "DARK_GRAY": "#444444",
    "DARK_GREY": "#444444", "GRAY_A": "#DDDDDD", "GRAY_B": "#BBBBBB", "GRAY_C": "#888888",
    "GRAY_D": "#444444", "GRAY_E": "#222222", "RED": "#FC6255", "RED_A": "#F7A1A3",
    "RED_B": "#FF8080", "RED_C": "#FC6255", "RED_D": "#E65A4C", "RED_E": "#CF5044",
    "MAROON": "#C55F73", "PINK": "#D147BD", "PURPLE": "#9A72AC", "BLUE": "#58C4DD",
    "BLUE_A": "#C7E9F1", "BLUE_B": "#9CDCEB", "BLUE_C": "#58C4DD", "BLUE_D": "#29ABCA",
    "BLUE_E": "#236B8E", "DARK_BLUE": "#236B8E", "TEAL": "#5CD0B3", "GREEN": "#83C167",
    "GREEN_A": "#C9E2AE", "GREEN_B": "#A6CF8C", "GREEN_C": "#83C167", "GREEN_D": "#77B05D",
    "GREEN_E": "#699C52", "YELLOW": "#FFFF00", "YELLOW_A": "#FFF1B6", "YELLOW_B": "#FFEA94",
    "YELLOW_C": "#F7D96F", "YELLOW_D": "#F4D345", "YELLOW_E": "#E8C11C", "GOLD": "#F0AC5F",
    "GOLD_A": "#F7C797", "GOLD_B": "#F9B775", "GOLD_C": "#F0AC5F", "GOLD_D": "#E1A158",
    "GOLD_E": "#C78D46", "ORANGE": "#FF862F", "LIGHT_PINK": "#DC75CD",
}  # fmt: skip

TEXT_ROLES = (("title", 60), ("math", 40), ("label", 30), ("annotation", 0))  # font_size -> role


class ManimImportError(NotImplementedError):
    """A Manim feature phasor's import does not have (the message says which)."""


@dataclass
class ManimImport:
    scene: PhasorScene
    warnings: list[str] = field(default_factory=list)
    duration: float = 0.0
    name: str = ""
    others: list[str] = field(default_factory=list)  # other Scene classes the file defines

    def document(self) -> Document:
        """The imported scene as a plain document: free to edit by hand or by an AI (a scene
        built by a script only takes tweaks), since the Manim script is not its source of
        truth any more."""
        doc = self.scene.engine.document
        raw = doc.model_dump(mode="json")
        raw["meta"]["source"] = "document"
        return Document.model_validate(raw)


def looks_like_manim(path: str | Path) -> bool:
    """Whether a .py file is a Manim script (it imports ``manim``)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(r"^\s*(from\s+manim(lib)?\s+import|import\s+manim)", text, re.M))


# ---------------------------------------------------------------- colors and sizes


def color_role(color: Any) -> str:
    """The nearest phasor emphasis role for a Manim color (hex string or name)."""
    if color is None:
        return "primary"
    c = str(color)
    c = _COLORS.get(c.upper(), c)
    if not (c.startswith("#") and len(c) in (4, 7)):
        return "primary"
    if len(c) == 4:
        c = "#" + "".join(ch * 2 for ch in c[1:])
    r, g, b = (int(c[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.18 or v < 0.2:
        return "primary" if v > 0.62 else "muted"
    hue = h * 360
    if hue < 18 or hue >= 300:
        return "danger"
    if hue < 75:
        return "highlight"
    return "secondary"


def text_role(font_size: float) -> str:
    return next(role for role, floor in TEXT_ROLES if font_size >= floor)


# ---------------------------------------------------------------- mobjects


class Mobject:
    """A thing on the Manim canvas. Leaves have points and become one phasor object; groups
    hold children and pass every change to them."""

    kind = "group"

    def __init__(self, *children: Mobject, color: Any = None, **kw: Any) -> None:
        self.children: list[Mobject] = list(children)
        self.color_role = color_role(color) if color is not None else None
        self.opacity = 1.0
        self.pts = np.zeros((0, 3))
        self.size = 0.0  # a dot's radius
        self._ctx: _Context | None = None
        self.obj_id: str | None = None  # the phasor object this is (leaves only)
        self.alias: Mobject | None = None  # after Transform: this is now that mobject's object
        self.target: Mobject | None = None
        self.unsupported = [k for k in ("fill_color",) if k in kw]
        fo = kw.get("fill_opacity")
        if fo:
            self._warn("filled shapes are drawn as outlines")

    # -- bookkeeping
    def _warn(self, msg: str) -> None:
        (_CTX.warnings if _CTX else []).append(msg)

    def leaves(self) -> list[Mobject]:
        if self.alias is not None:
            return self.alias.leaves()
        if not self.children:
            return [self] if self.kind != "group" else []
        return [leaf for ch in self.children for leaf in ch.leaves()]

    @property
    def animate(self) -> _AnimateBuilder:
        return _AnimateBuilder(self)

    def copy(self) -> Mobject:
        c = copy.copy(self)
        c.children = [ch.copy() for ch in self.children]
        c.pts = self.pts.copy()
        c.obj_id, c.alias, c.target = None, None, None
        return c

    def generate_target(self) -> Mobject:
        self.target = self.copy()
        return self.target

    # -- geometry
    def _all_pts(self) -> np.ndarray:
        parts = [leaf.pts for leaf in self.leaves() if len(leaf.pts)]
        return np.vstack(parts) if parts else np.zeros((1, 3))

    def _bbox(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = None, None
        for leaf in self.leaves():
            if not len(leaf.pts):
                continue
            pad = np.array([leaf.size, leaf.size, 0.0])
            if leaf.kind == "text":
                w, h = leaf._text_extent()
                pad = np.array([w / 2, h / 2, 0.0])
            a, b = leaf.pts.min(axis=0) - pad, leaf.pts.max(axis=0) + pad
            lo = a if lo is None else np.minimum(lo, a)
            hi = b if hi is None else np.maximum(hi, b)
        if lo is None:
            z = np.zeros(3)
            return z, z
        return lo, hi

    def get_center(self) -> np.ndarray:
        lo, hi = self._bbox()
        return (lo + hi) / 2

    def get_left(self):
        return np.array([self._bbox()[0][0], self.get_center()[1], 0.0])

    def get_right(self):
        return np.array([self._bbox()[1][0], self.get_center()[1], 0.0])

    def get_top(self):
        return np.array([self.get_center()[0], self._bbox()[1][1], 0.0])

    def get_bottom(self):
        return np.array([self.get_center()[0], self._bbox()[0][1], 0.0])

    get_start = get_left
    get_end = get_right

    @property
    def width(self) -> float:
        lo, hi = self._bbox()
        return float(hi[0] - lo[0])

    @property
    def height(self) -> float:
        lo, hi = self._bbox()
        return float(hi[1] - lo[1])

    def shift(self, *vecs: Sequence[float]) -> Mobject:
        d = np.sum([np.asarray(v, dtype=float) for v in vecs], axis=0)
        for leaf in self.leaves():
            leaf.pts = leaf.pts + d
        return self

    def move_to(self, target: Any, aligned_edge: Any = None) -> Mobject:
        p = target.get_center() if isinstance(target, Mobject) else np.asarray(target, float)
        return self.shift(p - self.get_center())

    def scale(self, factor: float, about_point: Any = None) -> Mobject:
        c = np.asarray(about_point, float) if about_point is not None else self.get_center()
        for leaf in self.leaves():
            leaf.pts = c + (leaf.pts - c) * factor
            leaf.size *= factor
            if leaf.kind == "text":
                leaf._warn("scaling text is not carried over (phasor text sizes are roles)")
        return self

    def rotate(self, angle: float, axis: Any = OUT, about_point: Any = None, **kw: Any) -> Mobject:
        c = np.asarray(about_point, float) if about_point is not None else self.get_center()
        ca, sa = math.cos(angle), math.sin(angle)
        rot = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        for leaf in self.leaves():
            leaf.pts = c + (leaf.pts - c) @ rot.T
        return self

    def next_to(self, other: Any, direction: Any = RIGHT, buff: float = 0.25, **kw: Any) -> Mobject:
        d = np.asarray(direction, float)
        if isinstance(other, Mobject):
            lo, hi = other._bbox()
            c = other.get_center()
            anchor = c + np.where(d > 0, hi - c, np.where(d < 0, lo - c, 0.0))
        else:
            anchor = np.asarray(other, float)
        mine_lo, mine_hi = self._bbox()
        half = (mine_hi - mine_lo) / 2
        target = anchor + d * buff + np.where(d != 0, d * half, 0.0)
        c = self.get_center()
        move = np.where(d != 0, target - c, 0.0)
        move = move + np.where(d == 0, anchor - c, 0.0) * np.array([1.0, 1.0, 0.0])
        return self.shift(move)

    def to_edge(self, edge: Any = LEFT, buff: float = 0.5) -> Mobject:
        e = np.asarray(edge, float)
        lo, hi = self._bbox()
        half = np.array([FRAME_WIDTH / 2, FRAME_HEIGHT / 2, 0.0])
        move = np.zeros(3)
        for i in (0, 1):
            if e[i] > 0:
                move[i] = half[i] - buff - hi[i]
            elif e[i] < 0:
                move[i] = -half[i] + buff - lo[i]
        return self.shift(move)

    def to_corner(self, corner: Any = UR, buff: float = 0.5) -> Mobject:
        return self.to_edge(corner, buff)

    def align_to(self, other: Any, direction: Any = UP) -> Mobject:
        d = np.asarray(direction, float)
        o = other if isinstance(other, Mobject) else None
        if o is None:
            return self
        lo, hi = self._bbox()
        olo, ohi = o._bbox()
        move = np.zeros(3)
        for i in (0, 1):
            if d[i] > 0:
                move[i] = ohi[i] - hi[i]
            elif d[i] < 0:
                move[i] = olo[i] - lo[i]
        return self.shift(move)

    def arrange(self, direction: Any = RIGHT, buff: float = 0.25, center: bool = True,
                **kw: Any) -> Mobject:  # fmt: skip
        prev = None
        for ch in self.children:
            if prev is not None:
                ch.next_to(prev, direction, buff)
            prev = ch
        if center and self.children:
            self.move_to(ORIGIN)
        return self

    # -- look
    def set_color(self, color: Any) -> Mobject:
        self.color_role = color_role(color)
        for ch in self.children:
            ch.set_color(color)
        return self

    def set_opacity(self, opacity: float) -> Mobject:
        self.opacity = float(opacity)
        for ch in self.children:
            ch.set_opacity(opacity)
        return self

    def set_fill(self, color: Any = None, opacity: float = 1.0) -> Mobject:
        if color is not None:
            self.set_color(color)
        self._warn("filled shapes are drawn as outlines")
        return self

    def set_stroke(self, color: Any = None, width: float | None = None, **kw: Any) -> Mobject:
        if color is not None:
            self.set_color(color)
        return self

    def set_z_index(self, z: int) -> Mobject:
        return self

    def add(self, *mobs: Mobject) -> Mobject:
        self.children.extend(mobs)
        return self

    def __getitem__(self, i: int) -> Mobject:
        return self.children[i]

    def __iter__(self):
        return iter(self.children)

    def __len__(self) -> int:
        return len(self.children)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise ManimImportError(f"{type(self).__name__}.{name} is not supported by phasor's "
                               "Manim import yet")  # fmt: skip

    # -- what phasor gets (leaves)
    def _style(self, **extra: str) -> dict[str, str]:
        """Style roles for the phasor object: the color role, plus ``extra`` (a text size)."""
        out = dict(extra)
        if self.color_role:
            out["color"] = self.color_role
        return out

    def phasor_object(self, ident: str):  # pragma: no cover (overridden)
        raise NotImplementedError

    def _text_extent(self) -> tuple[float, float]:
        return 0.0, 0.0


def _vec3(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=float).ravel()
    return np.array([a[0], a[1], a[2] if len(a) > 2 else 0.0])


def _tuples(a: np.ndarray) -> list[tuple[float, float, float]]:
    return [tuple(round(float(x), 9) for x in row) for row in a]


class VMobject(Mobject):
    pass


class VGroup(Mobject):
    kind = "group"

    def __init__(self, *mobs: Mobject, **kw: Any) -> None:
        super().__init__(*mobs, **kw)


Group = VGroup


class Dot(Mobject):
    kind = "dot"

    def __init__(self, point: Any = ORIGIN, radius: float = 0.08, color: Any = "#FFFFFF",
                 **kw: Any) -> None:  # fmt: skip
        super().__init__(color=color, **kw)
        self.pts = _vec3(point)[None, :]
        self.size = float(radius)

    def phasor_object(self, ident: str):
        from phasor import Dot as D

        return D(id=ident, pos=tuple(self.pts[0]), r=self.size, opacity=0.0,
                 style=self._style())  # fmt: skip


class _Path(Mobject):
    """Anything drawn as a polyline."""

    kind = "line"
    closed = False

    def __init__(self, pts: np.ndarray, color: Any = "#FFFFFF", **kw: Any) -> None:
        super().__init__(color=color, **kw)
        pts = np.asarray(pts, float)
        self.pts = np.vstack([pts, pts[:1]]) if self.closed else pts

    def phasor_object(self, ident: str):
        from phasor import Line

        return Line(id=ident, points=_tuples(self.pts), opacity=0.0, style=self._style())


def _ring(radius: float, center: Any = ORIGIN, n: int = 72) -> np.ndarray:
    t = np.linspace(0, TAU, n, endpoint=False)
    return _vec3(center) + radius * np.stack([np.cos(t), np.sin(t), np.zeros(n)], axis=1)


class Circle(_Path):
    closed = True

    def __init__(self, radius: float = 1.0, color: Any = "#FC6255", arc_center: Any = ORIGIN,
                 **kw: Any) -> None:  # fmt: skip
        super().__init__(_ring(radius, arc_center), color, **kw)


class Polygon(_Path):
    closed = True

    def __init__(self, *vertices: Any, color: Any = "#58C4DD", **kw: Any) -> None:
        super().__init__(np.array([_vec3(v) for v in vertices]), color, **kw)


class RegularPolygon(Polygon):
    def __init__(self, n: int = 6, radius: float = 1.0, color: Any = "#58C4DD", **kw: Any) -> None:
        t = np.pi / 2 + np.arange(n) * TAU / n
        super().__init__(*[(radius * math.cos(a), radius * math.sin(a), 0) for a in t],
                         color=color, **kw)  # fmt: skip


class Triangle(RegularPolygon):
    def __init__(self, **kw: Any) -> None:
        super().__init__(3, **kw)


class Rectangle(Polygon):
    def __init__(self, width: float = 4.0, height: float = 2.0, color: Any = "#FFFFFF",
                 **kw: Any) -> None:  # fmt: skip
        w, h = width / 2, height / 2
        super().__init__((-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0), color=color, **kw)


class Square(Rectangle):
    def __init__(self, side_length: float = 2.0, **kw: Any) -> None:
        super().__init__(side_length, side_length, **kw)


class Line(_Path):
    def __init__(
        self, start: Any = LEFT, end: Any = RIGHT, color: Any = "#FFFFFF", **kw: Any
    ) -> None:
        super().__init__(np.array([_vec3(start), _vec3(end)]), color, **kw)


DashedLine = Line


class Arrow(Mobject):
    kind = "arrow"

    def __init__(
        self, start: Any = LEFT, end: Any = RIGHT, color: Any = "#FFFFFF", **kw: Any
    ) -> None:
        super().__init__(color=color, **kw)
        self.pts = np.array([_vec3(start), _vec3(end)])

    def phasor_object(self, ident: str):
        from phasor import linalg

        return linalg.Arrow(id=ident, start=tuple(self.pts[0]), end=tuple(self.pts[1]),
                            opacity=0.0, style=self._style())  # fmt: skip


class Vector(Arrow):
    def __init__(self, direction: Any = RIGHT, **kw: Any) -> None:
        super().__init__(ORIGIN, direction, **kw)


class ParametricFunction(_Path):
    def __init__(self, function: Callable[[float], Any], t_range: Sequence[float] = (0, 1),
                 color: Any = "#FFFFFF", **kw: Any) -> None:  # fmt: skip
        lo, hi = float(t_range[0]), float(t_range[1])
        n = 240
        pts = np.array([_vec3(function(t)) for t in np.linspace(lo, hi, n)])
        super().__init__(pts, color, **kw)


class Text(Mobject):
    kind = "text"

    def __init__(self, *strings: str, font_size: float = 48, color: Any = "#FFFFFF",
                 math: bool = False, **kw: Any) -> None:  # fmt: skip
        super().__init__(color=color, **kw)
        raw = "".join(strings)
        self.role = text_role(font_size)
        self.math = math
        self.text = raw if not math or "$" in raw else f"${raw}$"
        try:
            text_metrics.check_math(self.text)
        except ValueError as e:
            self._warn(f"{raw!r} could not be typeset ({e}); shown as plain text")
            self.text = raw.replace("$", "").replace("\\", "")
        self.pts = np.zeros((1, 3))
        self._ref = FRAME_HEIGHT * {"title": 0.075, "math": 0.048, "label": 0.042,
                                    "annotation": 0.032}[self.role]  # fmt: skip

    def _text_extent(self) -> tuple[float, float]:
        return text_metrics.aspect(self.text) * self._ref, text_metrics.box(self.text) * self._ref

    def phasor_object(self, ident: str):
        from phasor import Text as T

        return T(id=ident, text=self.text, pos=tuple(self.pts[0]), opacity=0.0,
                 style=self._style(size=self.role))  # fmt: skip


class MathTex(Text):
    def __init__(self, *strings: str, **kw: Any) -> None:
        super().__init__(*strings, math=True, **kw)


Tex = MathTex
Title = Text
MarkupText = Text


class Axes(Mobject):
    kind = "axes"

    def __init__(
        self,
        x_range: Sequence[float] = (-7, 7, 1),
        y_range: Sequence[float] = (-4, 4, 1),
        x_length: float | None = None,
        y_length: float | None = None,
        color: Any = "#FFFFFF",
        **kw: Any,
    ) -> None:
        super().__init__(color=color, **kw)
        xr = [*list(x_range), 1][:3]
        yr = [*list(y_range), 1][:3]
        self.xr, self.yr = xr, yr
        self.sx = (x_length or 12.0) / (xr[1] - xr[0])
        self.sy = (y_length or 6.0) / (yr[1] - yr[0])
        if abs(self.sx - self.sy) > 1e-6:
            self._warn("axes with different x and y scales are drawn with the x scale")
        self.origin = np.array(
            [-(xr[0] + xr[1]) / 2 * self.sx, -(yr[0] + yr[1]) / 2 * self.sy, 0.0]
        )
        self.pts = np.array([self.origin])

    def c2p(self, x: float, y: float, z: float = 0.0) -> np.ndarray:
        return self.pts[0] + np.array([x * self.sx, y * self.sy, 0.0])

    coords_to_point = c2p

    def plot(self, func: Callable[[float], float], x_range: Sequence[float] | None = None,
             color: Any = "#58C4DD", **kw: Any) -> Mobject:  # fmt: skip
        lo, hi = (x_range or self.xr)[:2]
        xs = np.linspace(lo, hi, 240)
        with np.errstate(all="ignore"):
            ys = np.array([float(func(x)) for x in xs])
        ok = np.isfinite(ys)
        return _Path(np.array([self.c2p(x, y) for x, y in zip(xs[ok], ys[ok], strict=True)]), color)

    get_graph = plot

    def get_axis_labels(self, *a: Any, **kw: Any) -> Mobject:
        self._warn("axis labels are not carried over: add Text next to the axes")
        return VGroup()

    def phasor_object(self, ident: str):
        from phasor import geo

        o = self.pts[0]
        return geo.Axes(id=ident, dims=2, x_range=tuple(self.xr), y_range=tuple(self.yr),
                        pos=tuple(float(v) for v in o), scale=self.sx, opacity=0.0,
                        style=self._style())  # fmt: skip


class NumberPlane(Axes):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**{"x_range": (-7, 7, 1), "y_range": (-4, 4, 1), **kw})
        self._warn("a NumberPlane is drawn as plain axes with ticks (no grid)")


class ValueTracker:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)
        self._warned = False

    def get_value(self) -> float:
        return self.value

    def set_value(self, v: float) -> ValueTracker:
        self.value = float(v)
        return self

    @property
    def animate(self) -> Any:
        _CTX and _CTX.warnings.append("ValueTracker animations are not carried over (they only "
                                      "matter through updaters)")  # fmt: skip
        return types.SimpleNamespace(set_value=lambda v: Wait(0.0))


def always_redraw(func: Callable[[], Mobject]) -> Mobject:
    if _CTX:
        _CTX.warnings.append("always_redraw / updaters are baked: the object keeps its first look")
    return func()


# ---------------------------------------------------------------- animations


def smooth(t: float) -> float:  # the rate-function names, so scripts can pass them
    return t * t * (3 - 2 * t)


def linear(t: float) -> float:
    return t


def rush_into(t: float) -> float:
    return t * t


def rush_from(t: float) -> float:
    return 1 - (1 - t) ** 2


def there_and_back(t: float) -> float:
    return 1 - abs(2 * t - 1)


_EASE = {"smooth": "smooth", "linear": "linear", "rush_into": "in", "rush_from": "out",
         "ease_in_out_sine": "in_out", "ease_in_sine": "in", "ease_out_sine": "out"}  # fmt: skip


def _ease_of(rate_func: Any) -> str:
    return _EASE.get(getattr(rate_func, "__name__", "smooth"), "smooth")


class Animation:
    default_time = 1.0

    def __init__(self, mob: Mobject | None = None, run_time: float | None = None,
                 rate_func: Any = smooth, lag_ratio: float = 0.0, **kw: Any) -> None:  # fmt: skip
        self.mob = mob
        self.run_time = run_time
        self.rate_func = rate_func

    @property
    def duration(self) -> float:
        return self.run_time if self.run_time is not None else self.default_time

    def apply(self, ctx: _Context, t0: float, d: float) -> None:  # pragma: no cover
        raise NotImplementedError


def _fade(ctx: _Context, mob: Mobject, t0: float, d: float, a: float, b: float, ease: str) -> None:
    for leaf in mob.leaves():
        ctx.show(leaf, t0, opacity=a)
        ctx.track(leaf, "opacity", b, t0, d, ease, start=a)
        leaf.opacity = b


class FadeIn(Animation):
    def __init__(self, mob: Mobject, shift: Any = None, **kw: Any) -> None:
        super().__init__(mob, **kw)

    def apply(self, ctx, t0, d):
        _fade(ctx, self.mob, t0, d, 0.0, 1.0, _ease_of(self.rate_func))


class FadeOut(Animation):
    def __init__(self, mob: Mobject, shift: Any = None, **kw: Any) -> None:
        super().__init__(mob, **kw)

    def apply(self, ctx, t0, d):
        _fade(ctx, self.mob, t0, d, 1.0, 0.0, _ease_of(self.rate_func))
        ctx.gone(self.mob)


class Create(Animation):
    def apply(self, ctx, t0, d):
        ease = _ease_of(self.rate_func)
        for leaf in self.mob.leaves():
            if leaf.kind == "line":
                ctx.show(leaf, t0, progress=0.0)
                ctx.track(leaf, "progress", 1.0, t0, d, ease, start=0.0)
            else:
                _fade(ctx, leaf, t0, d, 0.0, 1.0, ease)


Write = DrawBorderThenFill = ShowCreation = Create


class Uncreate(Animation):
    def apply(self, ctx, t0, d):
        for leaf in self.mob.leaves():
            if leaf.kind == "line":
                ctx.track(leaf, "progress", 0.0, t0, d, _ease_of(self.rate_func), start=1.0)
            else:
                _fade(ctx, leaf, t0, d, 1.0, 0.0, _ease_of(self.rate_func))
        ctx.gone(self.mob)


class GrowFromCenter(Animation):
    def apply(self, ctx, t0, d):
        ease = _ease_of(self.rate_func)
        for leaf in self.mob.leaves():
            c = leaf.get_center()
            if leaf.kind == "dot":
                ctx.show(leaf, t0, r=0.0)
                ctx.track(leaf, "r", leaf.size, t0, d, ease, start=0.0)
            elif leaf.kind == "line":
                ctx.show(leaf, t0, points=_tuples(np.tile(c, (len(leaf.pts), 1))))
                ctx.track(leaf, "points", _tuples(leaf.pts), t0, d, ease)
            else:
                _fade(ctx, leaf, t0, d, 0.0, 1.0, ease)


GrowArrow = SpinInFromNothing = GrowFromCenter


class _MorphTo(Animation):
    """Move a mobject's geometry to a target's (same kind and point count), else crossfade."""

    def _morph(self, ctx, leaf, goal, t0, d):
        ease = _ease_of(self.rate_func)
        if (
            leaf.kind == goal.kind
            and leaf.kind in ("dot", "line", "arrow")
            and len(leaf.pts) == len(goal.pts)
        ):
            ctx.show(leaf, t0)
            if leaf.kind == "dot":
                ctx.track(leaf, "pos", tuple(goal.pts[0]), t0, d, ease)
                ctx.track(leaf, "r", goal.size, t0, d, ease)
            elif leaf.kind == "arrow":
                ctx.track(leaf, "start", tuple(goal.pts[0]), t0, d, ease)
                ctx.track(leaf, "end", tuple(goal.pts[1]), t0, d, ease)
            else:
                ctx.track(leaf, "points", _tuples(goal.pts), t0, d, ease)
            leaf.pts, leaf.size = goal.pts.copy(), goal.size
            return True
        return False


class MoveToTarget(_MorphTo):
    def apply(self, ctx, t0, d):
        goal = self.mob.target
        if goal is None:
            raise ManimImportError("MoveToTarget needs mob.generate_target() first")
        for leaf, g in zip(self.mob.leaves(), goal.leaves(), strict=False):
            self._morph(ctx, leaf, g, t0, d)


class Transform(_MorphTo):
    """``Transform(a, b)``: ``a`` becomes ``b``. Shapes with the same number of points morph;
    anything else crossfades. Afterwards ``a`` stands for what ``b`` was (as in Manim)."""

    def __init__(self, mob: Mobject, target: Mobject, **kw: Any) -> None:
        super().__init__(mob, **kw)
        self.goal = target

    def apply(self, ctx, t0, d):
        ease = _ease_of(self.rate_func)
        mine, theirs = self.mob.leaves(), self.goal.leaves()
        if len(mine) == 1 and len(theirs) == 1 and self._morph(ctx, mine[0], theirs[0], t0, d):
            mine[0].color_role = theirs[0].color_role or mine[0].color_role
            return
        for leaf in mine:
            _fade(ctx, leaf, t0, d, 1.0, 0.0, ease)
        for leaf in theirs:
            _fade(ctx, leaf, t0, d, 0.0, 1.0, ease)
        ctx.gone(self.mob)
        self.mob.alias = self.goal
        ctx.shown.add(id(self.goal))


class ReplacementTransform(Transform):
    pass


TransformMatchingTex = TransformMatchingShapes = FadeTransform = ReplacementTransform


class Rotate(_MorphTo):
    def __init__(self, mob: Mobject, angle: float = PI, about_point: Any = None, **kw: Any) -> None:
        super().__init__(mob, **kw)
        self.angle, self.about = angle, about_point

    def apply(self, ctx, t0, d):
        goal = self.mob.copy()
        goal.rotate(self.angle, about_point=self.about)
        for leaf, g in zip(self.mob.leaves(), goal.leaves(), strict=False):
            self._morph(ctx, leaf, g, t0, d)


class Indicate(Animation):
    def apply(self, ctx, t0, d):
        for leaf in self.mob.leaves():
            if leaf.kind == "dot":
                ctx.track(leaf, "r", leaf.size * 1.3, t0, d / 2, "smooth")
                ctx.track(leaf, "r", leaf.size, t0 + d / 2, d / 2, "smooth", start=leaf.size * 1.3)


class Wait(Animation):
    def apply(self, ctx, t0, d):
        return None


class AnimationGroup(Animation):
    def __init__(self, *anims: Animation, lag_ratio: float = 0.0, run_time: float | None = None,
                 **kw: Any) -> None:  # fmt: skip
        super().__init__(None, run_time=run_time)
        self.anims = [a.build() if isinstance(a, _AnimateBuilder) else a for a in anims]
        self.lag = lag_ratio

    def _starts(self) -> list[float]:
        out, t = [], 0.0
        for a in self.anims:
            out.append(t)
            t += a.duration * self.lag if self.lag else 0.0
        return out

    @property
    def duration(self) -> float:
        if self.run_time is not None:
            return self.run_time
        starts = self._starts()
        return max((s + a.duration for s, a in zip(starts, self.anims, strict=True)), default=0.0)

    def apply(self, ctx, t0, d):
        for s, a in zip(self._starts(), self.anims, strict=True):
            a.apply(ctx, t0 + s, a.duration)


class LaggedStart(AnimationGroup):
    def __init__(self, *anims: Animation, lag_ratio: float = 0.05, **kw: Any) -> None:
        super().__init__(*anims, lag_ratio=lag_ratio, **kw)


class Succession(AnimationGroup):
    def __init__(self, *anims: Animation, **kw: Any) -> None:
        super().__init__(*anims, lag_ratio=1.0, **kw)


class _AnimateBuilder:
    """``mob.animate.shift(RIGHT).set_color(RED)``: the target is what those calls do to a
    copy; playing it moves the mobject there."""

    def __init__(self, mob: Mobject) -> None:
        self.mob = mob
        self.calls: list[tuple[str, tuple, dict]] = []
        self.run_time: float | None = None
        self.rate_func: Any = smooth

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*a: Any, **kw: Any) -> _AnimateBuilder:
            self.calls.append((name, a, kw))
            return self

        return call

    def build(self) -> Animation:
        mob = self.mob
        goal = mob.copy()
        for name, a, kw in self.calls:
            getattr(goal, name)(*a, **kw)
        anim = _MorphToGoal(mob, goal, run_time=self.run_time, rate_func=self.rate_func)
        return anim


class _MorphToGoal(_MorphTo):
    def __init__(self, mob: Mobject, goal: Mobject, **kw: Any) -> None:
        super().__init__(mob, **kw)
        self.goal = goal

    def apply(self, ctx, t0, d):
        for leaf, g in zip(self.mob.leaves(), self.goal.leaves(), strict=False):
            self._morph(ctx, leaf, g, t0, d)
            if g.opacity != leaf.opacity:
                _fade(ctx, leaf, t0, d, leaf.opacity, g.opacity, "smooth")
            if g.color_role != leaf.color_role and g.color_role:
                leaf.color_role = g.color_role
                ctx.restyle(leaf)
                ctx.warnings.append("a color change is instant here (style roles do not animate)")


# ---------------------------------------------------------------- the scene


@dataclass
class _Context:
    """The phasor scene being built and the Manim clock."""

    scene: PhasorScene
    warnings: list[str] = field(default_factory=list)
    t: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    shown: set[int] = field(default_factory=set)  # python ids of leaves currently visible
    camera_calls: list[tuple] = field(default_factory=list)

    def ident(self, leaf: Mobject) -> str:
        name = type(leaf).__name__.lower().lstrip("_")
        self.counts[name] = self.counts.get(name, 0) + 1
        return f"{name}_{self.counts[name]}"

    def realize(self, leaf: Mobject, **initial: Any) -> str:
        """Make sure the leaf exists in the phasor scene; returns its object id. Objects start
        invisible (opacity 0): ``show`` switches them on at the time Manim first shows them."""
        if leaf.obj_id is None:
            leaf.obj_id = self.ident(leaf)
            self.scene.add(leaf.phasor_object(leaf.obj_id))
        return leaf.obj_id

    def restyle(self, leaf: Mobject) -> None:
        if leaf.obj_id and leaf.color_role:
            from phasor.core.objects import registered_types  # noqa: F401

            self.scene.engine.apply({"op": "set_style", "id": leaf.obj_id, "key": "color",
                                     "role": leaf.color_role}, origin="script")  # fmt: skip

    def show(self, leaf: Mobject, t: float, **start: Any) -> None:
        """Make the leaf visible from time ``t`` (with starting values for animated params)."""
        oid = self.realize(leaf)
        if id(leaf) in self.shown and not start:
            return
        first = leaf.obj_id and id(leaf) not in self.shown
        for k, v in start.items():
            self.scene.at(t).set(f"{oid}.{k}", v)
        if first and "opacity" not in start:
            self.scene.at(t).set(f"{oid}.opacity", leaf.opacity or 1.0)
        self.shown.add(id(leaf))

    def track(self, leaf: Mobject, param: str, value: Any, t0: float, d: float, ease: str,
              start: Any = None) -> None:  # fmt: skip
        oid = self.realize(leaf)
        path = f"{oid}.{param}"
        if start is not None:
            self.scene.at(t0).set(path, start)
        self.scene.at(t0).to(path, value, duration=max(d, 1e-3), ease=ease)

    def gone(self, mob: Mobject) -> None:
        for leaf in mob.leaves():
            self.shown.discard(id(leaf))


_CTX: _Context | None = None


class _Camera:
    def __init__(self, ctx_getter: Callable[[], _Context]) -> None:
        self._ctx = ctx_getter
        self.background_color = "#000000"
        self.frame = _Frame()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None


class _Frame(Mobject):
    kind = "frame"

    def __init__(self) -> None:
        super().__init__()
        self.pts = np.zeros((1, 3))
        self.size = FRAME_HEIGHT


class Scene:
    """Manim's ``Scene``: ``construct`` runs on ``render`` (here, on import)."""

    def __init__(self, **kw: Any) -> None:
        self.camera = _Camera(lambda: _CTX)  # type: ignore[arg-type]
        self.mobjects: list[Mobject] = []
        self.renderer = None

    # -- the clock and the scene
    @property
    def _c(self) -> _Context:
        assert _CTX is not None
        return _CTX

    def _animation(self, a: Any) -> Animation:
        if isinstance(a, _AnimateBuilder):
            return a.build()
        if isinstance(a, Animation):
            return a
        if isinstance(a, Mobject):
            raise ManimImportError("play() got a mobject: wrap it in an animation (Create, "
                                   "FadeIn ...)")  # fmt: skip
        raise ManimImportError(f"play() got {type(a).__name__}, which is not an animation")

    def add(self, *mobs: Mobject) -> None:
        for m in mobs:
            for leaf in m.leaves():
                self._c.show(leaf, self._c.t)
            if m not in self.mobjects:
                self.mobjects.append(m)

    def remove(self, *mobs: Mobject) -> None:
        for m in mobs:
            for leaf in m.leaves():
                if id(leaf) in self._c.shown and leaf.obj_id:
                    self._c.scene.at(self._c.t).set(f"{leaf.obj_id}.opacity", 0.0)
            self._c.gone(m)
            if m in self.mobjects:
                self.mobjects.remove(m)

    def clear(self) -> None:
        self.remove(*list(self.mobjects))

    def play(self, *anims: Any, run_time: float | None = None, rate_func: Any = None,
             lag_ratio: float = 0.0, **kw: Any) -> None:  # fmt: skip
        built = [self._animation(a) for a in anims]
        for a in built:
            if run_time is not None:
                a.run_time = run_time
            if rate_func is not None:
                a.rate_func = rate_func
        d = max((a.duration for a in built), default=0.0)
        for a in built:
            a.apply(self._c, self._c.t, a.duration)
            if isinstance(a, (FadeIn, Create, GrowFromCenter, Transform)) and a.mob is not None:
                for m in [a.mob] if not isinstance(a, Transform) else [a.mob, a.goal]:
                    if m not in self.mobjects and not isinstance(a, FadeOut):
                        self.mobjects.append(m)
        self._c.t += d

    def wait(self, duration: float = 1.0, **kw: Any) -> None:
        self._c.t += float(duration)

    def next_section(self, *a: Any, **k: Any) -> None:
        return None

    def add_sound(self, *a: Any, **k: Any) -> None:
        self._c.warnings.append("add_sound is not carried over (add the file as a cue on a "
                                "marker: s.cue(marker, 'file.wav'))")  # fmt: skip

    def construct(self) -> None:  # pragma: no cover
        pass

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise ManimImportError(f"Scene.{name} is not supported by phasor's Manim import yet")


class MovingCameraScene(Scene):
    pass


class ThreeDScene(Scene):
    """``set_camera_orientation(phi, theta)`` and ambient rotation become the orbit camera."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._rotation_rate = 0.0
        self._rotation_from: float | None = None

    def set_camera_orientation(self, phi: float | None = None, theta: float | None = None,
                               gamma: float | None = None, zoom: float | None = None,
                               **kw: Any) -> None:  # fmt: skip
        params: dict[str, Any] = {"distance": 18.0, "target": (0.0, 0.0, 0.0), "drift": 0.0}
        params["elevation"] = 90.0 - math.degrees(phi if phi is not None else 70 * DEGREES)
        params["azimuth"] = math.degrees(theta if theta is not None else -90 * DEGREES) + 90.0
        self._c.scene.use_camera("orbit", **params)

    def begin_ambient_camera_rotation(self, rate: float = 0.02, **kw: Any) -> None:
        self._rotation_rate, self._rotation_from = rate, self._c.t

    def stop_ambient_camera_rotation(self, **kw: Any) -> None:
        self._finish_rotation()

    def _finish_rotation(self) -> None:
        if self._rotation_from is None:
            return
        span = self._c.t - self._rotation_from
        if span > 0:
            deg = math.degrees(self._rotation_rate) * span * 60 / 60  # Manim: rad per second
            doc = self._c.scene.engine.document
            cam = doc.camera(doc.timeline.camera)
            az = float(cam.params.get("azimuth", 0.0)) if cam else 0.0
            self._c.scene.at(self._rotation_from).to("camera.azimuth", az + deg, duration=span,
                                                     ease="linear")  # fmt: skip
        self._rotation_from = None


# ---------------------------------------------------------------- the stand-in module


def _module() -> types.ModuleType:
    mod = types.ModuleType("manim")
    names = {k: v for k, v in globals().items() if not k.startswith("_")}
    names.update({k: v for k, v in _COLORS.items()})
    names["np"] = np  # Manim's star-import gives scripts numpy as ``np``
    names["config"] = types.SimpleNamespace(frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH,
                                            pixel_height=1080, pixel_width=1920)  # fmt: skip
    mod.__dict__.update(names)
    keep = ("config", "np", "smooth", "linear", "always_redraw")
    mod.__all__ = [k for k in names if k[0].isupper() or k in keep]  # type: ignore[attr-defined]

    def missing(name: str) -> Any:
        raise AttributeError(f"{name!r} is not in phasor's Manim stand-in yet (see docs/ai/API.md,"
                             " 'Manim import')")  # fmt: skip

    mod.__getattr__ = missing  # type: ignore[attr-defined]
    return mod


def import_manim(path: str | Path, scene: str | None = None) -> ManimImport:
    """Run the Manim script at ``path`` (a ``Scene`` subclass named ``scene``, or the first one
    it defines) and return the phasor scene it builds."""
    global _CTX
    path = Path(path)
    fake = _module()
    saved = {k: sys.modules.get(k) for k in ("manim", "manimlib")}
    sys.modules["manim"] = fake
    try:
        ns = runpy.run_path(str(path), run_name="manim_import")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    classes = [v for v in ns.values() if isinstance(v, type) and issubclass(v, Scene)
               and v.__module__ == "manim_import"]  # fmt: skip
    if not classes:
        raise ManimImportError(f"{path.name} defines no Manim Scene class")
    if scene is not None:
        match = [c for c in classes if c.__name__ == scene]
        if not match:
            raise ManimImportError(f"no Scene {scene!r} in {path.name}; it defines "
                                   f"{[c.__name__ for c in classes]}")  # fmt: skip
        cls = match[0]
    else:
        cls = classes[0]
    s = PhasorScene(cls.__name__, width=1920, height=1080)
    s.set("camera.height", FRAME_HEIGHT)
    s.set("camera.center", (0.0, 0.0, 0.0))
    ctx = _Context(scene=s)
    _CTX = ctx
    try:
        instance = cls()
        instance.construct()
        if isinstance(instance, ThreeDScene):
            instance._finish_rotation()
        s.marker("end", ctx.t)
    finally:
        _CTX = None
    return ManimImport(
        scene=s,
        warnings=list(dict.fromkeys(ctx.warnings)),
        duration=ctx.t,
        name=cls.__name__,
        others=[c.__name__ for c in classes if c is not cls],
    )
