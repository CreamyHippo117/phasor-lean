"""Point clouds: hundreds of thousands to millions of points, each a tiny
glowing splat, added together so dense regions build up light and fine structure appears.

    from phasor import Scene, cloud

    s.add(cloud.PointCloud(id="roots", source="roots", degree=12, size=(8, 4.5)))

The points come from a source in ``domains/cloud_math.py`` (the roots of every polynomial
with coefficients +-1, a strange attractor, the Buddhabrot) or are given directly. A point
sits at world (x, y) = ``pos`` + ``scale`` * (the source's point), so with ``pos`` at the
origin and ``scale`` 1 the roots are at their own places in the complex plane. The cloud is
drawn through ``ctx.field`` inside the ``size`` rectangle round ``pos`` by the ``splat``
kernel, which computes only the part on screen: zooming in keeps every point sharp.

Three things keep a cloud from sitting still: ``twist`` (animatable) turns one coefficient
of a root cloud round the origin, so every root glides and the fractal edge morphs;
``twinkle`` flickers each point's light a little (life, a pure function of (id, seed, t));
``white`` lets the densest parts run white-hot.

A bifurcation diagram (``source="bifurcation"``) is computed for the view: its columns of
r sit on a lattice fixed in r that doubles as the view narrows, so zooming keeps the
columns about a pixel apart and the columns already there never change (new ones fade in
between them). Zoomed in on x, a column shows more of its orbit.
"""

from __future__ import annotations

import functools
import math
import threading
from collections import OrderedDict
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.rng import rng
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import cloud_math as clm
from phasor.render.gpu.codegen import FunctionError

# Brightness is set against the cloud's mean density (points per pixel over the rectangle,
# at the current zoom): ``exposure`` 1 lights a uniform cloud to 1 - 1/e. The mean is held at
# least this, so a sparse cloud seen close up keeps single points as soft dots, not flares.
MIN_DENSITY = 0.02
ZOOM_LIGHT = 0.7  # 1: brightness held as the camera zooms in; 0: each point's light fixed
CACHE = 8  # point sets kept in memory (the latest ones)
CARD_CACHE = 2  # point sets kept on the CUDA card too (the latest ones)
TWINKLE_PERIOD = 2.5  # seconds, typical; each point (each polynomial's roots) has its own

# Bifurcation diagrams: level L of the lattice has columns at r = j * LATTICE / 2^L. A view
# draws the level whose columns are COLUMN_PX pixels apart or less, and fades in the columns
# of the next level by how far the zoom has gone toward it.
LATTICE = 2.0**-6
COLUMN_PX = 1.0  # spacing of the columns on screen, pixels (at most)
CHUNK = 256  # columns per cached block of orbits
ORBIT_BYTES = 1 << 28  # orbit blocks kept in memory (the latest ones), bytes
ROWS_MAX = 4.0  # zoomed in on x, a column shows up to ROWS_MAX times n_keep iterates
X_REFERENCE = 1.0  # the x span (map units) on screen at which a column shows n_keep iterates
SWEEP_EDGE = 0.03  # share of the window over which the sweep's front fades in
TWINKLE_GROUPS = 4096  # columns flicker in this many groups (by their place in r)


@functools.lru_cache(maxsize=CACHE)
def source_points(oid: str, seed: int, source: str, key: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(points (n, 2) float64, value (n,) in [0, 1)) for one source; cached per params."""
    gen = rng(oid, seed)
    if source == "roots":
        degree, coefficients, count, twist = key
        total = clm.every_polynomial_count(degree, coefficients)
        polys = None if total * degree <= count else max(1, count // degree)
        z, v = clm.polynomial_roots(degree, coefficients, polys, gen, twist=twist)
    elif source == "attractor":
        kind, constants, count = key
        z, v = clm.attractor(kind, count, gen, constants)
    elif source == "buddhabrot":
        max_iter, count = key
        z, v = clm.buddhabrot(count, max_iter, gen)
    else:
        pts, values = key
        z = np.array([complex(x, y) for x, y in pts], dtype=np.complex128)
        v = np.asarray(values, dtype=np.float64)
        if not values:  # spread along the list, so a hand-made cloud is not one colour
            v = np.linspace(0.0, 1.0, len(z), endpoint=False)
    out = np.ascontiguousarray(np.c_[z.real, z.imag], dtype=np.float64)
    out.flags.writeable = False
    v = np.asarray(v, dtype=np.float64) % 1.0
    v.flags.writeable = False
    return out, v


@functools.lru_cache(maxsize=CARD_CACHE)
def card_points(oid: str, seed: int, source: str, key: tuple) -> tuple[Any, Any] | None:
    """``source_points`` as CuPy arrays, uploaded once and kept on the card, or None
    without CUDA. Drawing from these keeps the per-point work (colour, placing, culling) on
    the card, in the same arithmetic as the host path (``_colors``, ``_inside``), so the
    picture is the same bits; only the finished pixels come back."""
    from phasor.render.gpu.registry import default_gpu

    gpu = default_gpu()
    if gpu is None:
        return None
    pts, v = source_points(oid, seed, source, key)
    return gpu.cp.asarray(pts), gpu.cp.asarray(v)


def array_module(a: Any) -> Any:
    """NumPy for a NumPy array, else CuPy (an array from ``card_points``)."""
    if isinstance(a, np.ndarray):
        return np
    from phasor.render.gpu.registry import default_gpu

    gpu = default_gpu()
    assert gpu is not None, "a device array without CUDA"
    return gpu.cp


@functools.lru_cache(maxsize=CACHE)
def _flicker(oid: str, seed: int, groups: int) -> tuple[np.ndarray, np.ndarray]:
    """(frequency, phase) per group of points that flicker together, from the seed."""
    gen = rng(f"{oid}/twinkle", seed)
    return gen.uniform(0.5, 1.5, groups) / TWINKLE_PERIOD, gen.uniform(0.0, 1.0, groups)


@register
class PointCloud(ObjectBase):
    """Many points of light, splatted additively: where they crowd, light builds up and
    saturates softly (``1 - exp(-density)``), so dense regions glow and the fine structure
    at their edges shows.

    ``source``: ``roots`` (the roots of every polynomial of ``degree`` with coefficients
    from ``coefficients``, or a random sample of them when there are more than ``count``
    roots), ``attractor`` (``kind`` clifford, dejong or lorenz, fitted to the square
    [-1, 1]^2; ``constants`` override the classic ones), ``buddhabrot`` (escaping orbits,
    ``max_iter``) or ``points`` (``points`` given directly, with optional ``values`` 0-1
    for colour). Colour is the ``colormap`` role at each point's value (``color_by``
    ``source``: what the source says, e.g. the polynomial's place in the list; ``phase``:
    the point's angle round ``pos``), turned by ``hue``. ``point_size`` is the splat's
    radius in pixels, ``exposure`` how much light a point gives; ``glow`` makes the bright
    parts shine and haze.

    Motion: ``twist`` (roots) turns the last coefficient of the set by that share of a turn
    (b -> b e^(2 pi i twist)); tween it with ``.to`` and every root glides continuously
    while the fractal edge morphs. ``twinkle`` (0-1, life) flickers each point's light by
    up to that share (the roots of one polynomial together), slowly and differently for
    each; ``white`` (0-1) turns the densest parts white-hot.

    ``source="bifurcation"``: the bifurcation diagram of x -> ``map``(r, x) (``logistic``,
    ``sine`` or a formula in r and x). The window ``r_range`` runs across the rectangle and
    ``x_range`` up it; each column of r draws the last ``n_keep`` of ``n_iter`` iterates of
    ``x0``, coloured by its Lyapunov exponent (cycles one hue, chaos another). The columns
    are computed for the view (about a pixel apart at any zoom; ``count`` is not used), so a
    camera zoom stays sharp. ``focus`` (animatable, 0-1) zooms the window geometrically to
    ``r_focus`` x ``x_focus`` (r and x by different factors, unlike the camera); ``sweep``
    (0-1) draws the diagram from the left up to that share of the window. The preset
    ``prism_bifurcation`` lights it in the prism look."""

    type: ClassVar[str] = "point_cloud"
    Life: ClassVar = frozenset({"hue", "glow", "twinkle"})
    Style: ClassVar = {"colormap": ("colormap", "spectrum"), "glow": ("glow", "none")}
    Presets: ClassVar = {
        # dark ground, palette hue, glow and haze (the look of the prism strokes)
        "prism": {"params": {"exposure": 0.06, "point_size": 0.9, "white": 0.8,
                             "twinkle": 0.6},
                  "style": {"colormap": "spectrum", "glow": "strong"}},
        "plain": {"params": {"exposure": 0.3, "point_size": 1.0},
                  "style": {"colormap": "sequential", "glow": "none"}},
        # a bifurcation diagram in the prism look: its cycles pile every iterate on one
        # pixel, so it takes more light per point than a cloud and less white
        "prism_bifurcation": {"params": {"source": "bifurcation", "exposure": 0.3,
                                         "point_size": 0.8, "white": 0.2, "twinkle": 0.3},
                              "style": {"colormap": "spectrum", "glow": "strong"}},
    }  # fmt: skip

    class Params(ParamsBase):
        source: Literal["roots", "attractor", "buddhabrot", "points", "bifurcation"] = "roots"
        degree: int = Field(10, ge=1, le=24)  # roots: the polynomials' degree
        coefficients: tuple[float, ...] = (-1.0, 1.0)  # roots: the coefficient set
        kind: Literal["clifford", "dejong", "lorenz"] = "clifford"  # attractor
        constants: tuple[float, ...] = ()  # attractor: () for the classic constants
        max_iter: int = Field(200, ge=8, le=5000)  # buddhabrot: longest orbit
        points: tuple[tuple[float, float], ...] = ()  # source "points": (x, y) each
        values: tuple[float, ...] = ()  # source "points": colour value 0-1 each (optional)
        count: int = Field(100_000, ge=1, le=20_000_000)  # how many points, at most
        seed: int = 0
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre of the rectangle and origin of the points
        size: tuple[float, float] = (8.0, 4.5)  # the rectangle the cloud is seen in
        scale: float = Field(1.0, gt=0)  # world units per unit of the source
        point_size: float = Field(1.0, ge=0.4, le=2.5)  # splat radius (sigma), pixels
        exposure: float = Field(0.3, gt=0)  # light per point, against the mean density
        hue: float = 0.0  # turns the palette (a share of it); life adds a little
        color_by: Literal["source", "phase"] = "source"
        twist: float = 0.0  # roots: turns the last coefficient by this share of a turn
        twinkle: float = Field(0.0, ge=0, le=1)  # flicker of each point's light (life)
        white: float = Field(0.0, ge=0, le=1)  # how white the densest parts run
        # bifurcation: x -> map(r, x), r across the rectangle, x up it
        map: str = "logistic"  # "logistic", "sine" or a formula in r and x
        r_range: tuple[float, float] = (2.8, 4.0)  # the window: r across the rectangle
        x_range: tuple[float, float] = (0.0, 1.0)  # and x up it
        r_focus: tuple[float, ...] = ()  # the window ``focus`` zooms to (() = the window)
        x_focus: tuple[float, ...] = ()
        focus: float = Field(0.0, ge=0, le=1)  # 0: the window, 1: the focus, between: a zoom
        x0: float = 0.5  # where every orbit starts
        n_iter: int = Field(600, ge=1, le=20_000)  # steps each orbit runs
        n_keep: int = Field(200, ge=1, le=5000)  # of which the last ones are drawn
        sweep: float = Field(1.0, ge=0, le=1)  # share of the window drawn, from the left

        @field_validator("map")
        @classmethod
        def _map(cls, v):
            try:
                clm.map_function(v)
            except FunctionError as e:
                raise ValueError(str(e)) from e
            return v

        @field_validator("r_range", "x_range", "r_focus", "x_focus")
        @classmethod
        def _range(cls, v):
            if v and (len(v) != 2 or not v[0] < v[1]):
                raise ValueError("a window is (low, high) with low < high")
            return v

        @field_validator("size")
        @classmethod
        def _positive(cls, v):
            if min(v) <= 0:
                raise ValueError("size must be positive")
            return v

        @field_validator("coefficients")
        @classmethod
        def _coeffs(cls, v):
            if not v or not any(c != 0 for c in v):
                raise ValueError("coefficients must include a nonzero number")
            return v

        @model_validator(mode="after")
        def _points_and_values(self):
            if self.values and len(self.values) != len(self.points):
                raise ValueError(f"{len(self.points)} points but {len(self.values)} values")
            if self.constants and len(self.constants) != len(clm.ATTRACTORS[self.kind]):
                n = len(clm.ATTRACTORS[self.kind])
                raise ValueError(f"a {self.kind} attractor takes {n} constants")
            if self.n_keep > self.n_iter:
                raise ValueError(f"n_keep ({self.n_keep}) is more than n_iter ({self.n_iter})")
            return self

    def cloud(self, state: SceneState) -> tuple[np.ndarray, np.ndarray]:
        """(points (n, 2) in world x, y; value (n,) 0-1) at this state, from the cache."""
        p = self.me(state).params
        if p.source == "bifurcation":  # the whole rectangle at 1080 pixels up it
            w, h = p.size
            ph = 1080
            pw = max(1, math.ceil(ph * w / h))
            d = _diagram(self.id, p, state.t, (w / 2, h / 2), h, pw, ph)
            corner = np.asarray(p.pos[:2]) - np.array([w / 2, h / 2])
            return d.points + corner, d.values
        pts, v = source_points(self.id, p.seed, p.source, _key(p))
        return pts * p.scale + np.asarray(p.pos[:2]), v

    def _rect(self, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w, h = p.size
        corner = np.asarray(p.pos, dtype=np.float64) - np.array([w / 2, h / 2, 0.0])
        return corner, np.array([w, 0.0, 0.0]), np.array([0.0, h, 0.0])

    def world_bounds(self, state: SceneState) -> Bounds:
        corner, r, u = self._rect(self.me(state).params)
        return Bounds.of_points([corner, corner + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        pts, _ = self.cloud(state)
        corner, r, u = self._rect(p)
        inside = _inside(pts, corner, r, u)
        out: dict[str, Any] = {"kind": "point_cloud", "backdrop": True, "source": p.source,
                               "points": len(pts), "in_rectangle": int(inside.sum())}  # fmt: skip
        if p.source == "roots":
            out.update(degree=p.degree, coefficients=list(p.coefficients), twist=p.twist)
        elif p.source == "attractor":
            out.update(attractor=p.kind)
        elif p.source == "buddhabrot":
            out.update(max_iter=p.max_iter)
        elif p.source == "bifurcation":
            (r0, r1), (x0, x1) = _window(p)
            out.update(map=p.map, window={"r": [round(r0, 9), round(r1, 9)],
                                          "x": [round(x0, 9), round(x1, 9)]},
                       n_iter=p.n_iter, n_keep=p.n_keep, sweep=round(p.sweep, 6),
                       focus=round(p.focus, 6))  # fmt: skip
        if len(pts):
            out["extent"] = [[round(float(x), 6) for x in pts.min(0)],
                             [round(float(x), 6) for x in pts.max(0)]]  # fmt: skip
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, r, u = self._rect(p)
        lut = np.asarray(me.style["colormap"], dtype=np.float32)
        bg = np.asarray(state.background[:3], dtype=np.float32)
        glow = float(me.style.get("glow", 1.0))

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            if p.source == "bifurcation":
                from phasor.render import gpu

                d = _diagram(self.id, p, state.t, center, height, pw, ph)
                gain = p.exposure / max(d.light, MIN_DENSITY)
                colors = _colors(d.values, p.hue, lut)
                out = gpu.render("splat", pw, ph, points=d.points, colors=colors, center=center,
                                 height=height, sigma=p.point_size, gain=gain,
                                 weights=d.weights, white=p.white)  # fmt: skip
                out[..., :3] += bg * (1.0 - out[..., 3:4])
                return out
            # on the card when CUDA is here (colouring by phase stays on the host: its
            # arctan2 is not the same bits on both)
            dev = None if p.color_by == "phase" else card_points(self.id, p.seed, p.source,
                                                                 _key(p))  # fmt: skip
            if dev is None:
                pts, v = self.cloud(state)
            else:  # as ``cloud`` does on the host
                pts, v = dev[0] * p.scale + _xp_vec(dev[0], p.pos[:2]), dev[1]
            xp = array_module(pts)
            inside = _inside(pts, corner, r, u)
            weights = None
            if p.twinkle > 0 and len(pts):
                group = p.degree if p.source == "roots" else 1  # a polynomial's roots
                freq, phase = _flicker(self.id, p.seed, -(-len(pts) // group))
                w = 1.0 + p.twinkle * np.sin(2 * np.pi * (freq * state.t + phase))  # host sin
                weights = xp.repeat(xp.asarray(w), group)[: len(pts)][inside]
            pts = pts[inside] - _xp_vec(pts, corner[:2])  # quad coordinates, like ``center``
            v = v[inside]
            if p.color_by == "phase":
                d = pts - (np.asarray(p.pos[:2]) - corner[:2])
                v = np.arctan2(d[:, 1], d[:, 0]) / (2 * np.pi)
            return splat_pixels(pts, v, lut, bg, pw, ph, center, height, size=p.size,
                                hue=p.hue, point_size=p.point_size, exposure=p.exposure,
                                white=p.white, weights=weights)  # fmt: skip

        ctx.field(corner, r, u, pixels=pixels, opacity=p.opacity, glow=glow)


def splat_pixels(
    pts: np.ndarray,
    v: np.ndarray,
    lut: np.ndarray,
    bg: np.ndarray,
    pw: int,
    ph: int,
    center: tuple[float, float],
    height: float,
    *,
    size: tuple[float, float],
    hue: float,
    point_size: float,
    exposure: float,
    white: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """The ``ctx.field`` pixels of points of light: ``pts`` (n, 2) in the coordinates of a
    ``size`` rectangle (lower-left corner at 0), already cut to it; ``v`` (n,) their colour
    values, read from ``lut`` turned by ``hue``. Shared by every object that draws a cloud
    (``PointCloud``, ``cx.Fractal``'s Buddhabrot)."""
    from phasor.render import gpu

    colors = _colors(v, hue, lut)  # on the device that holds ``v``
    per_px = len(pts) * (height / ph) ** 2 / (size[0] * size[1])
    # zooming in spreads the points over more pixels; light per point grows only as
    # the square root of that, so a dense region seen close up does not wash out
    whole = len(pts) * (size[1] / ph) ** 2 / (size[0] * size[1])
    base = max(whole, MIN_DENSITY)
    gain = exposure / base * (base / max(per_px, MIN_DENSITY)) ** ZOOM_LIGHT
    out = gpu.render("splat", pw, ph, points=pts, colors=colors, center=center,
                     height=height, sigma=point_size, gain=gain, weights=weights,
                     white=white)  # fmt: skip
    out[..., :3] += bg * (1.0 - out[..., 3:4])  # light on the scene's ground
    return out


def _key(p) -> tuple:
    if p.source == "roots":
        return (p.degree, tuple(p.coefficients), p.count, float(p.twist))
    if p.source == "attractor":
        return (p.kind, tuple(p.constants), p.count)
    if p.source == "buddhabrot":
        return (p.max_iter, p.count)
    return (tuple(p.points[: p.count]), tuple(p.values[: p.count]))


def _inside(pts: np.ndarray, corner, r, u) -> np.ndarray:
    x0, y0 = corner[0], corner[1]
    return ((pts[:, 0] >= x0) & (pts[:, 0] <= x0 + r[0])
            & (pts[:, 1] >= y0) & (pts[:, 1] <= y0 + u[1]))  # fmt: skip


def _xp_vec(like: Any, values) -> Any:
    """``values`` as a float64 array on the device of ``like``."""
    return array_module(like).asarray(np.asarray(values, dtype=np.float64))


def _colors(v: Any, hue: float, lut: np.ndarray) -> Any:
    """(n, 3) colours: the palette ``lut`` at each value, turned by ``hue``, interpolated;
    on the device that holds ``v`` (NumPy or CuPy, the same bits: ``y - floor(y)`` is
    ``y % 1.0`` exactly, and every other step is one correctly rounded operation)."""
    xp = array_module(v)
    lut = xp.asarray(lut)
    y = v + hue
    x = (y - xp.floor(y)) * len(lut)
    k0 = xp.minimum(x.astype(xp.int64), len(lut) - 1)
    t = (x - k0)[:, None].astype(xp.float32)
    return lut[k0] * (1 - t) + lut[(k0 + 1) % len(lut)] * t


# ---------------------------------------------------------------- bifurcation diagrams


def _window(p) -> tuple[tuple[float, float], tuple[float, float]]:
    """The (r, x) window the rectangle shows: ``r_range`` x ``x_range``, zoomed ``focus`` of
    the way to ``r_focus`` x ``x_focus``."""
    return (clm.zoom_window(p.r_range, p.r_focus or p.r_range, p.focus),
            clm.zoom_window(p.x_range, p.x_focus or p.x_range, p.focus))  # fmt: skip


_orbit_lock = threading.Lock()
_orbit_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
_orbit_bytes = [0]


def _orbit_block(map: str, x0: float, n_iter: int, n_keep: int, level: int, blocks):
    """For each block b of lattice ``level`` (columns j = b * CHUNK ... + CHUNK - 1, at
    r = j * LATTICE / 2^level): (iterates (ROWS_MAX * n_keep, CHUNK) from x_(n_iter - n_keep
    + 1) on, the columns' colour values (CHUNK,)). Cached; the missing blocks run together.
    A column's orbit is the same bits in any block at any level (``clm.orbits``)."""
    rows = int(ROWS_MAX * n_keep)
    out, missing = {}, []
    with _orbit_lock:
        for b in blocks:
            key = (map, x0, n_iter, n_keep, level, b)
            if key in _orbit_cache:
                _orbit_cache.move_to_end(key)
                out[b] = _orbit_cache[key]
            else:
                missing.append(b)
    if missing:
        step = LATTICE / 2.0**level
        j = (np.asarray(missing, dtype=np.int64)[:, None] * CHUNK + np.arange(CHUNK)).reshape(-1)
        r = j * step
        xs = clm.orbits(map, r, n_iter + rows - n_keep, x0, start=n_iter - n_keep + 1)
        value = clm.lyapunov_value(clm.lyapunov(map, r, xs[:n_keep]))
        with _orbit_lock:
            for i, b in enumerate(missing):
                cols = slice(i * CHUNK, (i + 1) * CHUNK)
                block = (np.ascontiguousarray(xs[:, cols]), value[cols])
                for a in block:
                    a.flags.writeable = False
                key = (map, x0, n_iter, n_keep, level, b)
                if key not in _orbit_cache:
                    _orbit_cache[key] = block
                    _orbit_bytes[0] += block[0].nbytes
                out[b] = _orbit_cache[key]
            while _orbit_bytes[0] > ORBIT_BYTES and len(_orbit_cache) > len(blocks):
                _, (old, _v) = _orbit_cache.popitem(last=False)
                _orbit_bytes[0] -= old.nbytes
    return [out[b] for b in blocks]


class _Diagram:
    """What one view of a bifurcation diagram splats: points (n, 2) in rectangle
    coordinates, their colour values (n,) and weights (n,), and ``light``, the mean density
    (points per pixel) the brightness is set against."""

    __slots__ = ("light", "points", "values", "weights")

    def __init__(self, points, values, weights, light):
        self.points, self.values, self.weights, self.light = points, values, weights, light


def _diagram(oid: str, p, t: float, center, height: float, pw: int, ph: int) -> _Diagram:
    """The part of the diagram a view sees (the view as ``DrawContext.field`` gives it, in
    rectangle coordinates: the rectangle runs from (0, 0) to ``size``)."""
    from phasor.render.gpu.kernels.splat import splat_reach

    empty = _Diagram(np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=np.float32), 1.0)
    (r0, r1), (xa, xb) = _window(p)
    w, h = p.size
    s = height / ph  # rectangle units per pixel
    pad = (splat_reach(p.point_size) + 1) * s
    q_lo = max(0.0, center[0] - s * pw / 2 - pad)
    q_hi = min(w, center[0] + s * pw / 2 + pad)
    y_lo = max(0.0, center[1] - height / 2 - pad)
    y_hi = min(h, center[1] + height / 2 + pad)
    if q_hi <= q_lo or y_hi <= y_lo:
        return empty
    # the columns: the lattice level whose spacing is COLUMN_PX pixels or less, and the odd
    # columns of the next level faded in by the fraction of the way to it
    px_per_r = w / (r1 - r0) / s
    level_f = max(0.0, math.log2(LATTICE * px_per_r / COLUMN_PX))
    level = int(level_f)
    fade = level_f - level
    fine = level + 1
    step = LATTICE / 2.0**fine
    rl, rh = r0 + q_lo / w * (r1 - r0), r0 + q_hi / w * (r1 - r0)
    j_lo, j_hi = math.ceil(rl / step), math.floor(rh / step)
    if j_hi < j_lo:
        return empty
    blocks = list(range(j_lo // CHUNK, j_hi // CHUNK + 1))
    got = _orbit_block(p.map, float(p.x0), p.n_iter, p.n_keep, fine, blocks)
    first = blocks[0] * CHUNK
    cols = slice(j_lo - first, j_hi - first + 1)
    xs = np.concatenate([g[0] for g in got], axis=1)[:, cols]
    value = np.concatenate([g[1] for g in got])[cols]
    j = np.arange(j_lo, j_hi + 1, dtype=np.int64)
    r = j * step
    col_w = np.where(j % 2 == 0, 1.0, fade)
    # light: columns per pixel times the iterates a column spreads over a pixel of x
    width_px = max((rh - rl) * px_per_r, 1.0)
    x_span = (y_hi - y_lo) / h * (xb - xa)  # x seen on screen
    more = min(max(X_REFERENCE / max(x_span, 1e-300), 1.0), ROWS_MAX)
    shown = p.n_keep * more  # iterates per column, a fraction for the last one
    light = col_w.sum() / width_px * shown * x_span / max(X_REFERENCE * (y_hi - y_lo) / s, 1.0)
    # the sweep draws the window from the left, its front a soft edge
    if p.sweep < 1.0:
        front = r0 + p.sweep * (1.0 + SWEEP_EDGE) * (r1 - r0)
        col_w = col_w * np.clip((front - r) / (SWEEP_EDGE * (r1 - r0)), 0.0, 1.0)
    if p.twinkle > 0:
        freq, phase = _flicker(oid, p.seed, TWINKLE_GROUPS)
        g = _column_group(j, fine)
        col_w = col_w * (1.0 + p.twinkle * np.sin(2 * np.pi * (freq[g] * t + phase[g])))
    n_full = int(shown)
    row_w = np.ones(min(n_full + 1, len(xs)))
    if n_full < len(xs):
        row_w[-1] = shown - n_full
    xs = xs[: len(row_w)]
    weight = row_w[:, None] * col_w[None, :]
    x_lo, x_hi = xa + y_lo / h * (xb - xa), xa + y_hi / h * (xb - xa)
    with np.errstate(invalid="ignore"):
        keep = (xs >= x_lo) & (xs <= x_hi) & (weight > 0)
    rr = np.broadcast_to(r[None, :], xs.shape)[keep]
    pts = np.c_[(rr - r0) / (r1 - r0) * w, (xs[keep] - xa) / (xb - xa) * h]
    vals = np.broadcast_to(value[None, :], xs.shape)[keep]
    return _Diagram(pts, vals, weight[keep].astype(np.float32), light)


def _column_group(j: np.ndarray, level: int) -> np.ndarray:
    """The twinkle group of each column j of a lattice level, from its place in r alone (the
    same r is j at one level and 2 j at the next, and keeps its group)."""
    j = j.copy()
    lv = np.full(len(j), level, dtype=np.int64)
    for _ in range(level):  # strip the factors of two: (odd j, the level it first appears)
        even = (j % 2 == 0) & (lv > 0)
        if not even.any():
            break
        j[even] //= 2
        lv[even] -= 1
    return ((j * 2654435761 + lv * 40503) % TWINKLE_GROUPS).astype(np.int64)
