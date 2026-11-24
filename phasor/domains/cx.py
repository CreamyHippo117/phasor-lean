"""Complex analysis: pictures of functions of z, drawn by GPU kernels.

    from phasor import Scene, cx

    s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(8, 4.5)))
    s.add(cx.Fractal(id="set", kind="mandelbrot", pos=(-0.6, 0), size=(3.2, 1.8)))

Both are rectangles in the plane z = 0 whose world coordinates are the complex plane
(x = Re, y = Im; contract 4), so the camera zooms into them like into anything else.
To the id and solo passes each is one quad with one id. Their inner structure (zeros and
poles) is reported through ``features()``, projected analytically.
"""

from __future__ import annotations

import functools
from fractions import Fraction
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import cloud
from phasor.domains import fractal_math as fm

F64_BELOW = 1e-4  # view heights under this render domain coloring in float64
SUPERSAMPLE = 2  # kernel pixels per output pixel, per side, for clean contour edges
BUDDHA_MAX_ITER = 5000  # the cloud mode keeps whole orbits in memory: this many steps at most


@functools.cache
def _parse(source: str):
    from phasor.render.gpu.codegen import ComplexFunction

    return ComplexFunction.parse(source, constant=True)  # ``a``: DomainColoring's constant


@functools.lru_cache(maxsize=256)
def _zeros_and_poles(
    source: str, a: complex = 0j, to: str = "", blend: float = 0.0
) -> tuple[tuple[complex, ...], tuple[complex, ...]]:
    """Zeros and poles of a rational function (with the constant ``a`` put in; with ``to``,
    of the blend (1 - blend) f + blend g), with multiplicity; () for anything else."""
    import sympy as sp
    from mpmath.libmp.libhyper import NoConvergence

    from phasor.render.gpu.codegen import A, Z

    expr = _parse(source).expr
    if to.strip() and blend >= 1:
        expr = _parse(to.strip()).expr
    elif to.strip() and blend > 0:
        b = sp.Float(blend)
        expr = (1 - b) * expr + b * _parse(to.strip()).expr
    expr = expr.subs(A, sp.Float(a.real) + sp.I * sp.Float(a.imag))
    num, den = sp.fraction(sp.cancel(sp.together(expr)))
    if not (num.is_polynomial(Z) and den.is_polynomial(Z)):
        return (), ()

    def roots(p):
        poly = sp.Poly(p, Z)
        if poly.degree() < 1:
            return ()
        try:
            return tuple(complex(r) for r in poly.nroots(n=15))
        except NoConvergence:  # a repeated float root (a filter's zeros at -1)
            return tuple(complex(r) for r in np.roots([complex(c) for c in poly.all_coeffs()]))

    return roots(num), roots(den)


def _rect(p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lower-left corner, right edge vector, up edge vector) of a centered rectangle."""
    w, h = p.size
    corner = np.asarray(p.pos) - np.array([w / 2, h / 2, 0.0])
    return corner, np.array([w, 0.0, 0.0]), np.array([0.0, h, 0.0])


class _Plane(ObjectBase):
    """What both kernel-drawn rectangles share. Their bounds are the box of the one ``field``
    quad they draw, derived from the recorded draw (ADR 0005, ``core/record.py``)."""

    def _inside(self, state: SceneState, points: tuple[complex, ...]) -> np.ndarray:
        p = self.me(state).params
        corner, r, u = _rect(p)
        z = p.pos[2]
        pts = [(c.real, c.imag, z) for c in points
               if corner[0] <= c.real <= corner[0] + r[0]
               and corner[1] <= c.imag <= corner[1] + u[1]]  # fmt: skip
        return np.array(pts, dtype=np.float64).reshape(-1, 3)


@register
class DomainColoring(_Plane):
    """A function of z, shown as color: hue is the phase of f(z), and the brightness bands
    step every time |f(z)| doubles. Poles are white.

    With ``radius`` above 0 (the ``prism`` preset) it is the prism look instead: dark, with
    light where the function does something. Each zero and pole glows, out to about
    ``radius`` world units, with the phase wrapped round it as hue; ``ambient`` is the faint
    light everywhere else; the ``glow`` style role sets how bright the light is and how much
    it blooms."""

    type: ClassVar[str] = "domain_coloring"
    Life: ClassVar = frozenset({"hue", "glow"})
    Style: ClassVar = {"colormap": ("colormap", "phase"), "glow": ("glow", "none")}
    # Hue means phase here, so the full wheel stays the default; presets vary the rest.
    Presets: ClassVar = {
        "classic": {"params": {"softness": 0.0, "contour": 0.35}},  # hard contour edges
        "soft_phase": {"params": {"softness": 0.8, "contour": 0.25},
                       "style": {"colormap": "soft_phase"}},  # restrained three-hue wheel
        "vivid": {"params": {"softness": 0.5, "contour": 0.4}},
        # dark ground, glowing zeros and poles, haze (the look of the prism strokes)
        "prism": {"params": {"radius": 0.35, "ambient": 0.06, "contour": 0.0, "lines": 12},
                  "style": {"glow": "strong"}},
    }  # fmt: skip

    class Params(ParamsBase):
        function: str = Field(min_length=1)  # e.g. "1/(z**2 + 1)"; see docs/ai/API.md
        pos: Vec3 = (0.0, 0.0, 0.0)  # center of the rectangle
        size: tuple[float, float] = (8.0, 4.5)  # width and height in world units
        contour: float = Field(0.35, ge=0, le=1)  # how dark the modulus bands get
        softness: float = Field(0.5, ge=0, le=1)  # feathering of the band edges (0: hard)
        hue: float = 0.0  # turns the color wheel (a share of one turn); life adds a little
        radius: float = Field(0.0, ge=0)  # > 0: the prism look, lit this far round zeros/poles
        ambient: float = Field(0.06, ge=0, le=1)  # prism look: light away from zeros and poles
        lines: int = Field(0, ge=0, le=64)  # prism look: glowing rays of phase per turn, + rings
        # the constant ``a`` the function may use, as (re, im): animate it to move zeros/poles
        a: tuple[float, float] = (0.0, 0.0)
        # A second function and how far along to it (0: ``function``, 1: ``to``); between,
        # (1 - blend) f + blend g. Animate ``blend`` to change one function into the other.
        to: str = ""
        blend: float = Field(0.0, ge=0, le=1)
        line_glow: float = Field(0.6, ge=0, le=4)  # prism look: how bright rays and rings are
        light: float = Field(1.0, ge=0, le=4)  # prism look: scales the light the glow role adds

        @field_validator("function", "to")
        @classmethod
        def _check(cls, v: str, info) -> str:
            from phasor.render.gpu.codegen import FunctionError

            if info.field_name == "to" and not v.strip():
                return v  # no second function
            try:
                _parse(v.strip())
            except FunctionError as e:
                raise ValueError(str(e)) from None
            return v

        @field_validator("size")
        @classmethod
        def _positive(cls, v):
            if min(v) <= 0:
                raise ValueError("size must be positive")
            return v

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        zeros, poles = _zeros_and_poles(p.function.strip(), complex(*p.a), p.to.strip(),
                                        float(p.blend))  # fmt: skip
        return {
            "center": np.asarray(p.pos),
            "zeros": self._inside(state, zeros),
            "poles": self._inside(state, poles),
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _parse(p.function.strip())
        feats = self.features(state)
        return {
            "kind": "domain_coloring",
            "backdrop": True,
            "function": f.source,
            "a": [round(float(p.a[0]), 6), round(float(p.a[1]), 6)],
            "latex": f.latex(),
            **(
                {
                    "to": p.to.strip(),
                    "to_latex": _parse(p.to.strip()).latex(),
                    "blend": round(float(p.blend), 6),
                }
                if p.to.strip()
                else {}
            ),
            "zeros": [[round(float(x), 6), round(float(y), 6)] for x, y, _ in feats["zeros"]],
            "poles": [[round(float(x), 6), round(float(y), 6)] for x, y, _ in feats["poles"]],
        }

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
        me = self.me(state)
        p = me.params
        corner, r, u = _rect(p)
        lut = np.asarray(me.style["colormap"], dtype=np.float32)
        lut = np.roll(lut, -round(p.hue * len(lut)), axis=0)
        f = _parse(p.function.strip())
        # A second function only while it shows (blend 0 is the plain kernel, no extra cost)
        blend = {"to": _parse(p.to.strip()), "blend": float(p.blend)} if (
            p.to.strip() and p.blend > 0) else {}  # fmt: skip
        lit = {"line_glow": float(p.line_glow)} if p.line_glow != 0.6 else {}  # 0.6: default

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            from phasor.render import gpu

            view = {"center": (corner[0] + center[0], corner[1] + center[1]), "height": height,
                    "precision": "f64" if height < F64_BELOW else "f32"}  # fmt: skip
            if p.radius > 0:
                return gpu.render("domain_glow", pw, ph, supersample=SUPERSAMPLE, function=f,
                                  lut=lut, radius=p.radius, ambient=p.ambient,
                                  contour=p.contour, lines=p.lines, a=complex(*p.a),
                                  **blend, **lit, **view)  # fmt: skip
            return gpu.render("domain", pw, ph, supersample=SUPERSAMPLE, function=f,
                              contour=p.contour, soft=p.softness, lut=lut, a=complex(*p.a),
                              **blend, **view)  # fmt: skip

        glow = float(me.style.get("glow", 1.0))
        if p.light != 1.0:  # more (or less) of the light the glow role adds
            glow = 1.0 + max(glow - 1.0, 0.0) * p.light
        ctx.field(corner, r, u, pixels=pixels, opacity=p.opacity,
                  glow=glow if p.radius > 0 else 1.0)  # fmt: skip


@functools.lru_cache(maxsize=64)
def _point(text: str) -> fm.Point:
    return fm.parse_point(text)


@functools.lru_cache(maxsize=8)
def _orbit(c: fm.Point, bits: int, max_iter: int) -> np.ndarray:
    """The reference orbit, computed once per (point, precision, steps): a steady dive
    recomputes it only when the precision steps up (every 32 bits, about 10 digits)."""
    orbit = fm.reference_orbit(c, bits, max_iter)
    orbit.setflags(write=False)
    return orbit


STEADY_ROWS = 18  # the coarse sample of the whole rectangle that ``steady`` reads


def _escape_field(p, c: fm.Point, k: float, pw: int, ph: int, center, height) -> np.ndarray:
    """The escape field of a ``pw`` x ``ph`` view of a Fractal whose rectangle-relative
    ``center`` and ``height`` are world units; ``c`` and ``k`` from ``Fractal._view``."""
    from phasor.render import gpu

    w, h = p.size
    off, fh = ((center[0] - w / 2) * k, (center[1] - h / 2) * k), height * k  # from c
    if _precision(p.kind, fh, ph) == "perturbation":
        orbit = _orbit(c, fm.bits_for(fh / ph), p.max_iter)
        return gpu.render("deep", pw, ph, center=off, height=fh, orbit=orbit,
                          max_iter=p.max_iter)  # fmt: skip
    return gpu.render(
        "escape", pw, ph, center=(float(c[0]) + off[0], float(c[1]) + off[1]), height=fh,
        max_iter=p.max_iter, julia=p.c if p.kind == "julia" else None,
    )  # fmt: skip


def _precision(kind: str, height: float, rows: int) -> str:
    """f32, f64 or perturbation for a view ``height`` plane units tall in ``rows`` pixels.
    Julia sets have no perturbation path here; past that they stay float64 (blocky)."""
    from phasor.render.gpu.kernels.fractal import precision_for

    if kind == "mandelbrot" and height / rows < fm.DEEP_BELOW:
        return "perturbation"
    return precision_for(height)


@register
class Fractal(_Plane):
    """The Mandelbrot set, or the Julia set of ``c``, colored by escape time. Points in the
    set take the theme's background color.

    The rectangle shows the plane around ``center`` (exact decimal text, ``"re im"``; empty:
    the rectangle's own position), magnified ``10**zoom`` times. Precision follows the pixel
    size: float32, then float64 under 1e-5, then perturbation
    against an arbitrary-precision reference orbit under 1e-10 per pixel (Mandelbrot only),
    good to a zoom of about 1e280. For a dive, animate ``zoom`` and set ``steady=True``.

    With ``radius`` above 0 (the ``prism`` preset) it is the prism look instead: the set
    dark (the background), the edge of the set glowing out to about ``radius`` world units
    (of the rectangle as drawn, so the glow keeps its size on screen through a dive), and
    ``ambient`` light far from it; the ``glow`` style role sets how bright the light is and
    how much it blooms.

    ``kind="buddhabrot"`` is the cloud mode: not the escape time of each pixel but the orbits
    themselves. ``count`` points of the orbits z -> z^2 + c (from z = 0) of random c that
    escape within ``max_iter`` steps (at most 5000), every point of every such orbit, are
    splatted as light (the ``splat`` kernel, as ``cloud.PointCloud``): where orbits crowd,
    light builds up. Colour is the ``colormap`` at log(escape time) / log(max_iter), turned
    by ``hue`` (short and long orbits apart, the "nebulabrot"); ``exposure``, ``point_size``
    and ``white`` as in ``PointCloud``; the ``glow`` role makes it shine. The orbits come
    from ``rng(id, seed)`` and are kept per (id, seed, max_iter, count), so the same scene
    draws the same cloud; ``center`` and ``zoom`` move the view as for the other kinds.
    """

    type: ClassVar[str] = "fractal"
    Life: ClassVar = frozenset({"hue", "glow"})
    Style: ClassVar = {"colormap": ("colormap", "sequential"), "glow": ("glow", "none")}
    Presets: ClassVar = {
        "classic": {"style": {"colormap": "sequential"}},
        "ember": {"style": {"colormap": "ember"}, "params": {"density": 0.3}},
        "ocean": {"style": {"colormap": "ocean"}, "params": {"density": 0.3}},
        # the set dark, its edge glowing with haze (the look of the prism strokes)
        "prism": {"params": {"radius": 0.01, "ambient": 0.03, "exposure": 0.06, "white": 0.8},
                  "style": {"glow": "soft"}},
    }  # fmt: skip

    class Params(ParamsBase):
        kind: Literal["mandelbrot", "julia", "buddhabrot"] = "mandelbrot"
        c: tuple[float, float] = (-0.8, 0.156)  # the Julia constant (unused for mandelbrot)
        pos: Vec3 = (-0.6, 0.0, 0.0)
        size: tuple[float, float] = (3.2, 1.8)
        max_iter: int = Field(256, ge=1, le=100_000)
        density: float = Field(0.35, gt=0)  # color bands per sqrt(iteration)
        hue: float = 0.0  # shifts colors along the ramp (share of a cycle); life adds a little
        center: str = ""  # the plane point at the middle, exact: "-0.7756837680090537974 0.13"
        zoom: float = Field(0.0, ge=-3, le=280)  # magnification 10**zoom about ``center``
        # Keep the outermost color fixed as counts grow (a dive's colors then stay put while
        # the structure flows); off: colors cycle with the escape count itself.
        steady: bool = False
        radius: float = Field(0.0, ge=0)  # > 0: the prism look, the edge lit this far
        ambient: float = Field(0.05, ge=0, le=1)  # prism look: light far from the edge
        # buddhabrot (the cloud mode): how many orbit points, their seed and their light
        count: int = Field(1_000_000, ge=1, le=20_000_000)
        seed: int = 0
        exposure: float = Field(0.15, gt=0)  # light per point, against the mean density
        point_size: float = Field(0.9, ge=0.4, le=2.5)  # splat radius (sigma), pixels
        white: float = Field(0.0, ge=0, le=1)  # how white the densest parts run

        @field_validator("center")
        @classmethod
        def _point(cls, v: str) -> str:
            if v.strip():
                fm.parse_point(v)  # ValueError names the format
            return v

        @model_validator(mode="after")
        def _orbits(self):
            if self.kind == "buddhabrot" and self.max_iter > BUDDHA_MAX_ITER:
                raise ValueError(f"a buddhabrot takes max_iter up to {BUDDHA_MAX_ITER}")
            return self

    def _view(self, p) -> tuple[fm.Point, float]:
        """(the exact plane point at the rectangle's middle, plane units per world unit)."""
        c = _point(p.center) if p.center.strip() else (Fraction(p.pos[0]), Fraction(p.pos[1]))
        return c, 10.0**-p.zoom

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos)}

    def orbits(self, state: SceneState, card: bool = False) -> tuple[Any, Any]:
        """Buddhabrot: (orbit points (n, 2) in the rectangle's coordinates, lower-left corner
        at 0, only those inside it; value (n,) 0-1, the colour), from the cache. ``card``:
        CuPy arrays placed and cut on the card when CUDA is here (the same bits)."""
        p = self.me(state).params
        key = (p.max_iter, p.count)
        dev = cloud.card_points(self.id, p.seed, "buddhabrot", key) if card else None
        pts, v = dev or cloud.source_points(self.id, p.seed, "buddhabrot", key)
        xp = cloud.array_module(pts)
        c, k = self._view(p)
        w, h = p.size
        q = (pts - xp.asarray([float(c[0]), float(c[1])])) / k + xp.asarray([w / 2, h / 2])
        inside = (q[:, 0] >= 0) & (q[:, 0] <= w) & (q[:, 1] >= 0) & (q[:, 1] <= h)
        return q[inside], v[inside]

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        c, k = self._view(p)
        out: dict[str, Any] = {"kind": "fractal", "backdrop": True, "set": p.kind,
                               "max_iter": p.max_iter}  # fmt: skip
        if p.kind == "buddhabrot":
            out["points"] = len(cloud.source_points(self.id, p.seed, "buddhabrot",
                                                    (p.max_iter, p.count))[0])  # fmt: skip
            out["in_rectangle"] = len(self.orbits(state)[0])
            if p.center.strip() or p.zoom:
                out.update(center=fm.format_point(c), zoom=p.zoom, view_height=p.size[1] * k)
            return out
        if p.kind == "julia":
            out["c"] = list(p.c)
        if p.center.strip() or p.zoom:
            out["center"] = fm.format_point(c)
            out["zoom"] = p.zoom
            out["view_height"] = p.size[1] * k
            # at a 1080-row render of the whole rectangle
            out["precision"] = _precision(p.kind, p.size[1] * k, 1080)
        return out

    def draw(self, ctx: DrawContext, state: SceneState) -> None:
        me = self.me(state)
        p = me.params
        corner, r, u = _rect(p)
        lut = np.asarray(me.style["colormap"], dtype=np.float32)
        inside = tuple(state.background[:3])

        c, k = self._view(p)
        if p.kind == "buddhabrot":  # the cloud mode: orbits splatted as light
            bg = np.asarray(inside, dtype=np.float32)

            def cloud_pixels(pw: int, ph: int, center: tuple[float, float],
                             height: float) -> np.ndarray:  # fmt: skip
                pts, v = self.orbits(state, card=True)
                return cloud.splat_pixels(pts, v, lut, bg, pw, ph, center, height, size=p.size,
                                          hue=p.hue, point_size=p.point_size,
                                          exposure=p.exposure, white=p.white)  # fmt: skip

            ctx.field(corner, r, u, pixels=cloud_pixels, opacity=p.opacity,
                      glow=float(me.style.get("glow", 1.0)))  # fmt: skip
            return

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            from phasor.render import gpu
            from phasor.render.gpu.bridge import to_host

            s = SUPERSAMPLE if p.radius > 0 else 1  # the glowing edge is anti-aliased
            field = _escape_field(p, c, k, pw * s, ph * s, center, height)
            offset = p.hue
            if p.steady:  # hold the color of the rectangle's outermost (lowest) count
                w, h = p.size
                coarse = _escape_field(p, c, k, STEADY_ROWS * 16 // 9, STEADY_ROWS,
                                       (w / 2, h / 2), h)  # fmt: skip
                out_ = coarse[coarse >= 0]
                offset -= p.density * float(np.sqrt(out_.min())) if out_.size else 0.0
            if p.radius <= 0:
                out = gpu.run("palette", gpu.Tile.full(pw, ph), field=field, lut=lut,
                              density=p.density, offset=offset, inside=inside)  # fmt: skip
                return to_host(out)
            out = gpu.run("fractal_glow", gpu.Tile.full(pw * s, ph * s), field=field, lut=lut,
                          density=p.density, offset=offset, inside=inside,
                          radius=p.radius / height * ph * s,  # world units -> pixels
                          ambient=p.ambient)  # fmt: skip
            return to_host(gpu.box_mean(out, s))  # averaged on the card, then sent

        ctx.field(corner, r, u, pixels=pixels, opacity=p.opacity,
                  glow=float(me.style.get("glow", 1.0)) if p.radius > 0 else 1.0)  # fmt: skip


# ---------------------------------------------------------------- Riemann surfaces

_BRANCHES = {  # kind -> (turns of the angle the surface needs, f(r, theta) as a complex number)
    "sqrt": (2, lambda r, th: np.sqrt(r) * np.exp(0.5j * th)),
    "cbrt": (3, lambda r, th: np.cbrt(r) * np.exp(1j * th / 3)),
    "log": (2, lambda r, th: np.log(r) + 1j * th),
}
_CUTS = {"sqrt": "negative real axis", "cbrt": "negative real axis", "log": "every turn"}


@register
class RiemannSurface(ObjectBase):
    """The Riemann surface of a multivalued function over the complex plane: each sheet is
    one branch, and walking once around the branch point carries you onto the next sheet.

    Height is ``Re f(z)`` (or ``Im``) times ``scale``; color is the phase of ``f(z)``, so each
    sheet has its own colors. Centered on ``pos``, out to ``radius``. A lit 3D surface: give
    the scene an orbit camera (``s.use_camera("orbit")``).
    """

    type: ClassVar[str] = "riemann_surface"
    Style: ClassVar = {"colormap": ("colormap", "phase"), "material": ("surface", "solid")}
    Presets: ClassVar = {
        "glass": {"style": {"material": "glass"}},  # sheets that do not hide each other
        "soft_glass": {"style": {"material": "glass", "colormap": "soft_phase"}},
    }

    class Params(ParamsBase):
        kind: Literal["sqrt", "cbrt", "log"] = "sqrt"
        pos: Vec3 = (0.0, 0.0, 0.0)  # the branch point
        radius: float = Field(2.0, gt=0)
        height: Literal["re", "im"] = "re"
        scale: float = Field(1.0, gt=0)
        rings: int = Field(28, ge=2, le=400)  # mesh resolution along the radius
        segments: int = Field(90, ge=8, le=2000)  # per turn around the branch point

    def _grid(self, state: SceneState):
        """(points (R, T, 3), r (R,), theta (T,), f (R, T) complex)."""
        p = self.me(state).params
        turns, fn = _BRANCHES[p.kind]
        r = np.linspace(p.radius * 0.01, p.radius, p.rings + 1)
        th = np.linspace(0.0, 2 * np.pi * turns, p.segments * turns + 1)
        if p.kind == "log":
            th -= 2 * np.pi * turns / 2
        rr, tt = np.meshgrid(r, th, indexing="ij")
        f = fn(rr, tt)
        h = (f.real if p.height == "re" else f.imag) * p.scale
        x0, y0, z0 = p.pos
        pts = np.stack([x0 + rr * np.cos(tt), y0 + rr * np.sin(tt), z0 + h], axis=-1)
        return pts, r, th, f

    def world_bounds(self, state: SceneState) -> Bounds:
        pts = self._grid(state)[0].reshape(-1, 3)
        return Bounds.of_points(pts)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        """``branch_point``, and ``sheets``: one point on each sheet, above the point
        ``0.75 * radius * i`` of the plane (where the sheets are furthest apart)."""
        p = self.me(state).params
        turns, fn = _BRANCHES[p.kind]
        r = 0.75 * p.radius
        out = []
        for k in range(turns):
            th = np.pi / 2 + 2 * np.pi * k - (2 * np.pi * turns / 2 if p.kind == "log" else 0)
            f = fn(r, th)
            h = (f.real if p.height == "re" else f.imag) * p.scale
            out.append((p.pos[0] + r * np.cos(th), p.pos[1] + r * np.sin(th), p.pos[2] + h))
        return {"branch_point": np.asarray(p.pos, dtype=np.float64), "sheets": np.array(out)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        turns, _ = _BRANCHES[p.kind]
        return {
            "kind": "riemann_surface",
            "function": {"sqrt": "sqrt(z)", "cbrt": "z**(1/3)", "log": "log(z)"}[p.kind],
            "sheets": turns,
            "branch_points": [list(p.pos)],
            "branch_cut": _CUTS[p.kind],
            "height": f"{'Re' if p.height == 're' else 'Im'} f(z)",
            "material": self.me(state).roles["material"],
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        pts, _r, th, f = self._grid(state)
        nr, nt = pts.shape[:2]
        # Normals from the grid's tangents (one-sided at the edges).
        du = np.gradient(pts, axis=0)
        dv = np.gradient(pts, axis=1)
        normals = np.cross(du, dv)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        phase = (np.angle(f) % (2 * np.pi)) / (2 * np.pi)
        rgb = lut[np.minimum((phase * len(lut)).astype(int), len(lut) - 1)]
        colors = np.concatenate([rgb, np.ones((*rgb.shape[:-1], 1))], axis=-1)
        uv = np.stack(np.meshgrid(np.linspace(0, 1, nr), th / (2 * np.pi), indexing="ij"),
                      axis=-1)[..., ::-1]  # fmt: skip
        k = np.arange(nr - 1)[:, None] * nt + np.arange(nt - 1)[None, :]
        k = k.ravel()
        tris = np.vstack([np.c_[k, k + nt, k + 1], np.c_[k + 1, k + nt, k + nt + 1]])
        ctx.surface(pts.reshape(-1, 3), tris, normals=normals.reshape(-1, 3),
                    colors=colors.reshape(-1, 4), uv=uv.reshape(-1, 2),
                    material=me.roles["material"],
                    opacity=me.params.opacity * me.style["material"])  # fmt: skip


# ---------------------------------------------------------------- ComplexMap


@register
class ComplexMap(ObjectBase):
    """A grid in the complex plane pushed through ``z -> f(z)``: the picture of what a
    function does to the plane. ``amount`` slides from the plain grid (0) to its image (1)
    along ``(1 - amount) z + amount f(z)``: animate it (``morph("map.amount", 1)``) to
    watch the plane bend. The grid covers ``extent`` around the complex point ``center``,
    and is drawn with ``center`` at ``pos``: a point w of the image sits at
    ``pos + (w - center)`` (so with both at 0, world coordinates are the complex plane, as
    for ``DomainColoring``). Lines break at poles."""

    type: ClassVar[str] = "complex_map"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),  # the vertical lines (Re z fixed)
        "cross": ("emphasis", "primary"),  # the horizontal lines (Im z fixed)
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {
        "glowing": {"style": {"glow": "soft"}},
        "fine": {"params": {"lines": (21, 21)}, "style": {"stroke": "hairline"}},
    }

    class Params(ParamsBase):
        function: str = Field("z**2", min_length=1)
        pos: Vec3 = (0.0, 0.0, 0.0)  # where the grid's centre is drawn
        center: tuple[float, float] = (0.0, 0.0)  # the complex point at the grid's centre
        extent: tuple[float, float] = (4.0, 4.0)
        lines: tuple[int, int] = (11, 11)  # vertical, horizontal
        samples: int = Field(120, ge=8, le=4000)  # points per line
        amount: float = Field(1.0, ge=0, le=1)  # 0: the plain grid; 1: its image under f
        limit: float = Field(12.0, gt=0)  # lines are cut where |f| grows past this

        @field_validator("function")
        @classmethod
        def _check(cls, v: str) -> str:
            from phasor.render.gpu.codegen import A, FunctionError

            try:
                f = _parse(v.strip())
            except FunctionError as e:
                raise ValueError(str(e)) from None
            if f.expr.has(A):
                raise ValueError("the constant a is for cx.DomainColoring only")
            return v

        @field_validator("lines")
        @classmethod
        def _lines(cls, v):
            if min(v) < 0 or max(v) > 200 or sum(v) == 0:
                raise ValueError("lines is (vertical, horizontal), each 0 to 200, not both 0")
            return v

    def _runs(self, p) -> tuple[tuple[int, np.ndarray], ...]:
        """((family 0 vertical / 1 horizontal, (K, 3) world points), ...), read-only."""
        return _map_runs(p)

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        runs = self._runs(p)
        if not runs:
            return Bounds.of_points([p.pos])
        return Bounds.of_points(np.vstack([r for _, r in runs]), pad=0.02)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _parse(p.function.strip())
        return {"kind": "complex_map", "function": f.source, "latex": f.latex(),
                "amount": round(p.amount, 6), "lines": int(sum(p.lines)),
                "pieces": len(self._runs(p))}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        from phasor.domains.strokes import polyline, with_alpha

        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        cols = (with_alpha(me.style["color"], p.opacity), with_alpha(me.style["cross"], p.opacity))
        for fam, pts in self._runs(p):
            polyline(ctx, cam, pts, me.style["stroke"] * cam.units_per_height(pts),
                     color=cols[fam], glow=me.style["glow"])  # fmt: skip


@functools.lru_cache(maxsize=32)
def _map_runs(p) -> tuple[tuple[int, np.ndarray], ...]:
    """The deformed grid of a ComplexMap's parameters, kept: bounds, semantics and drawing
    all ask for it at every moment, and building it means evaluating f on every line."""
    from phasor.domains import cx_math as xm

    f = _parse(p.function.strip()).numpy()
    z0 = complex(*p.center)
    shift = complex(p.pos[0], p.pos[1]) - z0
    out = []
    for k, line in enumerate(xm.grid_lines(z0, p.extent, p.lines, p.samples)):
        fam = 0 if k < p.lines[0] else 1
        w = xm.deform(f, line, p.amount)
        for run in xm.split_runs(w, p.limit):
            run = run + shift
            pts = np.stack([run.real, run.imag, np.full(len(run), p.pos[2])], 1)
            pts.setflags(write=False)
            out.append((fam, pts))
    return tuple(out)
