"""Heat maps: a grid of values drawn as colour through ``ctx.field``, shared by every type
that shows one (``pde.Field``, ``pde.Slice``, ``acoustics.PressureSlice``,
``fourier.Spectrogram``).

The plain picture is what those types always drew. The prism look adds,
each optional:

- ``smooth``: Catmull-Rom (bicubic) sampling between grid values instead of straight-line
  blending, so a coarse grid reads as a smooth field;
- ``isolines``: that many contour levels drawn as thin light about a pixel wide at any zoom
  (the distance to a level is measured in pixels through the local gradient);
- a glow weight per pixel (alpha), so with the object's ``glow`` role the strong parts of the
  field shine and haze. See ``DrawContext.field``;
- ``hue_cycles``: colour from the colormap read as a wheel (the prism ``spectrum``), its hue
  running ``hue_cycles`` times over the value range the way it runs along a prism stroke, so
  each isoline has the hue of its level; brightness comes from the glow weight alone;
- ``fade``: the field melts into the background over that share of the panel's shorter side
  at its edges (no hard rectangle), glow weight included, so the haze has no edge either.

Every type that draws a heat map has these parameters (``HeatParams``); their defaults draw
what the plain pictures always drew.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import Field

from phasor.core.objects import ParamsBase

Pixels = Callable[[int, int, tuple[float, float], float], np.ndarray]
AMBIENT = 0.12  # with a glow weight: the share of colour a quiet (weight 0) pixel keeps


class HeatParams(ParamsBase):
    """The heat-map parameters every type that draws one shares (the prism preset sets them)."""

    smooth: bool = False  # bicubic between grid values (the prism look); else as before
    isolines: int = Field(0, ge=0, le=64)  # contour levels drawn as thin light
    # > 0: colour from the colormap as a wheel, this many turns over the value range (prism)
    hue_cycles: float = Field(0.0, ge=0, le=8)
    hue: float = 0.0  # with hue_cycles: turns the colours (share of a turn); life adds a little
    fade: float = Field(0.0, ge=0, le=0.5)  # edges melt into the background over this share
    ambient: float = Field(AMBIENT, ge=0, le=1)  # with glow: the colour a quiet pixel keeps
    # with glow: the weight is raised to this power, so > 1 keeps the light to the strongest
    # parts and lets the rest fall dark (structure glowing out of darkness)
    contrast: float = Field(1.0, ge=0.25, le=8)

    def drawing(self) -> dict:
        """The ``heat_pixels`` keywords these parameters set."""
        return {"isolines": self.isolines, "ambient": self.ambient, "fade": self.fade,
                "contrast": self.contrast}  # fmt: skip


# The prism preset of every heat-map type: smooth, dark where the field is
# quiet and glowing where strong, in hues of the spectrum that run with the value, contour
# lines of light in the hue of their level, edges melting into the background. The prism
# looks repeat these values in their tables.
PRISM: dict = {
    "params": {
        "smooth": True,
        "isolines": 8,
        "hue_cycles": 1.6,
        "fade": 0.1,
        "ambient": 0.0,
        "contrast": 2.0,
    },
    "style": {"glow": "soft", "colormap": "spectrum"},
}


def _catmull_rom(t: np.ndarray) -> list[np.ndarray]:
    t2, t3 = t * t, t * t * t
    return [
        0.5 * (-t3 + 2 * t2 - t),
        0.5 * (3 * t3 - 5 * t2 + 2),
        0.5 * (-3 * t3 + 4 * t2 + t),
        0.5 * (t3 - t2),
    ]


def sample(grid: np.ndarray, g: np.ndarray, h: np.ndarray, how: str) -> np.ndarray:
    """``grid[x, y]`` at fractional grid coordinates ``g`` (x, length pw) and ``h`` (y,
    length ph), clamped to the grid: ``(ph, pw)``. ``how``: ``nearest`` (the cell's value),
    ``linear`` (bilinear) or ``smooth`` (Catmull-Rom bicubic)."""
    nx, ny = grid.shape
    g, h = np.clip(g, 0, nx - 1), np.clip(h, 0, ny - 1)
    if how == "nearest":  # the cell the point is in (a cell spans its centre +- 1/2)
        ix = np.minimum(np.floor(g + 0.5).astype(int), nx - 1)
        iy = np.minimum(np.floor(h + 0.5).astype(int), ny - 1)
        return grid[ix][:, iy].T
    x0, y0 = np.floor(g).astype(int), np.floor(h).astype(int)
    fx, fy = g - x0, h - y0
    if how == "linear":
        x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
        fx, fy = fx[None, :], fy[:, None]
        return (grid[x0][:, y0].T * (1 - fx) * (1 - fy) + grid[x1][:, y0].T * fx * (1 - fy)
                + grid[x0][:, y1].T * (1 - fx) * fy + grid[x1][:, y1].T * fx * fy)  # fmt: skip
    if how != "smooth":
        raise ValueError(f"how must be nearest, linear or smooth, got {how!r}")
    wx, wy = _catmull_rom(fx), _catmull_rom(fy)
    out = np.zeros((len(h), len(g)))
    for k in range(4):
        ix = np.clip(x0 + k - 1, 0, nx - 1)
        col = grid[ix]  # (pw, ny)
        for m in range(4):
            iy = np.clip(y0 + m - 1, 0, ny - 1)
            out += wy[m][:, None] * wx[k][None, :] * col[:, iy].T
    return out


def isoline_weight(level: np.ndarray, count: int) -> np.ndarray:
    """1 on the ``count`` contour levels of ``level`` (values 0..1), falling to 0 about a
    pixel away from them, whatever the zoom: the distance to the nearest level, in level
    units, over how much ``level`` changes per pixel."""
    if count <= 0:
        return np.zeros_like(level)
    s = level * count
    frac = np.abs(s - np.round(s))
    gy, gx = np.gradient(s)
    per_px = np.hypot(gx, gy)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(per_px > 1e-9, frac / per_px, np.inf)
    inside = (s > 0.5) & (s < count - 0.5)  # not the floor or ceiling of the range
    return np.where(inside, np.exp(-d * d / 0.8), 0.0)


def heat_pixels(
    grid: np.ndarray,
    size: tuple[float, float],
    colour: Callable[[np.ndarray], np.ndarray],
    *,
    cells: bool = False,
    how: str = "linear",
    level: Callable[[np.ndarray], np.ndarray] | None = None,
    isolines: int = 0,
    weight: Callable[[np.ndarray], np.ndarray] | None = None,
    line_glow: float = 0.7,
    background: np.ndarray | None = None,
    ambient: float = AMBIENT,
    fade: float = 0.0,
    contrast: float = 1.0,
    colour_at: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Pixels:
    """``ctx.field`` pixels for ``grid[x, y]`` spanning a ``size`` rectangle.

    ``cells``: grid values sit at cell centres (``x / w * nx - 0.5``), else at nodes from edge
    to edge (``x / w * (nx - 1)``). ``colour(value) -> (ph, pw, 3)`` linear RGB, or
    ``colour_at(value, u, v)`` where ``u`` (pw,) and ``v`` (ph,) are the pixels' places across
    and up the rectangle (0..1). ``level`` maps a value to 0..1 for the isolines (default: the
    value). ``weight(value)`` is the glow weight 0..1 (default: none, alpha 1 as before): with
    one, colour scales from ``ambient`` at weight 0 to full at 1, from ``background`` (the
    scene's, so a quiet field melts into it), the weight first raised to ``contrast``; the
    isolines add to the weight and brighten.
    ``fade``: colour and weight fall to the background over that share of the shorter side at
    the rectangle's edges."""
    nx, ny = grid.shape
    sx, sy = size

    def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
        half_w = height * pw / ph / 2
        xs = center[0] + (np.arange(pw) + 0.5) / pw * 2 * half_w - half_w
        ys = center[1] + (0.5 - (np.arange(ph) + 0.5) / ph) * height  # row 0 = top
        if cells:
            g, h = xs / sx * nx - 0.5, ys / sy * ny - 0.5
        else:
            g, h = xs / sx * (nx - 1), ys / sy * (ny - 1)
        val = sample(grid, g, h, how)
        if colour_at is None:
            rgb = np.asarray(colour(val), dtype=np.float64)
        else:
            rgb = np.asarray(colour_at(val, xs / sx, ys / sy), dtype=np.float64)
        return shade_values(val, rgb, xs / sx, ys / sy, size, level=level, isolines=isolines,
                            weight=weight, line_glow=line_glow, background=background,
                            ambient=ambient, fade=fade, contrast=contrast)  # fmt: skip

    return pixels


def shade_values(
    val: np.ndarray,
    rgb: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    size: tuple[float, float],
    *,
    level: Callable[[np.ndarray], np.ndarray] | None = None,
    isolines: int = 0,
    weight: Callable[[np.ndarray], np.ndarray] | None = None,
    line_glow: float = 0.7,
    background: np.ndarray | None = None,
    ambient: float = AMBIENT,
    fade: float = 0.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """The ``ctx.field`` pixels (ph, pw, 4) of values ``val`` (ph, pw) already coloured
    ``rgb`` (ph, pw, 3), at places ``u`` (pw,) across and ``v`` (ph,) up a ``size``
    rectangle (0..1): isolines, glow weight and edge fade as in ``heat_pixels``, which samples
    a grid and calls this; a field computed per pixel (``waves``) calls it directly."""
    ph, pw = val.shape
    out = np.ones((ph, pw, 4), dtype=np.float32)
    lines = isoline_weight(np.clip((level or (lambda x: x))(val), 0, 1), isolines)
    if weight is None and not isolines and not fade:
        out[..., :3] = rgb
        return out
    w = np.zeros_like(val) if weight is None else np.clip(weight(val), 0, 1)
    if contrast != 1:
        w = w**contrast
    lit = w + line_glow * lines * (1 - w)
    # With a weight, quiet parts fall toward black and strong parts keep their colour
    # (the prism look: light out of darkness); the contour lines are light on top.
    bg = np.zeros(3) if background is None else np.asarray(background[:3], dtype=np.float64)
    shade = 1.0 if weight is None else ambient + (1 - ambient) * w
    col = bg + (rgb - bg) * np.asarray(shade)[..., None] + rgb * (line_glow * lines)[..., None]
    if fade > 0:
        win = edge_window(u, v, size, fade)
        col = bg + (col - bg) * win[..., None]
        lit = lit * win
    out[..., :3] = col
    out[..., 3] = lit
    return out


def edge_window(u: np.ndarray, v: np.ndarray, size: tuple[float, float], fade: float):
    """(len(v), len(u)): 1 inside the rectangle, falling smoothly to 0 at its edges over
    ``fade`` of its shorter side. ``u``, ``v``: places across and up it (0..1)."""
    short = max(min(size), 1e-12)

    def ramp_(d: np.ndarray) -> np.ndarray:
        x = np.clip(d / (fade * short), 0, 1)
        return x * x * (3 - 2 * x)

    across = ramp_(np.minimum(u, 1 - u) * size[0])
    up = ramp_(np.minimum(v, 1 - v) * size[1])
    return up[:, None] * across[None, :]


def wheel_colour(colormap, p: HeatParams) -> Callable[[np.ndarray], np.ndarray]:
    """The prism colouring of ``p`` (``hue_cycles`` > 0): a value's place 0..1 (its level) to
    a colour read round ``colormap`` as a wheel, ``hue_cycles`` turns over the range."""
    wheel = lut_colour(np.asarray(colormap, dtype=np.float64)[:, :3], "wrap")
    return lambda x: wheel(p.hue_cycles * x + p.hue)


def lut_colour(lut: np.ndarray, how: str = "nearest") -> Callable[[np.ndarray], np.ndarray]:
    """A colour function from a (n, 3) colormap table for values in 0..1: the table entry
    (``nearest``, as the plain pictures always did), blended between entries (``linear``, no
    visible steps), or the table as a wheel (``wrap``: any value, its fraction read round the
    table and blended, last entry into the first, as hue runs along a prism stroke)."""
    lut = np.asarray(lut)

    def colour(v: np.ndarray) -> np.ndarray:
        if how == "wrap":
            x = np.mod(v, 1.0) * len(lut)
            k = np.floor(x).astype(int) % len(lut)
            t = (x - np.floor(x))[..., None]
            return lut[k] * (1 - t) + lut[(k + 1) % len(lut)] * t
        x = np.clip(v, 0, 1) * (len(lut) - 1)
        if how == "nearest":
            return lut[np.clip(x.astype(int), 0, len(lut) - 1)]
        k = np.minimum(np.floor(x).astype(int), len(lut) - 2)
        t = (x - k)[..., None]
        return lut[k] * (1 - t) + lut[k + 1] * t

    return colour
