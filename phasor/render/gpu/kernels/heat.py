"""Heat maps: a grid of values, or a field already computed per pixel, turned into the
``ctx.field`` pixels of a heat map in one pass.

For each pixel: the value (a grid sampled ``nearest``, ``linear`` or Catmull-Rom ``smooth``,
or a pixel of an ``image`` another kernel made, e.g. ``wave_field``), its colour (a colormap
table, the table as a hue wheel, or two colours by sign), the isolines (thin light on the
contour levels, about a pixel wide: the distance to a level over how much the level changes
per pixel, from the neighbouring pixels), the glow weight in alpha, and the edge fade.
``domains/heatmap.py`` describes the picture; a ``Recipe`` says how a type turns a value
into level, colour and weight.

The NumPy functions here (``sample``, ``isoline_weight``, ``shade_values``, ``edge_window``,
``lut_colour``) are the code every heat map was drawn with before the kernel: they are its
reference, unchanged. The kernel works in double and in the reference's order of
operations, with the pixel coordinates computed as ``heat_pixels`` always did, so the two
differ only where the maths library's ``tanh``, ``exp`` and ``pow`` round differently from
NumPy's (the last bit, before the result is stored in float32).

Output: float32 (h, w, 4), linear RGB and the glow weight. Plain arithmetic per pixel, no
atomics: the same bits every run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from phasor.render.gpu.budget import Tile
from phasor.render.gpu.registry import KernelError, kernel

AMBIENT = 0.12  # with a glow weight: the share of colour a quiet (weight 0) pixel keeps

TRANSFORMS = ("none", "tanh", "sqrt", "wave")
LEVELS = ("value", "signed")
COLOURS = ("lut", "wheel", "sign")
HUES = ("level", "inverse", "height")
WEIGHTS = (None, "value", "abs", "centre")
HOWS = ("nearest", "linear", "smooth")


# ---------------------------------------------------------------- the reference maths


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
    rectangle (0..1): isolines, glow weight and edge fade as in ``heatmap.heat_pixels``."""
    ph, pw = val.shape
    out = np.ones((ph, pw, 4), dtype=np.float32)
    if weight is None and not isolines and not fade:
        out[..., :3] = rgb
        return out
    lines = isoline_weight(np.clip((level or (lambda x: x))(val), 0, 1), isolines)
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


# ---------------------------------------------------------------- what a type draws


@dataclass(frozen=True, eq=False)
class Recipe:
    """How a heat-map type turns a value into the signal ``s`` it draws, and ``s`` into
    level, colour and glow weight: the choices the kernel knows, so the whole picture is
    made in one pass.

    ``transform``: ``none`` (s is the value), ``tanh`` (s = tanh(gain * value), -1..1),
    ``sqrt`` (s = sqrt(value clipped to 0..1)) or ``wave`` (an ``image`` of ``wave_field``:
    s = tanh(gain * field), blended by ``share`` with tanh(gain * |A|) squared).
    ``level`` (0..1, for the isolines and the hue): ``value`` (s) or ``signed`` (s runs
    -1..1: 0.5 + 0.5 s, blended by ``share`` with s itself).
    ``colour``: ``lut`` (the table ``lut`` (n, 3) read at s, the nearest entry, or with
    ``blend`` between entries), ``wheel`` (``lut`` as a wheel, ``hue_cycles`` turns over
    ``hue_of``: the ``level``, its ``inverse`` 1 - level, or the pixel's ``height`` up the
    rectangle, turned by ``hue``) or ``sign`` (``positive`` where s >= ``sign_at``, else
    ``negative``; with ``mix``, from the background by \\|s\\|).
    ``weight`` (the glow weight): None, ``value`` (s), ``abs`` (\\|s\\|) or ``centre``
    (\\|2 s - 1\\|, for a value whose rest is 0.5)."""

    transform: str = "none"
    gain: float = 1.0
    share: float = 0.0
    level: str = "value"
    colour: str = "lut"
    lut: Any = None
    blend: bool = False
    hue_cycles: float = 0.0
    hue: float = 0.0
    hue_of: str = "level"
    positive: Any = (1.0, 1.0, 1.0)
    negative: Any = (1.0, 1.0, 1.0)
    sign_at: float = 0.0
    mix: bool = False
    weight: str | None = None

    def __post_init__(self) -> None:
        for name, value, known in (
            ("transform", self.transform, TRANSFORMS), ("level", self.level, LEVELS),
            ("colour", self.colour, COLOURS), ("hue_of", self.hue_of, HUES),
            ("weight", self.weight, WEIGHTS),
        ):  # fmt: skip
            if value not in known:
                raise KernelError(f"heat recipe: {name} must be one of {known}, got {value!r}")
        if self.colour != "sign":
            lut = None if self.lut is None else np.asarray(self.lut)
            need = 2 if self.colour == "lut" and self.blend else 1
            if lut is None or lut.ndim != 2 or lut.shape[1] < 3 or len(lut) < need:
                raise KernelError(f"heat recipe: colour {self.colour!r} needs a (n >= {need}, "
                                  f"3) lut table")  # fmt: skip

    def table(self) -> np.ndarray:
        """The colour table in float64, (n, 3); one unused row for ``sign``."""
        if self.colour == "sign":
            return np.zeros((1, 3))
        return np.ascontiguousarray(np.asarray(self.lut)[:, :3], dtype=np.float64)

    def signal(self, raw: np.ndarray) -> np.ndarray:
        """``s`` from sampled values (ph, pw), or from an image (ph, pw[, channels])."""
        first = raw if raw.ndim == 2 else raw[..., 0]
        if self.transform == "none":
            return first
        if self.transform == "tanh":
            return np.tanh(self.gain * first)
        if self.transform == "sqrt":
            return np.sqrt(np.clip(first, 0, 1))
        field = first.astype(np.float64)
        s = np.tanh(self.gain * field)
        if self.share > 0:  # the intensity goes as |A|^2: (tanh of the envelope) squared
            envelope = np.sqrt(2.0 * np.maximum(raw[..., 1].astype(np.float64), 0.0))  # |A|
            s = (1.0 - self.share) * s + self.share * np.tanh(self.gain * envelope) ** 2
        return s

    def level_of(self, s: np.ndarray) -> np.ndarray:
        if self.level == "value":
            return s
        return (1.0 - self.share) * (0.5 + 0.5 * s) + self.share * s

    def weight_of(self) -> Callable[[np.ndarray], np.ndarray] | None:
        return {None: None, "value": lambda s: s, "abs": np.abs,
                "centre": lambda s: np.abs(2 * s - 1)}[self.weight]  # fmt: skip

    def rgb(self, s: np.ndarray, v: np.ndarray, background) -> np.ndarray:
        """The colour (ph, pw, 3) of ``s`` (ph, pw); ``v`` (ph,): the rows' height 0..1."""
        if self.colour == "lut":
            table = np.asarray(self.lut)[:, :3]
            return lut_colour(table, "linear" if self.blend else "nearest")(s)
        if self.colour == "wheel":
            wheel = lut_colour(np.asarray(self.lut, dtype=np.float64)[:, :3], "wrap")
            if self.hue_of == "height":
                row = wheel(self.hue_cycles * v + self.hue)
                return np.broadcast_to(row[:, None], (*s.shape, 3))
            x = self.level_of(s)
            return wheel(self.hue_cycles * (x if self.hue_of == "level" else 1.0 - x) + self.hue)
        pos, neg = np.asarray(self.positive)[:3], np.asarray(self.negative)[:3]
        tint = np.where((s >= self.sign_at)[..., None], pos, neg)
        if not self.mix:
            return tint
        bg = np.zeros(3) if background is None else np.asarray(background)[:3]
        return bg + (tint - bg) * np.abs(s)[..., None]


def _source(recipe: Recipe, grid, image, origin, tile: Tile) -> None:
    if (grid is None) == (image is None):
        raise KernelError("heat_shade: give a grid or an image, one of them")
    if grid is not None:
        if np.ndim(grid) != 2 or 0 in np.shape(grid):
            raise KernelError(f"heat_shade: grid must be (nx, ny), got shape {np.shape(grid)}")
        if recipe.transform == "wave":
            raise KernelError("heat_shade: the wave transform reads a wave_field image")
        return
    shape = image.shape
    two = len(shape) == 3 and shape[2] >= 2
    if len(shape) not in (2, 3) or (recipe.transform == "wave" and not two):
        raise KernelError(f"heat_shade: image must be (h, w[, channels]), two channels or more "
                          f"for the wave transform; got shape {shape}")  # fmt: skip
    x0, y0, x1, y1 = neighbourhood(tile)
    if not (origin[0] <= x0 and origin[1] <= y0 and x1 <= origin[0] + shape[1]
            and y1 <= origin[1] + shape[0]):  # fmt: skip
        raise KernelError(f"heat_shade: the image (origin {tuple(origin)}, shape {shape}) "
                          f"must cover the tile and the pixels next to it, ({x0}, {y0}) to "
                          f"({x1}, {y1})")  # fmt: skip


def neighbourhood(tile: Tile) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the tile and the pixels next to it that lie in the picture (the
    isolines read each pixel's neighbours)."""
    right, low = tile.x0 + tile.w + 1, tile.y0 + tile.h + 1
    return max(tile.x0 - 1, 0), max(tile.y0 - 1, 0), min(right, tile.full_w), min(low, tile.full_h)


def heat_shade_reference(
    tile: Tile,
    center: tuple[float, float],
    height: float,
    size: tuple[float, float],
    recipe: Recipe,
    grid: np.ndarray | None = None,
    image: Any = None,
    origin: tuple[int, int] = (0, 0),
    cells: bool = False,
    how: str = "linear",
    isolines: int = 0,
    line_glow: float = 0.7,
    background: Any = None,
    ambient: float = AMBIENT,
    fade: float = 0.0,
    contrast: float = 1.0,
) -> np.ndarray:
    _source(recipe, grid, image, origin, tile)
    pw, ph = tile.full_w, tile.full_h
    x0, y0, x1, y1 = neighbourhood(tile)
    sx, sy = size
    half_w = height * pw / ph / 2
    xs = center[0] + (np.arange(x0, x1) + 0.5) / pw * 2 * half_w - half_w
    ys = center[1] + (0.5 - (np.arange(y0, y1) + 0.5) / ph) * height  # row 0 = top
    if grid is not None:
        grid = np.asarray(grid)
        nx, ny = grid.shape
        if cells:
            g, h = xs / sx * nx - 0.5, ys / sy * ny - 0.5
        else:
            g, h = xs / sx * (nx - 1), ys / sy * (ny - 1)
        raw = sample(grid, g, h, how)
    else:
        ox, oy = origin
        raw = np.asarray(image)[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    s = recipe.signal(raw)
    rgb = np.asarray(recipe.rgb(s, ys / sy, background), dtype=np.float64)
    if x1 - x0 < 2 or y1 - y0 < 2:  # a picture one pixel wide has no neighbours: no lines
        isolines = 0
    out = shade_values(s, rgb, xs / sx, ys / sy, size, level=recipe.level_of,
                       isolines=isolines, weight=recipe.weight_of(), line_glow=line_glow,
                       background=background, ambient=ambient, fade=fade,
                       contrast=contrast)  # fmt: skip
    top, left = tile.y0 - y0, tile.x0 - x0
    return np.ascontiguousarray(out[top : top + tile.h, left : left + tile.w])


# ---------------------------------------------------------------- the kernel

# q: gain, share, hue_cycles, hue, sign_at, line_glow, ambient, fade, contrast,
#    positive rgb, negative rgb, background rgb
SOURCE = """
struct HeatSrc {
    int full_w, full_h;
    double cx, cy, height, sx, sy;
    const double* grid; int nx, ny, cells, how;
    const float* image; int img_x0, img_y0, img_w, img_c;
    int transform, level, count;
    double gain, share;
};

static __device__ double heat_x(const HeatSrc& h, int px) {
    double half_w = h.height * h.full_w / h.full_h / 2;
    return h.cx + (px + 0.5) / h.full_w * 2 * half_w - half_w;
}

static __device__ double heat_y(const HeatSrc& h, int py) {
    return h.cy + (0.5 - (py + 0.5) / h.full_h) * h.height;
}

static __device__ double heat_clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static __device__ int heat_clampi(int k, int lo, int hi) {
    return k < lo ? lo : (k > hi ? hi : k);
}

static __device__ void heat_catmull(double t, double* w) {
    double t2 = t * t, t3 = t * t * t;
    w[0] = 0.5 * (-t3 + 2 * t2 - t);
    w[1] = 0.5 * (3 * t3 - 5 * t2 + 2);
    w[2] = 0.5 * (-3 * t3 + 4 * t2 + t);
    w[3] = 0.5 * (t3 - t2);
}

// grid[x, y] at the plane point (x, y) of the rectangle
static __device__ double heat_grid(const HeatSrc& h, double x, double y) {
    int nx = h.nx, ny = h.ny;
    double g, e;
    if (h.cells) { g = x / h.sx * nx - 0.5; e = y / h.sy * ny - 0.5; }
    else { g = x / h.sx * (nx - 1); e = y / h.sy * (ny - 1); }
    g = heat_clip(g, 0, nx - 1);
    e = heat_clip(e, 0, ny - 1);
    if (h.how == 0) {
        int ix = (int)floor(g + 0.5), iy = (int)floor(e + 0.5);
        if (ix > nx - 1) ix = nx - 1;
        if (iy > ny - 1) iy = ny - 1;
        return h.grid[(long long)ix * ny + iy];
    }
    int x0 = (int)floor(g), y0 = (int)floor(e);
    double fx = g - x0, fy = e - y0;
    if (h.how == 1) {
        int x1 = x0 + 1 < nx - 1 ? x0 + 1 : nx - 1, y1 = y0 + 1 < ny - 1 ? y0 + 1 : ny - 1;
        const double* a = h.grid + (long long)x0 * ny;
        const double* b = h.grid + (long long)x1 * ny;
        return a[y0] * (1 - fx) * (1 - fy) + b[y0] * fx * (1 - fy)
             + a[y1] * (1 - fx) * fy + b[y1] * fx * fy;
    }
    double wx[4], wy[4];
    heat_catmull(fx, wx);
    heat_catmull(fy, wy);
    double out = 0.0;
    for (int k = 0; k < 4; k++) {
        const double* col = h.grid + (long long)heat_clampi(x0 + k - 1, 0, nx - 1) * ny;
        for (int m = 0; m < 4; m++)
            out += wy[m] * wx[k] * col[heat_clampi(y0 + m - 1, 0, ny - 1)];
    }
    return out;
}

// the signal s the type draws, at pixel (px, py)
static __device__ double heat_signal(const HeatSrc& h, int px, int py) {
    double v, second = 0.0;
    if (h.nx > 0) {
        v = heat_grid(h, heat_x(h, px), heat_y(h, py));
    } else {
        const float* p = h.image
            + ((long long)(py - h.img_y0) * h.img_w + (px - h.img_x0)) * h.img_c;
        v = (double)p[0];
        if (h.transform == 3) second = (double)p[1];
    }
    if (h.transform == 0) return v;
    if (h.transform == 1) return tanh(h.gain * v);
    if (h.transform == 2) return sqrt(heat_clip(v, 0, 1));
    double s = tanh(h.gain * v);
    if (h.share > 0) {
        double env = tanh(h.gain * sqrt(2.0 * (second > 0.0 ? second : 0.0)));
        s = (1.0 - h.share) * s + h.share * (env * env);
    }
    return s;
}

static __device__ double heat_level(const HeatSrc& h, double s) {
    if (h.level == 0) return s;
    return (1.0 - h.share) * (0.5 + 0.5 * s) + h.share * s;
}

// the level of pixel (px, py) in isoline units: 0..count
static __device__ double heat_steps(const HeatSrc& h, int px, int py) {
    return heat_clip(heat_level(h, heat_signal(h, px, py)), 0, 1) * h.count;
}

static __device__ double heat_ramp(double d, double over) {
    double x = heat_clip(d / over, 0, 1);
    return x * x * (3 - 2 * x);
}

extern "C" __global__ void phasor_heat(
    int n, int tile_x0, int tile_y0, int tile_w, int full_w, int full_h,
    double cx, double cy, double height, double sx, double sy,
    const double* grid, int nx, int ny, int cells, int how,
    const float* image, int img_x0, int img_y0, int img_w, int img_c,
    const double* lut, int nlut, const double* q,
    int transform, int level, int colour, int blend, int hue_of, int mix, int weight,
    int count, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int px = tile_x0 + i % tile_w;
    int py = tile_y0 + i / tile_w;
    double gain = q[0], share = q[1], hue_cycles = q[2], hue = q[3], sign_at = q[4];
    double line_glow = q[5], ambient = q[6], fade = q[7], contrast = q[8];
    const double* pos = q + 9;
    const double* neg = q + 12;
    const double* bg = q + 15;
    HeatSrc h = {full_w, full_h, cx, cy, height, sx, sy, grid, nx, ny, cells, how,
                 image, img_x0, img_y0, img_w, img_c, transform, level, count, gain, share};
    double s = heat_signal(h, px, py);
    double lev = heat_level(h, s);
    double v = heat_y(h, py) / sy;

    double rgb[3];
    if (colour == 0) {
        double x = heat_clip(s, 0, 1) * (nlut - 1);
        if (blend) {
            int k = (int)floor(x);
            if (k > nlut - 2) k = nlut - 2;
            double t = x - k;
            for (int c = 0; c < 3; c++)
                rgb[c] = lut[3 * k + c] * (1 - t) + lut[3 * (k + 1) + c] * t;
        } else {
            int k = heat_clampi((int)x, 0, nlut - 1);
            for (int c = 0; c < 3; c++) rgb[c] = lut[3 * k + c];
        }
    } else if (colour == 1) {
        double a = hue_cycles * (hue_of == 2 ? v : (hue_of == 1 ? 1.0 - lev : lev)) + hue;
        double r = fmod(a, 1.0);  // then as numpy.mod: the sign of the divisor
        if (r != 0.0 && r < 0.0) r += 1.0;
        double x = r * nlut;
        double fl = floor(x);
        int k = ((int)fl) % nlut;
        int k1 = (k + 1) % nlut;
        double t = x - fl;
        for (int c = 0; c < 3; c++) rgb[c] = lut[3 * k + c] * (1 - t) + lut[3 * k1 + c] * t;
    } else {
        const double* tint = s >= sign_at ? pos : neg;
        for (int c = 0; c < 3; c++)
            rgb[c] = mix ? bg[c] + (tint[c] - bg[c]) * fabs(s) : tint[c];
    }

    float* o = out + 4 * (long long)i;
    if (weight == 0 && count == 0 && !(fade != 0.0)) {
        o[0] = (float)rgb[0]; o[1] = (float)rgb[1]; o[2] = (float)rgb[2]; o[3] = 1.0f;
        return;
    }

    double lines = 0.0;
    if (count > 0) {
        double st = heat_clip(lev, 0, 1) * count;
        int xm = px > 0 ? px - 1 : 0, xp = px < full_w - 1 ? px + 1 : full_w - 1;
        int ym = py > 0 ? py - 1 : 0, yp = py < full_h - 1 ? py + 1 : full_h - 1;
        double gx = (heat_steps(h, xp, py) - heat_steps(h, xm, py)) / (double)(xp - xm);
        double gy = (heat_steps(h, px, yp) - heat_steps(h, px, ym)) / (double)(yp - ym);
        double per_px = hypot(gx, gy);
        if (per_px > 1e-9 && st > 0.5 && st < count - 0.5) {
            double d = fabs(st - rint(st)) / per_px;
            lines = exp(-d * d / 0.8);
        }
    }
    double w = 0.0;
    if (weight == 1) w = heat_clip(s, 0, 1);
    else if (weight == 2) w = heat_clip(fabs(s), 0, 1);
    else if (weight == 3) w = heat_clip(fabs(2 * s - 1), 0, 1);
    if (contrast == 2.0) w = w * w;  // as NumPy squares for the power 2
    else if (contrast != 1.0) w = pow(w, contrast);
    double lit = w + line_glow * lines * (1 - w);
    double shade = weight == 0 ? 1.0 : ambient + (1 - ambient) * w;
    double win = 1.0;
    if (fade > 0) {
        double u = heat_x(h, px) / sx;
        double smaller = sx < sy ? sx : sy;
        double over = fade * (smaller > 1e-12 ? smaller : 1e-12);
        double across = heat_ramp((u < 1 - u ? u : 1 - u) * sx, over);
        double up = heat_ramp((v < 1 - v ? v : 1 - v) * sy, over);
        win = up * across;
        lit = lit * win;
    }
    for (int c = 0; c < 3; c++) {
        double col = bg[c] + (rgb[c] - bg[c]) * shade + rgb[c] * (line_glow * lines);
        if (fade > 0) col = bg[c] + (col - bg[c]) * win;
        o[c] = (float)col;
    }
    o[3] = (float)lit;
}
"""


@kernel(
    "heat_shade",
    "(tile, center, height, size, recipe, grid=None, image=None, origin=(0, 0), cells=False, "
    "how='linear', isolines=0, line_glow=0.7, background=None, ambient=0.12, fade=0.0, "
    "contrast=1.0) -> float32[h, w, 4] (linear RGB, glow weight)",
    reference=heat_shade_reference,
)
def heat_shade(
    gpu,
    tile: Tile,
    center: tuple[float, float],
    height: float,
    size: tuple[float, float],
    recipe: Recipe,
    grid: np.ndarray | None = None,
    image: Any = None,
    origin: tuple[int, int] = (0, 0),
    cells: bool = False,
    how: str = "linear",
    isolines: int = 0,
    line_glow: float = 0.7,
    background: Any = None,
    ambient: float = AMBIENT,
    fade: float = 0.0,
    contrast: float = 1.0,
):
    """The heat-map pixels of a ``size`` rectangle whose corner is the plane's origin, seen
    in the view (``center``, ``height``), as ``recipe`` colours them. The values are
    ``grid[x, y]`` over the rectangle (at cell centres with ``cells``, else at nodes from
    edge to edge; sampled ``how``), or the pixels of ``image`` ((h, w[, channels]), on the
    device or the host), whose top-left pixel is pixel ``origin`` of the picture and which
    covers the tile and the pixels next to it (``neighbourhood``)."""
    _source(recipe, grid, image, origin, tile)
    if how not in HOWS:
        raise ValueError(f"how must be nearest, linear or smooth, got {how!r}")
    cp = gpu.cp
    if tile.full_w < 2 or tile.full_h < 2:  # one pixel wide: no neighbours, no lines
        isolines = 0
    if grid is not None:
        nx, ny = np.shape(grid)
        grid_d = cp.asarray(np.ascontiguousarray(grid, dtype=np.float64))
        image_d, shape = cp.zeros((1,), dtype=cp.float32), (1, 1, 1)
    else:
        nx = ny = 0
        grid_d = cp.zeros((1,), dtype=cp.float64)
        image_d = cp.ascontiguousarray(cp.asarray(image), dtype=cp.float32)
        shape = (*image_d.shape, 1)[:3]
    bg = np.zeros(3) if background is None else np.asarray(background, dtype=np.float64)[:3]
    pos = np.asarray(recipe.positive, dtype=np.float64)[:3]
    neg = np.asarray(recipe.negative, dtype=np.float64)[:3]
    look = (line_glow, ambient, fade, contrast)
    q = np.array([recipe.gain, recipe.share, recipe.hue_cycles, recipe.hue, recipe.sign_at,
                  *look, *pos, *neg, *bg], dtype=np.float64)  # fmt: skip
    table = recipe.table()
    compiled = gpu.compile(SOURCE, "phasor_heat")
    out = cp.empty((tile.h, tile.w, 4), dtype=cp.float32)
    args = (
        np.int32(tile.n), np.int32(tile.x0), np.int32(tile.y0), np.int32(tile.w),
        np.int32(tile.full_w), np.int32(tile.full_h),
        np.float64(center[0]), np.float64(center[1]), np.float64(height),
        np.float64(size[0]), np.float64(size[1]),
        grid_d, np.int32(nx), np.int32(ny), np.int32(bool(cells)), np.int32(HOWS.index(how)),
        image_d, np.int32(origin[0]), np.int32(origin[1]), np.int32(shape[1]),
        np.int32(shape[2]),
        cp.asarray(table), np.int32(len(table)), cp.asarray(q),
        np.int32(TRANSFORMS.index(recipe.transform)), np.int32(LEVELS.index(recipe.level)),
        np.int32(COLOURS.index(recipe.colour)), np.int32(bool(recipe.blend)),
        np.int32(HUES.index(recipe.hue_of)), np.int32(bool(recipe.mix)),
        np.int32(WEIGHTS.index(recipe.weight)), np.int32(isolines), out,
    )  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out
