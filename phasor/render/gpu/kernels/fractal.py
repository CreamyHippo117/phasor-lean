"""Escape-time fractals: Mandelbrot and Julia sets of z^2 + c.

Output is the smooth (fractional) escape count per pixel, float32, and -1 for points that
never escape. Coloring is a later step, so one field can be re-colored for free.
Precision policy: float32 by default, float64 once the view is under about 1e-5.
"""

from __future__ import annotations

import numpy as np

from phasor.render.gpu.budget import Tile
from phasor.render.gpu.registry import VIEW_BODY, VIEW_PARAMS, kernel, plane, view_args

BAILOUT = 65536.0  # |z|^2 at escape; a large radius makes the smooth count accurate
F64_BELOW = 1e-5  # view heights under this need float64


def precision_for(height: float) -> str:
    return "f64" if height < F64_BELOW else "f32"


SOURCE = (
    """
extern "C" __global__ void phasor_escape("""
    + VIEW_PARAMS
    + """, int julia, double jx, double jy, int max_iter, float* out)
{"""
    + VIEW_BODY
    + """
    real zr, zi, cr, ci;
    if (julia) { zr = x; zi = y; cr = (real)jx; ci = (real)jy; }
    else       { zr = R(0); zi = R(0); cr = x; ci = y; }
    real r2 = zr * zr + zi * zi;
    int k = 0;
    while (k < max_iter && r2 <= R(65536.0)) {
        real t = zr * zr - zi * zi + cr;
        zi = R(2) * zr * zi + ci;
        zr = t;
        r2 = zr * zr + zi * zi;
        k++;
    }
    if (r2 <= R(65536.0)) { out[i] = -1.0f; return; }
    // smooth count: k + 1 - log2(log|z|), with log|z| = log(r2) / 2
    out[i] = (float)((double)k + 1.0 - log2(0.5 * log((double)r2)));
}
"""
)


def escape_reference(
    tile: Tile,
    center: tuple[float, float],
    height: float,
    max_iter: int = 256,
    julia: tuple[float, float] | None = None,
    precision: str | None = None,
) -> np.ndarray:
    real = np.float64 if (precision or precision_for(height)) == "f64" else np.float32
    x, y = (a.astype(real) for a in plane(tile, center, height))
    if julia is None:
        zr, zi, cr, ci = np.zeros_like(x), np.zeros_like(y), x, y
    else:
        zr, zi = x.copy(), y.copy()
        cr, ci = np.full_like(x, julia[0]), np.full_like(y, julia[1])
    two, bail = real(2), real(BAILOUT)
    # Iterate only the points that have not escaped yet; escaped ones keep their last z.
    zr, zi, cr, ci = (a.ravel().copy() for a in (zr, zi, cr, ci))
    count = np.zeros(zr.shape, dtype=np.int32)
    r2 = zr * zr + zi * zi
    live = np.flatnonzero(r2 <= bail)
    for _ in range(max_iter):
        if live.size == 0:
            break
        a, b = zr[live], zi[live]
        t = a * a - b * b + cr[live]
        b = two * a * b + ci[live]
        zr[live], zi[live] = t, b
        r2[live] = t * t + b * b
        count[live] += 1
        live = live[r2[live] <= bail]
    r2 = r2.reshape(x.shape)
    count = count.reshape(x.shape)
    escaped = r2 > bail
    with np.errstate(all="ignore"):
        smooth = count + 1.0 - np.log2(0.5 * np.log(r2.astype(np.float64)))
    return np.where(escaped, smooth, -1.0).astype(np.float32)


@kernel(
    "escape",
    "(tile, center, height, max_iter=256, julia=None, precision=None) -> float32[h, w]",
    reference=escape_reference,
)
def escape(
    gpu,
    tile: Tile,
    center: tuple[float, float],
    height: float,
    max_iter: int = 256,
    julia: tuple[float, float] | None = None,
    precision: str | None = None,
):
    """Mandelbrot set (``julia=None``) or the Julia set of ``c = julia``."""
    compiled = gpu.compile(SOURCE, "phasor_escape", precision or precision_for(height))
    out = gpu.cp.empty((tile.h, tile.w), dtype=gpu.cp.float32)
    jx, jy = julia if julia is not None else (0.0, 0.0)
    args = (*view_args(tile, center, height), np.int32(julia is not None),
            np.float64(jx), np.float64(jy), np.int32(max_iter), out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out


DEEP = (
    """
extern "C" __global__ void phasor_deep("""
    + VIEW_PARAMS
    + """, const double* orbit, int orbit_n, int max_iter, int rebase, float* out)
{"""
    + VIEW_BODY
    + """
    // x, y are this pixel's offset dc from the reference point; z = Z_m + dz.
    double dcr = (double)x, dci = (double)y, dzr = 0.0, dzi = 0.0, r2 = 0.0;
    int m = 0, k = 0;
    while (k < max_iter) {
        double ar = 2.0 * orbit[2 * m] + dzr, ai = 2.0 * orbit[2 * m + 1] + dzi;
        double nr = ar * dzr - ai * dzi + dcr;
        double ni = ar * dzi + ai * dzr + dci;
        dzr = nr; dzi = ni; m++; k++;
        double zr = orbit[2 * m] + dzr, zi = orbit[2 * m + 1] + dzi;
        r2 = zr * zr + zi * zi;
        if (r2 > 65536.0) break;
        // Rebase onto the start of the reference when z is nearer 0 than dz (the reference
        // has stopped describing this pixel: a glitch) or when the reference has run out.
        if (m == orbit_n - 1 || (rebase && r2 < dzr * dzr + dzi * dzi)) {
            dzr = zr; dzi = zi; m = 0;
        }
    }
    if (r2 <= 65536.0) { out[i] = -1.0f; return; }
    out[i] = (float)((double)k + 1.0 - log2(0.5 * log(r2)));
}
"""
)


def _orbit(orbit) -> np.ndarray:
    o = np.ascontiguousarray(orbit, dtype=np.float64)
    if o.ndim != 2 or o.shape[1] != 2 or len(o) < 2 or o[0, 0] != 0 or o[0, 1] != 0:
        raise ValueError("orbit must be (L + 1, 2) float64 starting at Z_0 = 0, with L >= 1")
    return o


def deep_reference(
    tile: Tile,
    center: tuple[float, float],
    height: float,
    orbit: np.ndarray,
    max_iter: int = 1000,
    rebase: bool = True,
) -> np.ndarray:
    o = _orbit(orbit)
    x, y = plane(tile, center, height)
    dcr, dci = x.ravel().copy(), y.ravel().copy()
    n = dcr.size
    dzr, dzi, r2 = np.zeros(n), np.zeros(n), np.zeros(n)
    m = np.zeros(n, dtype=np.int64)
    k = np.zeros(n, dtype=np.int64)
    live = np.arange(n)
    last = len(o) - 1
    zr_all, zi_all = o[:, 0], o[:, 1]
    for _ in range(max_iter):
        if live.size == 0:
            break
        ml = m[live]
        a, b = dzr[live], dzi[live]
        ar, ai = 2.0 * zr_all[ml] + a, 2.0 * zi_all[ml] + b
        nr = ar * a - ai * b + dcr[live]
        ni = ar * b + ai * a + dci[live]
        ml = ml + 1
        zr, zi = zr_all[ml] + nr, zi_all[ml] + ni
        rr = zr * zr + zi * zi
        k[live] += 1
        r2[live] = rr
        go = rr <= BAILOUT
        back = (ml == last) | (rebase & (rr < nr * nr + ni * ni))
        nr, ni = np.where(back, zr, nr), np.where(back, zi, ni)
        dzr[live], dzi[live], m[live] = nr, ni, np.where(back, 0, ml)
        live = live[go]
    with np.errstate(all="ignore"):
        smooth = k + 1.0 - np.log2(0.5 * np.log(r2))
    return np.where(r2 > BAILOUT, smooth, -1.0).astype(np.float32).reshape(tile.h, tile.w)


@kernel(
    "deep",
    "(tile, center, height, orbit, max_iter=1000, rebase=True) -> float32[h, w]",
    reference=deep_reference,
)
def deep(gpu, tile: Tile, center, height, orbit, max_iter=1000, rebase=True):
    """Perturbation: the escape field of the Mandelbrot set
    around a reference point whose orbit ``Z_0..Z_L`` (float64, from
    ``domains/fractal_math.reference_orbit``) is given. ``center`` and ``height`` are offsets
    from that point, so a 1e-30 view is ordinary float64. Same output as ``escape``."""
    o = _orbit(orbit)
    compiled = gpu.compile(DEEP, "phasor_deep", "f64")
    out = gpu.cp.empty((tile.h, tile.w), dtype=gpu.cp.float32)
    args = (*view_args(tile, center, height), gpu.cp.asarray(o.ravel()), np.int32(len(o)),
            np.int32(max_iter), np.int32(bool(rebase)), out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out


PALETTE = """
extern "C" __global__ void phasor_palette(int n, const float* field, const float* lut, int lut_n,
                                          float density, float offset, const float* inside,
                                          float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* o = out + 4 * (long long)i;
    float f = field[i];
    if (f < 0.0f) { o[0] = inside[0]; o[1] = inside[1]; o[2] = inside[2]; o[3] = 1.0f; return; }
    float t = sqrtf(f) * density + offset;
    t = t - floorf(t);
    float g = t * (float)lut_n;
    int k0 = (int)g;
    if (k0 >= lut_n) k0 = lut_n - 1;
    int k1 = (k0 + 1) % lut_n;
    float u = g - (float)k0;
    for (int c = 0; c < 3; c++) o[c] = lut[3 * k0 + c] * (1.0f - u) + lut[3 * k1 + c] * u;
    o[3] = 1.0f;
}
"""


def _palette_args(field, lut, inside):
    if field.ndim != 2 or str(field.dtype) != "float32":
        raise ValueError("palette needs a float32 (h, w) field from the escape kernel")
    table = np.ascontiguousarray(lut, dtype=np.float32)
    if table.ndim != 2 or table.shape[1] != 3 or len(table) < 2:
        raise ValueError("lut must be an (n, 3) table of linear RGB with n >= 2")
    return table, np.asarray(inside, dtype=np.float32)[:3]


def palette_reference(
    tile: Tile,
    field: np.ndarray,
    lut: np.ndarray,
    density: float = 0.35,
    offset: float = 0.0,
    inside: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    table, inside_rgb = _palette_args(field, lut, inside)
    t = np.sqrt(np.maximum(field, 0)).astype(np.float32) * np.float32(density) + np.float32(offset)
    t = (t - np.floor(t)).astype(np.float32)
    g = t * np.float32(len(table))
    k0 = np.minimum(g.astype(np.int64), len(table) - 1)
    u = (g - k0.astype(np.float32))[..., None]
    rgb = table[k0] * (np.float32(1) - u) + table[(k0 + 1) % len(table)] * u
    out = np.ones((*field.shape, 4), dtype=np.float32)
    out[..., :3] = np.where((field < 0)[..., None], inside_rgb, rgb)
    return out


@kernel(
    "palette",
    "(tile, field, lut, density=0.35, offset=0.0, inside=(0,0,0)) -> float32[h, w, 4]",
    reference=palette_reference,
)
def palette(gpu, tile: Tile, field, lut, density=0.35, offset=0.0, inside=(0.0, 0.0, 0.0)):
    """Color an escape field: a cyclic table indexed by sqrt(count), so bands stay even as
    counts grow; points in the set get ``inside``. ``tile`` is the field's own extent."""
    table, inside_rgb = _palette_args(field, lut, inside)
    src = gpu.cp.ascontiguousarray(gpu.cp.asarray(field))
    out = gpu.cp.empty((*src.shape, 4), dtype=gpu.cp.float32)
    compiled = gpu.compile(PALETTE, "phasor_palette")
    args = (np.int32(src.size), src, gpu.cp.asarray(table), np.int32(len(table)),
            np.float32(density), np.float32(offset), gpu.cp.asarray(inside_rgb), out)  # fmt: skip
    gpu.launch(compiled, int(src.size), args)
    return out


# ---------------------------------------------------------------- the prism look
#
# Light along the edge of the set. The smooth count is nu = 1 - log2(G)
# plus a constant, with G = log|z_n| / 2^n the potential, and the distance to the set is about
# G / |grad G| (exact for the unit circle, the Julia set of c = 0). So the distance in pixels is
# d = 1 / (ln 2 |grad nu|), with grad nu in counts per pixel: no derivative carried through the
# iteration, and the same for the ``escape`` and ``deep`` fields. Weight w = 1 / (1 + (d / r)^2),
# 1 on the edge and a long soft tail (the haze); a pixel next to the set has w = 1. Colour is the
# palette ramp, as in ``palette``, at ambient + (1 - ambient) w; alpha carries w, the glow weight
# the renderer turns into light and bloom. Points in the set take ``inside`` and alpha 0.

GLOW = """
extern "C" __global__ void phasor_fractal_glow(int n, int w, int h, const float* field,
                                               const float* lut, int lut_n, float density,
                                               float offset, const float* inside, float radius,
                                               float ambient, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* o = out + 4 * (long long)i;
    float f = field[i];
    if (f < 0.0f) { o[0] = inside[0]; o[1] = inside[1]; o[2] = inside[2]; o[3] = 0.0f; return; }
    int x = i % w, y = i / w;
    int l = x > 0 ? i - 1 : i, r = x < w - 1 ? i + 1 : i;
    int u = y > 0 ? i - w : i, d = y < h - 1 ? i + w : i;
    float fl = field[l], fr = field[r], fu = field[u], fd = field[d];
    float wt = 1.0f;
    if (fl >= 0.0f && fr >= 0.0f && fu >= 0.0f && fd >= 0.0f) {
        int sx = (x > 0) + (x < w - 1), sy = (y > 0) + (y < h - 1);
        float gx = sx > 0 ? (fr - fl) / (float)sx : 0.0f;
        float gy = sy > 0 ? (fd - fu) / (float)sy : 0.0f;
        float g = sqrtf(gx * gx + gy * gy);
        wt = 0.0f;
        if (g > 0.0f) {
            float q = 1.0f / (0.6931472f * g) / radius;
            wt = 1.0f / (1.0f + q * q);
        }
    }
    float t = sqrtf(f) * density + offset;
    t = t - floorf(t);
    float s = t * (float)lut_n;
    int k0 = (int)s;
    if (k0 >= lut_n) k0 = lut_n - 1;
    int k1 = (k0 + 1) % lut_n;
    float v = s - (float)k0;
    float shade = ambient + (1.0f - ambient) * wt;
    for (int c = 0; c < 3; c++) o[c] = shade * (lut[3 * k0 + c] * (1.0f - v) + lut[3 * k1 + c] * v);
    o[3] = wt;
}
"""


def fractal_glow_reference(
    tile: Tile,
    field: np.ndarray,
    lut: np.ndarray,
    density: float = 0.35,
    offset: float = 0.0,
    inside: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 2.0,
    ambient: float = 0.08,
) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radius must be positive")
    f32 = np.float32
    f = np.asarray(field, dtype=np.float32)
    h, w = f.shape
    # neighbours, clamped at the picture's edges (a one-sided difference there)
    fl, fr = f[:, np.r_[0, 0 : w - 1]], f[:, np.r_[1:w, w - 1]]
    fu, fd = f[np.r_[0, 0 : h - 1]], f[np.r_[1:h, h - 1]]
    xs, ys = np.arange(w), np.arange(h)
    sx = ((xs > 0).astype(np.int32) + (xs < w - 1))[None, :].astype(np.float32)
    sy = ((ys > 0).astype(np.int32) + (ys < h - 1))[:, None].astype(np.float32)
    with np.errstate(all="ignore"):
        gx = np.where(sx > 0, (fr - fl) / sx, f32(0))
        gy = np.where(sy > 0, (fd - fu) / sy, f32(0))
        g = np.sqrt(gx * gx + gy * gy)
        q = f32(1) / (f32(0.6931472) * g) / f32(radius)
        wt = np.where(g > 0, f32(1) / (f32(1) + q * q), f32(0)).astype(np.float32)
    edge = (fl < 0) | (fr < 0) | (fu < 0) | (fd < 0)
    wt = np.where(edge, f32(1), wt)
    out = palette_reference(tile, f, lut, density, offset, inside)
    shade = (f32(ambient) + (f32(1) - f32(ambient)) * wt)[..., None]
    inside_ = (f < 0)[..., None]
    out[..., :3] = np.where(inside_, out[..., :3], shade * out[..., :3])
    out[..., 3] = np.where(f < 0, f32(0), wt)
    return out


@kernel(
    "fractal_glow",
    "(tile, field, lut, density=0.35, offset=0.0, inside=(0,0,0), radius=2.0, ambient=0.08) "
    "-> float32[h, w, 4] linear RGB, alpha = glow",
    reference=fractal_glow_reference,
)
def fractal_glow(gpu, tile: Tile, field, lut, density=0.35, offset=0.0, inside=(0.0, 0.0, 0.0),
                 radius=2.0, ambient=0.08):  # fmt: skip
    """The prism look of an escape field (from ``escape`` or ``deep``): the palette ramp as
    in ``palette``, lit along the edge of the set out to about ``radius`` pixels and
    ``ambient`` far from it; points in the set get ``inside``; alpha is the glow weight."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    table, inside_rgb = _palette_args(field, lut, inside)
    src = gpu.cp.ascontiguousarray(gpu.cp.asarray(field))
    h, w = src.shape
    out = gpu.cp.empty((h, w, 4), dtype=gpu.cp.float32)
    compiled = gpu.compile(GLOW, "phasor_fractal_glow")
    args = (np.int32(src.size), np.int32(w), np.int32(h), src, gpu.cp.asarray(table),
            np.int32(len(table)), np.float32(density), np.float32(offset),
            gpu.cp.asarray(inside_rgb), np.float32(radius), np.float32(ambient), out)  # fmt: skip
    gpu.launch(compiled, int(src.size), args)
    return out
