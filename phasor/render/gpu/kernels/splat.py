"""Point splatting: many points, each a tiny Gaussian of light, added together.

Every point adds a normalised Gaussian footprint (``sigma`` pixels, cut off ``reach`` pixels
from its nearest pixel) into an accumulation image with integer ``atomicAdd`` (fixed point, so the
sum is the same bits whatever order the threads add in): its colour times its
weight into rgb, its weight into a density channel. Dense regions build up light; a point's
footprint always sums to its weight (light is conserved). A second pass per pixel applies
the tone curve ``a = 1 - exp(-gain * density)``, so dense areas saturate softly instead of
clipping: rgb becomes the mean colour of the points there times ``a``, and alpha = ``a`` is
the glow weight (``DrawContext.field``). With ``white`` above 0 that colour turns toward
white by ``white * a^4``, so only the densest parts run white-hot; rgb stays at most ``a``
(for colours at most 1). With ``curve=False`` the raw sums are returned.

Points are in the view's plane coordinates, like the pixels of the other kernels (the view
is ``center`` and ``height``). A tile skips points whose footprint misses it, so tiles
stitch exactly. Output: float32 (h, w, 4) on a black ground.
"""

from __future__ import annotations

import math

import numpy as np

from phasor.render.gpu.budget import Tile
from phasor.render.gpu.registry import KernelError, kernel

MAX_REACH = 7  # footprint half-width the kernel's fixed arrays allow
MAX_SIGMA = MAX_REACH / 2.5
# Light is summed in 64-bit fixed point (units of 2^-32): integer atomic adds give the same
# bits whatever order the threads run in (float atomics do not). A pixel holds up to 2^31
# units of light, two billion points at full weight, and resolves 2.3e-10.
FIXED = 2.0**32


def splat_reach(sigma: float) -> int:
    """Footprint half-width in pixels: 2.5 sigma, so the cut-off edge is 4% of the peak."""
    return max(1, math.ceil(2.5 * sigma))


SOURCE = (
    "#define MAXW "
    + str(2 * MAX_REACH + 1)
    + "\n#define FIXED "
    + repr(float(FIXED))
    + """
extern "C" __global__ void phasor_splat(int n, int tile_x0, int tile_y0, int tile_w,
    int tile_h, int full_w, int full_h, double cx, double cy, double height,
    const double* pts, const float* rgb, const float* wt, float sigma, int reach,
    unsigned long long* acc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s = height / (double)full_h;  // world units per pixel
    double fx = (pts[2 * i] - cx) / s + 0.5 * (double)full_w - 0.5 - (double)tile_x0;
    double fy = (cy - pts[2 * i + 1]) / s + 0.5 * (double)full_h - 0.5 - (double)tile_y0;
    // whole footprint off this tile (or not a number): nothing to add
    if (!(fx > (double)(-reach - 1) && fx < (double)(tile_w + reach)
          && fy > (double)(-reach - 1) && fy < (double)(tile_h + reach))) return;
    float w = wt[i];
    if (!(w > 0.0f)) return;
    int bx = (int)floor(fx + 0.5) - reach;
    int by = (int)floor(fy + 0.5) - reach;
    int span = 2 * reach + 1;
    double gx[MAXW], gy[MAXW];
    double sx = 0.0, sy = 0.0;
    double q = -0.5 / ((double)sigma * (double)sigma);
    for (int j = 0; j < span; j++) {
        double ex = (double)(bx + j) - fx, ey = (double)(by + j) - fy;
        gx[j] = exp(q * ex * ex);
        gy[j] = exp(q * ey * ey);
        sx += gx[j];
        sy += gy[j];
    }
    double scale = (double)w / (sx * sy);
    float r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
    for (int l = 0; l < span; l++) {
        int py = by + l;
        if (py < 0 || py >= tile_h) continue;
        for (int j = 0; j < span; j++) {
            int px = bx + j;
            if (px < 0 || px >= tile_w) continue;
            float v = (float)(gx[j] * gy[l] * scale);
            unsigned long long* o = acc + 4 * ((long long)py * tile_w + px);
            // fixed point: integer sums do not depend on the order threads add in
            atomicAdd(o, (unsigned long long)llrint((double)(v * r) * FIXED));
            atomicAdd(o + 1, (unsigned long long)llrint((double)(v * g) * FIXED));
            atomicAdd(o + 2, (unsigned long long)llrint((double)(v * b) * FIXED));
            atomicAdd(o + 3, (unsigned long long)llrint((double)v * FIXED));
        }
    }
}

extern "C" __global__ void phasor_splat_tone(int n, float gain, float white, int curve,
    const unsigned long long* acc, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long* in = acc + 4 * (long long)i;
    float* o = out + 4 * (long long)i;
    double c[4];
    for (int k = 0; k < 4; k++) c[k] = (double)(long long)in[k] / FIXED;
    if (!curve) {
        for (int k = 0; k < 4; k++) o[k] = (float)c[k];
        return;
    }
    double d = c[3];
    if (!(d > 0.0)) { o[0] = 0.0f; o[1] = 0.0f; o[2] = 0.0f; o[3] = 0.0f; return; }
    double a = -expm1(-(double)gain * d);
    double sw = (double)white * (a * a) * (a * a);  // white only where light nears the ceiling
    for (int k = 0; k < 3; k++) {
        double m = c[k] / d;  // the mean colour of the points here
        o[k] = (float)((m + (1.0 - m) * sw) * a);
    }
    o[3] = (float)a;
}
"""
)


def _on_host(*arrays) -> bool:
    return all(a is None or not hasattr(a, "__cuda_array_interface__") for a in arrays)


def _inputs(points, colors, weights, xp=np):
    """The kernel's arrays: NumPy, or on the card (``xp`` CuPy) when the caller keeps them
    there (``cloud.card_points``), so they are not copied back and forth."""
    pts = xp.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    rgb = xp.ascontiguousarray(colors, dtype=np.float32).reshape(-1, 3)
    if len(rgb) == 1 and n != 1:
        rgb = xp.ascontiguousarray(xp.broadcast_to(rgb, (n, 3)))
    wt = (xp.ones(n, dtype=np.float32) if weights is None
          else xp.ascontiguousarray(weights, dtype=np.float32).reshape(-1))  # fmt: skip
    if len(rgb) != n or len(wt) != n:
        raise KernelError(f"splat: {n} points but {len(rgb)} colours and {len(wt)} weights")
    return pts, rgb, wt


def _check_sigma(sigma: float) -> None:
    if not 0 < sigma <= MAX_SIGMA:
        raise KernelError(f"splat: sigma must be in (0, {MAX_SIGMA:.1f}] pixels, got {sigma}")


def splat_reference(
    tile: Tile,
    points: np.ndarray,
    colors: np.ndarray,
    center: tuple[float, float],
    height: float,
    weights: np.ndarray | None = None,
    sigma: float = 1.0,
    gain: float = 1.0,
    curve: bool = True,
    white: float = 0.0,
) -> np.ndarray:
    _check_sigma(sigma)
    pts, rgb, wt = _inputs(points, colors, weights)
    reach = splat_reach(sigma)
    s = height / tile.full_h
    fx = (pts[:, 0] - center[0]) / s + 0.5 * tile.full_w - 0.5 - tile.x0
    fy = (center[1] - pts[:, 1]) / s + 0.5 * tile.full_h - 0.5 - tile.y0
    with np.errstate(invalid="ignore"):
        keep = ((fx > -reach - 1) & (fx < tile.w + reach) & (fy > -reach - 1)
                & (fy < tile.h + reach) & (wt > 0))  # fmt: skip
    fx, fy, rgb, wt = fx[keep], fy[keep], rgb[keep], wt[keep].astype(np.float64)
    bx = np.floor(fx + 0.5).astype(np.int64) - reach
    by = np.floor(fy + 0.5).astype(np.int64) - reach
    q = -0.5 / (float(sigma) * float(sigma))
    offs = np.arange(2 * reach + 1)
    gx = np.exp(q * ((bx[:, None] + offs) - fx[:, None]) ** 2)
    gy = np.exp(q * ((by[:, None] + offs) - fy[:, None]) ** 2)
    scale = wt / (gx.sum(1) * gy.sum(1))
    acc = np.zeros((4, tile.h * tile.w))
    for ly in offs:
        py = by + ly
        for jx in offs:
            px = bx + jx
            on = (px >= 0) & (px < tile.w) & (py >= 0) & (py < tile.h)
            if not on.any():
                continue
            idx = (py * tile.w + px)[on]
            v = (gx[on, jx] * gy[on, ly] * scale[on]).astype(np.float32)
            # the kernel's fixed-point units: each float32 product rounded to a whole unit
            for c in range(3):
                units = np.rint((v * rgb[on, c]).astype(np.float64) * FIXED)
                acc[c] += np.bincount(idx, units, minlength=tile.n)
            acc[3] += np.bincount(idx, np.rint(v.astype(np.float64) * FIXED), minlength=tile.n)
    out = acc.T.reshape(tile.h, tile.w, 4) / FIXED
    if curve:
        d = out[..., 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(d > 0, -np.expm1(-float(gain) * d), 0.0)
            m = np.where(d[..., None] > 0, out[..., :3] / d[..., None], 0.0)
        s = (float(white) * (a * a) * (a * a))[..., None]
        out = np.concatenate([(m + (1.0 - m) * s) * a[..., None], a[..., None]], axis=-1)
    return out.astype(np.float32)


@kernel(
    "splat",
    "(tile, points, colors, center, height, weights=None, sigma=1.0, gain=1.0, curve=True, "
    "white=0.0) -> float32[h, w, 4] rgb = (mean colour toward white by white * a^4) * a, "
    "alpha a = 1 - exp(-gain * density)",
    reference=splat_reference,
)
def splat(
    gpu,
    tile: Tile,
    points: np.ndarray,
    colors: np.ndarray,
    center: tuple[float, float],
    height: float,
    weights: np.ndarray | None = None,
    sigma: float = 1.0,
    gain: float = 1.0,
    curve: bool = True,
    white: float = 0.0,
):
    """Splat ``points`` ((n, 2) plane coordinates) with ``colors`` ((n, 3) linear RGB, or one)
    and ``weights`` ((n,), default 1) as Gaussians ``sigma`` pixels wide over the view.
    ``white`` (0-1) turns the colour toward white where the light nears the curve's ceiling
    (by ``white * a^4``): a white-hot core in the densest parts, none in the sparse ones."""
    _check_sigma(sigma)
    # host arrays as always (the emulator and CPU backend allow only a few array calls);
    # arrays already on the card stay there (only real CuPy can hold them)
    on_card = not _on_host(points, colors, weights)
    pts, rgb, wt = _inputs(points, colors, weights, gpu.cp if on_card else np)
    acc = gpu.cp.zeros((tile.h, tile.w, 4), dtype=gpu.cp.uint64)
    if len(pts):
        compiled = gpu.compile(SOURCE, "phasor_splat")
        args = (np.int32(len(pts)), np.int32(tile.x0), np.int32(tile.y0), np.int32(tile.w),
                np.int32(tile.h), np.int32(tile.full_w), np.int32(tile.full_h),
                np.float64(center[0]), np.float64(center[1]), np.float64(height),
                gpu.cp.asarray(pts), gpu.cp.asarray(rgb), gpu.cp.asarray(wt),
                np.float32(sigma), np.int32(splat_reach(sigma)), acc)  # fmt: skip
        gpu.launch(compiled, len(pts), args)
    out = gpu.cp.empty((tile.h, tile.w, 4), dtype=gpu.cp.float32)
    tone = gpu.compile(SOURCE, "phasor_splat_tone")
    gpu.launch(tone, tile.n, (np.int32(tile.n), np.float32(gain), np.float32(white),
                              np.int32(bool(curve)), acc, out))  # fmt: skip
    return out
