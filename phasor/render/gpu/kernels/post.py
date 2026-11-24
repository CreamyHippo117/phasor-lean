"""Post-processing kernels on a whole linear RGBA float32 image: tone map and blur.

They work on the full frame (the tile must be the whole image): a blur reads neighbors, so
it cannot be cut into independent tiles, and a 4K frame is only about 133 MB.
"""

from __future__ import annotations

import math

import numpy as np

from phasor.render.gpu.budget import Tile
from phasor.render.gpu.registry import KernelError, kernel

SOURCE = """
extern "C" __global__ void phasor_tonemap(int n, float exposure, float knee, const float* in,
                                          const float* bloom, float bloom_strength, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int c = 0; c < 3; c++) {
        float x = (in[4 * (long long)i + c] + bloom_strength * bloom[4 * (long long)i + c])
                  * exposure;
        if (!(x > 0.0f)) x = 0.0f;
        // Shoulder: identity up to the knee, then an exponential roll-off towards 1.
        // Paint colors stay exactly as the theme defines them; only light above the knee
        // (glow, bloom) is compressed. The curve is smooth (slope 1) at the knee.
        float s = 1.0f - knee;
        float y = x <= knee ? x : knee + s * (1.0f - expf(-(x - knee) / s));
        out[4 * (long long)i + c] = y;
    }
    out[4 * (long long)i + 3] = in[4 * (long long)i + 3];
}

extern "C" __global__ void phasor_gauss(int n, int w, int h, int axis, int radius,
                                        float sigma, const float* in, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int px = i % w, py = i / w;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float total = 0.0f;
    for (int k = -radius; k <= radius; k++) {
        int qx = px + (axis == 0 ? k : 0);
        int qy = py + (axis == 1 ? k : 0);
        qx = qx < 0 ? 0 : (qx >= w ? w - 1 : qx);   // clamp at the edges
        qy = qy < 0 ? 0 : (qy >= h ? h - 1 : qy);
        float g = expf(-0.5f * (float)(k * k) / (sigma * sigma));
        const float* p = in + 4 * ((long long)qy * w + qx);
        for (int c = 0; c < 4; c++) acc[c] += g * p[c];
        total += g;
    }
    for (int c = 0; c < 4; c++) out[4 * (long long)i + c] = acc[c] / total;
}

extern "C" __global__ void phasor_bright(int n, float threshold, const float* in, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = in + 4 * (long long)i;
    float r = p[0] > 0.0f ? p[0] : 0.0f;
    float g = p[1] > 0.0f ? p[1] : 0.0f;
    float b = p[2] > 0.0f ? p[2] : 0.0f;
    float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float keep = lum > threshold ? (lum - threshold) / lum : 0.0f;
    float* o = out + 4 * (long long)i;
    o[0] = r * keep; o[1] = g * keep; o[2] = b * keep; o[3] = 1.0f;
}

extern "C" __global__ void phasor_resize(int n, int in_w, int in_h, int out_w, int out_h,
                                         const float* in, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ox = i % out_w, oy = i / out_w;
    float u = ((float)ox + 0.5f) * (float)in_w / (float)out_w - 0.5f;
    float v = ((float)oy + 0.5f) * (float)in_h / (float)out_h - 0.5f;
    u = u < 0.0f ? 0.0f : (u > (float)(in_w - 1) ? (float)(in_w - 1) : u);
    v = v < 0.0f ? 0.0f : (v > (float)(in_h - 1) ? (float)(in_h - 1) : v);
    int x0 = (int)u, y0 = (int)v;
    int x1 = x0 + 1 < in_w ? x0 + 1 : x0, y1 = y0 + 1 < in_h ? y0 + 1 : y0;
    float fx = u - (float)x0, fy = v - (float)y0;
    for (int c = 0; c < 4; c++) {
        long long r0 = (long long)y0 * in_w, r1 = (long long)y1 * in_w;
        float a = in[4 * (r0 + x0) + c], b = in[4 * (r0 + x1) + c];
        float d = in[4 * (r1 + x0) + c], e = in[4 * (r1 + x1) + c];
        float top = a + (b - a) * fx, bottom = d + (e - d) * fx;
        out[4 * (long long)i + c] = top + (bottom - top) * fy;
    }
}
"""


def _check(tile: Tile, image) -> None:
    if (tile.x0, tile.y0, tile.w, tile.h) != (0, 0, tile.full_w, tile.full_h):
        raise KernelError("post-processing kernels need the whole image, not a tile")
    if tuple(image.shape) != (tile.h, tile.w, 4) or str(image.dtype) != "float32":
        raise KernelError(
            f"image must be float32 with shape {(tile.h, tile.w, 4)}, "
            f"got {image.dtype} {tuple(image.shape)}"
        )


def tonemap_reference(
    tile: Tile,
    image: np.ndarray,
    exposure: float = 1.0,
    bloom: np.ndarray | None = None,
    bloom_strength: float = 0.0,
    knee: float = 0.9,
) -> np.ndarray:
    _check(tile, image)
    rgb = image[..., :3]
    if bloom is not None:
        _check(tile, bloom)
        rgb = rgb + np.float32(bloom_strength) * bloom[..., :3]
    x = np.maximum(np.nan_to_num((rgb * np.float32(exposure)).astype(np.float32), nan=0.0), 0)
    x = x.astype(np.float32)
    k, s = np.float32(knee), np.float32(1.0) - np.float32(knee)
    with np.errstate(over="ignore"):
        y = np.where(x <= k, x, k + s * (np.float32(1) - np.exp(-(x - k) / s)))
    return np.concatenate([y, image[..., 3:]], axis=-1).astype(np.float32)


@kernel(
    "tonemap",
    "(tile, image, exposure=1.0, bloom=None, bloom_strength=0.0, knee=0.9) -> float32[h, w, 4]; "
    "shoulder curve of image + strength * bloom, linear in, linear 0..1 out",
    reference=tonemap_reference,
)
def tonemap(
    gpu, tile: Tile, image, exposure: float = 1.0, bloom=None, bloom_strength=0.0, knee=0.9
):
    """Map linear HDR to linear display range: unchanged up to ``knee``, then rolled off
    smoothly towards 1. The sRGB encode still happens once, later."""
    if not 0.0 < knee < 1.0:
        raise KernelError("knee must be between 0 and 1")
    _check(tile, image)
    src = gpu.cp.ascontiguousarray(gpu.cp.asarray(image))
    if bloom is None:
        glow, bloom_strength = src, 0.0
    else:
        _check(tile, bloom)
        glow = gpu.cp.ascontiguousarray(gpu.cp.asarray(bloom))
    out = gpu.cp.empty_like(src)
    compiled = gpu.compile(SOURCE, "phasor_tonemap")
    args = (np.int32(tile.n), np.float32(exposure), np.float32(knee), src, glow,
            np.float32(bloom_strength), out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out


def gauss_radius(sigma: float) -> int:
    return max(1, math.ceil(3.0 * sigma))


def gauss_reference(tile: Tile, image: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    _check(tile, image)
    r = gauss_radius(sigma)
    pad = [(0, 0)] * 3
    pad[1 - axis] = (r, r)  # axis 0 blurs along x, which is array axis 1
    padded = np.pad(image, pad, mode="edge")
    acc = np.zeros_like(image, dtype=np.float32)
    total = np.float32(0)
    size = image.shape[1 - axis]
    for k in range(-r, r + 1):
        g = np.exp(np.float32(-0.5) * np.float32(k * k) / np.float32(sigma * sigma))
        window = [slice(None)] * 3
        window[1 - axis] = slice(r + k, r + k + size)
        acc += g.astype(np.float32) * padded[tuple(window)]
        total += g.astype(np.float32)
    return (acc / total).astype(np.float32)


@kernel(
    "gauss",
    "(tile, image, sigma, axis) -> float32[h, w, 4]; one pass of a separable Gaussian blur",
    reference=gauss_reference,
)
def gauss(gpu, tile: Tile, image, sigma: float, axis: int):
    """Blur along x (``axis=0``) or y (``axis=1``). Run both for a full blur (bloom)."""
    _check(tile, image)
    if axis not in (0, 1) or sigma <= 0:
        raise KernelError("gauss needs axis 0 or 1 and sigma > 0")
    src = gpu.cp.ascontiguousarray(gpu.cp.asarray(image))
    out = gpu.cp.empty_like(src)
    compiled = gpu.compile(SOURCE, "phasor_gauss")
    args = (np.int32(tile.n), np.int32(tile.w), np.int32(tile.h), np.int32(axis),
            np.int32(gauss_radius(sigma)), np.float32(sigma), src, out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out


def bright_reference(tile: Tile, image: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    _check(tile, image)
    rgb = np.maximum(image[..., :3], 0).astype(np.float32)
    lum = (np.float32(0.2126) * rgb[..., 0] + np.float32(0.7152) * rgb[..., 1]
           + np.float32(0.0722) * rgb[..., 2])  # fmt: skip
    with np.errstate(all="ignore"):
        keep = np.where(lum > threshold, (lum - np.float32(threshold)) / lum, 0).astype(np.float32)
    out = np.ones_like(image, dtype=np.float32)
    out[..., :3] = rgb * keep[..., None]
    return out


@kernel(
    "bright",
    "(tile, image, threshold=1.0) -> float32[h, w, 4]; the light above threshold, for bloom",
    reference=bright_reference,
)
def bright(gpu, tile: Tile, image, threshold: float = 1.0):
    """Keep only what is brighter than ``threshold`` (luminance), scaled so the excess is
    kept and the hue is unchanged. Linear HDR: only emissive things exceed 1."""
    _check(tile, image)
    src = gpu.cp.ascontiguousarray(gpu.cp.asarray(image))
    out = gpu.cp.empty_like(src)
    compiled = gpu.compile(SOURCE, "phasor_bright")
    gpu.launch(compiled, tile.n, (np.int32(tile.n), np.float32(threshold), src, out))
    return out


def _resize_check(tile: Tile, image) -> None:
    if (tile.x0, tile.y0, tile.w, tile.h) != (0, 0, tile.full_w, tile.full_h):
        raise KernelError("resize needs the whole output image as its tile")
    if image.ndim != 3 or image.shape[2] != 4 or str(image.dtype) != "float32":
        raise KernelError("resize needs a float32 (h, w, 4) image")


def resize_reference(tile: Tile, image: np.ndarray) -> np.ndarray:
    _resize_check(tile, image)
    in_h, in_w = image.shape[:2]
    ox = np.arange(tile.w, dtype=np.float32)
    oy = np.arange(tile.h, dtype=np.float32)
    u = np.clip((ox + np.float32(0.5)) * np.float32(in_w) / np.float32(tile.w) - np.float32(0.5),
                0, in_w - 1).astype(np.float32)  # fmt: skip
    v = np.clip((oy + np.float32(0.5)) * np.float32(in_h) / np.float32(tile.h) - np.float32(0.5),
                0, in_h - 1).astype(np.float32)  # fmt: skip
    x0, y0 = u.astype(np.int64), v.astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, in_w - 1), np.minimum(y0 + 1, in_h - 1)
    fx = (u - x0.astype(np.float32))[None, :, None]
    fy = (v - y0.astype(np.float32))[:, None, None]
    a, b = image[y0][:, x0], image[y0][:, x1]
    d, e = image[y1][:, x0], image[y1][:, x1]
    top, bottom = a + (b - a) * fx, d + (e - d) * fx
    return (top + (bottom - top) * fy).astype(np.float32)


@kernel(
    "resize",
    "(tile, image) -> float32[tile.h, tile.w, 4]; bilinear, sampled at pixel centers",
    reference=resize_reference,
)
def resize(gpu, tile: Tile, image):
    """Resample a whole image to the tile's size. Halving a dimension averages pixel pairs
    exactly, so repeated halving builds a box-filtered pyramid (bloom)."""
    _resize_check(tile, image)
    src = gpu.cp.ascontiguousarray(gpu.cp.asarray(image))
    out = gpu.cp.empty((tile.h, tile.w, 4), dtype=gpu.cp.float32)
    compiled = gpu.compile(SOURCE, "phasor_resize")
    args = (np.int32(tile.n), np.int32(src.shape[1]), np.int32(src.shape[0]),
            np.int32(tile.w), np.int32(tile.h), src, out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out


# ---------------------------------------------------------------- lens: fringe and vignette

LENS = """
__device__ float phasor_sample(const float* img, int w, int h, float x, float y, int c)
{
    x = x < 0.0f ? 0.0f : (x > (float)(w - 1) ? (float)(w - 1) : x);
    y = y < 0.0f ? 0.0f : (y > (float)(h - 1) ? (float)(h - 1) : y);
    int x0 = (int)x, y0 = (int)y;
    int x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
    float fx = x - (float)x0, fy = y - (float)y0;
    float a = img[4 * ((long long)y0 * w + x0) + c], b = img[4 * ((long long)y0 * w + x1) + c];
    float d = img[4 * ((long long)y1 * w + x0) + c], e = img[4 * ((long long)y1 * w + x1) + c];
    float top = a + (b - a) * fx, bottom = d + (e - d) * fx;
    return top + (bottom - top) * fy;
}

extern "C" __global__ void phasor_fringe(int n, int w, int h, float amount, const float* in,
                                         float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float px = (float)(i % w), py = (float)(i / w);
    float cx = 0.5f * (float)(w - 1), cy = 0.5f * (float)(h - 1);
    // Red is sampled from further out and blue from further in (a lens's lateral color),
    // plus a small fixed diagonal shift so a glow in the middle of the frame splits too.
    float sx = amount * ((px - cx) + 0.25f * (float)h);
    float sy = amount * ((py - cy) - 0.25f * (float)h);
    float* o = out + 4 * (long long)i;
    o[0] = phasor_sample(in, w, h, px - sx, py - sy, 0);
    o[1] = in[4 * (long long)i + 1];
    o[2] = phasor_sample(in, w, h, px + sx, py + sy, 2);
    o[3] = in[4 * (long long)i + 3];
}

extern "C" __global__ void phasor_vignette(int n, int w, int h, float strength, const float* in,
                                           float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float u = ((float)(i % w) + 0.5f) / (float)w - 0.5f;
    float v = ((float)(i / w) + 0.5f) / (float)h - 0.5f;
    float r2 = (u * u + v * v) * 2.0f;  // 0 in the middle, 1 in the corners
    float k = 1.0f - strength * r2 * r2;
    k = k < 0.0f ? 0.0f : k;
    for (int c = 0; c < 3; c++) out[4 * (long long)i + c] = in[4 * (long long)i + c] * k;
    out[4 * (long long)i + 3] = in[4 * (long long)i + 3];
}
"""


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray, c: int) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1).astype(np.float32)
    y = np.clip(y, 0, h - 1).astype(np.float32)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0.astype(np.float32), y - y0.astype(np.float32)
    ch = img[..., c]
    top = ch[y0, x0] + (ch[y0, x1] - ch[y0, x0]) * fx
    bottom = ch[y1, x0] + (ch[y1, x1] - ch[y1, x0]) * fx
    return (top + (bottom - top) * fy).astype(np.float32)


def fringe_reference(tile: Tile, image: np.ndarray, amount: float = 0.004) -> np.ndarray:
    _check(tile, image)
    h, w = image.shape[:2]
    py, px = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.float32(amount)
    cx, cy = np.float32(0.5) * np.float32(w - 1), np.float32(0.5) * np.float32(h - 1)
    sx = a * ((px - cx) + np.float32(0.25) * np.float32(h))
    sy = a * ((py - cy) - np.float32(0.25) * np.float32(h))
    out = image.copy()
    out[..., 0] = _bilinear(image, px - sx, py - sy, 0)
    out[..., 2] = _bilinear(image, px + sx, py + sy, 2)
    return out


@kernel(
    "fringe",
    "(tile, image, amount=0.004) -> float32[h, w, 4]; red out, blue in: a lens's color fringe",
    reference=fringe_reference,
)
def fringe(gpu, tile: Tile, image, amount: float = 0.004):
    """Lateral color on the glow (theme ``[bloom] fringe``): red is pulled outward and blue
    inward by ``amount`` of the distance from the center (plus a small fixed diagonal), so
    halos split into warm and cool rims. Applied to the bloom only: paint stays exact."""
    _check(tile, image)
    src = gpu.cp.ascontiguousarray(gpu.cp.asarray(image))
    out = gpu.cp.empty_like(src)
    compiled = gpu.compile(LENS, "phasor_fringe")
    gpu.launch(compiled, tile.n, (np.int32(tile.n), np.int32(tile.w), np.int32(tile.h),
                                  np.float32(amount), src, out))  # fmt: skip
    return out


def vignette_reference(tile: Tile, image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    _check(tile, image)
    h, w = image.shape[:2]
    u = (np.arange(w, dtype=np.float32) + np.float32(0.5)) / np.float32(w) - np.float32(0.5)
    v = (np.arange(h, dtype=np.float32) + np.float32(0.5)) / np.float32(h) - np.float32(0.5)
    r2 = (u[None, :] * u[None, :] + v[:, None] * v[:, None]) * np.float32(2)
    k = np.maximum(np.float32(1) - np.float32(strength) * r2 * r2, np.float32(0))
    out = image.copy()
    out[..., :3] = image[..., :3] * k[..., None]
    return out.astype(np.float32)


@kernel(
    "vignette",
    "(tile, image, strength=0.3) -> float32[h, w, 4]; corners darkened by strength",
    reference=vignette_reference,
)
def vignette(gpu, tile: Tile, image, strength: float = 0.3):
    """Darken toward the corners (theme ``[bloom] vignette``): 1 - strength r^4, r = 1 at a
    corner. After the tone map; the middle of the frame is left exactly as it was."""
    _check(tile, image)
    src = gpu.cp.ascontiguousarray(gpu.cp.asarray(image))
    out = gpu.cp.empty_like(src)
    compiled = gpu.compile(LENS, "phasor_vignette")
    gpu.launch(compiled, tile.n, (np.int32(tile.n), np.int32(tile.w), np.int32(tile.h),
                                  np.float32(strength), src, out))  # fmt: skip
    return out


# ---------------------------------------------------------------- display bytes

SRGB = """
extern "C" __global__ void phasor_srgb8(int n, const float* in, unsigned char* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int c = 0; c < 3; c++) {
        // Double precision and the same constants as core.color.linear_to_srgb, then
        // round half to even like np.round, so the card and NumPy agree byte for byte
        // (up to pow's last bit). NaN and negatives become 0, above 1 becomes 1.
        double x = (double)in[4 * (long long)i + c];
        if (!(x > 0.0)) x = 0.0;
        if (x > 1.0) x = 1.0;
        double y = x <= 0.0031308 ? x * 12.92 : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
        out[3 * (long long)i + c] = (unsigned char)rint(y * 255.0);
    }
}
"""


def srgb8_reference(tile: Tile, image: np.ndarray) -> np.ndarray:
    from phasor.render.output import encode_srgb8

    _check(tile, image)
    return encode_srgb8(image)


@kernel(
    "srgb8",
    "(tile, image) -> uint8[h, w, 3]; linear display RGBA (0..1) -> sRGB bytes",
    reference=srgb8_reference,
)
def srgb8(gpu, tile: Tile, image):
    """The sRGB encode of ``render.output.encode_srgb8`` on the card, after the tone map
    (contract 4: still once per pixel). The frame comes back as 6 MB of bytes at 1080p
    instead of 33 MB of float32."""
    _check(tile, image)
    src = gpu.cp.ascontiguousarray(gpu.cp.asarray(image))
    out = gpu.cp.empty((tile.h, tile.w, 3), dtype=gpu.cp.uint8)
    compiled = gpu.compile(SRGB, "phasor_srgb8")
    gpu.launch(compiled, tile.n, (np.int32(tile.n), src, out))
    return out


# ---------------------------------------------------------------- motion-blur sum

ACCUMULATE = """
extern "C" __global__ void phasor_accumulate(int n, float weight, const float* in,
                                             const float* acc, int has_acc, float* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = weight * in[i];
    out[i] = has_acc ? acc[i] + x : x;
}
"""


def _check_sum(tile: Tile, image, acc) -> None:
    if (tile.x0, tile.y0, tile.w, tile.h) != (0, 0, tile.full_w, tile.full_h):
        raise KernelError("post-processing kernels need the whole image, not a tile")
    if tuple(image.shape[:2]) != (tile.h, tile.w) or str(image.dtype) not in ("float16", "float32"):
        raise KernelError(f"image must be float16 or float32 with shape ({tile.h}, {tile.w}, "
                          f"c), got {image.dtype} {tuple(image.shape)}")  # fmt: skip
    if acc is not None and (tuple(acc.shape) != tuple(image.shape) or str(acc.dtype) != "float32"):
        raise KernelError(f"acc must be float32 with shape {tuple(image.shape)}, "
                          f"got {acc.dtype} {tuple(acc.shape)}")  # fmt: skip


def accumulate_reference(tile: Tile, image: np.ndarray, weight: float = 1.0,
                         acc: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    _check_sum(tile, image, acc)
    x = np.float32(weight) * image.astype(np.float32)
    return x if acc is None else (acc + x).astype(np.float32)  # a new array: inputs untouched


@kernel(
    "accumulate",
    "(tile, image, weight=1.0, acc=None) -> float32[h, w, c]; acc + weight * image",
    reference=accumulate_reference,
)
def accumulate(gpu, tile: Tile, image, weight: float = 1.0, acc=None):
    """One step of a weighted sum in float32 (``Engine.frame_blurred``: the motion-blur
    samples). A float16 ``image`` (the renderer's raw read) goes to the card at half the
    bytes and is widened there; the sum stays on the card for ``post.finish``. Any number of
    channels. Fixed order, no atomics: the same samples give the same bits."""
    _check_sum(tile, image, acc)
    src = gpu.cp.ascontiguousarray(gpu.cp.asarray(image).astype(np.float32))
    out = gpu.cp.empty_like(src)
    prev = src if acc is None else gpu.cp.ascontiguousarray(gpu.cp.asarray(acc))
    compiled = gpu.compile(ACCUMULATE, "phasor_accumulate")
    gpu.launch(compiled, int(src.size), (np.int32(src.size), np.float32(weight), src, prev,
                                         np.int32(acc is not None), out))  # fmt: skip
    return out
