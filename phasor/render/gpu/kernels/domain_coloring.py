"""Domain coloring: a picture of a complex function.

Hue is the phase of f(z), looked up in a cyclic color table (so a theme can swap the
wheel). Brightness steps once every time |f| doubles, which draws the modulus contours.
Poles (and anything not finite) are white; zeros fall to the dark end of a contour band.
``soft`` (0-1) feathers the band edge: the brightness ramp eases back to its start over the
last ``soft / 2`` of each band instead of dropping in one step, so contours do not clash.
Output: linear RGBA float32, shape (h, w, 4).
"""

from __future__ import annotations

import numpy as np

from phasor.core.color import phase_lut
from phasor.render.gpu.budget import Tile
from phasor.render.gpu.codegen import ComplexFunction
from phasor.render.gpu.registry import VIEW_BODY, VIEW_PARAMS, kernel, plane, view_args

POLE = 1e30  # |f| above this is drawn as a pole


# ---------------------------------------------------------------- blending two functions
#
# ``to`` and ``blend`` (0-1) show w = (1 - blend) f + blend g: a straight line between the two
# functions (a homotopy), so every frame is the picture of a holomorphic function. Zeros glide
# along continuous paths, a pole of g grows out of a point (its share of the plane scales with
# blend), and there are no seams. Blending in log space, f^(1-b) g^b, looks plausible but cuts
# the plane along the branch cuts of log f and log g, where the colours jump. At blend 0 and 1
# the pixels are exactly those of f and of g. Without ``to`` the source is the one-function
# source, unchanged; with it the kernel takes ``blend`` as an argument, so a blend that
# animates never recompiles.


def _prelude(function: ComplexFunction, to: ComplexFunction | None) -> str:
    """The device function ``phasor_f(z, a)`` the kernels call: ``function``, or with ``to``
    the blend of the two (a macro reading the kernel's ``blend`` argument)."""
    if to is None:
        return "static __device__ cplx phasor_f(cplx z, cplx a) { return " + function.c() + "; }"
    return (
        "static __device__ cplx phasor_f0(cplx z, cplx a) { return " + function.c() + "; }\n"
        "static __device__ cplx phasor_f1(cplx z, cplx a) { return " + to.c() + "; }\n"
        """static __device__ cplx phasor_mix(cplx z, cplx a, real b) {
    if (b <= R(0)) return phasor_f0(z, a);
    if (b >= R(1)) return phasor_f1(z, a);
    cplx p = phasor_f0(z, a), q = phasor_f1(z, a);
    return c_make((R(1) - b) * p.re + b * q.re, (R(1) - b) * p.im + b * q.im);
}
#define phasor_f(z, a) phasor_mix(z, a, (real)blend)"""
    )


def _blend_param(to: ComplexFunction | None) -> str:
    return "" if to is None else ", double blend"


def blended(function: ComplexFunction, to: ComplexFunction | None = None, blend: float = 0.0):
    """The NumPy counterpart of ``_prelude``: ``f(z, a)`` of the blend."""
    f = function.numpy()
    if to is None or blend <= 0:
        return f
    g = to.numpy()
    if blend >= 1:
        return g

    def mix(z, a=0j):
        with np.errstate(all="ignore"):
            return (1 - blend) * f(z, a) + blend * g(z, a)

    return mix


def _blend_arg(to: ComplexFunction | None, blend: float) -> tuple:
    return () if to is None else (np.float64(blend),)


def source_for(function: ComplexFunction, to: ComplexFunction | None = None) -> str:
    return (
        _prelude(function, to)
        + """

extern "C" __global__ void phasor_domain("""
        + VIEW_PARAMS
        + """, double a_re, double a_im"""
        + _blend_param(to)
        + """,
    const float* lut, int lut_n, float contour, float soft, float* out)
{"""
        + VIEW_BODY
        + """
    cplx a = c_make((real)a_re, (real)a_im);
    cplx w = phasor_f(c_make(x, y), a);
    float* o = out + 4 * (long long)i;
    real m = c_abs(w);
    o[3] = 1.0f;
    if (!(m <= R(1e30))) { o[0] = 1.0f; o[1] = 1.0f; o[2] = 1.0f; return; }  // pole or NaN
    real band = R(0);
    if (m > R(0)) { real l = (real)log2(m); band = l - (real)floor(l); }
    real wd = (real)soft * R(0.5);
    if (wd > R(0) && band > R(1) - wd) {  // feather: ease back to the band's start value
        real q = (band - (R(1) - wd)) / wd;
        q = q * q * (R(3) - R(2) * q);
        band = band * (R(1) - q);
    }
    real shade = R(1) - (real)contour * (R(1) - band);
    real h = c_arg(w) / R(2.0 * PHASOR_PI);
    h = h - (real)floor(h);
    real f = h * (real)lut_n;
    int k0 = (int)f;
    if (k0 >= lut_n) k0 = lut_n - 1;
    real u = f - (real)k0;
    int k1 = (k0 + 1) % lut_n;
    for (int c = 0; c < 3; c++)
        o[c] = (float)(shade * ((real)lut[3 * k0 + c] * (R(1) - u) + (real)lut[3 * k1 + c] * u));
}
"""
    )


def domain_reference(
    tile: Tile,
    function: ComplexFunction,
    center: tuple[float, float],
    height: float,
    contour: float = 0.35,
    lut: np.ndarray | None = None,
    precision: str = "f32",
    soft: float = 0.0,
    a: complex = 0j,
    to: ComplexFunction | None = None,
    blend: float = 0.0,
) -> np.ndarray:
    lut = phase_lut() if lut is None else np.asarray(lut, dtype=np.float32)
    x, y = plane(tile, center, height)
    w = blended(function, to, blend)(x + 1j * y, a)
    with np.errstate(all="ignore"):
        m = np.abs(w)
        pole = ~(m <= POLE)
        log = np.log2(np.where(m > 0, m, 1.0))
        band = np.where(m > 0, log - np.floor(log), 0.0)
        wd = soft * 0.5
        if wd > 0:
            q = np.clip((band - (1 - wd)) / wd, 0.0, 1.0)
            band = band * (1 - q * q * (3 - 2 * q))
        shade = 1.0 - contour * (1.0 - band)
        h = np.angle(w) / (2 * np.pi)
        h = np.where(pole, 0.0, h - np.floor(h))
    f = h * len(lut)
    k0 = np.minimum(f.astype(np.int64), len(lut) - 1)
    u = (f - k0)[..., None]
    rgb = shade[..., None] * (lut[k0] * (1 - u) + lut[(k0 + 1) % len(lut)] * u)
    out = np.ones((tile.h, tile.w, 4), dtype=np.float32)
    out[..., :3] = np.where(pole[..., None], 1.0, rgb)
    return out


@kernel(
    "domain",
    "(tile, function, center, height, contour=0.35, lut=None, precision='f32', soft=0.0, a=0j, "
    "to=None, blend=0.0) "
    "-> float32[h, w, 4] linear RGBA",
    reference=domain_reference,
)
def domain(
    gpu,
    tile: Tile,
    function: ComplexFunction,
    center: tuple[float, float],
    height: float,
    contour: float = 0.35,
    lut: np.ndarray | None = None,
    precision: str = "f32",
    soft: float = 0.0,
    a: complex = 0j,
    to: ComplexFunction | None = None,
    blend: float = 0.0,
):
    """Color ``function`` over the view. ``contour`` (0-1) is how dark modulus bands get;
    ``soft`` (0-1) feathers the band edge;
    ``lut`` is an (n, 3) cyclic table of linear RGB, default ``core.color.phase_lut()``.
    With ``to``, the picture is of (1 - ``blend``) function + ``blend`` to (see ``_prelude``)."""
    table = np.ascontiguousarray(phase_lut() if lut is None else lut, dtype=np.float32)
    compiled = gpu.compile(source_for(function, to), "phasor_domain", precision)
    out = gpu.cp.empty((tile.h, tile.w, 4), dtype=gpu.cp.float32)
    args = (*view_args(tile, center, height), np.float64(complex(a).real),
            np.float64(complex(a).imag), *_blend_arg(to, blend), gpu.cp.asarray(table),
            np.int32(len(table)), np.float32(contour), np.float32(soft), out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out


# ---------------------------------------------------------------- the prism look
#
# Light where the function does something: each pixel's distance to the nearest zero or pole
# is estimated by one Newton step, d = |f / f'| (the same for a pole, since 1/f has the same
# ratio), with no root finding, so it works for any formula. Weight w = 1 / (1 + (d / r)^2):
# 1 at a zero or pole, a long soft tail (the haze) away from them. Colour is the phase from the
# table, as in the classic picture, at ambient + (1 - ambient) w, with soft sinusoidal modulus
# bands; alpha carries w, the glow weight the renderer turns into light and bloom.


def glow_source_for(function: ComplexFunction, to: ComplexFunction | None = None) -> str:
    return (
        _prelude(function, to)
        + """

extern "C" __global__ void phasor_domain_glow("""
        + VIEW_PARAMS
        + """, double a_re, double a_im"""
        + _blend_param(to)
        + """,
    const float* lut, int lut_n, float radius, float ambient, float contour,
    int lines, float line_glow, float* out)
{"""
        + VIEW_BODY
        + """
    cplx a = c_make((real)a_re, (real)a_im);
    float* o = out + 4 * (long long)i;
    cplx z = c_make(x, y);
    cplx w = phasor_f(z, a);
    real m = c_abs(w);
    if (!(m <= R(1e30))) { o[0] = 1.0f; o[1] = 1.0f; o[2] = 1.0f; o[3] = 1.0f; return; }
    real hs = (real)(0.5 * height / (double)full_h);  // half a pixel
    cplx dw = c_sub(phasor_f(c_make(x + hs, y), a), phasor_f(c_make(x - hs, y), a));
    real dm = c_abs(dw) / (R(2) * hs);                 // |f'|
    real wt = R(0);
    if (m == R(0)) wt = R(1);
    else if (dm > R(0) && dm <= R(1e30)) {
        real q = (m / dm) / (real)radius;
        wt = R(1) / (R(1) + q * q);
    }
    real band = R(0.5);
    if (m > R(0)) band = R(0.5) - R(0.5) * (real)cos(R(2.0 * PHASOR_PI) * (real)log2(m));
    real h = c_arg(w) / R(2.0 * PHASOR_PI);
    h = h - (real)floor(h);
    // Lines of constant phase (rays) and constant |f| (rings), about a pixel wide anywhere:
    // log f changes by |f'/f| per world unit, so a pixel spans g = |f'/f| px of it.
    real ln = R(0);
    if (lines > 0 && m > R(0) && dm > R(0) && dm <= R(1e30)) {
        real g = (dm / m) * R(2) * hs;
        real ph = h * (real)lines;
        real dp = R(0.5) - (real)fabs(ph - (real)floor(ph) - R(0.5));        // to a ray
        real dist = dp * R(2.0 * PHASOR_PI) / (real)lines / g;              // in pixels
        real lm = (real)log2(m);
        real dl = R(0.5) - (real)fabs(lm - (real)floor(lm) - R(0.5));        // to a ring
        real dist2 = dl * R(0.6931471805599453) / g;
        real d1 = dist < dist2 ? dist : dist2;
        ln = (real)exp(-d1 * d1 / R(0.8));
    }
    real lit = wt + (real)line_glow * ln * (R(1) - wt);
    real shade = ((real)ambient + (R(1) - (real)ambient) * lit) * (R(1) - (real)contour * band);
    real f = h * (real)lut_n;
    int k0 = (int)f;
    if (k0 >= lut_n) k0 = lut_n - 1;
    real u = f - (real)k0;
    int k1 = (k0 + 1) % lut_n;
    for (int c = 0; c < 3; c++)
        o[c] = (float)(shade * ((real)lut[3 * k0 + c] * (R(1) - u) + (real)lut[3 * k1 + c] * u));
    o[3] = (float)lit;
}
"""
    )


def domain_glow_reference(
    tile: Tile,
    function: ComplexFunction,
    center: tuple[float, float],
    height: float,
    lut: np.ndarray | None = None,
    radius: float = 0.3,
    ambient: float = 0.08,
    contour: float = 0.15,
    precision: str = "f32",
    lines: int = 0,
    line_glow: float = 0.6,
    a: complex = 0j,
    to: ComplexFunction | None = None,
    blend: float = 0.0,
) -> np.ndarray:
    lut = phase_lut() if lut is None else np.asarray(lut, dtype=np.float32)
    x, y = plane(tile, center, height)
    f0 = blended(function, to, blend)

    def f(z):
        return f0(z, a)

    hs = 0.5 * height / tile.full_h
    with np.errstate(all="ignore"):
        w = f(x + 1j * y)
        m = np.abs(w)
        pole = ~(m <= POLE)
        dm = np.abs(f((x + hs) + 1j * y) - f((x - hs) + 1j * y)) / (2 * hs)
        q = (m / dm) / radius
        wt = np.where(m == 0, 1.0, np.where((dm > 0) & (dm <= POLE), 1 / (1 + q * q), 0.0))
        band = np.where(m > 0, 0.5 - 0.5 * np.cos(2 * np.pi * np.log2(np.where(m > 0, m, 1))),
                        0.5)  # fmt: skip
        h = np.angle(w) / (2 * np.pi)
        h = np.where(pole, 0.0, h - np.floor(h))
        ln = np.zeros_like(wt)
        if lines > 0:
            g = (dm / m) * 2 * hs
            ok = (m > 0) & (dm > 0) & (dm <= POLE)
            ph = h * lines
            dist = (0.5 - np.abs(ph - np.floor(ph) - 0.5)) * 2 * np.pi / lines / g
            lm = np.log2(np.where(m > 0, m, 1.0))
            dist2 = (0.5 - np.abs(lm - np.floor(lm) - 0.5)) * np.log(2) / g
            d1 = np.minimum(dist, dist2)
            ln = np.where(ok, np.exp(-d1 * d1 / 0.8), 0.0)
        lit = wt + line_glow * ln * (1 - wt)
        shade = (ambient + (1 - ambient) * lit) * (1 - contour * band)
    fl = h * len(lut)
    k0 = np.minimum(fl.astype(np.int64), len(lut) - 1)
    u = (fl - k0)[..., None]
    rgb = shade[..., None] * (lut[k0] * (1 - u) + lut[(k0 + 1) % len(lut)] * u)
    out = np.ones((tile.h, tile.w, 4), dtype=np.float32)
    out[..., :3] = np.where(pole[..., None], 1.0, rgb)
    out[..., 3] = np.where(pole, 1.0, lit)
    return out


@kernel(
    "domain_glow",
    "(tile, function, center, height, lut=None, radius=0.3, ambient=0.08, contour=0.15, "
    "precision='f32', lines=0, line_glow=0.6, a=0j, to=None, blend=0.0) "
    "-> float32[h, w, 4] linear RGB, alpha = glow",
    reference=domain_glow_reference,
)
def domain_glow(
    gpu,
    tile: Tile,
    function: ComplexFunction,
    center: tuple[float, float],
    height: float,
    lut: np.ndarray | None = None,
    radius: float = 0.3,
    ambient: float = 0.08,
    contour: float = 0.15,
    precision: str = "f32",
    lines: int = 0,
    line_glow: float = 0.6,
    a: complex = 0j,
    to: ComplexFunction | None = None,
    blend: float = 0.0,
):
    """The prism look of ``function``: phase as hue, lit near its zeros and poles (within
    about ``radius`` world units), ``ambient`` light elsewhere, soft modulus bands of depth
    ``contour``; ``lines`` rays of constant phase per turn plus a ring each time |f|
    doubles, about a pixel wide, lit by ``line_glow``; alpha is the glow weight. With ``to``,
    the picture is of (1 - ``blend``) function + ``blend`` to."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    table = np.ascontiguousarray(phase_lut() if lut is None else lut, dtype=np.float32)
    compiled = gpu.compile(glow_source_for(function, to), "phasor_domain_glow", precision)
    out = gpu.cp.empty((tile.h, tile.w, 4), dtype=gpu.cp.float32)
    args = (*view_args(tile, center, height), np.float64(complex(a).real),
            np.float64(complex(a).imag), *_blend_arg(to, blend), gpu.cp.asarray(table),
            np.int32(len(table)), np.float32(radius), np.float32(ambient),
            np.float32(contour), np.int32(lines), np.float32(line_glow), out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out
