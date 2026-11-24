"""Wave fields: the sum of N point sources of one frequency (interference, beamforming).

For each pixel x, the complex amplitude

    A(x) = sum_k  amp_k * s_k(x) * e^{i (k |x - p_k| + phase_k)} / sqrt(max(|x - p_k|, core))

(2D spreading: the amplitude falls as one over the square root of the distance, held at
``core`` near a source so it stays finite). ``s_k`` switches source k on: 1 inside its
``reach`` (how far its first wave has travelled), falling smoothly to 0 over ``edge`` world
units at the front, so a source that starts sends out a ring; a reach of ``ALWAYS`` or more
means always on. The instantaneous field is Re(A e^{-i w t}) = A.re cos(wt) + A.im sin(wt):
time enters only through ``cos(wt)`` and ``sin(wt)``, which the launcher takes in float64,
so the waves move by run-time arguments (animating never recompiles) and a large ``wt`` does
not lose precision. The time-averaged intensity is |A|^2 / 2.

Output: float32 (h, w, 4) = (field, intensity, A.re, A.im). Colouring is the object's job
(``domains/waves.py``). Plain arithmetic per pixel, no atomics: the same bits every run.
"""

from __future__ import annotations

import numpy as np

from phasor.render.gpu.budget import Tile
from phasor.render.gpu.registry import VIEW_BODY, VIEW_PARAMS, KernelError, kernel, plane, view_args

ALWAYS = 1e30  # a reach at least this: the source has always been on (no front)
COLUMNS = 5  # per source: x, y, amplitude, phase (radians), reach

SOURCE = (
    """
extern "C" __global__ void phasor_wave("""
    + VIEW_PARAMS
    + """,
    const double* src, int ns, double wavenumber, double cos_wt, double sin_wt, double core,
    double edge, float* out)
{"""
    + VIEW_BODY
    + """
    real re = R(0), im = R(0);
    real kw = (real)wavenumber;
    for (int j = 0; j < ns; j++) {
        const double* s = src + 5 * j;
        real dx = x - (real)s[0], dy = y - (real)s[1];
        real r = (real)sqrt(dx * dx + dy * dy);
        real w = (real)s[2];
        if (s[4] < 1e30) {  // switched on at some time: its front has gone ``reach`` far
            real reach = (real)s[4];
            if (edge > 0.0) {
                real f = (reach - r) / (real)edge;
                if (f <= R(0)) continue;
                if (f < R(1)) w = w * (f * f * (R(3) - R(2) * f));
            } else if (!(r < reach)) continue;
        }
        real rr = r > (real)core ? r : (real)core;
        real a = w / (real)sqrt(rr);
        real ph = kw * r + (real)s[3];
        re += a * (real)cos(ph);
        im += a * (real)sin(ph);
    }
    float* o = out + 4 * (long long)i;
    o[0] = (float)(re * (real)cos_wt + im * (real)sin_wt);
    o[1] = (float)(R(0.5) * (re * re + im * im));
    o[2] = (float)re;
    o[3] = (float)im;
}
"""
)


def _sources(sources) -> np.ndarray:
    src = np.ascontiguousarray(sources, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != COLUMNS or not len(src):
        raise KernelError(f"wave_field: sources must be (n >= 1, {COLUMNS}): x, y, amplitude, "
                          f"phase, reach; got shape {src.shape}")  # fmt: skip
    return src


def amplitude(x, y, sources, wavenumber: float, core: float = 0.05, edge: float = 0.0):
    """The complex amplitude A at points ``x``, ``y`` (arrays of one shape), in float64: the
    specification of the kernel's sum (``waves_math`` evaluates the field anywhere with it)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    re, im = np.zeros(np.broadcast(x, y).shape), np.zeros(np.broadcast(x, y).shape)
    for sx, sy, amp, phase, reach in _sources(sources):
        r = np.hypot(x - sx, y - sy)
        w = np.full_like(r, amp)
        if reach < ALWAYS:
            if edge > 0:
                f = np.clip((reach - r) / edge, 0.0, 1.0)
                w = w * f * f * (3 - 2 * f)
            else:
                w = np.where(r < reach, w, 0.0)
        a = w / np.sqrt(np.maximum(r, core))
        ph = wavenumber * r + phase
        re += a * np.cos(ph)
        im += a * np.sin(ph)
    return re + 1j * im


def wave_field_reference(
    tile: Tile,
    sources: np.ndarray,
    center: tuple[float, float],
    height: float,
    wavenumber: float,
    omega_t: float = 0.0,
    core: float = 0.05,
    edge: float = 0.0,
    precision: str = "f32",
) -> np.ndarray:
    x, y = plane(tile, center, height)
    a = amplitude(x, y, sources, wavenumber, core, edge)
    re, im = a.real, a.imag
    out = np.empty((tile.h, tile.w, 4), dtype=np.float32)
    out[..., 0] = re * np.cos(omega_t) + im * np.sin(omega_t)
    out[..., 1] = 0.5 * (re * re + im * im)
    out[..., 2] = re
    out[..., 3] = im
    return out


@kernel(
    "wave_field",
    "(tile, sources, center, height, wavenumber, omega_t=0.0, core=0.05, edge=0.0, "
    "precision='f32') -> float32[h, w, 4] (field, intensity, Re A, Im A)",
    reference=wave_field_reference,
)
def wave_field(
    gpu,
    tile: Tile,
    sources: np.ndarray,
    center: tuple[float, float],
    height: float,
    wavenumber: float,
    omega_t: float = 0.0,
    core: float = 0.05,
    edge: float = 0.0,
    precision: str = "f32",
):
    """The field of point ``sources`` ((n, 5): x, y, amplitude, phase in radians, reach in
    world units, ``ALWAYS`` for always on) of one ``wavenumber`` (2 pi / wavelength) over the
    view, at time phase ``omega_t`` (w t, radians). ``core``: the distance under which the
    1/sqrt(r) spreading is held; ``edge``: how soft a switching-on front is (world units)."""
    src = _sources(sources)
    compiled = gpu.compile(SOURCE, "phasor_wave", precision)
    out = gpu.cp.empty((tile.h, tile.w, 4), dtype=gpu.cp.float32)
    args = (*view_args(tile, center, height), gpu.cp.asarray(src), np.int32(len(src)),
            np.float64(wavenumber), np.float64(np.cos(omega_t)), np.float64(np.sin(omega_t)),
            np.float64(core), np.float64(edge), out)  # fmt: skip
    gpu.launch(compiled, tile.n, args)
    return out
