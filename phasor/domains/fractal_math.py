"""Deep fractal zooms: a reference orbit in arbitrary precision.

Past about 1e-10 per pixel, float64 starts to lose neighbouring pixels. Perturbation
fixes that: one point C of the view is iterated exactly (here), and every pixel c = C + dc
iterates only its small difference from it, in float64, on the GPU (``deep`` kernel):

    z_n = Z_n + dz_n,     dz_{n+1} = (2 Z_n + dz_n) dz_n + dc

Arbitrary precision is fixed point on Python's exact integers: a number x is the integer
floor(x * 2**bits). The arithmetic is exact apart from one floor per product, identical on
every machine, and needs no extra library (plain integers are
fast enough: a 20 000-step orbit at 200 bits takes a few tens of milliseconds).

Points are text (``"-0.7436438870371587047521915 0.1318259042053119704931320"``) because a
float64 holds only 16 digits and a 1e-30 zoom needs more than 30.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np

BAILOUT = 65536.0  # |z|^2 at escape; the same radius as the escape kernel
# Pixel sizes under this use perturbation. Measured against
# exact iteration, plain float64 already gets ~1% of pixels visibly wrong at 1e-10 per pixel
# near the set (rounding c itself, amplified over hundreds of steps), perturbation none.
DEEP_BELOW = 1e-10
MARGIN = 48  # bits beyond the pixel size, so the reference outlives rounding
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_POINT = re.compile(rf"^\s*({_NUMBER})\s*[,\s]\s*({_NUMBER})\s*$")

Point = tuple[Fraction, Fraction]


def parse_point(text: str) -> Point:
    """``"re im"`` or ``"re, im"`` in decimal or scientific notation, exactly."""
    m = _POINT.match(text)
    if m is None:
        raise ValueError(
            f"a point is two decimal numbers, 're im' (for example '-0.75 0.1'); got {text!r}"
        )
    return Fraction(m.group(1)), Fraction(m.group(2))


def format_point(p: Point, digits: int = 40) -> str:
    """Decimal text for a point, ``digits`` after the point (enough for 1e-30 views)."""

    def one(x: Fraction) -> str:
        scaled = round(x * 10**digits)
        sign = "-" if scaled < 0 else ""
        s = str(abs(scaled)).rjust(digits + 1, "0")
        return f"{sign}{s[:-digits]}.{s[-digits:]}".rstrip("0").rstrip(".")

    return f"{one(p[0])} {one(p[1])}"


def bits_for(pixel: float) -> int:
    """Fixed-point bits for a view whose pixels are ``pixel`` wide, in steps of 32."""
    need = max(0.0, -math.log2(pixel)) + MARGIN
    return max(64, 32 * math.ceil(need / 32))


def to_fixed(x: Fraction, bits: int) -> int:
    return math.floor(x * (1 << bits))


def _to_float(v: int, bits: int) -> float:
    return math.ldexp(float(v), -bits) if abs(v) < (1 << 1000) else math.copysign(math.inf, v)


def reference_orbit(c: Point, bits: int, max_iter: int) -> np.ndarray:
    """Z_0 = 0, Z_{n+1} = Z_n^2 + c in ``bits``-bit fixed point, rounded to float64, shape
    (L + 1, 2). It stops at the first Z_L that escapes (kept: the kernel needs to see it
    leave) or at L = max_iter."""
    one = 1 << bits
    bail = int(BAILOUT) * one
    cr, ci = to_fixed(c[0], bits), to_fixed(c[1], bits)
    zr = zi = 0
    out = [(0.0, 0.0)]
    for _ in range(max_iter):
        zr2, zi2 = (zr * zr) >> bits, (zi * zi) >> bits
        zi = ((zr * zi) >> (bits - 1)) + ci
        zr = zr2 - zi2 + cr
        out.append((_to_float(zr, bits), _to_float(zi, bits)))
        if ((zr * zr + zi * zi) >> bits) > bail:
            break
    return np.asarray(out, dtype=np.float64)


def direct_escape(c: Point, bits: int, max_iter: int) -> float:
    """The smooth escape count of one point, iterated entirely in fixed point: the ground
    truth the perturbation kernel is tested against. -1 when it never escapes."""
    orbit = reference_orbit(c, bits, max_iter)
    zr, zi = orbit[-1]
    r2 = zr * zr + zi * zi
    if len(orbit) - 1 >= max_iter and r2 <= BAILOUT:
        return -1.0
    k = len(orbit) - 1
    return float(k + 1.0 - math.log2(0.5 * math.log(r2)))


def escapes_at(orbit: np.ndarray, max_iter: int) -> int | None:
    """The step at which a reference orbit escaped, or None if it lasted ``max_iter``."""
    zr, zi = orbit[-1]
    return len(orbit) - 1 if zr * zr + zi * zi > BAILOUT else None
