"""Digital filters: pure NumPy, no drawing.

A filter is its zeros, poles and gain in the z plane, H(z) = k prod(z - z_i) / prod(z - p_i)
(causal: no more zeros than poles). Butterworth designs (low-pass, high-pass, band-pass,
band-stop) go through the analog prototype, the frequency transform and the bilinear transform
with the cutoff pre-warped, so the half-power frequency lands exactly where it was asked:
|H(e^{j w})|^2 = 1 / (1 + (tan(w/2) / tan(w_c/2))^(2 n)) for a low-pass. A notch puts zeros on
the unit circle at the notch frequency and poles just inside it. The response is H on the unit
circle, ``z = e^{j 2 pi f / fs}``; signals are filtered as cascaded second-order sections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

KINDS = ("lowpass", "highpass", "bandpass", "bandstop", "notch", "custom")


class FilterError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class Filter:
    """Zeros, poles and gain of a causal filter with real coefficients, at sample rate ``fs``."""

    zeros: np.ndarray
    poles: np.ndarray
    gain: float
    fs: float

    @property
    def order(self) -> int:
        return len(self.poles)

    @property
    def stable(self) -> bool:
        return bool(len(self.poles) == 0 or np.abs(self.poles).max() < 1.0)

    def at(self, z) -> np.ndarray:
        """H(z) for complex ``z`` (any shape)."""
        z = np.asarray(z, dtype=np.complex128)
        num = np.ones_like(z)
        for r in self.zeros:
            num = num * (z - r)
        den = np.ones_like(z)
        for r in self.poles:
            den = den * (z - r)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.gain * num / den

    def response(self, freq) -> np.ndarray:
        """H on the unit circle at ``freq`` Hz."""
        return self.at(np.exp(2j * math.pi * np.asarray(freq, dtype=np.float64) / self.fs))

    def sos(self) -> np.ndarray:
        return signal.zpk2sos(self.zeros, self.poles, self.gain)

    def apply(self, x) -> np.ndarray:
        """Filter ``x`` (from rest) with the difference equation, as cascaded biquads."""
        return signal.sosfilt(self.sos(), np.asarray(x, dtype=np.float64))


def _conjugate_closed(r: np.ndarray, what: str) -> None:
    for v in r:
        if abs(v.imag) < 1e-9:
            continue
        if not np.any(np.abs(r - np.conj(v)) < 1e-6):
            raise FilterError(f"{what} must come in conjugate pairs ({v:.4g} has no partner)")


def custom(zeros, poles, gain: float, fs: float) -> Filter:
    """A filter from given zeros and poles (complex numbers or (re, im) pairs)."""
    z = _as_complex(zeros)
    p = _as_complex(poles)
    if len(z) > len(p):
        raise FilterError("a causal filter needs at least as many poles as zeros")
    _conjugate_closed(z, "zeros")
    _conjugate_closed(p, "poles")
    return Filter(z, p, float(gain), float(fs))


def _as_complex(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(-1, 2) if len(v) else np.zeros((0, 2))
    return a[:, 0] + 1j * a[:, 1]


# ---------------------------------------------------------------- Butterworth design


def _prototype(n: int) -> np.ndarray:
    """Poles of the analog Butterworth low-pass of order ``n`` with cutoff 1 rad/s."""
    k = np.arange(n)
    return np.exp(1j * math.pi * (2 * k + n + 1) / (2 * n))


def _prewarp(f: float, fs: float) -> float:
    """The analog frequency (rad/s) the bilinear transform sends to ``f`` Hz."""
    return 2 * fs * math.tan(math.pi * f / fs)


def _bilinear(z, p, k, fs):
    fs2 = 2 * fs
    degree = len(p) - len(z)
    zd = (fs2 + z) / (fs2 - z)
    pd = (fs2 + p) / (fs2 - p)
    zd = np.append(zd, -np.ones(degree))
    kd = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))
    return zd, pd, kd


def _lp2lp(z, p, k, wo):
    return z * wo, p * wo, k * wo ** (len(p) - len(z))


def _lp2hp(z, p, k, wo):
    degree = len(p) - len(z)
    zh = np.append(wo / z, np.zeros(degree))
    return zh, wo / p, k * np.real(np.prod(-z) / np.prod(-p))


def _lp2bp(z, p, k, wo, bw):
    degree = len(p) - len(z)
    zl, pl = z * bw / 2, p * bw / 2
    zb = np.concatenate([zl + np.sqrt(zl**2 - wo**2 + 0j), zl - np.sqrt(zl**2 - wo**2 + 0j)])
    pb = np.concatenate([pl + np.sqrt(pl**2 - wo**2 + 0j), pl - np.sqrt(pl**2 - wo**2 + 0j)])
    return np.append(zb, np.zeros(degree)), pb, k * bw**degree


def _lp2bs(z, p, k, wo, bw):
    degree = len(p) - len(z)
    zh, ph = (bw / 2) / z, (bw / 2) / p
    zb = np.concatenate([zh + np.sqrt(zh**2 - wo**2 + 0j), zh - np.sqrt(zh**2 - wo**2 + 0j)])
    pb = np.concatenate([ph + np.sqrt(ph**2 - wo**2 + 0j), ph - np.sqrt(ph**2 - wo**2 + 0j)])
    zb = np.concatenate([zb, np.full(degree, 1j * wo), np.full(degree, -1j * wo)])
    return zb, pb, k * np.real(np.prod(-z) / np.prod(-p))


def butterworth(kind: str, order: int, cutoff: float, fs: float,
                cutoff2: float | None = None) -> Filter:  # fmt: skip
    """The Butterworth filter of ``order`` (per band edge for the band kinds) with half-power
    frequency ``cutoff`` Hz (and ``cutoff2`` for the band kinds, the upper edge)."""
    if kind not in ("lowpass", "highpass", "bandpass", "bandstop"):
        raise FilterError(f"unknown kind {kind!r}; kinds are {KINDS}")
    nyq = fs / 2
    if not 0 < cutoff < nyq:
        raise FilterError(f"cutoff {cutoff} Hz must be between 0 and {nyq} Hz (fs / 2)")
    p = _prototype(order)
    z = np.zeros(0, dtype=np.complex128)
    k = 1.0
    wc = _prewarp(cutoff, fs)
    if kind == "lowpass":
        z, p, k = _lp2lp(z, p, k, wc)
    elif kind == "highpass":
        z, p, k = _lp2hp(z, p, k, wc)
    else:
        if cutoff2 is None or not cutoff < cutoff2 < nyq:
            raise FilterError(f"a band needs {cutoff} < cutoff2 < {nyq} Hz")
        w2 = _prewarp(cutoff2, fs)
        wo, bw = math.sqrt(wc * w2), w2 - wc
        z, p, k = (_lp2bp if kind == "bandpass" else _lp2bs)(z, p, k, wo, bw)
    z, p, k = _bilinear(z, p, k, fs)
    return Filter(z.astype(np.complex128), p.astype(np.complex128), float(k), float(fs))


def notch(freq: float, radius: float, fs: float) -> Filter:
    """Zeros on the unit circle at ``freq`` Hz, poles at ``radius`` times them; unit gain at
    DC. The closer the radius is to 1, the narrower the notch."""
    if not 0 < freq < fs / 2:
        raise FilterError(f"notch frequency {freq} Hz must be between 0 and {fs / 2} Hz")
    if not 0 < radius < 1:
        raise FilterError("the pole radius must be between 0 and 1")
    e = np.exp(2j * math.pi * freq / fs)
    z = np.array([e, np.conj(e)])
    p = radius * z
    k = float(np.real(np.prod(1 - p) / np.prod(1 - z)))
    return Filter(z, p, k, float(fs))


def design(kind: str, order: int, cutoff: float, fs: float, cutoff2: float | None = None,
           radius: float = 0.95, zeros=(), poles=(), gain: float = 1.0) -> Filter:  # fmt: skip
    if kind == "notch":
        return notch(cutoff, radius, fs)
    if kind == "custom":
        return custom(zeros, poles, gain, fs)
    return butterworth(kind, order, cutoff, fs, cutoff2)


def cutoffs(kind: str, cutoff: float, cutoff2: float) -> tuple[float, ...]:
    """The frequencies the design is specified at (what the plots mark)."""
    if kind in ("bandpass", "bandstop"):
        return (cutoff, cutoff2)
    return () if kind == "custom" else (cutoff,)


# ---------------------------------------------------------------- the response


def gain_db(h, floor: float = -120.0) -> np.ndarray:
    """20 log10 |h|, never below ``floor`` (a zero on the circle is minus infinity)."""
    return np.maximum(20 * np.log10(np.maximum(np.abs(h), 1e-300)), floor)


def phase(h) -> np.ndarray:
    """The unwrapped phase of ``h`` (radians), starting from the first sample's wrapped value."""
    return np.unwrap(np.angle(h))


def curve_runs(x, y, jump: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a curve where ``y`` jumps by more than ``jump`` (a zero on the circle flips the
    phase by pi: do not draw the jump) or is not finite."""
    x, y = np.asarray(x), np.asarray(y)
    bad = ~np.isfinite(y[1:]) | ~np.isfinite(y[:-1]) | (np.abs(np.diff(y)) > jump)
    cut = np.flatnonzero(bad) + 1
    return [(x[a:b], y[a:b]) for a, b in zip([0, *cut], [*cut, len(x)], strict=True) if b - a > 1]


def minus_3db(f: Filter, lo: float = 0.0, hi: float | None = None, n: int = 4096):
    """Frequencies (Hz) where the magnitude crosses 1/sqrt(2) of its peak, found on a fine grid
    and refined by interpolation."""
    hi = f.fs / 2 if hi is None else hi
    fr = np.linspace(lo, hi, n)
    m = np.abs(f.response(fr))
    target = m.max() / math.sqrt(2)
    s = np.sign(m - target)
    out = []
    for i in np.flatnonzero(s[:-1] * s[1:] < 0):
        t = (target - m[i]) / (m[i + 1] - m[i])
        out.append(float(fr[i] + t * (fr[i + 1] - fr[i])))
    return out


# ---------------------------------------------------------------- the transfer function as text


def _num(c: complex) -> str:
    return f"({c.real!r}{'+' if c.imag >= 0 else '-'}{abs(c.imag)!r}j)"


def _factors(roots, shifted: str) -> str:
    """``(shifted - r)`` per distinct root, with its multiplicity as a power."""
    seen: dict[tuple[float, float], list] = {}
    for r in roots:
        key = (round(float(r.real), 9) + 0.0, round(float(r.imag), 9) + 0.0)
        seen.setdefault(key, []).append(r)
    out = []
    for rs in seen.values():
        r = rs[0]
        term = f"({shifted} - {_num(complex(r))})"
        out.append(term if len(rs) == 1 else f"{term}**{len(rs)}")
    return "*".join(out) if out else "1"


def transfer_source(f: Filter, pos: tuple[float, float] = (0.0, 0.0), scale: float = 1.0) -> str:
    """H as the text ``cx.DomainColoring`` reads, drawn so the z plane sits at world ``pos``
    with ``scale`` world units per unit of z: the function of the world point w is
    H((w - pos) / scale), so zeros and poles appear at ``pos + scale z``."""
    c = complex(*pos)
    zeros = c + scale * f.zeros
    poles = c + scale * f.poles
    k = f.gain * scale ** (len(f.poles) - len(f.zeros))
    return f"{k!r}*{_factors(zeros, 'z')}/({_factors(poles, 'z')})"


# ---------------------------------------------------------------- signals and spectra


def tone_amplitudes_after(f: Filter, freqs, amps) -> np.ndarray:
    """What each steady-state tone's amplitude becomes: ``a |H(f)|``."""
    return np.asarray(amps, dtype=np.float64) * np.abs(f.response(np.asarray(freqs)))


def filtered_tones(f: Filter, freqs, amps, n: int):
    """A sum of sinusoids and what the filter makes of it, ``n`` samples at ``f.fs``."""
    t = np.arange(n) / f.fs
    x = sum((a * np.sin(2 * math.pi * fr * t) for fr, a in zip(freqs, amps, strict=True)),
            np.zeros(n))  # fmt: skip
    return x, f.apply(x)
