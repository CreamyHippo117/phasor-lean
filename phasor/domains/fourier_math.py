"""Fourier numerics: pure NumPy.

Epicycles from a closed curve, series partial sums (and their Gibbs overshoot), windows and
leakage, the spectrum of a few sinusoids, spectrograms, and FFT convolution (the convolution
theorem: convolving in time is multiplying spectra).
"""

from __future__ import annotations

import math

import numpy as np

WINDOWS = ("rect", "hann", "hamming", "blackman")
SERIES = ("square", "sawtooth", "triangle")


# ---------------------------------------------------------------- epicycles


def epicycle_coefficients(points, terms: int) -> list[tuple[int, complex]]:
    """The ``terms`` biggest Fourier coefficients ``(k, c_k)`` of a closed curve given as
    points (complex numbers or (N, 2) / (N, 3) rows, resampled uniformly by index): the curve
    is ``sum c_k e^{2 pi i k s}`` for ``s`` in [0, 1). Largest circle first (k = 0 is the
    centre), so truncating keeps the biggest circles."""
    z = np.asarray(points)
    if z.ndim == 2:
        z = z[:, 0] + 1j * z[:, 1]
    z = z.astype(np.complex128)
    n = len(z)
    if n < 4:
        raise ValueError("a curve needs at least 4 points")
    spec = np.fft.fft(z) / n
    k = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    order = sorted(range(n), key=lambda i: (-abs(spec[i]), abs(k[i]), k[i]))
    chosen = sorted(order[: max(1, min(terms, n))], key=lambda i: (-abs(spec[i]), abs(k[i]), k[i]))
    return [(int(k[i]), complex(spec[i])) for i in chosen]


def epicycle_chain(coeffs: list[tuple[int, complex]], s: float) -> np.ndarray:
    """The joints of the chain of circles at parameter ``s``: element 0 is the first centre,
    the last is the tip (the point the full series draws at ``s``). Complex array."""
    parts = np.array([c * np.exp(2j * math.pi * k * s) for k, c in coeffs])
    return np.concatenate([[0j], np.cumsum(parts)])


def epicycle_curve(coeffs: list[tuple[int, complex]], samples: int = 400) -> np.ndarray:
    k = np.array([k for k, _ in coeffs])
    c = np.array([c for _, c in coeffs])
    s = np.linspace(0.0, 1.0, samples, endpoint=False)
    return (c[None, :] * np.exp(2j * math.pi * np.outer(s, k))).sum(axis=1)


# ---------------------------------------------------------------- series


def series_partial_sum(kind: str, terms: int, x) -> np.ndarray:
    """Partial sum of the Fourier series of a unit-amplitude wave with period 2 pi, using
    ``terms`` harmonics. ``square``: +-1; ``sawtooth``: rising from -1 to 1; ``triangle``:
    between -1 and 1."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    if kind == "square":
        for j in range(1, terms + 1):
            m = 2 * j - 1
            out += 4 / (math.pi * m) * np.sin(m * x)
    elif kind == "sawtooth":
        for m in range(1, terms + 1):
            out += -2 / math.pi * (-1) ** m / m * np.sin(m * x)
    elif kind == "triangle":
        for j in range(terms):
            m = 2 * j + 1
            out += 8 / (math.pi**2) * (-1) ** j / m**2 * np.sin(m * x)
    else:
        raise ValueError(f"unknown series {kind!r}; series are {SERIES}")
    return out


def series_target(kind: str, x) -> np.ndarray:
    """The wave the series converges to."""
    x = np.mod(np.asarray(x, dtype=np.float64) + math.pi, 2 * math.pi) - math.pi
    if kind == "square":
        return np.where(x > 0, 1.0, np.where(x < 0, -1.0, 0.0))
    if kind == "sawtooth":
        return x / math.pi
    if kind == "triangle":
        return np.where(x >= 0, 1.0, -1.0) * 0 + (2 / math.pi) * (
            np.where(np.abs(x) <= math.pi / 2, x, np.sign(x) * math.pi - x)
        )
    raise ValueError(f"unknown series {kind!r}; series are {SERIES}")


def gibbs_overshoot(kind: str, terms: int, samples: int = 20000) -> float:
    """How far the partial sum rises above the wave next to its jump, as a share of the jump.
    For the square wave this tends to about 0.0895 however many terms you add (Gibbs); the
    triangle wave has no jump and no overshoot."""
    if kind == "triangle":
        return 0.0
    x = np.linspace(0.0, math.pi / 2, samples)
    s = series_partial_sum(kind, terms, x)
    return float((s.max() - 1.0) / 2.0)


# ---------------------------------------------------------------- windows, spectrum, spectrogram


def window(name: str, n: int) -> np.ndarray:
    if name == "rect":
        return np.ones(n)
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    raise ValueError(f"unknown window {name!r}; windows are {WINDOWS}")


def sinusoids(freqs, amps, n: int, fs: float) -> np.ndarray:
    t = np.arange(n) / fs
    return sum((a * np.sin(2 * math.pi * f * t) for f, a in zip(freqs, amps, strict=True)),
               np.zeros(n))  # fmt: skip


def wound_signal(signal, freq, fs: float) -> np.ndarray:
    """Multiply samples by the clockwise winding phasor at each test frequency."""
    t = np.arange(len(signal)) / fs
    return np.asarray(signal) * np.exp(-2j * np.pi * np.asarray(freq)[..., None] * t)


def winding_mean(signal, freq, fs: float):
    """Centroid of the wound samples, equal to DFT / N at the FFT bins."""
    return wound_signal(signal, freq, fs).mean(axis=-1)


def spectrum(freqs, amps, win: str, n: int, fs: float) -> dict[str, np.ndarray]:
    """Magnitude spectrum of a few sinusoids seen through a window (amplitude-corrected so a
    tone on a bin reads its true amplitude). Returns ``freq`` (Hz) and ``mag``."""
    x = sinusoids(freqs, amps, n, fs)
    w = window(win, n)
    mag = np.abs(np.fft.rfft(x * w)) / (w.sum() / 2)
    return {"freq": np.fft.rfftfreq(n, 1 / fs), "mag": mag}


def leakage(freqs, amps, win: str, n: int, fs: float, guard: int = 2) -> float:
    """The share of spectral energy that is *not* within ``guard`` bins of a true tone: how
    much the window smears the line spectrum. Zero for a tone exactly on a bin with a
    rectangular window; large for an off-bin tone."""
    sp = spectrum(freqs, amps, win, n, fs)
    e = sp["mag"] ** 2
    keep = np.zeros(len(e), dtype=bool)
    df = fs / n
    for f in freqs:
        b = round(f / df)
        keep[max(b - guard, 0) : b + guard + 1] = True
    return float(e[~keep].sum() / max(e.sum(), 1e-300))


def chirp(f0: float, f1: float, duration: float, fs: float) -> np.ndarray:
    n = int(duration * fs)
    t = np.arange(n) / fs
    k = (f1 - f0) / duration
    return np.sin(2 * math.pi * (f0 * t + 0.5 * k * t**2))


def sampled(fn, duration: float, fs: float) -> np.ndarray:
    """``fn(t)`` (a scalar function of seconds) at ``t = k / fs``, ``duration * fs`` samples."""
    n = int(duration * fs)
    return np.array([float(fn(k / fs)) for k in range(n)], dtype=np.float64)


def spectrogram(x, fs: float, nwin: int = 256, hop: int | None = None,
                win: str = "hann") -> dict[str, np.ndarray]:  # fmt: skip
    """Short-time Fourier magnitudes: ``mag`` has shape (frequencies, frames), amplitude-
    corrected so a tone on a bin reads its amplitude. ``hop`` defaults to ``nwin // 4``;
    ``win`` is one of ``WINDOWS``. A signal shorter than the window is zero-padded to one."""
    x = np.asarray(x, dtype=np.float64)
    hop = hop or nwin // 4
    w = window(win, nwin)
    if len(x) < nwin:
        x = np.pad(x, (0, nwin - len(x)))
    frames = range(0, len(x) - nwin + 1, hop)
    mag = np.array([np.abs(np.fft.rfft(x[i : i + nwin] * w)) for i in frames]).T
    times = np.array([(i + nwin / 2) / fs for i in frames])
    return {"freq": np.fft.rfftfreq(nwin, 1 / fs), "time": times, "mag": mag / (w.sum() / 2)}


def dominant_frequency(sg: dict[str, np.ndarray], t: float) -> float:
    """The strongest frequency (Hz) in the frame nearest time ``t``, refined between bins by
    a parabola through the peak and its two neighbours."""
    col = sg["mag"][:, int(np.argmin(np.abs(sg["time"] - t)))]
    k = int(np.argmax(col))
    df = float(sg["freq"][1] - sg["freq"][0]) if len(sg["freq"]) > 1 else 0.0
    if 0 < k < len(col) - 1:
        a, b, c = np.log(np.maximum(col[k - 1 : k + 2], 1e-300))
        denom = a - 2 * b + c
        if denom < 0:
            return float(sg["freq"][k] + 0.5 * (a - c) / denom * df)
    return float(sg["freq"][k])


# ---------------------------------------------------------------- convolution


def convolve(a, b) -> np.ndarray:
    """Linear convolution by FFT (the convolution theorem): multiply the spectra."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)[:n]


# ---------------------------------------------------------------- sampling and aliasing
#
# A tone is ``a sin(2 pi f t + phase)``, as in ``sinusoids``; samples are taken at ``t = k / fs``.


def fold(f, fs: float) -> np.ndarray:
    """The signed frequency a tone appears at once sampled at ``fs``: ``f`` moved by a whole
    number of ``fs`` into [-fs / 2, fs / 2). Per sample the tone's phasor turns ``fold / fs`` of
    a turn the short way round: below zero it seems to turn backwards (the wagon wheel)."""
    f = np.asarray(f, dtype=np.float64)
    return np.mod(f + fs / 2, fs) - fs / 2


def alias_tone(
    f: float, fs: float, amp: float = 1.0, phase: float = 0.0
) -> tuple[float, float, float]:
    """The lowest-frequency sinusoid through the samples of ``amp sin(2 pi f t + phase)`` at
    ``t = k / fs``, as ``(frequency, amplitude, phase)`` with the frequency in [0, fs / 2].

    With ``r = f mod fs``: below ``fs / 2`` it is the same tone at ``r``; above, the tone at
    ``fs - r`` with its phase reflected (``pi - phase``: the folded wave runs the other way);
    exactly at ``fs / 2`` every sample is ``amp sin(phase) (-1)^k``, so the alias is the cosine
    ``amp sin(phase) cos(pi fs t)`` (amplitude ``amp sin(phase)``, phase pi / 2). A whole multiple
    of ``fs`` aliases to the constant ``amp sin(phase)`` (frequency 0)."""
    if fs <= 0:
        raise ValueError("the sample rate must be positive")
    if f < 0:
        f, amp, phase = -f, -amp, -phase  # sin(-x) = -sin(x)
    r = math.fmod(f, fs)
    tol = 1e-12 * max(fs, f)
    if r > fs - tol:  # a whole multiple, up to rounding
        r = 0.0
    half = fs / 2
    if abs(r - half) <= tol:
        return half, amp * math.sin(phase), math.pi / 2
    if r < half:
        fa, pa = r, phase
    else:
        fa, pa = fs - r, math.pi - phase
    return fa, amp, math.remainder(pa, 2 * math.pi)


def tone_sum(freqs, amps, phases, t) -> np.ndarray:
    """``sum a sin(2 pi f t + phase)`` at the times ``t`` (seconds)."""
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t)
    for f, a, ph in zip(freqs, amps, phases, strict=True):
        out = out + a * np.sin(2 * math.pi * f * t + ph)
    return out


def aliases(freqs, amps, phases, fs: float) -> list[tuple[float, float, float]]:
    """``alias_tone`` of each tone: what the samples of the sum show, tone by tone."""
    return [alias_tone(f, fs, a, ph) for f, a, ph in zip(freqs, amps, phases, strict=True)]


def alias_signal(freqs, amps, phases, fs: float, t) -> np.ndarray:
    """The sum of the aliases at the times ``t``: the slowest wave through every sample of the
    sum of tones (the band-limited one, ``sinc_reconstruct`` of infinitely many samples)."""
    al = aliases(freqs, amps, phases, fs)
    return tone_sum([a[0] for a in al], [a[1] for a in al], [a[2] for a in al], t)


def sample_times(fs: float, start: float, stop: float) -> np.ndarray:
    """The sampling instants ``k / fs`` in [start, stop]."""
    k0, k1 = math.ceil(start * fs - 1e-9), math.floor(stop * fs + 1e-9)
    return np.arange(k0, k1 + 1) / fs


def sinc_reconstruct(x, fs: float, t, t0: float = 0.0) -> np.ndarray:
    """Whittaker-Shannon interpolation: ``sum_n x[n] sinc(fs (t - t0) - n)`` for samples
    ``x[n]`` taken at ``t0 + n / fs``. It is the original signal if that held no frequency at
    or above ``fs / 2`` (up to the truncation of the sum near its ends), and the aliases if
    it did."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    u = fs * (t - t0)
    return np.sinc(u[..., None] - np.arange(len(x))) @ x
