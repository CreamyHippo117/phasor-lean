"""The sound mix: the voice, beds and cues on one stereo bus.
Pure NumPy/SciPy (and ``soundfile`` to decode files); no GPU, GL or video library.

A document's ``sound`` section (core.schema ``Bed``, ``Cue``) is resolved to seconds by
``compile_document`` through the TimeMap, like tracks, so re-narrating moves every cue and
every bed end anchored to a marker. ``mix`` turns that into samples, a pure function of the
document and the files it names:

- **A narration-only scene is the narration, untouched**: the int16 track ``mix_track``
  makes, at its own rate, copied to both channels. Nothing below runs.
- Otherwise everything runs on a **stereo float32 bus at 48 kHz**. Files at another rate
  are resampled once (polyphase, cached per file version); mono is centred.
- **Loudness, not peaks** (ITU-R BS.1770-4: K-weighting, 400 ms blocks, gates at -70 LUFS
  and -10 LU). The voice is set to the theme's ``[sound] target``; a bed to ``target + gain``
  (its integrated loudness), a cue to ``target + gain`` (its loudest 400 ms), ``gain``
  defaulting to ``[sound] bed`` / ``[sound] cue``.
- **Ducking** from the narration's word times, not from the signal: while a block speaks,
  beds dip by ``[sound] duck`` dB, starting ``attack`` s before the first word and
  recovering over ``release`` s after the last (smoothstep ramps). Exact and deterministic.
- **Bus**: a final trim brings the whole mix to ``target`` LUFS, then a soft-knee limiter
  (``knee`` dB wide) holds the true peak (4x oversampled) at ``ceiling`` dBTP. Its gain is
  a running minimum then a running mean over the same window, which never exceeds the gain
  each sample needs, so the ceiling holds without a click.
- **Built-in cues** (``tick``, ``chime``, ``whoosh``, ``pop``, ``swell``, ``tone``) are
  synthesized from (cue id, scene seed) through ``rng`` and cached, so a scene needs no
  downloads and two renders agree to the bit.
"""

from __future__ import annotations

import functools
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator  # the type only; generators come from core.rng

from phasor.core import files
from phasor.core.rng import rng
from phasor.core.schema import BUILTIN_SOUNDS, Bed, Cue, Narration

RATE = 48000  # Hz: the bus
BLOCK = 0.4  # s: a loudness block (BS.1770 gating block, and "momentary" loudness)
HOP = 0.1  # s: blocks overlap by 75 %
ABS_GATE = -70.0  # LUFS
REL_GATE = -10.0  # LU below the absolutely gated loudness
OVERSAMPLE = 4  # true-peak estimate (BS.1770-4 Annex 2)
LIMIT_WINDOW = 0.005  # s: the limiter looks this far ahead and holds this long
STRESSED_LETTERS = 4  # lint cue_on_word: a word this long (or longer) carries a stress
MAX_TIMES = 12
PEAK_MARGIN = 6.0  # dB: a chunk this far under a peak floor is not oversampled (``true_peak``)

# Partials of each ``tone`` voice: (ratio to the fundamental, amplitude, decay in 1/s; 0 = steady)
VOICES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "sine": ((1.0, 1.0, 0.0),),
    "soft": ((1.0, 1.0, 0.0), (2.0, 0.35, 0.0), (3.0, 0.12, 0.0), (4.0, 0.05, 0.0)),
    "pluck": tuple((float(k), 1.0 / k, 3.0 + 2.5 * k) for k in range(1, 9)),
    "bell": ((1.0, 1.0, 1.5), (2.76, 0.6, 2.5), (5.4, 0.35, 4.0), (8.93, 0.2, 6.0)),
}
# The knobs a built-in uses when the cue leaves them unset, and its length in seconds.
DEFAULTS: dict[str, dict[str, Any]] = {
    "tick": {"dur": 0.03},
    "chime": {"midi": 84.0, "dur": 1.6},
    "whoosh": {"dur": 0.6},
    "pop": {"midi": 76.0, "dur": 0.09},
    "swell": {"midi": 60.0, "dur": 2.0},
    "tone": {"midi": 69.0, "dur": 0.5, "voice": "sine"},
}
if set(DEFAULTS) != set(BUILTIN_SOUNDS):  # the schema names what is built here
    raise ImportError("time/sound.py and core.schema.BUILTIN_SOUNDS name different built-ins")


@dataclass(frozen=True)
class PlacedCue:
    """A cue with its start resolved to seconds on the scene timeline."""

    id: str
    t: float
    cue: Cue


@dataclass(frozen=True)
class PlacedBed:
    """A bed with its ends resolved to seconds (``t1`` is the scene's end when it has none)."""

    id: str
    t0: float
    t1: float
    bed: Bed


def midi_hz(midi: float) -> float:
    """Equal temperament, A4 (69) = 440 Hz."""
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def builtin_seconds(cue: Cue) -> float:
    """How long a built-in cue plays (known without synthesizing it)."""
    return float(cue.dur if cue.dur is not None else DEFAULTS[cue.sound]["dur"])


# ---------------------------------------------------------------- loudness (BS.1770-4)


@functools.cache
def k_weighting(rate: int) -> np.ndarray:
    """The K-weighting filter (a high shelf, then a high pass) as second-order sections at
    ``rate``. The analogue prototypes are the ones that give BS.1770's 48 kHz table."""
    g, q1, f1 = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    k = math.tan(math.pi * f1 / rate)
    vh = 10 ** (g / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q1 + k * k
    shelf = [(vh + vb * k / q1 + k * k) / a0, 2 * (k * k - vh) / a0,
             (vh - vb * k / q1 + k * k) / a0, 1.0, 2 * (k * k - 1) / a0,
             (1 - k / q1 + k * k) / a0]  # fmt: skip
    q2, f2 = 0.5003270373238773, 38.13547087602444
    k = math.tan(math.pi * f2 / rate)
    a0 = 1 + k / q2 + k * k
    high = [1.0, -2.0, 1.0, 1.0, 2 * (k * k - 1) / a0, (1 - k / q2 + k * k) / a0]
    return np.array([shelf, high])


def _stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


def block_energies(x: np.ndarray, rate: int, hop: float = HOP) -> np.ndarray:
    """The mean square of the K-weighted signal, summed over channels, in each ``BLOCK``
    starting every ``hop`` seconds (a partial last block is dropped, as BS.1770 says)."""
    from scipy.signal import sosfilt

    y = sosfilt(k_weighting(rate), _stereo(x).astype(np.float64), axis=0)
    power = (y * y).sum(axis=1)
    nb, nh = round(BLOCK * rate), max(1, round(hop * rate))
    if len(power) < nb:
        return np.zeros(0)
    c = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, len(power) - nb + 1, nh)
    return (c[starts + nb] - c[starts]) / nb


def lufs(z: np.ndarray | float) -> np.ndarray | float:
    """Loudness of a block energy (LUFS); -inf for silence."""
    with np.errstate(divide="ignore"):
        return -0.691 + 10 * np.log10(np.maximum(z, 0.0))


def gated(z: np.ndarray) -> float:
    """Integrated loudness from block energies: the absolute gate, then the relative one."""
    z = np.asarray(z, dtype=np.float64)
    z = z[lufs(z) > ABS_GATE] if z.size else z
    if not z.size:
        return -math.inf
    z = z[lufs(z) > float(lufs(z.mean())) + REL_GATE]
    return float(lufs(z.mean()))


def integrated(x: np.ndarray, rate: int) -> float:
    """BS.1770-4 integrated loudness of ``x`` (mono is heard on both channels), in LUFS."""
    return gated(block_energies(x, rate))


def momentary_max(x: np.ndarray, rate: int) -> float:
    """The loudest 400 ms of ``x`` (sampled every 10 ms), in LUFS: a one-shot's level. A
    sound shorter than 400 ms is measured as itself followed by silence."""
    x = _stereo(x)
    nb = round(BLOCK * rate)
    if len(x) < nb:
        x = np.concatenate([x, np.zeros((nb - len(x), 2), x.dtype)])
    z = block_energies(x, rate, hop=0.01)
    return float(lufs(z.max())) if z.size else -math.inf


def true_peak(x: np.ndarray, floor: float | None = None) -> np.ndarray:
    """Each sample's true-peak estimate (the largest |value| of the 4x oversampled signal
    around it, or the sample itself), over both channels. Computed in chunks of 4096
    samples; with ``floor``, a chunk whose samples all stay ``PEAK_MARGIN`` dB under it is
    not oversampled (its sample peaks are kept), since no inter-sample peak of real audio
    rises that far over its samples."""
    from scipy.signal import resample_poly

    x = _stereo(x).astype(np.float64)
    n = len(x)
    out = np.abs(x).max(axis=1) if n else np.zeros(0)
    chunk, pad = 4096, 64
    skip = -1.0 if floor is None else floor * 10 ** (-PEAK_MARGIN / 20)
    for a in range(0, n, chunk):
        if out[a : a + chunk].max() < skip:
            continue
        lo, hi = max(0, a - pad), min(n, a + chunk + pad)
        up = resample_poly(x[lo:hi], OVERSAMPLE, 1, axis=0)
        env = np.abs(up).reshape(hi - lo, OVERSAMPLE, 2).max(axis=(1, 2))
        b = min(n, a + chunk)
        out[a:b] = np.maximum(out[a:b], env[a - lo : b - lo])
    return out


def limit(x: np.ndarray, rate: int, ceiling: float, knee: float) -> np.ndarray:
    """A soft-knee limiter: the true peak never passes ``ceiling`` dBTP. Below
    ``ceiling - knee/2`` the signal is untouched; through the knee the reduction grows
    quadratically, above it the peak is held at the ceiling. The gain is smooth (a running
    minimum then a running mean over ``2 LIMIT_WINDOW``), never above what a sample needs."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    with np.errstate(divide="ignore"):
        level = 20 * np.log10(np.maximum(true_peak(x, 10 ** ((ceiling - knee / 2) / 20)), 1e-12))
    over = level - ceiling
    if knee > 0:
        knee_cut = -((over + knee / 2) ** 2) / (2 * knee)
        cut = np.where(over <= -knee / 2, 0.0, np.where(over >= knee / 2, -over, knee_cut))
    else:
        cut = np.minimum(0.0, -over)
    need = 10 ** (cut / 20)
    if np.all(need >= 1.0):
        return x
    size = 2 * max(1, round(LIMIT_WINDOW * rate)) + 1
    gain = uniform_filter1d(minimum_filter1d(need, size, mode="nearest"), size, mode="nearest")
    gain = np.minimum(gain, need)  # float rounding in the mean must never tip it over
    return (x * gain[:, None]).astype(x.dtype)


# ---------------------------------------------------------------- the narration's timing


def speech_spans(narration: Narration | None) -> tuple[tuple[float, float], ...]:
    """Where the narration speaks: each spoken block from its first word to its last (the
    block itself when it has no word times), in seconds on the scene timeline."""
    if narration is None:
        return ()
    spans = []
    for b in narration.blocks:
        if not b.text:
            continue
        if b.words:
            spans.append((b.start + b.words[0].start, b.start + max(w.end for w in b.words)))
        else:
            spans.append((b.start, b.start + b.duration))
    return tuple(spans)


def stressed_onsets(narration: Narration | None) -> tuple[tuple[float, str], ...]:
    """The onset of every word that carries a stress (``STRESSED_LETTERS`` letters or more:
    short function words do not), with the word."""
    if narration is None:
        return ()
    return tuple((b.start + w.start, w.text) for b in narration.blocks for w in b.words
                 if len(re.findall(r"\w", w.text)) >= STRESSED_LETTERS)  # fmt: skip


def duck_envelope(spans: Sequence[tuple[float, float]], n: int, rate: int, depth: float,
                  attack: float, release: float) -> np.ndarray:  # fmt: skip
    """The beds' gain (linear, ``n`` samples): 1 away from speech, ``-depth`` dB during it,
    ramping down over ``attack`` s before each span and back up over ``release`` s after.
    Spans closer than ``attack + release`` are one: the bed stays down between sentences
    instead of pumping."""
    merged: list[list[float]] = []
    for t0, t1 in sorted(spans):
        if merged and t0 - merged[-1][1] < attack + release:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    w = np.zeros(n)
    for t0, t1 in merged:
        lo = max(0, math.floor((t0 - attack) * rate))
        hi = min(n, math.ceil((t1 + release) * rate) + 1)
        if hi <= lo:
            continue
        t = np.arange(lo, hi) / rate
        up = np.clip((t - (t0 - attack)) / attack, 0, 1) if attack > 0 else (t >= t0) * 1.0
        down = np.clip((t1 + release - t) / release, 0, 1) if release > 0 else (t <= t1) * 1.0
        u = np.minimum(up, down)
        w[lo:hi] = np.maximum(w[lo:hi], u * u * (3 - 2 * u))  # smoothstep
    return 10 ** (-depth * w / 20)


# ---------------------------------------------------------------- files


def _stamp(base_dir: str | Path, path: str) -> tuple[str, int, int] | None:
    try:
        p = files.resolve(base_dir, path)
        st = os.stat(p)
    except (OSError, ValueError):
        return None
    return str(p), st.st_mtime_ns, st.st_size


@functools.lru_cache(maxsize=32)
def _decode(path: str, mtime_ns: int, size: int) -> tuple[np.ndarray, int] | None:
    """A file as float32 stereo and its rate; None when it cannot be decoded."""
    import soundfile

    try:
        data, rate = soundfile.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        return None
    out = _stereo(data)
    out.setflags(write=False)
    return out, int(rate)


@functools.lru_cache(maxsize=32)
def _at_rate(path: str, mtime_ns: int, size: int, rate: int) -> np.ndarray | None:
    """The file at ``rate`` (polyphase resampling, cached per file version)."""
    got = _decode(path, mtime_ns, size)
    if got is None:
        return None
    x, src = got
    if src != rate:
        from scipy.signal import resample_poly

        g = math.gcd(rate, src)
        x = resample_poly(x.astype(np.float64), rate // g, src // g, axis=0).astype(np.float32)
        x.setflags(write=False)
    return x


@functools.lru_cache(maxsize=64)
def _file_level(path: str, mtime_ns: int, size: int, rate: int, one_shot: bool) -> float:
    x = _at_rate(path, mtime_ns, size, rate)
    if x is None:
        return -math.inf
    return momentary_max(x, rate) if one_shot else integrated(x, rate)


def read_audio(base_dir: str | Path, path: str, rate: int = RATE) -> np.ndarray | None:
    """The file ``path`` names in the scene's folder as float32 stereo at ``rate``; None when
    it is missing or cannot be decoded (lint says which). Re-read when the file changes."""
    stamp = _stamp(base_dir, path)
    return None if stamp is None else _at_rate(*stamp, rate)


def file_seconds(base_dir: str | Path, path: str) -> float | None:
    """How long the file plays; None when it is missing or cannot be decoded."""
    stamp = _stamp(base_dir, path)
    got = None if stamp is None else _decode(*stamp)
    return None if got is None else len(got[0]) / got[1]


# ---------------------------------------------------------------- built-in effects


def _ramp(n: int, attack: int, release: int) -> np.ndarray:
    """Linear fade-in from exactly 0 over ``attack`` samples, fade-out to 0 over ``release``."""
    env = np.ones(n)
    a, r = max(1, min(n // 2, attack)), max(1, min(n // 2, release))
    env[:a] = np.arange(a) / a
    env[n - r :] = np.minimum(env[n - r :], np.arange(r - 1, -1, -1) / r)
    return env


def _band_noise(noise: np.ndarray, rate: int, center: np.ndarray, octaves: float) -> np.ndarray:
    """``noise`` through a band (Gaussian in log frequency, ``octaves`` wide) whose centre
    moves: ``center`` gives it in Hz at each 1024-sample frame (Hann, 50 % overlap-add)."""
    frame, hop = 1024, 512
    n = len(noise)
    count = max(1, math.ceil(n / hop) + 1)
    padded = np.concatenate([np.zeros(hop), noise, np.zeros(count * hop + frame)])
    idx = np.arange(count)[:, None] * hop + np.arange(frame)[None, :]
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)  # periodic: sums to 1
    spec = np.fft.rfft(padded[idx] * win, axis=1)
    f = np.maximum(np.fft.rfftfreq(frame, 1 / rate), 1.0)
    c = np.interp(np.arange(count) * hop, np.linspace(0, n, len(center)), center)[:, None]
    mask = np.exp(-0.5 * (np.log2(f[None, :] / c) / octaves) ** 2)
    frames = np.fft.irfft(spec * mask, n=frame, axis=1)
    out = np.zeros(len(padded))
    np.add.at(out, idx.ravel(), frames.ravel())
    return out[hop : hop + n]


def _tone(midi: float, dur: float, voice: str, rate: int) -> np.ndarray:
    n = max(1, round(dur * rate))
    t = np.arange(n) / rate
    f0 = midi_hz(midi)
    x = np.zeros(n)
    for ratio, amp, decay in VOICES[voice]:
        if f0 * ratio < rate / 2:  # no partial above Nyquist (it would alias)
            part = amp * np.sin(2.0 * np.pi * f0 * ratio * t)
            x += part * np.exp(-decay * t) if decay else part
    return x * _ramp(n, round(0.005 * rate), round(0.030 * rate))


def _synth(name: str, gen: Generator, midi: float, dur: float, voice: str,
           rate: int) -> np.ndarray:  # fmt: skip
    n = max(1, round(dur * rate))
    t = np.arange(n) / rate
    if name == "tick":  # a click and a short high ring
        x = 0.5 * gen.standard_normal(n) * np.exp(-t / 0.0015)
        x += np.sin(2 * np.pi * 3200 * t) * np.exp(-t / 0.006)
        return x * _ramp(n, round(0.0005 * rate), round(0.003 * rate))
    if name == "chime":  # a small bell, every partial a hair detuned
        f0, x = midi_hz(midi), np.zeros(n)
        for ratio, amp, decay in ((1.0, 1.0, 2.2), (2.76, 0.5, 3.5), (5.4, 0.3, 5.0),
                                  (8.93, 0.15, 7.0)):  # fmt: skip
            f = f0 * ratio * 2 ** (gen.uniform(-3, 3) / 1200)
            if f < rate / 2:
                phase = gen.uniform(0, 2 * np.pi)
                x += amp * np.sin(2 * np.pi * f * t + phase) * np.exp(-decay * t)
        return x * _ramp(n, round(0.003 * rate), round(0.05 * rate))
    if name == "whoosh":  # noise through a band sweeping up, loudest at 70 %
        u = t / dur
        center = np.geomspace(350, 2800, 64)
        x = _band_noise(gen.standard_normal(n), rate, center, 0.6)
        env = np.where(u < 0.7, (u / 0.7) ** 2, np.cos(0.5 * np.pi * (u - 0.7) / 0.3) ** 2)
        return x * env * _ramp(n, 1, round(0.01 * rate))
    if name == "pop":  # a pitch that drops an octave into its note, very short
        f = midi_hz(midi) * (1 + np.exp(-t / 0.008))
        x = np.sin(2 * np.pi * np.cumsum(f) / rate) * np.exp(-t / 0.025)
        x += 0.15 * gen.standard_normal(n) * np.exp(-t / 0.001)
        return x * _ramp(n, round(0.001 * rate), round(0.01 * rate))
    if name == "swell":  # an open fifth and a rising breath of noise, growing to the end
        f0, x = midi_hz(midi), np.zeros(n)
        for ratio, amp in ((1.0, 1.0), (1.5, 0.6), (2.0, 0.4)):
            for cents in (-4.0, 4.0):
                f = f0 * ratio * 2 ** (cents / 1200)
                x += amp * np.sin(2 * np.pi * f * t + gen.uniform(0, 2 * np.pi))
        air = _band_noise(gen.standard_normal(n), rate, np.geomspace(300, 1500, 64), 0.8)
        x = x / 4 + 0.6 * air / max(1e-9, float(np.abs(air).max()))
        return x * (t / dur) ** 2.5 * _ramp(n, 1, round(0.06 * rate))
    return _tone(midi, dur, voice, rate)


@functools.lru_cache(maxsize=256)
def _builtin(name: str, cue_id: str, seed: int, midi: float, dur: float, voice: str,
             rate: int) -> np.ndarray:  # fmt: skip
    x = _synth(name, rng(cue_id, seed), midi, dur, voice, rate)
    peak = float(np.abs(x).max())
    x = (x / peak if peak > 0 else x).astype(np.float32)
    x.setflags(write=False)
    return x


def builtin(cue: Cue, seed: int, rate: int = RATE) -> np.ndarray:
    """A built-in cue as float32 mono at ``rate``, peak 1: a pure function of the cue (its
    id, sound and knobs) and the scene's seed, cached."""
    d = DEFAULTS[cue.sound]
    midi = float(cue.midi if cue.midi is not None else d.get("midi", 69.0))
    voice = cue.voice or d.get("voice", "sine")
    return _builtin(cue.sound, cue.id, seed, midi, builtin_seconds(cue), voice, rate)


@functools.lru_cache(maxsize=256)
def _builtin_level(name: str, cue_id: str, seed: int, midi: float, dur: float, voice: str,
                   rate: int) -> float:  # fmt: skip
    return momentary_max(_builtin(name, cue_id, seed, midi, dur, voice, rate), rate)


def _cue_source(cue: Cue, base_dir: str | Path, seed: int) -> tuple[np.ndarray, float] | None:
    """A cue's samples (mono or stereo, at ``RATE``) and their level in LUFS."""
    if cue.builtin:
        x = builtin(cue, seed)
        d = DEFAULTS[cue.sound]
        midi = float(cue.midi if cue.midi is not None else d.get("midi", 69.0))
        level = _builtin_level(cue.sound, cue.id, seed, midi, builtin_seconds(cue),
                               cue.voice or d.get("voice", "sine"), RATE)  # fmt: skip
        return x, level
    stamp = _stamp(base_dir, cue.sound)
    x = None if stamp is None else _at_rate(*stamp, RATE)
    return None if x is None else (x, _file_level(*stamp, RATE, True))


# ---------------------------------------------------------------- the mix


@dataclass(frozen=True)
class Lane:
    """What one source contributes, measured: its level before gain (LUFS), the gain that
    set it, and its block energies as mixed (before the final trim), every ``HOP``."""

    id: str
    kind: str  # "voice", "bed" or "cue"
    level: float
    gain_db: float
    energies: np.ndarray = field(repr=False)
    t0: float = 0.0
    t1: float = 0.0
    peak_t: float | None = None  # cues: when the cue is loudest

    def loudness(self, t0: float | None = None, t1: float | None = None) -> float:
        """Integrated loudness of the lane as mixed between ``t0`` and ``t1`` (whole lane by
        default), before the final trim. The 400 ms blocks inside the span count; a span
        shorter than a block is measured by the block centred on it."""
        z = self.energies
        if t0 is None or t1 is None or not z.size:
            return gated(z)
        starts = np.arange(len(z)) * HOP
        inside = z[(starts >= t0 - 1e-9) & (starts + BLOCK <= t1 + 1e-9)]
        if not inside.size:
            k = int(np.clip(round(((t0 + t1) / 2 - BLOCK / 2) / HOP), 0, len(z) - 1))
            inside = z[k : k + 1]
        return gated(inside)


@dataclass(frozen=True)
class Mix:
    """The scene's audio: int16 stereo ``samples`` at ``rate``, and how it was made."""

    samples: np.ndarray = field(repr=False)
    rate: int
    mixed: bool  # False: the narration alone, untouched
    lanes: tuple[Lane, ...] = ()
    spans: tuple[tuple[float, float], ...] = ()
    duck: np.ndarray | None = field(default=None, repr=False)  # the beds' gain, every sample
    levels: Mapping[str, float] = field(default_factory=dict)
    trim_db: float = 0.0
    pre_peak_db: float = -math.inf  # sample peak before the limiter (after the trim)
    over: tuple[float, ...] = ()  # where the mix passes full scale by ``[sound] clip`` dB
    loudness: float = -math.inf  # integrated loudness of the result
    peak_dbtp: float = -math.inf  # true peak of the result

    def lane(self, lane_id: str) -> Lane | None:
        return next((x for x in self.lanes if x.id == lane_id), None)

    def voice_level(self, t0: float, t1: float) -> float:
        """The voice's loudness over a span; the target (where it will sit) when the
        narration has no audio yet (estimated timing)."""
        v = self.lane("voice")
        return v.loudness(t0, t1) if v is not None else float(self.levels.get("target", -16.0))

    def duck_db(self, t0: float, t1: float) -> float:
        """The beds' mean dip over a span, in dB (0 when nothing ducks)."""
        if self.duck is None:
            return 0.0
        a, b = round(t0 * self.rate), max(round(t0 * self.rate) + 1, round(t1 * self.rate))
        seg = self.duck[a:b]
        return float(20 * np.log10(seg.mean())) if seg.size else 0.0

    def report(self) -> dict[str, Any]:
        """Numbers an AI can read instead of listening: loudness per lane per narration
        block (as heard, LUFS), the ducking, the trim and the peaks."""

        def num(v: float) -> float | None:
            return round(float(v), 2) if math.isfinite(v) else None

        lanes = []
        for x in self.lanes:
            heard = x.loudness() + self.trim_db
            row: dict[str, Any] = {"id": x.id, "kind": x.kind, "loudness": num(heard),
                                   "source_lufs": num(x.level), "gain_db": num(x.gain_db),
                                   "blocks": [num(x.loudness(a, b) + self.trim_db)
                                              for a, b in self.spans]}  # fmt: skip
            if x.kind == "bed":
                row["duck_db"] = [num(self.duck_db(a, b)) for a, b in self.spans]
                row["under_voice_lu"] = [num(self.voice_level(a, b) - x.loudness(a, b))
                                         for a, b in self.spans]  # fmt: skip
            if x.kind == "cue":
                row["t"], row["peak_t"] = round(x.t0, 4), num(x.peak_t or 0.0)
            else:
                row["start"], row["end"] = round(x.t0, 4), round(x.t1, 4)
            lanes.append(row)
        return {
            "mixed": self.mixed, "rate": self.rate, "channels": 2,
            "seconds": round(len(self.samples) / self.rate, 4),
            "loudness": num(self.loudness), "true_peak_dbtp": num(self.peak_dbtp),
            "target": self.levels.get("target"), "trim_db": num(self.trim_db),
            "peak_before_limiter_dbfs": num(self.pre_peak_db),
            "duck": {"depth_db": self.levels.get("duck"), "attack": self.levels.get("attack"),
                     "release": self.levels.get("release")},
            "blocks": [{"start": round(a, 4), "end": round(b, 4)} for a, b in self.spans],
            "lanes": lanes,
        }  # fmt: skip


def to_int16(x: np.ndarray) -> np.ndarray:
    """Float samples (full scale 1.0 = 32768) to int16, the inverse of reading a WAV."""
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def _placed_energies(x: np.ndarray, i0: int, n: int, rate: int) -> np.ndarray:
    """Block energies over the whole track of a lane that is ``x`` from sample ``i0`` and
    silent elsewhere, filtering only from the block boundary before it (the filter starts
    at rest, as it would in the silence)."""
    nh, nb = round(HOP * rate), round(BLOCK * rate)
    total = max(0, (n - nb) // nh + 1) if n >= nb else 0
    out = np.zeros(total)
    if not len(x) or not total:
        return out
    a = (i0 // nh) * nh
    local = np.zeros((min(n, i0 + len(x) + nb) - a, 2), np.float32)
    local[i0 - a : i0 - a + len(x)] = _stereo(x)
    z = block_energies(local, rate)
    k0 = a // nh
    m = min(len(z), total - k0)
    if m > 0:
        out[k0 : k0 + m] = z[:m]
    return out


def _pan(x: np.ndarray, pan: float) -> np.ndarray:
    """Constant power: centred is the signal itself on both channels."""
    theta = (pan + 1) * math.pi / 4
    gains = np.array([math.sqrt(2) * math.cos(theta), math.sqrt(2) * math.sin(theta)], np.float32)
    return _stereo(x) * gains


def _fades(m: int, fade_in: float, fade_out: float, rate: int) -> np.ndarray:
    """Equal-power fades (sin in, cos out), so two overlapping fades crossfade evenly."""
    a, b = round(fade_in * rate), round(fade_out * rate)
    if a + b > m:  # too short for both: shrink them in proportion
        a, b = (m * a) // max(1, a + b), m - (m * a) // max(1, a + b)
    env = np.ones(m, np.float32)
    if a:
        env[:a] = np.sin(0.5 * np.pi * np.arange(a) / a)
    if b:
        env[m - b :] *= np.cos(0.5 * np.pi * np.arange(1, b + 1) / b)
    return env


def mix(
    voice: tuple[np.ndarray, int] | None,
    narration: Narration | None,
    beds: Sequence[PlacedBed],
    cues: Sequence[PlacedCue],
    base_dir: str | Path,
    duration: float,
    levels: Mapping[str, float],
    seed: int = 0,
) -> Mix | None:
    """The scene's audio (see the module docstring); None when there is nothing to hear.
    ``voice`` is ``narration.mix_track``'s mono int16 track and its rate."""
    spans = speech_spans(narration)
    if not beds and not cues:
        if voice is None:
            return None
        v, r = voice
        lane = Lane("voice", "voice", integrated(v / 32768.0, r), 0.0,
                    block_energies(v / 32768.0, r), 0.0, len(v) / r)  # fmt: skip
        out = np.stack([v, v], axis=1)
        return Mix(out, r, False, (lane,), spans, None, dict(levels),
                   loudness=lane.loudness(), peak_dbtp=_peak(v / 32768.0))  # fmt: skip

    rate, target = RATE, float(levels["target"])
    n = math.ceil(duration * rate - 1e-9)
    v = None
    if voice is not None:
        v = voice[0].astype(np.float64) / 32768.0
        if voice[1] != rate:
            from scipy.signal import resample_poly

            g = math.gcd(rate, voice[1])
            v = resample_poly(v, rate // g, voice[1] // g)
        n = max(n, len(v))
    bus = np.zeros((n, 2), np.float32)
    lanes: list[Lane] = []
    if v is not None:
        level = integrated(v, rate)
        g_db = target - level if math.isfinite(level) else 0.0
        bus[: len(v)] += _stereo(v * 10 ** (g_db / 20)).astype(np.float32)
        lanes.append(Lane("voice", "voice", level, g_db, block_energies(bus, rate), 0.0,
                          len(v) / rate))  # fmt: skip
    duck = None
    if beds:
        duck = duck_envelope(spans, n, rate, levels["duck"], levels["attack"], levels["release"])
    for p in beds:
        stamp = _stamp(base_dir, p.bed.path)
        x = None if stamp is None else _at_rate(*stamp, rate)
        i0, i1 = round(p.t0 * rate), min(round(p.t1 * rate), n)
        if x is None or not len(x) or i1 <= i0:
            lanes.append(Lane(p.id, "bed", -math.inf, 0.0, np.zeros(0), p.t0, p.t1))
            continue
        level = _file_level(*stamp, rate, False)
        want = target + (p.bed.gain if p.bed.gain is not None else levels["bed"])
        g_db = want - level if math.isfinite(level) else 0.0
        m = i1 - i0
        seg = np.tile(x, (math.ceil(m / len(x)), 1))[:m] if p.bed.loop else x[:m]
        env = _fades(len(seg), p.bed.fade_in, p.bed.fade_out, rate) * duck[i0 : i0 + len(seg)]
        seg = seg * (10 ** (g_db / 20) * env)[:, None].astype(np.float32)
        bus[i0 : i0 + len(seg)] += seg
        lanes.append(Lane(p.id, "bed", level, g_db, _placed_energies(seg, i0, n, rate),
                          p.t0, (i0 + len(seg)) / rate))  # fmt: skip
    for p in cues:
        got = _cue_source(p.cue, base_dir, seed)
        i0 = round(p.t * rate)
        if got is None or i0 >= n:
            lanes.append(Lane(p.id, "cue", -math.inf, 0.0, np.zeros(0), p.t, p.t, None))
            continue
        x, level = got
        want = target + (p.cue.gain if p.cue.gain is not None else levels["cue"])
        g_db = want - level if math.isfinite(level) else 0.0
        seg = _pan(x[: n - i0], p.cue.pan) * np.float32(10 ** (g_db / 20))
        bus[i0 : i0 + len(seg)] += seg
        peak = i0 + int(np.argmax(np.abs(seg).max(axis=1))) if len(seg) else i0
        lanes.append(Lane(p.id, "cue", level, g_db, _placed_energies(seg, i0, n, rate),
                          p.t, (i0 + len(seg)) / rate, peak / rate))  # fmt: skip

    # The trim makes the program ``target`` loud. A narration with no audio yet (estimated
    # timing) leaves every lane where it sits relative to the voice it will have.
    whole = integrated(bus, rate)
    trim = target - whole if math.isfinite(whole) and (v is not None or not spans) else 0.0
    bus *= np.float32(10 ** (trim / 20))
    mag = np.abs(bus).max(axis=1)
    pre_peak = _db(mag.max()) if n else -math.inf
    loud = np.flatnonzero(mag > 10 ** (levels["clip"] / 20))
    over = tuple(round(float(i) / rate, 3) for i in loud[np.diff(loud, prepend=-rate) > rate // 10]
                 [:MAX_TIMES])  # fmt: skip
    out = limit(bus, rate, levels["ceiling"], levels["knee"])
    return Mix(to_int16(out), rate, True, tuple(lanes), spans, duck, dict(levels), trim,
               pre_peak, over, integrated(out, rate), _peak(out))  # fmt: skip


def _peak(x: np.ndarray) -> float:
    """True peak in dBTP (only the chunks near the sample peak are oversampled)."""
    if not len(x):
        return -math.inf
    top = float(np.abs(x).max())
    return _db(true_peak(x, top).max()) if top > 0 else -math.inf


def _db(x: float) -> float:
    return 20 * math.log10(x) if x > 0 else -math.inf
