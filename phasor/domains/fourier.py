"""Fourier analysis objects.

    from phasor import fourier

    s.add(fourier.Epicycles(id="heart", points=curve, terms=24, phase=expr("t / 10")))
    s.add(fourier.PartialSum(id="wave", kind="square", terms=9))
    s.add(fourier.Spectrum(id="spec", frequencies=(100.5,), amplitudes=(1.0,), window="hann"))
    s.add(fourier.Spectrogram(id="sg", f0=200, f1=1200))
    s.add(fourier.Aliasing(id="al", frequencies=(5.0,), fs=7.0))  # sampling and aliasing

Numerics are in ``fourier_math``; these draw them. Everything is a pure function of the
parameters.
"""

from __future__ import annotations

import functools
import itertools
import math
import os
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core import files
from phasor.core.color import ramp
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import fourier_math as fm
from phasor.domains.heatmap import (
    PRISM,
    HeatParams,
    Recipe,
    heat_pixels,
    wheel_recipe,
)
from phasor.domains.strokes import (
    circle,
    disc,
    disc_batch,
    hue_colors,
    polyline,
    segment_batch,
    with_alpha,
)


def _ydir(me, state: SceneState) -> float:
    """+1 in a world layer, -1 on the overlay (y runs down): plots grow upward either way."""
    return 1.0 if state.camera_for(me).up[1] >= 0 else -1.0


# ---------------------------------------------------------------- Epicycles


@functools.lru_cache(maxsize=16)
def _coeffs(points: tuple, terms: int):
    return fm.epicycle_coefficients(np.asarray(points, dtype=np.float64), terms)


@register
class Epicycles(ObjectBase):
    """A closed curve drawn by a chain of rotating circles: each circle is one Fourier
    coefficient. ``phase`` (0-1) is how far around the curve the tip is; ``trace`` (0-1)
    keeps the path the tip has drawn so far visible."""

    type: ClassVar[str] = "epicycles"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "accent": ("emphasis", "highlight"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {
        "glowing": {"style": {"glow": "soft"}},
    }

    class Params(ParamsBase):
        points: tuple[Vec3, ...] = Field(min_length=4)  # the closed curve (x, y)
        terms: int = Field(12, ge=1, le=400)
        phase: float = Field(0.0, ge=0, le=1)
        trace: float = Field(1.0, ge=0, le=1)  # share of the path drawn behind the tip

        @field_validator("phase")
        @classmethod
        def _wrap(cls, v):
            return v

    def _co(self, state: SceneState):
        p = self.me(state).params
        return _coeffs(tuple(p.points), p.terms)

    def _chain(self, state: SceneState) -> np.ndarray:
        p = self.me(state).params
        return fm.epicycle_chain(self._co(state), p.phase)

    def world_bounds(self, state: SceneState) -> Bounds:
        co = self._co(state)
        r = sum(abs(c) for k, c in co if k != 0)
        c0 = next((c for k, c in co if k == 0), 0j)
        z = self.me(state).params.points[0][2]
        return Bounds((c0.real - r, c0.imag - r, z), (c0.real + r, c0.imag + r, z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        chain = self._chain(state)
        z = self.me(state).params.points[0][2]
        pts = np.stack([chain.real, chain.imag, np.full(len(chain), z)], axis=1)
        return {"tip": pts[-1], "centers": pts[:-1]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        co = self._co(state)
        target = np.asarray(p.points)
        approx = fm.epicycle_curve(co, len(target))
        err = float(np.abs(approx - (target[:, 0] + 1j * target[:, 1])).max())
        radii = [round(abs(c), 4) for _, c in co[:8]]
        return {
            "kind": "epicycles",
            "terms": len(co),
            "phase": p.phase,
            "max_error": round(err, 6),
            "radii": radii,
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        z = p.points[0][2]
        co = self._co(state)
        base = with_alpha(me.style["color"], p.opacity)
        accent = with_alpha(me.style["accent"], p.opacity)
        chain = self._chain(state)

        def xyz(zs):
            return np.stack([zs.real, zs.imag, np.full(len(zs), z)], axis=-1)

        w = me.style["stroke"] * float(cam.units_per_height())
        if p.trace > 0:  # the path drawn so far: the curve the full chain traces up to phase
            n = max(int(160 * p.trace), 2)
            s = np.linspace(max(p.phase - p.trace, 0.0), p.phase, n) if p.phase > 0 else np.zeros(2)
            path = np.array([fm.epicycle_chain(co, v)[-1] for v in s])
            polyline(ctx, cam, xyz(path), w * 3, color=accent, glow=me.style["glow"])
        for (k, c), centre in zip(co, chain[:-1], strict=True):
            if k == 0 or abs(c) * 1.0 < 1e-9:
                continue
            circle(ctx, cam, xyz(np.array([centre]))[0], abs(c), w, color=with_alpha(base, 0.45))
        polyline(ctx, cam, xyz(chain), w * 2, color=base)
        disc(ctx, cam, xyz(chain[-1:])[0], 3.0 * w, color=accent, glow=me.style["glow"])


# ---------------------------------------------------------------- PartialSum


@register
class PartialSum(ObjectBase):
    """A Fourier series partial sum over the wave it converges to, with the Gibbs overshoot
    at the jumps. ``pos`` is the origin of the plot, ``extent`` its (width, height) in world
    units: x spans one period, y spans -1.2 to 1.2."""

    type: ClassVar[str] = "partial_sum"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "target": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {"glowing": {"style": {"glow": "soft"}}}

    class Params(ParamsBase):
        kind: Literal["square", "sawtooth", "triangle"] = "square"
        terms: int = Field(5, ge=1, le=500)
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre of the plot
        extent: tuple[float, float] = (6.0, 2.4)
        reveal: float = Field(1.0, ge=0, le=1)  # share of the period drawn

    def _xs(self, p):
        return np.linspace(-math.pi, math.pi, 600)

    def _curve(self, state: SceneState, p, y):
        xs = self._xs(p)
        px = p.pos[0] + xs / (2 * math.pi) * p.extent[0]
        py = p.pos[1] + y / 1.2 * p.extent[1] / 2
        n = max(int(len(xs) * p.reveal), 2)
        return np.stack([px, py, np.full_like(px, p.pos[2])], axis=1)[:n]

    def path_points(self, state: SceneState) -> np.ndarray:
        """The partial sum's curve (what a ``Tracer`` travels along)."""
        return self.features(state)["curve"]

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        return Bounds((x - p.extent[0] / 2, y - p.extent[1] / 2, z),
                      (x + p.extent[0] / 2, y + p.extent[1] / 2, z))  # fmt: skip

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        y = fm.series_partial_sum(p.kind, p.terms, self._xs(p))
        return {"curve": self._curve(state, p, y)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        xs = self._xs(p)
        err = np.abs(fm.series_partial_sum(p.kind, p.terms, xs) - fm.series_target(p.kind, xs))
        away = np.abs(np.sin(xs)) > 0.3  # not next to a jump
        return {"kind": "partial_sum", "series": p.kind, "terms": p.terms,
                "gibbs_overshoot": round(fm.gibbs_overshoot(p.kind, p.terms), 5),
                "max_error_away_from_jumps": round(float(err[away].max()), 5)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        w = me.style["stroke"] * float(cam.units_per_height())
        xs = self._xs(p)
        target = fm.series_target(p.kind, xs)
        # the target as separate strokes so the jump is not drawn as a vertical line
        jump = np.abs(np.diff(target)) > 0.5
        pts = self._curve(state, p, target)
        starts = [0, *(np.nonzero(jump)[0] + 1), len(pts)]
        for a, b in itertools.pairwise(starts):
            polyline(
                ctx, cam, pts[a:b], w * 0.6, color=with_alpha(me.style["target"], p.opacity * 0.7)
            )
        y = fm.series_partial_sum(p.kind, p.terms, xs)
        color = with_alpha(me.style["color"], p.opacity)
        polyline(ctx, cam, self._curve(state, p, y), w, color=color, glow=me.style["glow"])


# ---------------------------------------------------------------- Spectrum


@register
class Spectrum(ObjectBase):
    """The magnitude spectrum of a few sinusoids seen through a window: bars per frequency
    bin. Shows leakage (a tone between bins smears) and what a window does about it."""

    type: ClassVar[str] = "spectrum"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "accent": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        frequencies: tuple[float, ...] = (100.0,)
        amplitudes: tuple[float, ...] = (1.0,)
        window: Literal["rect", "hann", "hamming", "blackman"] = "rect"
        samples: int = Field(256, ge=16, le=8192)
        fs: float = Field(256.0, gt=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # lower-left corner
        extent: tuple[float, float] = (6.0, 2.5)
        bins: int = Field(0, ge=0)  # bins shown from DC (0: up to the highest tone + margin)

        @field_validator("amplitudes")
        @classmethod
        def _same(cls, v, info):
            f = info.data.get("frequencies", ())
            if len(v) != len(f):
                raise ValueError("amplitudes needs one value per frequency")
            return v

    def _spec(self, p):
        sp = fm.spectrum(p.frequencies, p.amplitudes, p.window, p.samples, p.fs)
        shown = p.bins or min(
            len(sp["mag"]), int(max(p.frequencies, default=1) / (p.fs / p.samples)) + 12
        )
        return sp["freq"][:shown], sp["mag"][:shown]

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        y2 = y + _ydir(self.me(state), state) * p.extent[1]
        return Bounds((x, min(y, y2), z), (x + p.extent[0], max(y, y2), z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        _, m = self._spec(p)
        x, y, z = p.pos
        h = p.extent[1] * _ydir(self.me(state), state)
        top = np.max(m) if len(m) else 1.0
        peak = int(np.argmax(m))
        return {
            "peak": np.array([x + (peak + 0.5) / len(m) * p.extent[0], y + m[peak] / top * h, z])
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f, m = self._spec(p)
        leak = fm.leakage(p.frequencies, p.amplitudes, p.window, p.samples, p.fs)
        return {
            "kind": "spectrum",
            "window": p.window,
            "bin_width_hz": round(p.fs / p.samples, 6),
            "peak_hz": round(float(f[int(np.argmax(m))]), 4),
            "leakage": round(leak, 6),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        _, m = self._spec(p)
        x, y, z = p.pos
        h = p.extent[1] * _ydir(me, state)
        top = max(float(np.max(m)), 1e-12)
        n = len(m)
        base = with_alpha(me.style["color"], p.opacity)
        accent = with_alpha(me.style["accent"], p.opacity)
        w = me.style["stroke"] * float(cam.units_per_height())
        polyline(ctx, cam, [(x, y, z), (x + p.extent[0], y, z)], w, color=with_alpha(base, 0.6))
        df = p.fs / p.samples
        tones = {round(t / df) for t in p.frequencies}
        bar = max(p.extent[0] / n * 0.7, w)
        for k in range(n):
            cx_ = x + (k + 0.5) / n * p.extent[0]
            height = m[k] / top * h
            if abs(height) < 1e-6:
                continue
            polyline(ctx, cam, [(cx_, y, z), (cx_, y + height, z)], bar,
                     color=accent if k in tones else base, glow=me.style["glow"])  # fmt: skip


# ---------------------------------------------------------------- Winding


@functools.lru_cache(maxsize=8)
def _winding_grid(frequencies, amplitudes, samples, fs, limit):
    signal = fm.sinusoids(frequencies, amplitudes, samples, fs)
    # Sample at least eight times per DFT bin so the connecting stroke resolves each peak.
    count = max(128, math.ceil(8 * limit * samples / fs))
    grid = np.linspace(0, limit, count + 1)
    # Keep the temporary frequency-by-time matrix bounded for long signals.
    mean = np.concatenate(
        [fm.winding_mean(signal, block, fs) for block in np.array_split(grid, count // 64 + 1)]
    )
    return signal, grid, np.abs(mean)


@register
class Winding(ObjectBase):
    """A signal wound clockwise, its centroid, and the spectrum up to the test frequency."""

    type: ClassVar[str] = "winding"
    Style: ClassVar = {
        "color": ("emphasis", "primary"),
        "accent": ("emphasis", "highlight"),
        "guide": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "phase"),
    }

    class Params(ParamsBase):
        frequencies: tuple[float, ...] = (3.0, 6.0)
        amplitudes: tuple[float, ...] = (1.0, 0.55)
        fs: float = Field(32.0, gt=0, allow_inf_nan=False)
        samples: int = Field(128, ge=16, le=2048)
        freq: float = Field(0.0, ge=0, allow_inf_nan=False)
        max_freq: float = Field(8.0, gt=0, allow_inf_nan=False)
        progress: float = Field(1.0, ge=0, le=1)
        size: float = Field(2.0, gt=0, allow_inf_nan=False)
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre of the winding diagram
        hue_along: float = Field(0.0, ge=0, allow_inf_nan=False)
        spectrum_guide: bool = True

        @model_validator(mode="after")
        def _signal(self):
            if not self.frequencies or len(self.frequencies) != len(self.amplitudes):
                raise ValueError("provide one amplitude per tone")
            if any(not 0 < f < self.fs / 2 for f in self.frequencies):
                raise ValueError("tone frequencies must be between DC and Nyquist")
            if not math.isfinite(sum(map(abs, self.amplitudes))):
                raise ValueError("amplitudes must have a finite sum of magnitudes")
            if not self.freq <= self.max_freq <= self.fs / 2:
                raise ValueError("freq must be at most max_freq, which cannot exceed Nyquist")
            return self

    def _data(self, p):
        signal, grid, magnitude = _winding_grid(
            p.frequencies, p.amplitudes, p.samples, p.fs, p.max_freq
        )
        wound = fm.wound_signal(signal, p.freq, p.fs)
        return wound, wound.mean(), grid, magnitude

    def _winding_points(self, p, ydir, values):
        x, y, z = p.pos
        scale = p.size / (sum(map(abs, p.amplitudes)) or 1.0)
        return np.stack(
            np.broadcast_arrays(x + values.real * scale, y + ydir * values.imag * scale, z),
            axis=-1,
        )

    def _spectrum_points(self, p, ydir, freq, magnitude):
        x, y, z = p.pos
        scale = p.size / (sum(map(abs, p.amplitudes)) or 1.0)
        return np.stack(
            np.broadcast_arrays(
                x + p.size * (1.5 + 3 * freq / p.max_freq),
                y + ydir * (-p.size + 2 * magnitude * scale),
                z,
            ),
            axis=-1,
        )

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        return Bounds((x - p.size, y - p.size, z), (x + 4.5 * p.size, y + p.size, z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        me = self.me(state)
        p = me.params
        _, mean, _, _ = self._data(p)
        ydir = _ydir(me, state)
        return {
            "centroid": self._winding_points(p, ydir, mean),
            "spectrum_tip": self._spectrum_points(p, ydir, p.freq, abs(mean)),
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        _, mean, _, _ = self._data(p)
        return {
            "kind": "winding",
            "backdrop": True,
            "frequency_hz": p.freq,
            "centroid": (float(mean.real), float(mean.imag)),
            "magnitude": float(abs(mean)),
            "bin_width_hz": p.fs / p.samples,
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        ydir = _ydir(me, state)
        wound, mean, grid, magnitude = self._data(p)
        style = me.style
        guide = with_alpha(style["guide"], p.opacity)
        color = with_alpha(style["color"], p.opacity)
        accent = with_alpha(style["accent"], p.opacity)
        width = style["stroke"] * float(cam.units_per_height())
        glow = style["glow"]

        circle(ctx, cam, p.pos, p.size, width, color=guide)
        x, y, z = p.pos
        if p.spectrum_guide:
            origin = self._spectrum_points(p, ydir, 0.0, 0.0)
            end = self._spectrum_points(p, ydir, p.max_freq, 0.0)
            polyline(ctx, cam, [end, origin, (origin[0], y + ydir * p.size, z)], width, color=guide)
        disc(ctx, cam, (x, y, z), width, color=guide)

        if p.progress > 0:
            points = self._winding_points(p, ydir, wound)
            index = p.progress * (len(points) - 1)
            whole = int(index)
            visible = points[: whole + 1]
            if index > whole:
                endpoint = points[whole] + (index - whole) * (points[whole + 1] - points[whole])
                visible = np.vstack((visible, endpoint))
            colors = None
            if p.hue_along and pass_ == "color":
                length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                colors = hue_colors(visible, style["colormap"], p.hue_along, total=length)
                colors[:, 3] *= p.opacity
            polyline(ctx, cam, visible, width, color=color, colors=colors, glow=glow)

        stop = np.searchsorted(grid, p.freq)
        traced = self._spectrum_points(
            p, ydir, np.r_[grid[:stop], p.freq], np.r_[magnitude[:stop], abs(mean)]
        )
        polyline(ctx, cam, traced, width, color=color, glow=glow)
        disc(ctx, cam, traced[-1], 2 * width, color=color, glow=glow)
        disc(ctx, cam, self._winding_points(p, ydir, mean), 3 * width, color=accent, glow=glow)


# ---------------------------------------------------------------- Spectrogram


MAX_FORMULA_SAMPLES = 400_000  # a formula is evaluated sample by sample
MAX_FRAMES = 1500  # the hop widens for long sounds so a wav never makes more columns than this


def _formula_in_t(v: str) -> str:
    """Check a signal formula: it may use ``t`` (seconds) and constants, not other parameters."""
    if v:
        from phasor.time.expr import ExprError, compile_expr

        try:
            c = compile_expr(v)
        except ExprError as e:
            raise ValueError(str(e)) from e
        if c.deps:
            raise ValueError(f"{v!r}: a signal is a formula in t only, not {sorted(c.deps)}")
    return v


@functools.lru_cache(maxsize=8)
def _stft(source: str, ref: str, mtime: float, f0: float, f1: float, duration: float, fs: float,
          nwin: int, hop: int, win: str) -> dict[str, Any]:  # fmt: skip
    """The STFT of one signal, cached by everything that makes it (``mtime`` is the wav's)."""
    if source == "wav":
        from phasor.domains.media import read_wav

        x, fs = read_wav(ref, mtime)
    elif source == "formula":
        from phasor.time.expr import compile_expr

        if duration * fs > MAX_FORMULA_SAMPLES:  # also when a track drives duration or fs
            raise ValueError(f"the signal is too long: {duration * fs:.0f} samples")
        fn = compile_expr(ref)
        x = fm.sampled(lambda t: fn({}, t), duration, fs)
    else:
        x = fm.chirp(f0, f1, duration, fs)
    hop = hop or max(nwin // 4, min(nwin, -(-len(x) // MAX_FRAMES)))
    sg = fm.spectrogram(x, fs, nwin, hop, win)
    return {**sg, "duration": len(x) / fs, "fs": fs, "source": source}


@register
class Spectrogram(ObjectBase):
    """A spectrogram: time across, frequency up, brightness for strength. A backdrop: labels
    may sit on it. The signal is, in this order, a ``.wav`` file next to the scene (``wav``,
    mono or mixed down), a formula in ``t`` seconds (``signal``, such as
    ``"sin(tau * 440 * t)"``), or a chirp sliding from ``f0`` to ``f1`` Hz. ``playhead``
    (0..1 of the duration; a track or an expression moves it) draws a glowing line, and the
    semantics name the strongest frequency under it."""

    type: ClassVar[str] = "spectrogram"
    Files: ClassVar = ("wav",)  # in the scene's folder (core/files.py); missing: lint
    Style: ClassVar = {"colormap": ("colormap", "ocean"), "glow": ("glow", "none"),
                       "playhead": ("emphasis", "highlight"),
                       "stroke": ("stroke", "hairline")}  # fmt: skip
    # the prism look (heatmap.PRISM); here the hue runs up the frequency axis, so a sliding
    # tone's trace runs through the spectrum the way hue runs along a prism stroke
    Presets: ClassVar = {"prism": PRISM}
    Life: ClassVar = frozenset({"hue", "glow"})

    class Params(HeatParams):
        f0: float = Field(200.0, gt=0)  # the chirp (no signal, no wav)
        f1: float = Field(1200.0, gt=0)
        signal: str = ""  # a formula in t (seconds), over ``duration`` at ``fs``
        wav: str = ""  # a .wav file next to the scene (then duration and fs are its own)
        duration: float = Field(1.0, gt=0)
        fs: float = Field(4000.0, gt=0)
        nwin: int = Field(256, ge=16, le=4096)
        hop: int = Field(0, ge=0)  # 0: nwin / 4 (wider for long sounds)
        window: Literal["rect", "hann", "hamming", "blackman"] = "hann"
        fmax: float = Field(0.0, ge=0)  # top of the frequency axis in Hz; 0: half the rate
        playhead: float = Field(-1.0, ge=-1, le=1)  # 0..1 of the duration; below 0: no line
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        size: tuple[float, float] = (6.0, 3.0)

        @field_validator("wav")
        @classmethod
        def _wav_file(cls, v: str) -> str:
            if v and not v.lower().endswith(".wav"):
                raise ValueError(f"{v!r} is not a .wav file")
            return v

        @field_validator("signal")
        @classmethod
        def _formula_in_t(cls, v: str) -> str:
            return _formula_in_t(v)

        @model_validator(mode="after")
        def _one_source(self):
            if self.signal and self.wav:
                raise ValueError("give a signal formula or a wav file, not both")
            if self.signal and self.duration * self.fs > MAX_FORMULA_SAMPLES:
                raise ValueError(
                    f"the signal is too long: {self.duration * self.fs:.0f} samples "
                    f"(at most {MAX_FORMULA_SAMPLES}; lower duration or fs)"
                )
            return self

    def _rect(self, p):
        c = np.asarray(p.pos, dtype=np.float64) - np.array([p.size[0] / 2, p.size[1] / 2, 0.0])
        return c, np.array([p.size[0], 0.0, 0.0]), np.array([0.0, p.size[1], 0.0])

    def _analysis(self, state: SceneState) -> dict[str, Any] | None:
        """The signal's STFT (``freq``, ``time``, ``mag``) cropped to ``fmax``, with its
        ``duration``, sample rate, ``source`` and the axis ``top``; None while the ``wav`` is
        missing or unreadable (lint ``missing_file``: draws nothing)."""
        p = self.me(state).params
        if p.wav:
            try:
                file = files.resolve(state.base_dir, p.wav)
                source, ref, mtime = "wav", str(file), os.path.getmtime(file)
            except (OSError, ValueError):
                return None
        elif p.signal:
            source, ref, mtime = "formula", p.signal, 0.0
        else:
            source, ref, mtime = "chirp", "", 0.0
        sg = _stft(source, ref, mtime, p.f0, p.f1, p.duration, p.fs, p.nwin, p.hop, p.window)
        top = p.fmax if 0 < p.fmax < sg["fs"] / 2 else sg["fs"] / 2
        keep = sg["freq"] <= top + 1e-9
        return {**sg, "freq": sg["freq"][keep], "mag": sg["mag"][keep], "top": float(top)}

    def world_bounds(self, state: SceneState) -> Bounds:
        c, r, u = self._rect(self.me(state).params)
        return Bounds.of_points([c, c + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        a = self._analysis(state)
        if a is None:
            return {"kind": "spectrogram", "backdrop": True, "source": "wav", "missing_file": p.wav}
        out = {"kind": "spectrogram", "backdrop": True, "source": a["source"],
               "f0": p.f0, "f1": p.f1, "duration": a["duration"], "max_frequency": a["top"],
               "playhead": p.playhead if p.playhead >= 0 else None, "playhead_time": None,
               "dominant_frequency": None}  # fmt: skip
        if p.playhead >= 0:
            at = p.playhead * a["duration"]
            out["playhead_time"] = at
            out["dominant_frequency"] = fm.dominant_frequency(a, at)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        sg = self._analysis(state)
        if sg is None:
            return
        mag = sg["mag"] / max(float(sg["mag"].max()), 1e-12)
        lut = ramp(me.style["colormap"], np.float32)  # strongest = brightest (ocean is cyclic)
        corner, right, up = self._rect(p)
        glow = float(me.style.get("glow", 1.0))
        # the signal is the square root of the magnitude: level, weight and (plain) colour
        how = {"transform": "sqrt", "weight": "value" if glow > 1 else None}
        if p.hue_cycles > 0:  # prism: the hue of the frequency (its place up the panel)
            recipe = wheel_recipe(me.style["colormap"], p, hue_of="height", **how)
        else:
            recipe = Recipe(lut=lut, blend=p.smooth, **how)
        pixels = heat_pixels(
            mag.T, p.size, recipe, cells=True, how="smooth" if p.smooth else "nearest",
            background=state.background, **p.drawing(),
        )  # fmt: skip
        ctx.field(corner, right, up, pixels=pixels, opacity=p.opacity, glow=glow)
        if p.playhead >= 0 and pass_ == "color":
            cam = state.camera_for(me)
            x = corner[0] + p.playhead * p.size[0]
            line = [(x, corner[1], corner[2]), (x, corner[1] + p.size[1], corner[2])]
            w = me.style["stroke"] * float(cam.units_per_height(np.asarray(line[0])))
            polyline(ctx, cam, line, w, color=with_alpha(me.style["playhead"], p.opacity),
                     glow=2.5)  # fmt: skip


# ---------------------------------------------------------------- Aliasing


MAX_ALIAS_SAMPLES = 4000  # samples in the window (duration * fs)
_SINC_PAD = 48  # samples taken beyond each end of the window for the sinc sum
_LOW = 0.30  # the spectrum and wheel strip, as a share of the height
_GAP = 0.12  # between the wave and the strip


@functools.lru_cache(maxsize=32)
def _formula_values(signal: str, times: tuple) -> np.ndarray:
    from phasor.time.expr import compile_expr

    fn = compile_expr(signal)
    return np.array([float(fn({}, t)) for t in times], dtype=np.float64)


@functools.lru_cache(maxsize=32)
def _alias_data(signal: str, tones: tuple, fs: float, start: float, duration: float,
                sinc: bool) -> dict[str, Any]:  # fmt: skip
    """The window's drawing grid, the true signal and the reconstruction on it, the sampling
    instants and their values. ``tones`` is ((f, a, phase), ...) when there is no formula."""
    stop = start + duration
    fastest = max((f for f, _, _ in tones), default=fs)
    n = int(np.clip(48 * max(fastest, fs / 2) * duration, 400, 6000))
    grid = np.linspace(start, stop, n)
    ts = fm.sample_times(fs, start, stop)
    f, a, ph = (tuple(v[i] for v in tones) for i in range(3)) if tones else ((), (), ())

    def true(t):
        t = np.asarray(t, dtype=np.float64)
        return _formula_values(signal, tuple(t.tolist())) if signal else fm.tone_sum(f, a, ph, t)

    x = true(ts)
    if sinc:  # samples beyond the window too, so the sum is not cut off at its edges
        k0 = math.ceil(start * fs - 1e-9) - _SINC_PAD
        wide = (k0 + np.arange(len(ts) + 2 * _SINC_PAD)) / fs
        xw = true(wide)
        rec = fm.sinc_reconstruct(xw, fs, grid, wide[0])
        at = fm.sinc_reconstruct(xw, fs, ts, wide[0])
    else:
        rec = fm.alias_signal(f, a, ph, fs, grid)
        at = fm.alias_signal(f, a, ph, fs, ts)
    wave = true(grid)
    top = max(sum(abs(v) for v in a) if tones else float(np.abs(wave).max()), 1e-9)
    return {"grid": grid, "wave": wave, "rec": rec, "ts": ts, "x": x, "at": at, "top": top}


@register
class Aliasing(ObjectBase):
    """Sampling and aliasing: a signal (a sum of tones ``a sin(2 pi f t + phase)``, or a formula
    in ``t`` seconds), the instants ``k / fs`` it is sampled at as stems with a beacon on each,
    and the slowest wave through those samples in a second hue. For tones that is the sum of
    their aliases (``reconstruct="alias"``: each tone folded into [0, fs / 2], its phase
    reflected when it folds from the upper half); ``"sinc"`` (always, for a formula) is the
    band-limited reconstruction from the samples. Below the wave, for tones: a spectrum
    (``spectrum``) with the true lines, their aliases mirrored about the Nyquist mark ``fs/2``
    and an arc from each folded line to its alias, and a wheel (``wheel``): the strongest
    tone's phasor at the last ``wheel_samples`` samples, with the turn it really makes in one
    sample (the inner spiral) and the short turn the eye sees (the outer arc; backwards when
    the alias runs backwards). ``fs`` can be swept: the alias slows, stops at ``fs = f`` and
    turns round. ``reveal`` (0..1 of the window) is how far the samples have been taken."""

    type: ClassVar[str] = "aliasing"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),  # the true signal
        "alias": ("emphasis", "highlight"),  # the wave the samples also fit
        "sample": ("emphasis", "primary"),  # the stems and beacons
        "axis": ("emphasis", "muted"),
        "text": ("emphasis", "muted"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        frequencies: tuple[float, ...] = (5.0,)  # the tones, Hz
        amplitudes: tuple[float, ...] = (1.0,)
        phases: tuple[float, ...] = ()  # radians, one per tone; empty: all 0
        signal: str = ""  # a formula in t (seconds) instead of the tones
        fs: float = Field(12.0, gt=0, allow_inf_nan=False)  # the sample rate, Hz
        start: float = Field(0.0, allow_inf_nan=False)  # the window: start .. start + duration
        duration: float = Field(2.0, gt=0, allow_inf_nan=False)
        reconstruct: Literal["alias", "sinc"] = "alias"
        reveal: float = Field(1.0, ge=0, le=1)  # share of the window sampled so far
        spectrum: bool = True  # the folding picture under the wave (tones only)
        wheel: bool = True  # the strobed phasor beside it (tones only)
        wheel_samples: int = Field(6, ge=2, le=24)
        fmax: float = Field(0.0, ge=0, allow_inf_nan=False)  # spectrum axis; 0: automatic
        pos: Vec3 = (0.0, 0.0, 0.0)  # the picture's centre
        extent: tuple[float, float] = (11.0, 6.0)

        @field_validator("signal")
        @classmethod
        def _formula(cls, v: str) -> str:
            return _formula_in_t(v)

        @model_validator(mode="after")
        def _tones(self):
            if len(self.amplitudes) != len(self.frequencies):
                raise ValueError("amplitudes needs one value per frequency")
            if self.phases and len(self.phases) != len(self.frequencies):
                raise ValueError("phases needs one value per frequency (or none)")
            if not self.signal and not self.frequencies:
                raise ValueError("give at least one tone or a signal formula")
            if any(f < 0 or not math.isfinite(f) for f in self.frequencies):
                raise ValueError("tone frequencies must be finite and not negative")
            if self.duration * self.fs > MAX_ALIAS_SAMPLES:
                raise ValueError(
                    f"too many samples: {self.duration * self.fs:.0f} in the window "
                    f"(at most {MAX_ALIAS_SAMPLES}; lower duration or fs)"
                )
            return self

    # -------------------------------------------------------- numbers

    @staticmethod
    def _tones_of(p) -> tuple:
        if p.signal:
            return ()
        ph = p.phases or (0.0,) * len(p.frequencies)
        return tuple(zip(p.frequencies, p.amplitudes, ph, strict=True))

    def _data(self, p) -> dict[str, Any]:
        sinc = bool(p.signal) or p.reconstruct == "sinc"
        return _alias_data(p.signal, self._tones_of(p), p.fs, p.start, p.duration, sinc)

    @staticmethod
    def _fmax(p) -> float:
        if p.fmax > 0:
            return p.fmax
        return 2.2 * max(max(p.frequencies, default=0.0), p.fs / 2, 1e-9)

    # -------------------------------------------------------- layout

    def _boxes(self, p) -> dict[str, tuple[float, float, float, float]]:
        """Each panel's (u0, v0, u1, v1) as shares of the extent, from the lower left."""
        low = not p.signal and (p.spectrum or p.wheel)
        out = {"wave": (0.0, _LOW + _GAP if low else 0.0, 1.0, 1.0)}
        if low:
            side = _LOW * p.extent[1] / p.extent[0]  # the wheel is square
            if p.wheel:
                out["wheel"] = (1.0 - side, 0.0, 1.0, _LOW)
            if p.spectrum:
                out["spectrum"] = (0.0, 0.0, 1.0 - (side + 0.05 if p.wheel else 0.0), _LOW)
        return out

    def _at(self, me, state: SceneState, box):
        """(u, v) shares of a panel -> world points (v grows upward on screen either way)."""
        p = me.params
        sign = _ydir(me, state)
        u0, v0, u1, v1 = box
        W, H = p.extent
        x0, yc, z = p.pos[0] - W / 2, p.pos[1], p.pos[2]

        def at(u, v):
            u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
            uu, vv = u0 + u * (u1 - u0), v0 + v * (v1 - v0)
            return np.stack([x0 + uu * W, yc + sign * (vv - 0.5) * H,
                             np.full(np.broadcast(uu, vv).shape, z)], axis=-1)  # fmt: skip

        return at

    def _wave_points(self, me, state, d, t, y):
        p = me.params
        at = self._at(me, state, self._boxes(p)["wave"])
        return at((np.asarray(t) - p.start) / p.duration, 0.5 + 0.44 * np.asarray(y) / d["top"])

    @staticmethod
    def _shown(p, d) -> int:
        """How many of the window's samples have been taken (``reveal``)."""
        edge = p.start + p.reveal * p.duration
        return int(np.searchsorted(d["ts"], edge + 1e-9, side="right"))

    # -------------------------------------------------------- contract

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c = np.asarray(p.pos, dtype=np.float64)
        half = np.array([p.extent[0] / 2, p.extent[1] / 2, 0.0])
        return Bounds(tuple(c - half), tuple(c + half))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        me = self.me(state)
        p = me.params
        d = self._data(p)
        out = {"center": np.asarray(p.pos, dtype=np.float64),
               "samples": self._wave_points(me, state, d, d["ts"], d["x"]),
               "alias_at_samples": self._wave_points(me, state, d, d["ts"], d["at"])}  # fmt: skip
        box = self._boxes(p).get("spectrum")
        if box is not None:
            out["nyquist"] = self._at(me, state, box)(min(p.fs / 2 / self._fmax(p), 1.0), 0.22)
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        d = self._data(p)
        out: dict[str, Any] = {
            "kind": "aliasing",
            "fs": p.fs,
            "nyquist": p.fs / 2,
            "samples": len(d["ts"]),
            "samples_shown": self._shown(p, d),
            "reconstruction": "sinc" if p.signal or p.reconstruct == "sinc" else "alias",
            "frequencies": None,
            "alias_frequencies": None,
            "reversed": None,
            "turns_per_sample": None,
            "aliased": None,
        }
        tones = self._tones_of(p)
        if tones:
            al = [fm.alias_tone(f, p.fs, a, ph) for f, a, ph in tones]
            folded = [float(fm.fold(f, p.fs)) for f, _, _ in tones]
            out.update(
                frequencies=[float(f) for f, _, _ in tones],
                alias_frequencies=[round(v[0], 9) for v in al],
                reversed=[v < 0 for v in folded],
                turns_per_sample=[round(v / p.fs, 9) for v in folded],
                aliased=any(f > 0 and f >= p.fs / 2 * (1 - 1e-12) for f, _, _ in tones),
            )
        return out

    # -------------------------------------------------------- drawing

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = float(cam.units_per_height(np.asarray(p.pos, dtype=np.float64)))
        w = me.style["stroke"] * unit
        glow = me.style["glow"]
        beacon = glow * 1.2 if glow > 1.0 else 1.0
        d = self._data(p)
        boxes = self._boxes(p)
        alias = with_alpha(me.style["alias"], p.opacity)
        sample = with_alpha(me.style["sample"], p.opacity)
        faint = with_alpha(me.style["axis"], p.opacity * 0.5)

        # the wave: its baseline, the true signal, the samples taken so far and their alias
        at = self._at(me, state, boxes["wave"])
        polyline(ctx, cam, at([0, 1], [0.5, 0.5]), w * 0.5, color=faint)
        m = self._shown(p, d)
        if m:  # the alias under the true wave: where they agree, both still show
            edge = d["ts"][m - 1] if p.reveal < 1 else p.start + p.duration
            keep = d["grid"] <= edge + 1e-12
            if keep.sum() >= 2:
                rec = np.clip(d["rec"][keep], -1.1 * d["top"], 1.1 * d["top"])
                pts = self._wave_points(me, state, d, d["grid"][keep], rec)
                polyline(ctx, cam, pts, w * 2.0, color=alias, glow=glow)
        polyline(ctx, cam, self._wave_points(me, state, d, d["grid"], d["wave"]), w,
                 color=with_alpha(me.style["color"], p.opacity * 0.9), glow=glow)  # fmt: skip
        if m:
            tops = self._wave_points(me, state, d, d["ts"][:m], d["x"][:m])
            feet = self._wave_points(me, state, d, d["ts"][:m], np.zeros(m))
            segment_batch(ctx, cam, feet, tops, me.style["stroke"] * 0.6,
                          colors=with_alpha(sample, 0.55), glow=1.0)  # fmt: skip
            disc_batch(ctx, cam, tops, unit * 0.011, colors=sample, glow=beacon)

        tones = self._tones_of(p)
        if "spectrum" in boxes:
            self._draw_spectrum(ctx, cam, me, state, boxes["spectrum"], tones, unit)
        if "wheel" in boxes and m:
            self._draw_wheel(ctx, cam, me, state, boxes["wheel"], tones, d["ts"][:m], unit)

    def _draw_spectrum(self, ctx, cam, me, state, box, tones, unit) -> None:
        p = me.params
        glow = me.style["glow"]
        w = me.style["stroke"] * unit
        at = self._at(me, state, box)
        true = with_alpha(me.style["color"], p.opacity)
        alias = with_alpha(me.style["alias"], p.opacity)
        faint = with_alpha(me.style["axis"], p.opacity * 0.5)
        text = with_alpha(me.style["text"], p.opacity)
        size = me.style["size"] * unit
        fmax = self._fmax(p)
        base, tall = 0.22, 0.62  # the axis and the tallest line, as shares of the panel

        def u(f):
            return np.asarray(f, dtype=np.float64) / fmax

        polyline(ctx, cam, at([0, 1], [base, base]), w * 0.5, color=faint)
        # what the samples can show, DC to Nyquist, lit faintly in the alias's hue
        polyline(ctx, cam, at([0, u(min(p.fs / 2, fmax))], [base, base]), w * 2.2,
                 color=with_alpha(alias, 0.3))  # fmt: skip
        for f, label in ((p.fs / 2, "fs/2"), (p.fs, "fs")):
            if f <= fmax:
                polyline(ctx, cam, at([u(f), u(f)], [base, base + tall * 1.25]), w * 0.5,
                         color=faint)  # fmt: skip
                ctx.text(label, at(u(f), base - 0.16), size, color=text, glow=1.0)
        top = max(sum(abs(a) for _, a, _ in tones), 1e-9)
        lines: dict[float, complex] = {}  # tones that fold onto one frequency add as phasors
        for f, a, ph in tones:
            fa, aa, pa = fm.alias_tone(f, p.fs, a, ph)
            key = round(fa, 9)
            lines[key] = lines.get(key, 0j) + aa * complex(math.cos(pa), math.sin(pa))
        for fa, c in lines.items():  # the aliases under the true lines, as in the wave
            if abs(c) > 1e-12 and fa <= fmax:
                h = base + tall * abs(c) / top
                polyline(ctx, cam, at([u(fa), u(fa)], [base, h]), w * 2.8, color=alias,
                         glow=glow)  # fmt: skip
        for f, a, ph in tones:
            fa = fm.alias_tone(f, p.fs, a, ph)[0]
            h = base + tall * abs(a) / top
            if f <= fmax:
                polyline(ctx, cam, at([u(f), u(f)], [base, h]), w * 1.6, color=true, glow=glow)
            if abs(fa - f) > 1e-9 * max(f, 1.0):  # folded: an arc from the line to its alias
                s = np.linspace(0.0, 1.0, 48)
                f0 = min(f, fmax)
                lift = h + 0.06 + 0.14 * np.sin(math.pi * s)
                polyline(ctx, cam, at(u(f0 + (fa - f0) * s), lift), w * 0.6,
                         color=with_alpha(alias, 0.55), glow=glow)  # fmt: skip

    def _draw_wheel(self, ctx, cam, me, state, box, tones, ts, unit) -> None:
        p = me.params
        glow = me.style["glow"]
        beacon = glow * 1.2 if glow > 1.0 else 1.0
        w = me.style["stroke"] * unit
        true = with_alpha(me.style["color"], p.opacity)
        alias = with_alpha(me.style["alias"], p.opacity)
        sample = with_alpha(me.style["sample"], p.opacity)
        faint = with_alpha(me.style["axis"], p.opacity * 0.5)
        f, _, ph = max(tones, key=lambda tone: abs(tone[1]))
        centre = self._at(me, state, box)(0.5, 0.5)
        radius = 0.34 * _LOW * p.extent[1]
        sign = _ydir(me, state)

        def ring(theta, r):
            theta = np.asarray(theta, dtype=np.float64)
            r = np.broadcast_to(np.asarray(r, dtype=np.float64), theta.shape)
            return np.stack([centre[0] + r * np.cos(theta),
                             centre[1] + sign * r * np.sin(theta),
                             np.full(theta.shape, centre[2])], axis=-1)  # fmt: skip

        circle(ctx, cam, centre, radius, w * 0.5, color=faint)
        polyline(ctx, cam, ring(np.array([math.pi, 0.0]), radius), w * 0.4, color=faint)
        ts = ts[-p.wheel_samples :]
        theta = 2 * math.pi * f * ts + ph  # the phasor's angle: its height is the sample
        k = len(ts)
        if k >= 2:  # the last step: the turn it really makes, and the short one it seems to
            step = 2 * math.pi * f / p.fs
            s = np.linspace(0.0, 1.0, max(int(48 * (1 + abs(step) / math.pi)), 24))
            spiral = ring(theta[-2] + step * s, radius * (0.4 + 0.42 * s))
            polyline(ctx, cam, spiral, w * 0.8, color=with_alpha(true, 0.8), glow=glow)
            seen = 2 * math.pi * float(fm.fold(f, p.fs)) / p.fs
            arc = ring(theta[-2] + seen * s, radius * 1.2)
            polyline(ctx, cam, arc, w * 1.6, color=alias, glow=glow)
            disc_batch(ctx, cam, arc[-1:], unit * 0.008, colors=alias, glow=beacon)
        dots = ring(theta, radius)
        polyline(ctx, cam, np.array([centre, dots[-1]]), w * 0.8, color=sample)
        light = np.tile(np.array(sample, dtype=np.float64), (k, 1))
        light[:, 3] *= np.linspace(0.3, 1.0, k)
        disc_batch(ctx, cam, dots, unit * 0.010, colors=light, glow=beacon)
