"""Filtering: a digital filter, its poles and zeros, its response, and what it
does to a signal and its spectrum.

    from phasor import cx, filt

    low = dict(kind="lowpass", order=4, cutoff=1000.0, fs=8000.0)
    s.add(filt.transfer_plane("plane", pos=(-4, 1), scale=2.0, **low))  # the coloring under it
    s.add(filt.PoleZero(id="pz", pos=(-4, 1), scale=2.0, z_index=1, **low))
    s.add(filt.FrequencyResponse(id="gain", plot="gain", **low))
    s.add(filt.FilteredSignal(id="wave", frequencies=(300, 2800), amplitudes=(1, 1), **low))
    s.add(filt.FilteredSpectrum(id="spec", frequencies=(300, 2800), amplitudes=(1, 1), **low))

Every object takes the same filter parameters (``kind``, ``order``, ``cutoff``, ``cutoff2``,
``fs``; ``custom`` takes ``zeros``, ``poles``, ``gain``), so they describe one filter. Numerics
are in ``filters_math``; the tones of ``FilteredSignal`` and ``FilteredSpectrum`` are sinusoids
(no randomness), and the spectrum is the existing ``fourier_math.spectrum`` (window and all).

``PhasorField`` is the other view of a filter: its modes, one turning hand ``r_k lambda_k^n`` per
pole, with the exact motion blur of the fast ones (it takes ``poles`` and ``residues`` of its
own, any number, not the design parameters):

    s.add(filt.PhasorField(id="modes", poles=filt.polar(rho, theta), tick=expr("30 * t"),
                           frame_ticks=30 / 60))
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import filters_math as fm
from phasor.domains import fourier_math as fo
from phasor.domains.cx import DomainColoring
from phasor.domains.fourier import _ydir
from phasor.domains.strokes import (
    disc_batch,
    lut_colors,
    polyline,
    polylines,
    segment_batch,
    with_alpha,
)

polar = fm.polar  # poles as (re, im) pairs from radii and angles, for PhasorField

_CACHE: dict = {}
_LIMIT = 64


def _memo(key, make):
    if key not in _CACHE:
        if len(_CACHE) >= _LIMIT:
            _CACHE.clear()
        _CACHE[key] = make()
    return _CACHE[key]


class FilterParams(ParamsBase):
    kind: Literal["lowpass", "highpass", "bandpass", "bandstop", "notch", "custom"] = "lowpass"
    order: int = Field(4, ge=1, le=12)  # Butterworth order (per band edge for the band kinds)
    cutoff: float = Field(1000.0, gt=0)  # half-power frequency in Hz (the notch frequency)
    cutoff2: float = Field(2000.0, gt=0)  # the upper edge of a band (band-pass, band-stop)
    fs: float = Field(8000.0, gt=0)  # sample rate in Hz
    radius: float = Field(0.95, gt=0, lt=1)  # notch: how close the poles sit to the circle
    zeros: tuple[tuple[float, float], ...] = ()  # custom: (re, im) of each, in conjugate pairs
    poles: tuple[tuple[float, float], ...] = ()
    gain: float = 1.0  # custom: the constant k in H(z) = k prod(z - z_i) / prod(z - p_i)

    @model_validator(mode="after")
    def _design(self):
        try:
            _filter(self)
        except fm.FilterError as e:
            raise ValueError(str(e)) from None
        return self


def _xyz(x: float, y: float) -> np.ndarray:
    return np.array([x, y, 0.0], dtype=np.float64)


def _beacon(glow: float) -> float:
    """A point of light shines a little brighter than the strokes round it, once the look glows."""
    return glow * 1.2 if glow > 1.0 else 1.0


def _filter(p) -> fm.Filter:
    key = ("filter", p.kind, p.order, p.cutoff, p.cutoff2, p.fs, p.radius, p.zeros, p.poles, p.gain)
    return _memo(key, lambda: fm.design(p.kind, p.order, p.cutoff, p.fs, p.cutoff2, p.radius,
                                        p.zeros, p.poles, p.gain))  # fmt: skip


def _cutoffs(p) -> tuple[float, ...]:
    return fm.cutoffs(p.kind, p.cutoff, p.cutoff2)


def transfer_plane(
    id: str,
    *,
    pos: Vec3 = (0.0, 0.0, 0.0),
    scale: float = 2.0,
    size: tuple[float, float] = (6.0, 5.0),
    **filter_params,
) -> DomainColoring:
    """The ``cx.DomainColoring`` of the filter's transfer function H(z), placed to sit under a
    ``PoleZero`` with the same ``pos`` and ``scale`` (the prism look lights its zeros and poles).
    ``filter_params`` are the filter's (``kind``, ``order``, ``cutoff``, ``fs``...)."""
    f = _filter(FilterParams(**filter_params))
    return DomainColoring(id=id, function=fm.transfer_source(f, tuple(pos[:2]), scale),
                          pos=pos, size=size)  # fmt: skip


# ---------------------------------------------------------------- PoleZero


@register
class PoleZero(ObjectBase):
    """The z plane of a filter: the unit circle, the zeros (rings) and poles (crosses) with their
    multiplicity, and a probe on the circle at ``probe`` (0 to 1: DC to half the sample rate) whose
    distances to the poles and zeros set the response there. ``scale`` is world units per unit of z
    and ``pos`` is z = 0, so it lines up with ``transfer_plane`` at the same ``pos`` and
    ``scale``."""

    type: ClassVar[str] = "pole_zero"
    Style: ClassVar = {
        "zero": ("emphasis", "primary"),
        "pole": ("emphasis", "primary"),
        "circle": ("emphasis", "muted"),
        "probe": ("emphasis", "highlight"),
        "text": ("emphasis", "primary"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(FilterParams):
        pos: Vec3 = (0.0, 0.0, 0.0)  # where z = 0 is
        size: tuple[float, float] = (6.0, 5.0)  # the window, as for the coloring under it
        scale: float = Field(2.0, gt=0)  # world units per unit of z
        probe: float = Field(-1.0, ge=-1, le=1)  # on the circle at w = pi * probe; below 0: hidden

    @staticmethod
    def _groups(roots) -> list[tuple[complex, int]]:
        out: dict[tuple[float, float], list] = {}
        for r in roots:
            out.setdefault(
                (round(float(r.real), 6) + 0.0, round(float(r.imag), 6) + 0.0), []
            ).append(r)
        return [(complex(*k), len(v)) for k, v in sorted(out.items())]

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        return Bounds((x - p.size[0] / 2, y - p.size[1] / 2, z),
                      (x + p.size[0] / 2, y + p.size[1] / 2, z))  # fmt: skip

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        f = _filter(p)
        o = np.asarray(p.pos, dtype=np.float64)

        def pts(roots):
            return np.array([o + _xyz(p.scale * r.real, p.scale * r.imag) for r in roots],
                            dtype=np.float64).reshape(-1, 3)  # fmt: skip

        return {"center": o, "zeros": pts(f.zeros), "poles": pts(f.poles)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _filter(p)

        def listed(roots):
            return [[round(r.real, 6), round(r.imag, 6), n] for r, n in self._groups(roots)]

        return {"kind": "pole_zero", "filter": p.kind, "order": f.order, "stable": f.stable,
                "zeros": listed(f.zeros), "poles": listed(f.poles), "gain": round(f.gain, 9),
                "max_pole_radius": round(float(np.abs(f.poles).max()), 6) if f.order else 0.0,
                }  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        f = _filter(p)
        o = np.asarray(p.pos, dtype=np.float64)
        unit = float(cam.units_per_height(o))
        w = me.style["stroke"] * unit
        glow = me.style["glow"]
        half = np.asarray(p.size) / 2

        def at(z):
            return o + np.array([p.scale * z.real, p.scale * z.imag, 0.0])

        def inside(z):
            return abs(p.scale * z.real) <= half[0] and abs(p.scale * z.imag) <= half[1]

        faint = with_alpha(me.style["circle"], p.opacity * 0.5)
        polyline(ctx, cam, [o - _xyz(half[0], 0), o + _xyz(half[0], 0)], w * 0.5, color=faint)
        polyline(ctx, cam, [o - _xyz(0, half[1]), o + _xyz(0, half[1])], w * 0.5, color=faint)
        a = np.linspace(0, 2 * math.pi, 160)
        ring = o + p.scale * np.c_[np.cos(a), np.sin(a), np.zeros(len(a))]
        polyline(ctx, cam, ring, w * 0.8, color=with_alpha(me.style["circle"], p.opacity * 0.9),
                 glow=glow)  # fmt: skip
        r = unit * 0.02
        zc = with_alpha(me.style["zero"], p.opacity)
        for z, n in self._groups(f.zeros):
            if not inside(z):
                continue
            c = at(z)
            t = np.linspace(0, 2 * math.pi, 28)
            polyline(ctx, cam, c + r * np.c_[np.cos(t), np.sin(t), np.zeros(len(t))], w,
                     color=zc, glow=glow)  # fmt: skip
            if n > 1:
                ctx.text(str(n), c + _xyz(0, r * 2.4), me.style["size"] * unit, color=zc, glow=1.0)
        pc = with_alpha(me.style["pole"], p.opacity)
        for z, n in self._groups(f.poles):
            if not inside(z):
                continue
            c = at(z)
            for d in ([1, 1, 0], [1, -1, 0]):
                d = np.asarray(d, dtype=np.float64) * r * 0.9
                polyline(ctx, cam, [c - d, c + d], w, color=pc, glow=glow)
            if n > 1:
                ctx.text(str(n), c + _xyz(0, r * 2.4), me.style["size"] * unit, color=pc, glow=1.0)
        if p.probe >= 0:
            e = at(complex(math.cos(math.pi * p.probe), math.sin(math.pi * p.probe)))
            pc = with_alpha(me.style["probe"], p.opacity)
            polyline(ctx, cam, [o, e], w * 0.7, color=with_alpha(pc, 0.7), glow=glow)
            disc_batch(ctx, cam, e[None], unit * 0.014, colors=pc, glow=_beacon(glow))


# ---------------------------------------------------------------- the plots


class _PlotParams(FilterParams):
    pos: Vec3 = (0.0, 0.0, 0.0)  # the plot's centre
    extent: tuple[float, float] = (7.0, 2.0)


def _plot_bounds(me, state: SceneState) -> Bounds:
    p = me.params
    x, y, z = p.pos
    return Bounds((x - p.extent[0] / 2, y - p.extent[1] / 2, z),
                  (x + p.extent[0] / 2, y + p.extent[1] / 2, z))  # fmt: skip


def _grid(p, n: int = 480) -> np.ndarray:
    return np.linspace(0.0, p.fs / 2, n)


def _plotter(me, state: SceneState):
    """(x(share 0..1 of the width), y(share 0..1 of the height)) -> world points."""
    p = me.params
    sign = _ydir(me, state)
    x0, y0, z = p.pos[0] - p.extent[0] / 2, p.pos[1], p.pos[2]

    def at(u, v):
        u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        return np.stack([x0 + u * p.extent[0], y0 + sign * (v - 0.5) * p.extent[1],
                         np.full(np.broadcast(u, v).shape, z)], axis=-1)  # fmt: skip

    return at


# ---------------------------------------------------------------- FrequencyResponse


@register
class FrequencyResponse(ObjectBase):
    """The gain (in dB, or linear) or the unwrapped phase of the filter from DC to half the sample
    rate, as a curve; ``reveal`` 0 -> 1 traces it with a bright point at its head (the same
    sweep as ``PoleZero.probe``). The cutoffs are marked."""

    type: ClassVar[str] = "frequency_response"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "guide": ("emphasis", "muted"),
        "text": ("emphasis", "muted"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(_PlotParams):
        plot: Literal["gain", "phase"] = "gain"
        scale: Literal["db", "linear"] = "db"  # the gain's axis
        floor_db: float = Field(-60.0, lt=0)  # the bottom of a dB plot
        reveal: float = Field(1.0, ge=0, le=1)  # the share of the frequency axis drawn

    def _curve(self, p):
        f = _filter(p)

        def make():
            fr = _grid(p)
            h = f.response(fr)
            return fr, h

        return _memo(("resp", _filter_key(p)), make)

    def _values(self, p) -> tuple[np.ndarray, np.ndarray, tuple[float, float], list[float]]:
        """(frequency, y in 0..1 units of the plot, (low, high) of the axis, guide levels)."""
        fr, h = self._curve(p)
        if p.plot == "phase":
            ph = fm.phase(h)
            lo = math.floor(ph.min() / (math.pi / 2)) * math.pi / 2
            hi = max(math.ceil(ph.max() / (math.pi / 2)) * math.pi / 2, lo + math.pi)
            return fr, (ph - lo) / (hi - lo), (lo, hi), [0.0] if lo < 0 < hi else []
        if p.scale == "db":
            top = 6.0
            db = fm.gain_db(h, p.floor_db)
            return fr, (db - p.floor_db) / (top - p.floor_db), (p.floor_db, top), [0.0, -3.0]
        top = max(1.0, float(np.abs(h).max())) * 1.1
        return fr, np.abs(h) / top, (0.0, top), [1.0]

    def world_bounds(self, state: SceneState) -> Bounds:
        return _plot_bounds(self.me(state), state)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        me = self.me(state)
        p = me.params
        at = _plotter(me, state)
        fr, y, _, _ = self._values(p)
        k = min(int(p.reveal * (len(fr) - 1)), len(fr) - 1)
        return {"center": np.asarray(p.pos, dtype=np.float64),
                "head": at(fr[k] / (p.fs / 2), y[k])}  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _filter(p)
        cuts = _cutoffs(p)
        out: dict[str, Any] = {"kind": "frequency_response", "plot": p.plot, "filter": p.kind,
                               "cutoffs_hz": [float(c) for c in cuts],
                               "stable": f.stable}  # fmt: skip
        if cuts:
            h = f.response(np.array(cuts))
            out["gain_db_at_cutoff"] = [round(float(v), 4) for v in fm.gain_db(h)]
            out["phase_at_cutoff"] = [round(float(self._phase_at(f, c)), 6) for c in cuts]
        _, hh = self._curve(p)
        out["peak_gain_db"] = round(float(fm.gain_db(hh).max()), 4)
        out["dc_gain_db"] = round(float(fm.gain_db(hh[:1])[0]), 4)
        out["nyquist_gain_db"] = round(float(fm.gain_db(hh[-1:])[0]), 4)
        return out

    @staticmethod
    def _phase_at(f: fm.Filter, freq: float) -> float:
        """The unwrapped phase at ``freq``, counted from DC."""
        fr = np.linspace(0.0, freq, 4001)
        return float(fm.phase(f.response(fr))[-1])

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = float(cam.units_per_height(np.asarray(p.pos, dtype=np.float64)))
        w = me.style["stroke"] * unit
        glow = me.style["glow"]
        at = _plotter(me, state)
        faint = with_alpha(me.style["guide"], p.opacity * 0.45)
        polyline(ctx, cam, at([0, 1], [0, 0]), w * 0.5, color=faint)  # the frequency axis
        polyline(ctx, cam, at([0, 0], [0, 1]), w * 0.5, color=faint)
        fr, y, (lo, hi), guides = self._values(p)
        for g in guides:  # 0 dB, -3 dB, unity gain, zero phase
            v = (g - lo) / (hi - lo)
            if 0 <= v <= 1:
                polyline(ctx, cam, at([0, 1], [v, v]), w * 0.4, color=with_alpha(faint, 0.6))
        text = with_alpha(me.style["text"], p.opacity)
        size = me.style["size"] * unit
        for c in _cutoffs(p):
            u = c / (p.fs / 2)
            polyline(ctx, cam, at([u, u], [0, 1]), w * 0.5, color=with_alpha(faint, 0.8))
            ctx.text(f"{c:g} Hz", at(u, -0.12), size, color=text, glow=1.0)
        label = {"gain": "gain (dB)" if p.scale == "db" else "gain", "phase": "phase"}[p.plot]
        ctx.text(label, at(0.0, 1.14), size, color=text, anchor="left", glow=1.0)
        n = max(round(p.reveal * (len(fr) - 1)), 1) + 1 if p.reveal > 0 else 0
        if n < 2:
            return
        color = with_alpha(me.style["color"], p.opacity)
        jump = 0.4 if p.plot == "phase" else 0.5
        for u, v in fm.curve_runs(fr[:n] / (p.fs / 2), np.clip(y[:n], -0.05, 1.05), jump):
            polyline(ctx, cam, at(u, v), w * 1.4, color=color, glow=glow)
        if p.reveal < 1:
            disc_batch(ctx, cam, at(fr[n - 1] / (p.fs / 2), y[n - 1])[None], unit * 0.012,
                       colors=color, glow=_beacon(glow))  # fmt: skip


def _filter_key(p) -> tuple:
    return (p.kind, p.order, p.cutoff, p.cutoff2, p.fs, p.radius, p.zeros, p.poles, p.gain)


# ---------------------------------------------------------------- tones through the filter


class _ToneParams(_PlotParams):
    frequencies: tuple[float, ...] = (300.0, 1800.0)  # the sinusoids in the signal, Hz
    amplitudes: tuple[float, ...] = (1.0, 1.0)

    @model_validator(mode="after")
    def _tones(self):
        if len(self.amplitudes) != len(self.frequencies):
            raise ValueError("amplitudes needs one value per frequency")
        if any(not 0 <= f < self.fs / 2 for f in self.frequencies):
            raise ValueError(f"every frequency must be between 0 and {self.fs / 2} Hz (fs / 2)")
        return self


# ---------------------------------------------------------------- FilteredSignal


@register
class FilteredSignal(ObjectBase):
    """A sum of sinusoids before the filter (faint) and after it (bright), from rest, so the
    start-up transient shows. ``progress`` 0 -> 1 traces both along time."""

    type: ClassVar[str] = "filtered_signal"
    Style: ClassVar = {
        "before": ("emphasis", "muted"),
        "after": ("emphasis", "highlight"),
        "text": ("emphasis", "muted"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(_ToneParams):
        extent: tuple[float, float] = (7.0, 2.4)
        duration: float = Field(0.03, gt=0)  # seconds shown
        progress: float = Field(1.0, ge=0, le=1)

    def _signals(self, p):
        n = max(round(p.duration * p.fs), 8)
        key = ("sig", _filter_key(p), p.frequencies, p.amplitudes, n)
        return _memo(key, lambda: fm.filtered_tones(_filter(p), p.frequencies, p.amplitudes, n))

    def world_bounds(self, state: SceneState) -> Bounds:
        return _plot_bounds(self.me(state), state)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        x, y = self._signals(p)
        h = len(x) // 2  # past the transient
        rms = lambda v: float(np.sqrt(np.mean(np.square(v))))  # noqa: E731
        return {"kind": "filtered_signal", "filter": p.kind, "samples": len(x),
                "input_rms": round(rms(x), 6), "output_rms": round(rms(y), 6),
                "late_gain": round(rms(y[h:]) / max(rms(x[h:]), 1e-12), 6),
                "progress": round(p.progress, 6)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = float(cam.units_per_height(np.asarray(p.pos, dtype=np.float64)))
        w = me.style["stroke"] * unit
        glow = me.style["glow"]
        at = _plotter(me, state)
        x, y = self._signals(p)
        top = max(float(np.abs(x).max()), 1e-12)
        faint = with_alpha(me.style["before"], p.opacity * 0.4)
        polyline(ctx, cam, at([0, 1], [0.5, 0.5]), w * 0.5, color=faint)
        size = me.style["size"] * unit
        before = with_alpha(me.style["before"], p.opacity)
        after = with_alpha(me.style["after"], p.opacity)
        ctx.text("before", at(0.0, 1.1), size, color=before, anchor="left", glow=1.0)
        ctx.text("after", at(0.12, 1.1), size, color=after, anchor="left", glow=1.0)
        n = max(round(p.progress * len(x)), 0)
        if n < 2:
            return
        u = np.arange(n) / (len(x) - 1)
        polyline(ctx, cam, at(u, 0.5 + 0.46 * x[:n] / top), w, color=with_alpha(before, 0.8))
        polyline(
            ctx,
            cam,
            at(u, 0.5 + np.clip(0.46 * y[:n] / top, -0.5, 0.5)),
            w * 1.5,
            color=after,
            glow=glow,
        )


# ---------------------------------------------------------------- FilteredSpectrum


@register
class FilteredSpectrum(ObjectBase):
    """The spectrum of the same tones through a window, before and after the filter: what the
    filter keeps is drawn solid and what it removes on top of it in a second color, which fades
    out as ``fade`` goes 0 -> 1. A faint curve is the filter's gain."""

    type: ClassVar[str] = "filtered_spectrum"
    Style: ClassVar = {
        "kept": ("emphasis", "highlight"),
        "removed": ("emphasis", "danger"),
        "envelope": ("emphasis", "muted"),
        "text": ("emphasis", "muted"),
        "size": ("text", "annotation"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(_ToneParams):
        extent: tuple[float, float] = (7.0, 2.4)
        window: Literal["rect", "hann", "hamming", "blackman"] = "hann"
        samples: int = Field(512, ge=16, le=8192)  # the length of the analysed block
        fade: float = Field(0.0, ge=0, le=1)  # 0: the removed part shown; 1: gone
        envelope: bool = True  # the gain curve behind the bars

    def _spectra(self, p):
        f = _filter(p)

        def make():
            amps_after = fm.tone_amplitudes_after(f, p.frequencies, p.amplitudes)
            before = fo.spectrum(p.frequencies, p.amplitudes, p.window, p.samples, p.fs)
            after = fo.spectrum(p.frequencies, amps_after, p.window, p.samples, p.fs)
            return before["freq"], before["mag"], np.minimum(after["mag"], before["mag"])

        return _memo(("spec", _filter_key(p), p.frequencies, p.amplitudes, p.window, p.samples),
                     make)  # fmt: skip

    def world_bounds(self, state: SceneState) -> Bounds:
        return _plot_bounds(self.me(state), state)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f = _filter(p)
        _, before, after = self._spectra(p)
        gains = np.abs(f.response(np.asarray(p.frequencies)))
        e0 = float(np.sum(before**2))
        return {
            "kind": "filtered_spectrum", "filter": p.kind, "window": p.window,
            "removed_energy_share": round(1 - float(np.sum(after**2)) / max(e0, 1e-300), 6),
            "tones": [{"hz": float(h), "kept": round(float(g), 6)}
                      for h, g in zip(p.frequencies, gains, strict=True)],
            "fade": round(p.fade, 6),
        }  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = float(cam.units_per_height(np.asarray(p.pos, dtype=np.float64)))
        w = me.style["stroke"] * unit
        glow = me.style["glow"]
        at = _plotter(me, state)
        freq, before, after = self._spectra(p)
        top = max(float(before.max()), 1e-12)
        faint = with_alpha(me.style["envelope"], p.opacity * 0.45)
        polyline(ctx, cam, at([0, 1], [0, 0]), w * 0.5, color=faint)
        if p.envelope:
            fr = _grid(p)
            gain = np.abs(_filter(p).response(fr))
            polyline(ctx, cam, at(fr / (p.fs / 2), np.clip(gain, 0, 1.05) * 0.96 + 0.0),
                     w * 0.8, color=with_alpha(faint, 0.9), glow=glow)  # fmt: skip
        u = freq / (p.fs / 2)
        bar = max(0.7 * p.extent[0] / max(len(freq) - 1, 1), w) / unit
        live = before / top > 1e-4
        ul = u[live]
        base = at(ul, np.zeros(len(ul)))
        kept = at(ul, 0.96 * after[live] / top)
        full = at(ul, 0.96 * before[live] / top)
        kept_color = with_alpha(me.style["kept"], p.opacity)
        segment_batch(ctx, cam, base, kept, bar, colors=kept_color, glow=glow)
        left = 1.0 - p.fade
        if left > 0:
            segment_batch(ctx, cam, kept, full, bar,
                          colors=with_alpha(me.style["removed"], p.opacity * 0.85 * left),
                          glow=glow)  # fmt: skip
        label = with_alpha(me.style["text"], p.opacity)
        ctx.text("spectrum", at(0.0, 1.1), me.style["size"] * unit, color=label, anchor="left",
                 glow=1.0)  # fmt: skip


# ---------------------------------------------------------------- PhasorField

_BLUR_MIN = 0.5  # a hand is smeared once its tip sweeps this many stroke widths in a shutter
_LIST_MAX = 32  # semantics list at most this many hand indices


class _PhasorParams(ParamsBase):
    poles: tuple[tuple[float, float], ...] = Field(min_length=1, max_length=4096)  # (re, im)
    residues: tuple[tuple[float, float], ...] = ()  # (re, im), one per pole; none: all 1
    tick: float = 0.0  # n: the hands are r_k lambda_k^n; fractional ticks turn smoothly
    pos: Vec3 = (0.0, 0.0, 0.0)  # the centre of the seats
    layout: Literal["spiral", "grid", "line"] = "spiral"
    order: Literal["angle", "index"] = "angle"  # angle: the slowest hand in the first seat
    spacing: float = Field(1.0, gt=0)  # world units between neighbouring seats
    scale: float = Field(0.0, ge=0)  # world units per unit of |r|; 0: longest hand 0.45 spacing
    decay: Literal["length", "glow", "none"] = "length"  # how rho^n shows
    hue: float = 0.0  # turns the colour wheel (a share of one turn)
    frame_ticks: float = Field(0.0, ge=0)  # ticks per frame (rate / fps); 0: no smear
    shutter: float = Field(0.5, ge=0, le=1)  # share of a frame the shutter is open
    rings: float = Field(0.2, ge=0, le=1)  # opacity of each hand's |r| circle; 0: none

    @model_validator(mode="after")
    def _matching(self):
        if self.residues and len(self.residues) != len(self.poles):
            raise ValueError("residues needs one (re, im) per pole, or none (all 1)")
        return self


@register
class PhasorField(ObjectBase):
    """The modes of a filter as hands: pole ``lambda_k = rho_k e^{i theta_k}`` with residue
    ``r_k`` is a hand ``r_k lambda_k^n`` that turns ``theta_k`` and shrinks by ``rho_k`` each
    tick ``n``. The hands sit on a sunflower spiral (slowest in the middle), a grid or a line;
    each takes the hue of the angle it points at (the phase wheel). With ``frame_ticks`` set,
    a hand that turns far in one frame is drawn as the exact smear a camera with the shutter
    open ``shutter`` of a frame would record: the same light spread over the swept sector, so a
    fast hand is a faint fan or disc instead of a strobing stick, and a slow one is unchanged."""

    type: ClassVar[str] = "phasor_field"
    Life: ClassVar = frozenset({"glow"})  # angle and length are the maths: they never drift
    Style: ClassVar = {
        "colormap": ("colormap", "phase"),
        "seat": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Params: ClassVar = _PhasorParams

    @staticmethod
    def _poles(p) -> np.ndarray:
        return fm._as_complex(p.poles)

    def _seats(self, p) -> np.ndarray:
        lam = self._poles(p)
        order = np.argsort(np.abs(np.angle(lam)), kind="stable") if p.order == "angle" else None
        key = ("seats", len(lam), p.layout, p.spacing, None if order is None else tuple(order))
        xy = _memo(key, lambda: fm.seat_offsets(len(lam), p.layout, p.spacing, order))
        return np.c_[xy, np.zeros(len(lam))] + np.asarray(p.pos, dtype=np.float64)

    @staticmethod
    def _unit(p) -> float:
        """World units per unit of |r|."""
        if p.scale > 0:
            return p.scale
        r = np.abs(fm._as_complex(p.residues)) if p.residues else np.ones(1)
        return 0.45 * p.spacing / max(float(r.max()), 1e-300)

    def _hands(self, p, n: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(angle, length in world units, light) of every hand at tick ``n``."""
        z = fm.mode_hands(p.poles, p.residues, n)
        r = np.abs(fm.mode_hands(p.poles, p.residues, 0.0)) * self._unit(p)
        rho = np.abs(self._poles(p))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            g = np.nan_to_num(np.clip(np.where(rho > 0, rho ** float(n), 0.0), 0.0, 8.0))
        if p.decay == "length":
            return np.angle(z), r * g, np.ones(len(z))
        light = np.clip(g, 0.0, 1.0) if p.decay == "glow" else np.ones(len(z))
        return np.angle(z), r, light

    @staticmethod
    def _sweep_ticks(p) -> float:
        return p.frame_ticks * p.shutter

    @staticmethod
    def _width(me, state: SceneState) -> float:
        cam = state.camera_for(me)
        return me.style["stroke"] * float(cam.units_per_height(np.asarray(me.params.pos)))

    def _blurred(self, me, state: SceneState) -> np.ndarray:
        """Which hands are drawn smeared: those whose tip sweeps over half a stroke width."""
        p = me.params
        s = self._sweep_ticks(p)
        _, length, _ = self._hands(p, p.tick)
        sweep = np.abs(np.angle(self._poles(p))) * s
        return (s > 0) & (length * sweep > _BLUR_MIN * self._width(me, state))

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        seats = self._seats(p)
        _, length, _ = self._hands(p, p.tick)
        ring = np.abs(fm.mode_hands(p.poles, p.residues, 0.0)) * self._unit(p)
        reach = max(float(length.max()), float(ring.max()) if p.rings > 0 else 0.0)
        pad = np.array([reach, reach, 0.0])
        return Bounds(tuple(seats.min(axis=0) - pad), tuple(seats.max(axis=0) + pad))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        seats = self._seats(p)
        angle, length, _ = self._hands(p, p.tick)
        tips = seats + np.c_[length * np.cos(angle), length * np.sin(angle), np.zeros(len(angle))]
        return {"center": np.asarray(p.pos, dtype=np.float64), "seats": seats, "tips": tips}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        me = self.me(state)
        p = me.params
        lam = self._poles(p)
        speed = np.abs(np.angle(lam))  # radians per tick
        per_frame = np.degrees(speed * p.frame_ticks)
        blurred = np.flatnonzero(self._blurred(me, state))
        aliased = np.flatnonzero(per_frame > 180.0)  # the wagon wheel: it seems to turn back
        total = complex(fm.mode_hands(p.poles, p.residues, p.tick).sum())
        fastest = int(np.argmax(speed))
        return {
            "kind": "phasor_field", "count": len(lam), "tick": round(p.tick, 6),
            "layout": p.layout, "decay": p.decay, "stable": bool(np.all(np.abs(lam) < 1)),
            "fastest": fastest, "fastest_deg_per_tick": round(float(np.degrees(speed[fastest])), 4),
            "max_turn_per_frame_deg": round(float(per_frame.max()), 4),
            "blurred_count": len(blurred), "blurred": blurred[:_LIST_MAX].tolist(),
            "aliased_count": len(aliased), "aliased": aliased[:_LIST_MAX].tolist(),
            "sum": [round(total.real, 9), round(total.imag, 9)],
        }  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        glow = me.style["glow"]
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        w = self._width(me, state)
        px = float(cam.units_per_height(np.asarray(p.pos))) / max(state.height, 1)
        seats = self._seats(p)
        if p.rings > 0:  # each hand's starting length |r_k|: how far it has decayed
            a = np.linspace(0, 2 * math.pi, 49)
            reach = np.abs(fm.mode_hands(p.poles, p.residues, 0.0)) * self._unit(p)
            ring = np.c_[np.cos(a), np.sin(a), np.zeros(len(a))]
            polylines(ctx, cam, seats[:, None, :] + reach[:, None, None] * ring[None], w * 0.5,
                      color=with_alpha(me.style["seat"], p.opacity * p.rings))  # fmt: skip
        disc_batch(ctx, cam, seats, w * 0.9, colors=with_alpha(me.style["seat"], p.opacity))
        angle, length, light = self._hands(p, p.tick)
        blur = self._blurred(me, state)
        sharp = np.flatnonzero(~blur & (length > 0))
        if len(sharp):
            d = np.c_[np.cos(angle[sharp]), np.sin(angle[sharp]), np.zeros(len(sharp))]
            ends = np.stack([seats[sharp], seats[sharp] + length[sharp, None] * d], axis=1)
            cols = lut_colors(lut, fm.phase_hue(angle[sharp], p.hue))
            cols[:, 3] = light[sharp]
            polylines(ctx, cam, ends, w, color=(1.0, 1.0, 1.0, p.opacity), glow=glow,
                      colors=np.repeat(cols[:, None], 2, axis=1))  # fmt: skip
        s = self._sweep_ticks(p)
        theta = np.angle(self._poles(p))
        opened, closed = self._hands(p, p.tick - s / 2), self._hands(p, p.tick + s / 2)
        pos, col, tris, base = [], [], [], 0
        for k in np.flatnonzero(blur):
            sweep = float(theta[k] * s)
            tip = max(opened[1][k], closed[1][k]) / px  # its length in pixels
            columns = int(np.clip(math.ceil(abs(sweep) * tip / 4), 2, 256))
            xy, lit, ang = fm.swept_hand(float(opened[0][k]), sweep, float(opened[1][k]),
                                         float(closed[1][k]), w, columns, 20 if tip > 40 else 10,
                                         float(opened[2][k]), float(closed[2][k]))  # fmt: skip
            pos.append(np.c_[xy.reshape(-1, 2), np.zeros(lit.size)] + seats[k])
            c = lut_colors(lut, fm.phase_hue(ang.ravel(), p.hue))
            c[:, 3] = np.clip(lit.ravel(), 0.0, 1.0)
            col.append(c)
            tris.append(fm.grid_triangles(*lit.shape) + base)
            base += lit.size
        if pos:
            ctx.triangles(np.vstack(pos), np.vstack(tris), color=(1.0, 1.0, 1.0, p.opacity),
                          glow=glow, colors=np.vstack(col))  # fmt: skip
