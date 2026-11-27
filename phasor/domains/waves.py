"""Wave fields of point sources, drawn per pixel on the GPU.

    from phasor import Scene, ac, expr

    s.add(ac.Interference(id="two", sources=((-1, 0), (1, 0)), time=expr("t")))
    s.add(ac.PhasedArray(id="array", center=(-3, 0), count=4, steer=30, time=expr("t")))

Both are rectangles (``pos``, ``size``) in the plane z = 0 through which the field is seen;
sources are at their own world (x, y), like the complex plane of ``cx`` (so the camera
zooms into the field like into anything else). The ``wave_field`` kernel sums the sources
for each pixel; the colour is a heat map (``heatmap.shade_values``): in the prism look,
crests and troughs glow in hues of the palette that run with the value, and the lines where
the waves cancel stay dark.

Motion is all through parameters that tween, and every frame is a pure function of them:
``time`` (usually ``expr("t")``) moves the waves outward; ``onsets`` switch sources on one by
one (each sends out a ring that spreads at ``speed``); ``blend`` fades between the field and
its time-averaged intensity; moving ``sources`` slides the fringes; ``steer`` swings a phased
array's beam. Wavelength is ``speed / frequency``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import waves_math as wm
from phasor.domains.heatmap import AMBIENT, Recipe, field_pixels, wheel_recipe
from phasor.domains.strokes import disc_batch, polyline, with_alpha

MAX_SOURCES = 64
CORE = 0.1  # wavelengths: the spreading 1 / sqrt(r) is held at this distance from a source
FRONT = 0.5  # wavelengths: how soft the ring a source sends out when it starts is
BEACON = 0.014  # beacon radius, a share of the frame height
PULSE = 0.3  # how much a beacon's light rises and falls with its source's oscillation

# The prism preset (the prism looks repeat these values in their tables): dark where the
# waves cancel, crests and troughs glowing in spectrum hues half a wheel apart, contour
# lines of light, edges melting into the dark.
PRISM: dict = {
    "params": {"hue_cycles": 0.5, "isolines": 5, "fade": 0.08, "ambient": 0.0,
               "contrast": 1.6, "gain": 1.2},
    "style": {"glow": "soft", "colormap": "spectrum"},
}  # fmt: skip


class WaveParams(ParamsBase):
    """What both wave fields share."""

    frequency: float = Field(1.0, gt=0)  # Hz (cycles per second of ``time``)
    speed: float = Field(1.0, gt=0)  # world units per second; wavelength = speed / frequency
    time: float = 0.0  # seconds; ``expr("t")`` sets the waves moving
    mode: Literal["field", "intensity"] = "field"  # the instantaneous field, or its average
    blend: float = Field(0.0, ge=0, le=1)  # fades from ``mode`` to the other (tween it)
    amplitudes: tuple[float, ...] = ()  # one per source (default 1 each)
    onsets: tuple[float, ...] = ()  # one per source: the ``time`` it starts (default: always on)
    pos: Vec3 = (0.0, 0.0, 0.0)  # centre of the rectangle the field is seen in
    size: tuple[float, float] = (8.0, 4.5)
    gain: float = Field(1.0, gt=0)  # how strongly the field turns into colour
    markers: bool = True  # the sources as glowing beacons
    # colour (as the heat maps' ``HeatParams``): > 0 runs the colormap as a wheel this many
    # turns over the value range (prism); 0: two tints, crests and troughs
    hue_cycles: float = Field(0.0, ge=0, le=8)
    hue: float = 0.0  # turns the colours (share of a turn); life adds a little
    isolines: int = Field(0, ge=0, le=64)  # contour levels drawn as thin light
    fade: float = Field(0.0, ge=0, le=0.5)  # edges melt into the background over this share
    ambient: float = Field(AMBIENT, ge=0, le=1)  # with glow: the colour a quiet pixel keeps
    contrast: float = Field(1.0, ge=0.25, le=8)  # with glow: the weight to this power

    @field_validator("size")
    @classmethod
    def _positive(cls, v):
        if min(v) <= 0:
            raise ValueError("size must be positive")
        return v


def _per_source(n: int, values: tuple[float, ...], name: str) -> None:
    if values and len(values) != n:
        raise ValueError(f"{n} sources but {len(values)} {name}")


class _WaveField(ObjectBase):
    """The field, the beacons and what both types report. Subclasses give the sources."""

    Life: ClassVar = frozenset({"hue", "glow"})
    Style: ClassVar = {
        "positive": ("emphasis", "highlight"),  # crests, without hue_cycles
        "negative": ("emphasis", "secondary"),  # troughs
        "colormap": ("colormap", "spectrum"),  # with hue_cycles
        "glow": ("glow", "none"),
        "beacon": ("emphasis", "highlight"),
        "beacon_glow": ("glow", "strong"),
    }
    Presets: ClassVar = {"prism": PRISM}

    # -- what subclasses give
    def _points(self, p) -> np.ndarray:  # (n, 2) world positions
        raise NotImplementedError

    def _phases(self, p, pts: np.ndarray) -> np.ndarray:  # radians
        raise NotImplementedError

    # -- the sources at this state
    @staticmethod
    def _wave(p) -> tuple[float, float, float]:
        """(wavelength, wavenumber, w t reduced to one turn)."""
        lam = wm.wavelength(p.speed, p.frequency)
        return lam, 2 * math.pi / lam, (2 * math.pi * ((p.frequency * p.time) % 1.0))

    def _table(self, p) -> np.ndarray:
        pts = self._points(p)
        n = len(pts)
        amps = np.asarray(p.amplitudes, dtype=np.float64) if p.amplitudes else np.ones(n)
        if p.onsets:
            reach = p.speed * (p.time - np.asarray(p.onsets, dtype=np.float64))
        else:
            reach = np.full(n, wm.ALWAYS)
        return wm.source_table(pts, amps, self._phases(p, pts), reach)

    def _rect(self, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w, h = p.size
        corner = np.asarray(p.pos, dtype=np.float64) - np.array([w / 2, h / 2, 0.0])
        return corner, np.array([w, 0.0, 0.0]), np.array([0.0, h, 0.0])

    def world_bounds(self, state: SceneState) -> Bounds:
        corner, r, u = self._rect(self.me(state).params)
        return Bounds.of_points([corner, corner + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        pts = self._points(p)
        return {"center": np.asarray(p.pos, dtype=np.float64),
                "sources": np.c_[pts, np.full(len(pts), p.pos[2])]}  # fmt: skip

    def _semantics(self, p) -> dict[str, Any]:
        table = self._table(p)
        lam, _, _ = self._wave(p)
        on = table[:, 4] > 0
        return {
            "backdrop": True,
            "mode": p.mode,
            "blend": round(float(p.blend), 6),
            "wavelength": round(lam, 6),
            "frequency": p.frequency,
            "speed": p.speed,
            "time": round(float(p.time), 6),
            "sources": [[round(float(x), 6), round(float(y), 6)] for x, y in table[:, :2]],
            "sources_on": int(on.sum()),
        }

    # -- drawing
    def pixels_for(self, state: SceneState):
        """The ``ctx.field`` pixel function of this state (tests call it directly)."""
        me = self.me(state)
        p = me.params
        corner, _, _ = self._rect(p)
        table = self._table(p)
        lam, k, wt = self._wave(p)
        glow = float(me.style.get("glow", 1.0))
        amps = np.abs(table[:, 2])
        # brightness against the sources' total amplitude a wavelength away, so two sources
        # and eight light the field alike; tanh keeps the strongest parts from clipping
        g = p.gain * math.sqrt(lam) / max(float(amps.sum()), 1e-9)
        share = p.blend if p.mode == "field" else 1.0 - p.blend  # of the intensity picture
        # The signal: tanh(g * field), blended by ``share`` with the intensity, which goes
        # as |A|^2: (tanh of the envelope) squared. Its level: the field runs -1..1 (troughs
        # to crests, silence in the middle), the intensity 0..1 (silence to loud), so it
        # uses the whole range too; blended in between.
        how = {"transform": "wave", "gain": g, "share": share, "level": "signed",
               "weight": "abs" if glow > 1 else None}  # fmt: skip
        if p.hue_cycles > 0:
            # read downward from the colormap's warm end: crests (and loud places) warm,
            # troughs cool, the quiet middle between them
            recipe = wheel_recipe(me.style["colormap"], p, hue_of="inverse", **how)
        else:
            recipe = Recipe(colour="sign", positive=me.style["positive"][:3],
                            negative=me.style["negative"][:3], mix=True, **how)  # fmt: skip

        def wave(tile, center: tuple[float, float], height: float):
            from phasor.render import gpu

            return gpu.run("wave_field", tile, sources=table, wavenumber=k, omega_t=wt,
                           center=(corner[0] + center[0], corner[1] + center[1]),
                           height=height, core=CORE * lam, edge=FRONT * lam)  # fmt: skip

        # the field stays where the kernel left it and is coloured there
        return field_pixels(wave, p.size, recipe, isolines=p.isolines,
                            background=state.background, ambient=p.ambient, fade=p.fade,
                            contrast=p.contrast)  # fmt: skip

    def _beacons(self, ctx: DrawContext, state: SceneState) -> None:
        """Each source that has started as a beacon, its light rising and falling with the
        source's own oscillation, fading in as its first ring leaves."""
        me = self.me(state)
        p = me.params
        if not p.markers:
            return
        table = self._table(p)
        lam, _, wt = self._wave(p)
        cam = state.camera_for(me)
        z = p.pos[2]
        centers = np.c_[table[:, :2], np.full(len(table), z)]
        on = np.clip(table[:, 4] / (FRONT * lam), 0.0, 1.0)
        amps = np.abs(table[:, 2])
        strength = on * amps / max(float(amps.max()), 1e-9)
        pulse = 1.0 - PULSE + PULSE * np.cos(table[:, 3] - wt)
        keep = strength > 0
        if not keep.any():
            return
        base = np.asarray(me.style["beacon"], dtype=np.float64)
        cols = np.repeat(base[None], len(table), axis=0)
        cols[:, :3] *= pulse[:, None]
        cols[:, 3] *= strength * p.opacity
        unit = float(cam.units_per_height(np.asarray(p.pos, dtype=np.float64)))
        disc_batch(ctx, cam, centers[keep], unit * BEACON, colors=cols[keep],
                   glow=float(me.style["beacon_glow"]))  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, r, u = self._rect(p)
        ctx.field(corner, r, u, pixels=self.pixels_for(state), opacity=p.opacity,
                  glow=float(me.style.get("glow", 1.0)))  # fmt: skip
        self._beacons(ctx, state)


@register
class Interference(_WaveField):
    """Point sources of one frequency and the field they make together: crests, troughs and
    the lines where they cancel (where the distances to two sources differ by half a
    wavelength, plus whole ones). ``sources`` are world (x, y) points; ``phases`` (degrees)
    and ``amplitudes`` one per source. ``time`` moves the waves (``expr("t")``); ``onsets``
    switch sources on one by one, each sending out a ring; ``mode`` ``intensity`` (or
    ``blend`` toward it) shows the time average, where the fringes stand still."""

    type: ClassVar[str] = "interference"

    class Params(WaveParams):
        sources: tuple[tuple[float, float], ...] = ((-1.0, 0.0), (1.0, 0.0))
        phases: tuple[float, ...] = ()  # degrees, one per source (default 0: in phase)

        @model_validator(mode="after")
        def _lengths(self):
            n = len(self.sources)
            if not 1 <= n <= MAX_SOURCES:
                raise ValueError(f"between 1 and {MAX_SOURCES} sources, got {n}")
            for name in ("phases", "amplitudes", "onsets"):
                _per_source(n, getattr(self, name), name)
            return self

    def _points(self, p) -> np.ndarray:
        return np.asarray(p.sources, dtype=np.float64).reshape(-1, 2)

    def _phases(self, p, pts: np.ndarray) -> np.ndarray:
        return np.radians(np.asarray(p.phases, dtype=np.float64)) if p.phases else np.zeros(
            len(pts))  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "interference", **self._semantics(p),
                "phases_deg": list(p.phases) or [0.0] * len(p.sources)}  # fmt: skip


@register
class PhasedArray(_WaveField):
    """A line (or ring) of ``count`` sources whose phases steer the beam: ``steer`` degrees
    off broadside (the direction ``axis`` - 90, where ``axis`` is the line's direction), so
    animating ``steer`` swings the beam. ``spacing`` is in wavelengths (half a wavelength:
    no grating lobes). Draws the field, the elements as beacons and the beam direction as a
    ray of light from the array's ``center``."""

    type: ClassVar[str] = "phased_array"
    Style: ClassVar = {**_WaveField.Style, "ray": ("emphasis", "primary"),
                       "stroke": ("stroke", "regular"), "ray_glow": ("glow", "strong")}  # fmt: skip

    class Params(WaveParams):
        center: tuple[float, float] = (-3.0, 0.0)  # world (x, y) of the array's middle
        count: int = Field(4, ge=1, le=MAX_SOURCES)
        spacing: float = Field(0.5, gt=0)  # wavelengths between neighbours
        layout: Literal["line", "ring"] = "line"
        axis: float = 90.0  # degrees: the line's direction (90: up, so the beam goes +x)
        steer: float = 0.0  # degrees off broadside, counterclockwise
        ray: bool = True  # the beam direction as a ray of light

        @model_validator(mode="after")
        def _lengths(self):
            for name in ("amplitudes", "onsets"):
                _per_source(self.count, getattr(self, name), name)
            return self

    @staticmethod
    def beam_direction(p) -> float:
        """Degrees counterclockwise from +x."""
        return p.axis - 90.0 + p.steer

    def _points(self, p) -> np.ndarray:
        lam = wm.wavelength(p.speed, p.frequency)
        return wm.elements(p.layout, p.count, p.spacing * lam, p.axis, p.center)

    def _phases(self, p, pts: np.ndarray) -> np.ndarray:
        _, k, _ = self._wave(p)
        return wm.steering_phases(pts, p.center, k, self.beam_direction(p))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return {**super().features(state), "array": np.r_[p.center, p.pos[2]]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        lam = wm.wavelength(p.speed, p.frequency)
        phases = np.degrees(self._phases(p, self._points(p)))
        out = {"kind": "phased_array", **self._semantics(p), "steer": p.steer,
               "beam_direction_deg": round(self.beam_direction(p) % 360.0, 6),
               "layout": p.layout, "count": p.count,
               "phases_deg": [round(float(v), 4) for v in phases]}  # fmt: skip
        if p.layout == "line":
            out["beamwidth_deg"] = round(wm.beamwidth(p.count, p.spacing * lam, lam, p.steer), 4)
        return out

    def _ray_end(self, p) -> np.ndarray:
        """Where the beam ray leaves the rectangle."""
        corner, r, u = self._rect(p)
        c = np.asarray(p.center, dtype=np.float64)
        d = wm.direction(self.beam_direction(p))
        lo, hi = corner[:2], corner[:2] + r[:2] + u[:2]
        ts = [((hi if d[i] > 0 else lo)[i] - c[i]) / d[i] for i in (0, 1) if abs(d[i]) > 1e-12]
        t = max(min(ts), 0.0) if ts else 0.0
        return c + d * t

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        super().draw(ctx, state, pass_)
        me = self.me(state)
        p = me.params
        if not p.ray:
            return
        cam = state.camera_for(me)
        z = p.pos[2]
        a, b = np.r_[p.center, z], np.r_[self._ray_end(p), z]
        if np.linalg.norm(b - a) < 1e-9:
            return
        unit = float(cam.units_per_height(a))
        # the ray starts clear of the beacons and fades out toward the edge
        pts = a + (b - a) * np.linspace(0.0, 1.0, 24)[:, None]
        fade = np.clip(np.linspace(0.0, 1.0, 24) * 8.0, 0.0, 1.0) * np.linspace(1.0, 0.25, 24)
        base = with_alpha(me.style["ray"], p.opacity)
        cols = np.repeat(np.asarray(base, dtype=np.float64)[None], 24, axis=0)
        cols[:, 3] *= fade
        polyline(ctx, cam, pts, me.style["stroke"] * unit, color=base, colors=cols,
                 glow=float(me.style["ray_glow"]))  # fmt: skip
