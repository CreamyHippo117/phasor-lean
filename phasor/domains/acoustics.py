"""Room acoustics: objects for the reverb story.

    from phasor import ac

    s.add(ac.Room(id="room", size=(8, 5, 3), material="plaster"))
    s.add(ac.Source(id="src", pos=(2, 2.5, 1.5)))
    s.add(ac.Listener(id="mic", pos=(6, 2, 1.5)))
    s.add(ac.Wavefront(id="wave", center=(2, 2.5, 1.5), radius=expr("343 * t")))
    s.add(ac.ImageSources(id="ghosts", size=(8, 5, 3), source=(2, 2.5, 1.5), order=2))
    s.add(ac.ImpulseResponse(id="ir", size=(8, 5, 3), source=(2, 2.5, 1.5), listener=(6, 2, 1.5),
                             layer="overlay", pos=(0.2, 0.62), extent=(0.8, 0.3)))
    s.add(ac.PressureSlice(id="field", size=(8, 5, 3), source=(2, 2.5, 1.5), time=expr("t")))

In 3D, a room of any shape and sound bouncing in it:

    s.add(ac.Room3D(id="room", size=(8, 5, 3)))            # or path="assets/l_room.obj"
    s.add(ac.SoundRays(id="rays", room="room", source=(2, 2.5, 1.5), time=expr("t - 1")))
    s.add(ac.Wavefront3D(id="front", room="room", source=(2, 2.5, 1.5), time=expr("t - 1")))

Everything is a pure function of its parameters (the numerics live in ``ac_math`` and the
pressure field is a checkpointed ``Simulation``), so the same frame always renders the same.
Walls are glass by default: translucent, so they never hide the source or the listener.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator

from phasor.core import files
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.simulation import runner_for
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import ac_math
from phasor.domains.geometry_math import sample_ramp
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
    flash_points,
    lut_colors,
    polyline,
    polylines,
    with_alpha,
)

SHOWN_FEATURES = 1000  # image sources listed in ``features`` at most (the lowest orders)
CHECKPOINT_EVERY = 50  # FDTD steps between cached checkpoints


def _vec3_positive(v):
    if min(v) <= 0:
        raise ValueError("size must be positive in every direction")
    return v


def _known_material(v: str) -> str:
    if v not in ac_math.MATERIALS:
        raise ValueError(f"unknown material {v!r}; materials are {sorted(ac_math.MATERIALS)}")
    return v


def _material(material: str, absorption: float) -> float:
    return ac_math.absorption_of(material, absorption)


# ---------------------------------------------------------------- cached numerics


@functools.lru_cache(maxsize=64)
def _arrivals(size, source, listener, order, alpha):
    return ac_math.arrivals(size, source, listener, order, alpha)


@functools.lru_cache(maxsize=32)
def _decay(size, source, listener, order, alpha, fs, length):
    arr = _arrivals(size, source, listener, order, alpha)
    ir = ac_math.impulse_response(arr, fs, length)
    try:
        return ac_math.rt60_fit(ir, fs)
    except ValueError:
        return None


# ---------------------------------------------------------------- Room


@register
class Room(ObjectBase):
    """A shoebox room with its corner at ``origin``: translucent walls and bright edges."""

    type: ClassVar[str] = "room"
    Style: ClassVar = {
        "color": ("emphasis", "muted"),
        "stroke": ("stroke", "hairline"),
        "material": ("surface", "glass"),
        "glow": ("glow", "none"),  # the prism look: the edges shine
    }
    Presets: ClassVar = {
        "glass": {},
        "wireframe": {"style": {"material": "wire"}},
        "solid": {"style": {"material": "solid"}},
    }

    class Params(ParamsBase):
        size: Vec3 = (8.0, 5.0, 3.0)
        origin: Vec3 = (0.0, 0.0, 0.0)
        material: str = "plaster"  # one of ``ac_math.MATERIALS``
        absorption: float = -1.0  # 0-1; negative: take it from ``material``

        @field_validator("size")
        @classmethod
        def _size(cls, v):
            return _vec3_positive(v)

        @field_validator("material")
        @classmethod
        def _mat(cls, v):
            return _known_material(v)

    def _lo_hi(self, state: SceneState):
        p = self.me(state).params
        lo = np.asarray(p.origin, dtype=np.float64)
        return lo, lo + np.asarray(p.size, dtype=np.float64)

    def world_bounds(self, state: SceneState) -> Bounds:
        lo, hi = self._lo_hi(state)
        return Bounds(tuple(lo), tuple(hi))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        lo, hi = self._lo_hi(state)
        return {"center": (lo + hi) / 2, "corners": Bounds(tuple(lo), tuple(hi)).corners()}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        alpha = _material(p.material, p.absorption)
        vol, area = ac_math.room_stats(p.size)
        return {
            "kind": "room", "size": list(p.size), "volume": round(vol, 3),
            "surface_area": round(area, 3), "material": p.material,
            "absorption": round(alpha, 4),
            "rt60_sabine": round(ac_math.sabine_rt60(p.size, alpha), 4),
            "rt60_eyring": round(ac_math.eyring_rt60(p.size, alpha), 4),
        }  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        lo, hi = self._lo_hi(state)
        cam = state.camera_for(me)
        color = me.style["color"]
        faces = []  # (4 corners counter-clockwise, outward normal)
        for ax in range(3):
            a, b = [i for i in range(3) if i != ax]
            for side, n in ((lo, -1.0), (hi, 1.0)):
                quad = np.zeros((4, 3))
                quad[:, ax] = side[ax]
                for k, (u, v) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                    quad[k, a] = (lo, hi)[u][a]
                    quad[k, b] = (lo, hi)[v][b]
                normal = np.zeros(3)
                normal[ax] = n
                faces.append((quad, normal))
        positions = np.vstack([q for q, _ in faces])
        normals = np.vstack([np.tile(n, (4, 1)) for _, n in faces])
        k = 4 * np.arange(6)
        indices = np.vstack([np.c_[k, k + 1, k + 2], np.c_[k, k + 2, k + 3]])
        rgba = np.tile(np.array(color), (len(positions), 1))
        ctx.surface(positions, indices, normals=normals, colors=rgba,
                    uv=np.tile([[0, 0], [1, 0], [1, 1], [0, 1]], (6, 1)).astype(float),
                    material=me.roles["material"],
                    opacity=me.params.opacity * float(me.style["material"]))  # fmt: skip
        corners = {(i, j, k): np.array([(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]])
                   for i in (0, 1) for j in (0, 1) for k in (0, 1)}  # fmt: skip
        edge_color = with_alpha(color, me.params.opacity)
        for (i, j, k0), a in corners.items():
            for axis in range(3):
                idx = [i, j, k0]
                if idx[axis] == 0:
                    idx[axis] = 1
                    b = corners[tuple(idx)]
                    mid = (a + b) / 2
                    w = me.style["stroke"] * float(cam.units_per_height(mid))
                    polyline(ctx, cam, [a, b], w, color=edge_color,
                             glow=float(me.style.get("glow", 1.0)))  # fmt: skip


# ---------------------------------------------------------------- Source and Listener


class _Marker(ObjectBase):
    """A small billboard marker at ``pos``."""

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c = np.asarray(p.pos, dtype=np.float64)
        return Bounds(tuple(c - p.r), tuple(c + p.r))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}


@register
class Source(_Marker):
    """A sound source: a glowing point."""

    type: ClassVar[str] = "source"
    Style: ClassVar = {"color": ("emphasis", "highlight"), "glow": ("glow", "soft")}

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(0.12, gt=0)

    def semantics(self, state: SceneState) -> dict[str, Any]:
        return {"kind": "point", "role": "source", "position": list(self.me(state).params.pos)}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        color = with_alpha(me.style["color"], me.params.opacity)
        disc(
            ctx,
            state.camera_for(me),
            me.params.pos,
            me.params.r,
            color=color,
            glow=me.style["glow"],
        )


@register
class Listener(_Marker):
    """A listener (microphone): a ring with a dot."""

    type: ClassVar[str] = "listener"
    Style: ClassVar = {"color": ("emphasis", "secondary"), "glow": ("glow", "none"),
                       "stroke": ("stroke", "regular")}  # fmt: skip

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = Field(0.16, gt=0)

    def semantics(self, state: SceneState) -> dict[str, Any]:
        return {"kind": "point", "role": "listener", "position": list(self.me(state).params.pos)}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        cam = state.camera_for(me)
        color = with_alpha(me.style["color"], me.params.opacity)
        w = me.style["stroke"] * float(cam.units_per_height(np.asarray(me.params.pos)))
        circle(ctx, cam, me.params.pos, me.params.r, w, color=color, glow=me.style["glow"])
        disc(ctx, cam, me.params.pos, me.params.r * 0.3, color=color, glow=me.style["glow"])


# ---------------------------------------------------------------- Wavefront


@register
class Wavefront(ObjectBase):
    """An expanding spherical wavefront, drawn as a lat-long wire sphere. Its ``radius``
    follows the speed of sound if you give it ``expr("343 * (t - 0.5)")``; before the emission
    time that is negative, and nothing is drawn."""

    type: ClassVar[str] = "wavefront"
    Style: ClassVar = {"color": ("emphasis", "secondary"), "stroke": ("stroke", "hairline"),
                       "glow": ("glow", "soft")}  # fmt: skip

    class Params(ParamsBase):
        center: Vec3 = (0.0, 0.0, 0.0)
        radius: float = 0.0  # a negative radius means "not emitted yet": nothing is drawn
        rings: int = Field(6, ge=2, le=24)  # latitude rings; twice as many meridians

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c = np.asarray(p.center, dtype=np.float64)
        r = max(p.radius, 1e-6)
        return Bounds(tuple(c - r), tuple(c + r))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.center, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        return {"kind": "wavefront", "radius": round(max(p.radius, 0.0), 4),
                "center": list(p.center), "time_to_reach_center": 0.0}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        if p.radius <= 1e-6:
            return
        cam = state.camera_for(me)
        c = np.asarray(p.center, dtype=np.float64)
        color = with_alpha(me.style["color"], me.params.opacity)
        a = np.linspace(0, 2 * np.pi, 49)
        for k in range(1, p.rings):  # latitude rings
            th = np.pi * k / p.rings
            ring = c + p.radius * np.stack(
                [np.sin(th) * np.cos(a), np.sin(th) * np.sin(a), np.full_like(a, np.cos(th))],
                axis=1,
            )
            w = me.style["stroke"] * float(cam.units_per_height(c))
            polyline(ctx, cam, ring, w, color=color, glow=me.style["glow"])
        t = np.linspace(0, np.pi, 25)
        for k in range(2 * p.rings):  # meridians
            ph = 2 * np.pi * k / (2 * p.rings)
            mer = c + p.radius * np.stack(
                [np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)], axis=1
            )
            polyline(
                ctx,
                cam,
                mer,
                me.style["stroke"] * float(cam.units_per_height(c)),
                color=color,
                glow=me.style["glow"],
            )


# ---------------------------------------------------------------- Doppler wavefronts


@register
class Doppler(ObjectBase):
    """Wavefronts from a source moving along +x at a constant Mach number.

    ``time`` is seconds since the source started at ``pos``. Sound travels ``size`` world
    units in one second. Each ring marks a crest at its retarded position.
    """

    type: ClassVar[str] = "doppler"
    Style: ClassVar = {
        "source": ("emphasis", "highlight"),
        "cone": ("emphasis", "primary"),
        "stroke": ("stroke", "regular"),
        "source_radius": ("stroke", "bold"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "magnitude"),
    }
    _history: ClassVar[int] = 16

    class Params(ParamsBase):
        speed: float = Field(0.6, ge=0, allow_inf_nan=False)
        frequency: float = Field(10.0, gt=0, allow_inf_nan=False)
        time: float = Field(0.0, ge=0, allow_inf_nan=False)
        pos: Vec3 = (0.0, 0.0, 0.0)
        size: float = Field(1.0, gt=0, allow_inf_nan=False)

    def _geometry(self, p):
        origin = np.asarray(p.pos, dtype=np.float64)
        centers, radii = ac_math.doppler_wavefronts(p.speed, p.frequency, p.time, self._history)
        source = origin + np.array([p.size * p.speed * p.time, 0, 0])
        centers = origin + p.size * np.c_[centers, np.zeros((len(centers), 2))]
        return source, centers, p.size * radii

    def world_bounds(self, state: SceneState) -> Bounds:
        source, centers, radii = self._geometry(self.me(state).params)
        extents = np.c_[radii, radii, np.zeros_like(radii)]
        lo = np.vstack([source, centers - extents]).min(axis=0)
        hi = np.vstack([source, centers + extents]).max(axis=0)
        return Bounds(tuple(lo), tuple(hi))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        source, centers, _ = self._geometry(self.me(state).params)
        return {"center": source, "source": source, "wavefront_centers": centers}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        source, _, radii = self._geometry(p)
        return {
            "kind": "doppler",
            "mach": p.speed,
            "emitted_frequency": p.frequency,
            "frequency_ahead": p.frequency / (1 - p.speed) if p.speed < 1 else None,
            "frequency_behind": p.frequency / (1 + p.speed),
            "mach_angle_degrees": float(np.degrees(np.arcsin(1 / p.speed)))
            if p.speed > 1
            else None,
            "source": source.tolist(),
            "wavefront_count": len(radii),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        source, centers, radii = self._geometry(p)
        scale = float(cam.units_per_height(source))
        width = me.style["stroke"] * scale
        glow = me.style["glow"]

        angle = np.linspace(0, 2 * np.pi, 97)
        cosine = np.cos(angle)
        rim = np.c_[cosine, np.sin(angle), np.zeros_like(angle)]
        heard = ac_math.doppler_frequency(p.speed, p.frequency, cosine)
        # Map the unshifted pitch to the middle, with the caustic at the warm end.
        hue = 1 / (1 + p.frequency / heard)
        rgb = sample_ramp(me.style["colormap"], hue)
        for center, radius in zip(centers, radii, strict=True):
            fade = 1 - (radius / p.size) * p.frequency / self._history
            colors = np.c_[rgb, np.full_like(angle, p.opacity * max(0.0, fade))]
            polyline(
                ctx,
                cam,
                center + radius * rim,
                width,
                color=(1.0, 1.0, 1.0, 1.0),
                colors=colors,
                glow=glow,
            )

        if len(radii) and p.speed > 1:
            oldest = centers[0] - source
            for x, y in ac_math.mach_cone_rays(p.speed):
                ray = np.array([x, y, 0.0])
                tangent = source + (oldest @ ray) * ray
                color = with_alpha(me.style["cone"], p.opacity)
                polyline(ctx, cam, [source, tangent], width, color=color, glow=glow)

        disc(
            ctx,
            cam,
            source,
            me.style["source_radius"] * scale,
            color=with_alpha(me.style["source"], p.opacity),
            glow=glow,
        )


# ---------------------------------------------------------------- Image sources


@register
class ImageSources(ObjectBase):
    """Ghost speakers behind the walls: the image-source method made visible. Brightness
    falls with each reflection (the gain of ``material`` or ``absorption``). ``order`` goes
    up to ``ac_math.MAX_ORDER`` (113, about 2 million images); at most ``max_points`` of them
    are drawn, the fewest reflections first. They are the same points as a
    ``sweep.SweepLattice`` with ``rule="mirror_box"`` (which makes them flash as a sphere of
    time sweeps them), drawn the same way, as steady glowing points."""

    type: ClassVar[str] = "image_sources"
    Style: ClassVar = {"color": ("emphasis", "muted"), "glow": ("glow", "none")}

    class Params(ParamsBase):
        size: Vec3 = (8.0, 5.0, 3.0)
        source: Vec3 = (2.0, 2.5, 1.5)
        order: int = Field(1, ge=0, le=ac_math.MAX_ORDER)
        material: str = "plaster"
        absorption: float = -1.0
        r: float = Field(0.09, gt=0)
        max_points: int = Field(60_000, ge=1, le=ac_math.MAX_IMAGES)

        @field_validator("size")
        @classmethod
        def _size(cls, v):
            return _vec3_positive(v)

    def _images(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return _images(tuple(p.size), tuple(p.source), p.order, _material(p.material, p.absorption))

    def world_bounds(self, state: SceneState) -> Bounds:
        im = self._images(state)
        return Bounds.of_points(im["pos"], pad=self.me(state).params.r)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        pos = self._images(state)["pos"]
        return {"points": pos[1 : 1 + SHOWN_FEATURES], "source": pos[:1]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        im = self._images(state)
        p = self.me(state).params
        count = int(len(im["pos"]) - 1)
        return {"kind": "image_sources", "backdrop": True, "count": count, "order": p.order,
                "shown": min(count, p.max_points),
                "weakest_gain": round(float(im["gain"].min()), 4)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        im = self._images(state)
        pos, gain = im["pos"][1 : 1 + p.max_points], im["gain"][1 : 1 + p.max_points]
        colors = np.tile(np.asarray(me.style["color"], dtype=np.float64), (len(pos), 1))
        colors[:, 3] *= gain
        flash_points(ctx, state.camera_for(me), pos, p.r, colors=colors,
                     glow=float(me.style["glow"]), opacity=p.opacity, segments=16)  # fmt: skip


@functools.lru_cache(maxsize=32)
def _images(size, source, order, alpha):
    return ac_math.image_sources(size, source, order, alpha)


# ---------------------------------------------------------------- Impulse response


@register
class ImpulseResponse(ObjectBase):
    """The room's impulse response as a plot: a spike for each arrival at the listener (the
    direct sound in the accent color), and optionally the Schroeder decay with its RT60 fit.

    A panel: put it on the overlay (``layer="overlay"``, ``pos`` the lower-left corner and
    ``extent`` the (width, height), in frame heights). ``reveal`` (0-1) draws the first share
    of ``duration`` seconds, so you can animate arrivals dropping in; ``show_rt60`` (0-1) fades
    in the decay curve and its fitted line.
    """

    type: ClassVar[str] = "impulse_response"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "accent": ("emphasis", "highlight"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        size: Vec3 = (8.0, 5.0, 3.0)
        source: Vec3 = (2.0, 2.5, 1.5)
        listener: Vec3 = (6.0, 2.0, 1.5)
        order: int = Field(12, ge=0, le=40)
        material: str = "plaster"
        absorption: float = -1.0
        pos: Vec3 = (0.1, 0.1, 0.0)  # lower-left corner of the panel
        extent: tuple[float, float] = (0.8, 0.3)
        duration: float = Field(0.4, gt=0)  # seconds shown across the panel
        reveal: float = Field(1.0, ge=0, le=1)
        show_rt60: float = Field(0.0, ge=0, le=1)

        @field_validator("size")
        @classmethod
        def _size(cls, v):
            return _vec3_positive(v)

    FS = 2000  # Hz, for the decay curve
    DECAY_LENGTH = 1.5  # seconds of IR used for the RT60 fit

    def _alpha(self, state):
        p = self.me(state).params
        return _material(p.material, p.absorption)

    def _arr(self, state):
        p = self.me(state).params
        return _arrivals(tuple(p.size), tuple(p.source), tuple(p.listener), p.order,
                         self._alpha(state))  # fmt: skip

    def _fit(self, state):
        p = self.me(state).params
        return _decay(tuple(p.size), tuple(p.source), tuple(p.listener), p.order,
                      self._alpha(state), self.FS, self.DECAY_LENGTH)  # fmt: skip

    def _ydir(self, state: SceneState) -> float:
        """+1 where screen-up is +y (a world layer), -1 on the overlay (y runs down): the panel
        always grows upward from its lower-left corner."""
        return 1.0 if state.camera_for(self.me(state)).up[1] >= 0 else -1.0

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        y2 = y + self._ydir(state) * p.extent[1]
        return Bounds((x, min(y, y2), z), (x + p.extent[0], max(y, y2), z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        arr = self._arr(state)
        p = self.me(state).params
        x, y, z = p.pos
        w, h = p.extent
        h = h * self._ydir(state)
        shown = arr["time"] <= p.duration
        xs = x + arr["time"][shown] / p.duration * w
        top = np.minimum(arr["amplitude"][shown] / arr["amplitude"][0], 1.0) * h
        return {"spikes": np.stack([xs, y + top, np.full(len(xs), z)], axis=1)[:200]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        arr = self._arr(state)
        fit = self._fit(state)
        p = self.me(state).params
        out: dict[str, Any] = {
            "kind": "impulse_response", "arrivals": len(arr["time"]),
            "first_arrival_ms": round(float(arr["time"][0]) * 1000, 3),
            "direct_distance": round(float(arr["distance"][0]), 4), "order": p.order,
            "rt60": None if fit is None else round(fit["rt60"], 4),
        }  # fmt: skip
        out["rt60_theory_eyring"] = round(ac_math.eyring_rt60(p.size, self._alpha(state)), 4)
        # Image sources up to ``order`` reflections describe the first order * (shortest wall) / c
        # seconds completely; a decay fitted beyond that is cut short, not the room's.
        complete = p.order * min(p.size) / ac_math.C
        out["complete_until"] = round(complete, 4)
        out["rt60_reliable"] = bool(fit is not None and fit["rt60"] <= complete)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        x, y, z = p.pos
        w, h = p.extent
        h = h * self._ydir(state)
        width = me.style["stroke"] * float(cam.units_per_height())
        base = with_alpha(me.style["color"], p.opacity)
        accent = with_alpha(me.style["accent"], p.opacity)
        polyline(ctx, cam, [(x, y, z), (x + w, y, z)], width, color=with_alpha(base, 0.6))  # axis
        arr = self._arr(state)
        shown = (arr["time"] <= p.duration * p.reveal) & (arr["time"] <= p.duration)
        a0 = float(arr["amplitude"][0])
        for k in np.nonzero(shown)[0][:400]:
            sx = x + arr["time"][k] / p.duration * w
            top = min(float(arr["amplitude"][k]) / a0, 1.0) * h
            polyline(ctx, cam, [(sx, y, z), (sx, y + top, z)], width,
                     color=accent if k == 0 else base, glow=me.style["glow"])  # fmt: skip
        if p.show_rt60 > 0:
            fit = self._fit(state)
            if fit is not None:
                tt = np.linspace(0, p.duration, 120)
                curve = np.interp(tt, fit["t"], fit["curve"])  # dB, 0 down to -60+
                ys = y + np.clip(1.0 + curve / 60.0, 0.0, 1.0) * h
                pts = np.stack([x + tt / p.duration * w, ys, np.full_like(tt, z)], axis=1)
                polyline(ctx, cam, pts, width, color=with_alpha(accent, p.show_rt60 * 0.9))
                line = np.interp(tt, fit["t"], fit["fit"])
                ys = y + np.clip(1.0 + line / 60.0, 0.0, 1.0) * h
                fitted = np.stack([x + tt / p.duration * w, ys, np.full_like(tt, z)], axis=1)
                polyline(ctx, cam, fitted, width * 0.8, color=with_alpha(base, p.show_rt60))


# ---------------------------------------------------------------- Pressure slice


@functools.lru_cache(maxsize=8)
def _fdtd(size, source, dx, alpha, frequency, seed):
    return ac_math.RoomFDTD(size, source, dx=dx, alpha=alpha, frequency=frequency, seed=seed)


@register
class PressureSlice(ObjectBase):
    """A horizontal slice of the true pressure field (FDTD) at height ``z``, shown as a plane:
    the source's color for compression, the other for rarefaction. Time is ``time`` (seconds,
    usually ``expr("t")``); the field is a checkpointed simulation, so scrubbing is cheap."""

    type: ClassVar[str] = "pressure_slice"
    Style: ClassVar = {"positive": ("emphasis", "highlight"), "negative": ("emphasis", "secondary"),
                       "glow": ("glow", "none"),
                       "colormap": ("colormap", "spectrum")}  # fmt: skip  # with hue_cycles
    Presets: ClassVar = {"prism": PRISM}  # glowing out of the dark in spectrum hues
    Life: ClassVar = frozenset({"hue", "glow"})

    class Params(HeatParams):
        size: Vec3 = (8.0, 5.0, 3.0)
        origin: Vec3 = (0.0, 0.0, 0.0)
        source: Vec3 = (2.0, 2.5, 1.5)
        z: float = 1.5
        time: float = Field(0.0, ge=0)
        dx: float = Field(0.2, gt=0.02)
        material: str = "plaster"
        absorption: float = -1.0
        frequency: float = Field(200.0, gt=1)
        gain: float = Field(1.0, gt=0)  # how strongly pressure turns into color
        seed: int = 0

        @field_validator("size")
        @classmethod
        def _size(cls, v):
            return _vec3_positive(v)

    def _sim(self, state):
        p = self.me(state).params
        # source is given in room coordinates; the simulation lives in them too
        return _fdtd(tuple(p.size), tuple(p.source), p.dx,
                     _material(p.material, p.absorption), p.frequency, p.seed)  # fmt: skip

    def _slab(self, state):
        sim = self._sim(state)
        p = self.me(state).params
        runner = runner_for(sim, every=CHECKPOINT_EVERY)
        return sim.slice_z(runner.state_at(p.time), p.z)

    def _rect(self, state):
        p = self.me(state).params
        o = np.asarray(p.origin, dtype=np.float64)
        return o + np.array([0, 0, p.z]), np.array([p.size[0], 0.0, 0.0]), np.array(
            [0.0, p.size[1], 0.0])  # fmt: skip

    def world_bounds(self, state: SceneState) -> Bounds:
        c, r, u = self._rect(state)
        return Bounds.of_points([c, c + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        return {
            "source": np.asarray(p.source, dtype=np.float64) + np.asarray(p.origin),
            "center": self._rect(state)[0] + (self._rect(state)[1] + self._rect(state)[2]) / 2,
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        slab = self._slab(state)
        p = self.me(state).params
        return {"kind": "pressure_slice", "backdrop": True, "time": round(p.time, 6),
                "peak_pressure": round(float(np.abs(slab).max()), 6),
                "grid": list(slab.shape), "z": p.z}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, right, up = self._rect(state)
        slab = self._slab(state)
        glow = float(me.style.get("glow", 1.0))
        # the signal is tanh(gain * pressure), -1..1; its level 0.5 + 0.5 s; plain: each
        # sign's colour out of the background by |s|
        how = {"transform": "tanh", "gain": p.gain * 8.0, "level": "signed",
               "weight": "abs" if glow > 1 else None}  # fmt: skip
        if p.hue_cycles > 0:  # prism: the hue of the level, brightness from the weight
            recipe = wheel_recipe(me.style["colormap"], p, **how)
        else:
            recipe = Recipe(colour="sign", positive=me.style["positive"][:3],
                            negative=me.style["negative"][:3], mix=True, **how)  # fmt: skip
        pixels = heat_pixels(
            slab, (p.size[0], p.size[1]), recipe, cells=True,
            how="smooth" if p.smooth else "linear",
            background=state.background, **p.drawing(),
        )  # fmt: skip
        ctx.field(corner, right, up, pixels=pixels, opacity=p.opacity, glow=glow)


# ---------------------------------------------------------------- 3D: a room of any shape

RAY_DENSITY = 400  # rays drawn at full strength; more rays share the light between them
FRONT_FADE = 0.12  # how fast a front dims as it spreads (per meter of radius)
SHELL_RINGS = 7  # latitude rings of a front's sphere, and twice as many meridians


@dataclass(frozen=True, eq=False)
class Geometry:
    """A room as a triangle mesh, and what tracing and drawing it need (read-only arrays)."""

    vertices: np.ndarray  # (N, 3)
    triangles: np.ndarray  # (M, 3)
    edges: np.ndarray  # (E, 2) the feature edges drawn as lines
    walls: tuple[np.ndarray, ...]  # the triangles grouped by plane
    box: tuple | None  # (size, origin) for a shoebox, else None
    key: tuple  # what it was made from (hashable)

    @property
    def lo(self) -> np.ndarray:
        return self.vertices.min(axis=0)

    @property
    def hi(self) -> np.ndarray:
        return self.vertices.max(axis=0)


def _read_only(*arrays) -> None:
    for a in arrays:
        a.setflags(write=False)


@functools.lru_cache(maxsize=16)
def _geometry(key: tuple) -> Geometry:
    if key[0] == "box":
        _, size, origin, angle = key
        v, f = ac_math.box_mesh(size, origin)
        box = (size, origin)
    else:
        _, path, _mtime, _nbytes, scale, origin, angle = key  # mtime/size: re-read on change
        v, f = ac_math.load_obj(path)
        v = v * scale + np.asarray(origin, dtype=np.float64)
        box = None
    edges = ac_math.feature_edges(v, f, angle)
    walls = tuple(ac_math.planes(v, f))
    _read_only(v, f, edges, *walls)
    return Geometry(v, f, edges, walls, box, key)


def _relative(v: str) -> str:
    return files.relative_path(v) if v else v


@register
class Room3D(ObjectBase):
    """A room in 3D, drawn as faint glowing lines along its edges, with optional faint
    glass walls. A shoebox (``size``, corner at ``origin``) or any closed mesh read from a
    Wavefront ``.obj`` file (``path``, relative to the scene's folder, scaled by ``scale`` and
    moved by ``origin``); for a mesh, the edges drawn are its creases (``edge_angle``), not
    the diagonals that split flat walls into triangles. ``SoundRays`` and ``Wavefront3D``
    name a room by its id and bounce off its walls."""

    type: ClassVar[str] = "room3d"
    Files: ClassVar = ("path",)
    Style: ClassVar = {
        "color": ("emphasis", "muted"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "soft"),
        "material": ("surface", "glass"),
    }
    Presets: ClassVar = {
        "glass": {},
        "outline": {"params": {"walls": 0.0}},
    }

    class Params(ParamsBase):
        size: Vec3 = (8.0, 5.0, 3.0)
        origin: Vec3 = (0.0, 0.0, 0.0)
        path: str = ""  # a Wavefront .obj next to the scene; empty: the shoebox ``size``
        scale: float = Field(1.0, gt=0)  # mesh units to meters
        material: str = "plaster"  # one of ``ac_math.MATERIALS``
        absorption: float = -1.0  # 0-1; negative: take it from ``material``
        walls: float = Field(0.25, ge=0, le=1)  # glass walls' share of the material; 0: lines
        edge_angle: float = Field(20.0, gt=0, lt=180)  # degrees: a crease drawn as a line

        @field_validator("size")
        @classmethod
        def _size(cls, v):
            return _vec3_positive(v)

        @field_validator("path")
        @classmethod
        def _path(cls, v):
            return _relative(v)

        @field_validator("material")
        @classmethod
        def _mat(cls, v):
            return _known_material(v)

    def geometry(self, state: SceneState) -> Geometry | None:
        """The room's mesh, or None while its file is missing or cannot be read."""
        p = self.me(state).params
        if not p.path:
            return _geometry(("box", tuple(p.size), tuple(p.origin), p.edge_angle))
        try:
            path = files.resolve(state.base_dir, p.path)
            st = path.stat()
            return _geometry(("mesh", str(path), st.st_mtime_ns, st.st_size, p.scale,
                              tuple(p.origin), p.edge_angle))  # fmt: skip
        except (OSError, ValueError):
            return None

    def alpha(self, state: SceneState) -> float:
        p = self.me(state).params
        return _material(p.material, p.absorption)

    def world_bounds(self, state: SceneState) -> Bounds:
        g = self.geometry(state)
        if g is None:
            p = self.me(state).params
            lo = np.asarray(p.origin, dtype=np.float64)
            return Bounds(tuple(lo), tuple(lo + np.asarray(p.size)))
        return Bounds.of_points(g.vertices)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        g = self.geometry(state)
        if g is None:
            return {}
        return {"center": (g.lo + g.hi) / 2, "corners": np.asarray(g.vertices[:200])}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        g = self.geometry(state)
        out: dict[str, Any] = {"kind": "room", "shape": "mesh" if p.path else "box",
                               "backdrop": True}  # fmt: skip
        if g is None:
            return {**out, "missing_file": p.path}
        alpha = self.alpha(state)
        vol, area = ac_math.mesh_stats(g.vertices, g.triangles)
        return {**out, "volume": round(vol, 3), "surface_area": round(area, 3),
                "walls": len(g.walls), "edges": len(g.edges), "triangles": len(g.triangles),
                "material": p.material, "absorption": round(alpha, 4),
                "rt60_sabine": round(ac_math.sabine_rt60(None, alpha, stats=(vol, area)), 4),
                "rt60_eyring": round(ac_math.eyring_rt60(None, alpha, stats=(vol, area)), 4),
                }  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        g = self.geometry(state)
        if g is None:
            return
        cam = state.camera_for(me)
        color = me.style["color"]
        if p.walls > 0:  # flat-shaded: every triangle its own three vertices
            tri = g.vertices[g.triangles]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
            ctx.surface(tri.reshape(-1, 3), np.arange(3 * len(tri)).reshape(-1, 3),
                        normals=np.repeat(normal, 3, axis=0),
                        colors=np.tile(np.asarray(color), (3 * len(tri), 1)),
                        material=me.roles["material"],
                        opacity=p.opacity * p.walls * float(me.style["material"]))  # fmt: skip
        segs = g.vertices[g.edges]  # (E, 2, 3)
        width = me.style["stroke"] * cam.units_per_height(segs.reshape(-1, 3)).reshape(-1, 2)
        polylines(ctx, cam, segs, width, color=with_alpha(color, p.opacity),
                  glow=me.style["glow"])  # fmt: skip


class _InRoom(ObjectBase):
    """Sound inside a ``Room3D`` (named by ``room``), leaving ``source`` when ``time`` is 0.

    ``time`` is in scene seconds since the sound left (usually ``expr("t - 1")``; negative:
    not yet). ``speed`` is the slow motion: seconds of sound per scene second, so the sound
    has travelled ``343 * speed * time`` meters."""

    def _room(self, state: SceneState) -> tuple[Geometry | None, float]:
        p = self.me(state).params
        if p.room not in state or not isinstance(state[p.room].impl, Room3D):
            return None, 0.0
        room = state[p.room].impl
        return room.geometry(state), room.alpha(state)

    def distance(self, state: SceneState) -> float:
        """Meters the sound has travelled (negative before it leaves)."""
        p = self.me(state).params
        return ac_math.C * p.speed * p.time

    def world_bounds(self, state: SceneState) -> Bounds:
        g, _ = self._room(state)
        if g is None:
            return Bounds.of_points([self.me(state).params.source])
        return Bounds.of_points(g.vertices)

    def _hues(self, state: SceneState, reflections) -> np.ndarray:
        """(N, 4) the palette color of each reflection count."""
        me = self.me(state)
        return lut_colors(me.style["colormap"], np.asarray(reflections) * me.params.hue_step)

    def _strokes(self, ctx, state, segs, colors) -> None:
        me = self.me(state)
        cam = state.camera_for(me)
        width = me.style["stroke"] * cam.units_per_height(segs.reshape(-1, 3)).reshape(-1, 2)
        polylines(ctx, cam, segs, width, color=with_alpha(me.style["color"], me.params.opacity),
                  glow=me.style["glow"], colors=colors)  # fmt: skip


class _SoundParams(ParamsBase):
    room: str = Field("room", min_length=1)  # the id of a Room3D
    source: Vec3 = (2.0, 2.5, 1.5)
    time: float = 0.0  # scene seconds since the sound left; negative: not yet
    speed: float = Field(0.01, gt=0)  # slow motion: seconds of sound per scene second
    hue_step: float = 0.14  # share of the palette between one reflection and the next


@functools.lru_cache(maxsize=16)
def _ray_paths(gkey, source, count, bounces, alpha, seed, key):
    g = _geometry(gkey)
    out = ac_math.rays((g.vertices, g.triangles), source, count, bounces, alpha, seed, key)
    _read_only(*out.values())
    return out


@register
class SoundRays(_InRoom):
    """Sound as rays: ``count`` rays leave ``source`` evenly in every direction and reflect
    specularly off the room's walls (angle in = angle out) ``bounces`` times. Each is drawn
    up to where the sound has got to, as a glowing streak ``trail`` meters long (0: the
    whole path so far); its brightness is the energy left, (1 - absorption) ** reflections,
    and its color comes from the palette by reflection count, so the late, reverberant
    sound reads as a different color from the first."""

    type: ClassVar[str] = "sound_rays"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "strong"),
        "colormap": ("colormap", "spectrum"),
    }

    class Params(_SoundParams):
        count: int = Field(1200, ge=1, le=ac_math.MAX_RAYS)
        bounces: int = Field(6, ge=0, le=ac_math.MAX_BOUNCES)
        trail: float = Field(1.5, ge=0)  # meters of each ray shown behind the front; 0: all
        seed: int = 0

    def paths(self, state: SceneState) -> dict[str, np.ndarray] | None:
        """The traced rays (``ac_math.trace``), or None without a room."""
        g, alpha = self._room(state)
        if g is None:
            return None
        p = self.me(state).params
        return _ray_paths(g.key, tuple(p.source), p.count, p.bounces, alpha, p.seed, self.id)

    def visible(self, state: SceneState) -> dict[str, np.ndarray] | None:
        """What is drawn at this time: ``ac_math.ray_segments`` at the distance travelled."""
        paths = self.paths(state)
        p = self.me(state).params
        if paths is None:
            return None
        return ac_math.ray_segments(paths, self.distance(state), p.trail)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        vis = self.visible(state)
        out = {"source": np.asarray(self.me(state).params.source, dtype=np.float64)}
        if vis is not None and len(vis["ray"]):
            head = vis["along"][:, 1] >= self.distance(state) - 1e-9
            out["heads"] = vis["segments"][head, 1][:200]
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        paths = self.paths(state)
        out: dict[str, Any] = {"kind": "sound_rays", "count": p.count, "bounces": p.bounces,
                               "distance": round(max(self.distance(state), 0.0), 4),
                               "sound_time": round(max(p.speed * p.time, 0.0), 6)}  # fmt: skip
        if paths is None:
            return {**out, "room_missing": p.room}
        d = self.distance(state)
        reached = (paths["length"][:, 1:] <= d) & paths["alive"][:, 1:]
        k = reached.sum(axis=1)  # reflections so far, per ray
        lost = ~paths["alive"][:, -1]
        return {**out, "reflections_so_far": round(float(k.mean()), 4),
                "energy_left": round(float(np.mean(paths["energy"][np.arange(p.count), k])), 6),
                "rays_lost": int(lost.sum()), "finished": bool(
                    d > float(paths["length"][:, -1].max()) + p.trail)}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        vis = self.visible(state)
        if vis is None or not len(vis["ray"]):
            return
        p = self.me(state).params
        paths = self.paths(state)
        d = self.distance(state)
        share = min(1.0, (RAY_DENSITY / p.count) ** 0.5)
        light = paths["energy"][vis["ray"], vis["bounce"]] * share
        alpha = np.repeat(light[:, None], 2, axis=1)
        if p.trail > 0:  # faint at the tail, bright at the head
            alpha = alpha * np.clip((vis["along"] - (d - p.trail)) / p.trail, 0.0, 1.0) ** 1.5
        colors = np.repeat(self._hues(state, vis["bounce"])[:, None], 2, axis=1)
        colors[..., 3] = alpha
        self._strokes(ctx, state, vis["segments"], colors)


@functools.lru_cache(maxsize=32)
def _fronts(gkey, source, order, alpha):
    """Centres (K, 3), reflections (K,) and mirror wall (K,; -1 for the direct front) of the
    fronts in a room: every image source up to ``order`` in a shoebox; in any other room
    the direct front and, from ``order`` 1, one image per wall (checked for visibility)."""
    g = _geometry(gkey)
    src = np.asarray(source, dtype=np.float64)
    if g.box is not None:
        size, origin = g.box
        o = np.asarray(origin, dtype=np.float64)
        im = _images(tuple(size), tuple(src - o), order, alpha)
        out = (im["pos"] + o, im["reflections"], np.full(len(im["pos"]), -1))
    else:
        mirrors = ac_math.mirror_images(g.vertices, g.triangles, g.walls, src)
        n = len(mirrors) if order >= 1 else 0
        out = (np.vstack([src[None], mirrors[:n]]), np.r_[0, np.ones(n, dtype=np.int64)],
               np.r_[-1, np.arange(n)])  # fmt: skip
    _read_only(*out)
    return out


@register
class Wavefront3D(_InRoom):
    """The expanding wavefront of a clap, drawn where it meets the walls: a circle on every
    wall it reaches, and the circles of the reflected fronts (the spheres around the image
    sources: every one up to ``order`` reflections in a shoebox, the first reflections off
    each wall in a room of another shape, where only what the source can really reach is
    drawn). ``shell`` (0-1) also draws the fronts themselves inside the room as faint wire
    spheres. Fronts fade as they spread and with every reflection; their color comes from
    the palette by reflection count, the same as ``SoundRays``."""

    type: ClassVar[str] = "wavefront3d"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "strong"),
        "colormap": ("colormap", "spectrum"),
    }

    class Params(_SoundParams):
        order: int = Field(2, ge=0, le=4)  # reflected fronts up to this many reflections
        shell: float = Field(0.12, ge=0, le=1)  # the fronts themselves, inside the room
        samples: int = Field(128, ge=16, le=1024)  # points around each circle

    def _fronts(self, state: SceneState):
        g, alpha = self._room(state)
        if g is None:
            return None, None
        p = self.me(state).params
        return g, _fronts(g.key, tuple(p.source), p.order, alpha)

    def _shown(self, g: Geometry, centers: np.ndarray, radius: float) -> np.ndarray:
        """Which fronts touch the room: closer to it than ``radius``, not yet past it."""
        near = np.linalg.norm(np.maximum(0.0, np.maximum(g.lo - centers, centers - g.hi)), axis=1)
        corners = Bounds(tuple(g.lo), tuple(g.hi)).corners()
        far = np.linalg.norm(centers[:, None] - corners[None], axis=2).max(axis=1)
        return np.flatnonzero((near < radius) & (far > radius))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.source, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        r = self.distance(state)
        out: dict[str, Any] = {"kind": "wavefront", "radius": round(max(r, 0.0), 4),
                               "center": list(p.source), "order": p.order}  # fmt: skip
        g, fronts = self._fronts(state)
        if g is None:
            return {**out, "room_missing": p.room}
        shown = self._shown(g, fronts[0], r) if r > 0 else np.zeros(0, dtype=np.int64)
        return {**out, "fronts": len(fronts[0]), "fronts_in_room": len(shown),
                "deepest_reflection": int(fronts[1][shown].max()) if len(shown) else 0}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        r = self.distance(state)
        g, fronts = self._fronts(state)
        if g is None or r <= 1e-6:
            return
        p = self.me(state).params
        centers, refl, mirror = fronts
        keep = self._shown(g, centers, r)
        if not len(keep):
            return
        v, f = g.vertices, g.triangles
        fade = 1.0 / (1.0 + FRONT_FADE * r)
        kept = 1.0 - self._room(state)[1]  # energy kept per reflection
        segs, who = ac_math.sphere_wall_arcs(v, f, centers[keep], r, p.samples, g.walls)
        who = keep[who]
        if g.box is None and len(segs):  # another shape: only what the sound can reach
            ok = ac_math.front_visible(v, f, g.walls, p.source, mirror[who], segs.mean(axis=1))
            segs, who = segs[ok], who[ok]
        parts = [(segs, who, 1.0)]
        if p.shell > 0:
            parts.append((*self._shell(g, centers, keep, r, p.source), p.shell))
        for pieces, owner, share in parts:
            if not len(pieces):
                continue
            light = share * fade * kept ** refl[owner]
            colors = np.repeat(self._hues(state, refl[owner])[:, None], 2, axis=1)
            colors[..., 3] = light[:, None]
            self._strokes(ctx, state, pieces, colors)

    def _shell(self, g: Geometry, centers, keep, r: float, source):
        """The fronts' spheres as wire (latitude rings and meridians), cut to the inside of
        the room: (S, 2, 3) pieces and the front each belongs to. In a room that is not a
        shoebox only the direct front is drawn, where the source can see it."""
        if g.box is None:
            keep = keep[keep == 0]
        if not len(keep):
            return np.zeros((0, 2, 3)), np.zeros(0, dtype=np.int64)
        a = np.linspace(0, 2 * np.pi, 65)
        lat = [np.stack([np.sin(th) * np.cos(a), np.sin(th) * np.sin(a),
                         np.full_like(a, np.cos(th))], axis=1)
               for th in np.pi * np.arange(1, SHELL_RINGS) / SHELL_RINGS]  # fmt: skip
        b = np.linspace(0, np.pi, 33)
        lon = [np.stack([np.sin(b) * np.cos(ph), np.sin(b) * np.sin(ph), np.cos(b)], axis=1)
               for ph in np.pi * np.arange(2 * SHELL_RINGS) / SHELL_RINGS]  # fmt: skip
        unit = np.concatenate([np.stack([c[:-1], c[1:]], axis=1) for c in lat + lon])  # (U, 2, 3)
        pieces = centers[keep][:, None, None, :] + r * unit[None]  # (K, U, 2, 3)
        flat = pieces.reshape(-1, 3)
        if g.box is not None:
            ok = np.all((flat >= g.lo - 1e-9) & (flat <= g.hi + 1e-9), axis=1)
        else:
            ok = ac_math.inside(g.vertices, g.triangles, flat)
            ok[ok] = ac_math.segments_clear(g.vertices, g.triangles,
                                            np.broadcast_to(np.asarray(source), (int(ok.sum()), 3)),
                                            flat[ok])  # fmt: skip
        ok = ok.reshape(len(keep), -1, 2).all(axis=2)
        k, u = np.nonzero(ok)
        return pieces[k, u], keep[k]


# ---------------------------------------------------------------- Directivity


@register
class Directivity(ObjectBase):
    """Polar pressure magnitude for a linear array, baffled piston, or cardioid."""

    type: ClassVar[str] = "directivity"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "guide": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "guide_stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        kind: Literal["array", "piston", "cardioid"] = "array"
        elements: int = Field(8, ge=1)
        spacing: float = Field(0.1715, gt=0, allow_inf_nan=False)  # meters
        radius: float = Field(0.343, gt=0, allow_inf_nan=False)  # piston radius in meters
        frequency: float = Field(1000.0, gt=0, allow_inf_nan=False)  # Hz
        steer: float = Field(0.0, allow_inf_nan=False)  # degrees from +x
        pos: Vec3 = (0.0, 0.0, 0.0)
        size: float = Field(2.0, gt=0, allow_inf_nan=False)  # unity-gain radius in world units

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        return Bounds((x - p.size, y - p.size, z), (x + p.size, y + p.size, z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        center = np.asarray(p.pos, dtype=np.float64)
        angle = np.deg2rad(p.steer % 360)
        peak = center + p.size * np.array([np.cos(angle), np.sin(angle), 0])
        return {"center": center, "peak": peak}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        data: dict[str, Any] = {"kind": p.kind, "steer": p.steer, "peak_gain": 1.0}
        if p.kind == "array":
            data.update(elements=p.elements, spacing=p.spacing, frequency=p.frequency)
        elif p.kind == "piston":
            data.update(radius=p.radius, frequency=p.frequency)
        return data

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        center = np.asarray(p.pos, dtype=np.float64)
        angle = np.linspace(0, 2 * np.pi, 1025)
        steer = np.deg2rad(p.steer % 360)
        if p.kind == "array":
            gain = ac_math.array_directivity(angle, p.elements, p.spacing, p.frequency, steer)
        elif p.kind == "piston":
            gain = ac_math.piston_directivity(angle - steer, p.radius, p.frequency)
        else:
            gain = ac_math.cardioid_directivity(angle - steer)

        unit = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=-1)
        rim = center + p.size * unit
        width = float(cam.units_per_height(center))
        guide = with_alpha(me.style["guide"], p.opacity)
        polyline(ctx, cam, rim, width * me.style["guide_stroke"], color=guide)

        points = center + (p.size * gain)[:, None] * unit
        color = with_alpha(me.style["color"], p.opacity)
        colors = np.broadcast_to(color, (len(angle), 4)).copy()
        colors[:, 3] *= np.sqrt(gain)
        polyline(
            ctx, cam, points, width * me.style["stroke"],
            color=color, glow=me.style["glow"], colors=colors,
        )  # fmt: skip


# Wave fields of point sources live in ``waves``; ``ac`` names them.
from phasor.domains.waves import Interference, PhasedArray  # noqa: E402, F401
