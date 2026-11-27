"""An N-dimensional lattice seen in 3D: the echoes of a
feedback-delay network with K loops.

    from phasor import Scene, expr, nd

    s.add(nd.StarLattice(id="echoes", loops=16, max_time=6.0, time=expr("t - 1"), arms=3.0))
    s.at(2.0).to("echoes.arms", 16.0, duration=8.0)  # the star gains arms; the frame fills

Each echo is the point n of N^K that says how many laps it took round each loop; it arrives
at sum n_i d_i. Each loop has an arm in 3D, and the echo sits at sum n_i v_i, so three loops
are the plain cubic grid and sixteen are a star that packs with points (``ndlattice_math``
has the numerics). An echo flashes when ``time`` reaches its arrival (the equal-time surface
is the hyperplane sum n_i d_i = t) and then glows on, faintly. ``arms`` fades the loops in one
by one, as a pure function of the parameter: arm i shows by clip(arms - i, 0, 1), and its
echoes slide out of the ones that do not use it.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import ndlattice_math as nlm
from phasor.domains.strokes import disc_batch, flash_points, lut_colors, polylines

DEFAULT_LOOPS = 3  # loops when neither ``loops`` nor ``delays`` says
MAX_DRAWN = 200_000  # echoes enumerated (and drawn) at most
VIVID_BRIGHT = 1.5  # flash_style "vivid": at arrival an echo is this much brighter...
VIVID_GROW = 1.2  # ...and this much bigger (its own look, before the shared flash)
SHOWN_FEATURES = 200  # echoes listed in ``features`` (the earliest), for the inspector


@register
class StarLattice(ObjectBase):
    """The echoes of K feedback loops: the lattice N^K flattened into 3D as a star.

    ``loops`` K (1-32; 0, the default: one per delay, or 3) and their ``delays`` d_i
    (default: 1 to 1.5, no two in a simple ratio). Each loop has an arm: ``directions`` (one
    3-vector per loop), or the ``fan`` ``grid`` (x, y, z first, so three loops are the cubic
    grid, then arms between them), ``cap`` or ``sphere``; ``spacing`` world units long, times
    d_i over the mean delay when ``by_delay``. Echo n sits at ``origin`` + sum n_i v_i.

    The echoes arriving by ``max_time`` (and with at most ``max_order`` laps in all) are
    shown, at most ``max_points`` of them: past that, the latest arrival times are dropped
    whole and ``complete_until`` in the semantics says up to when the set is complete.

    ``time`` (animatable, usually ``expr("t - 1")``): an echo flashes when ``time`` reaches
    its arrival, then glows on at ``afterglow`` of its light (``decay`` seconds to settle);
    the default lights every echo. The flash is the one every lattice of lights shares
    (``sweep_math.envelope``, drawn by ``strokes.flash_points``): a flashing echo grows by
    ``swell`` and runs white-hot by ``white``, with a soft halo in the glow role.
    ``lap_gain`` (0-1] dims an echo by lap_gain ** laps, the loss of each trip round a loop.
    ``arms`` (animatable, 0 to K; default all) fades the loops in one by one. Colour is the
    ``colormap`` at ``color_by``: ``arrival`` (time over
    ``max_time``), ``order`` (laps in all), ``arm`` (the loop it went round most) or
    ``value`` (sum n_i ``arm_values``_i, scaled to 0-1), over ``hue_span`` of the palette
    from ``hue``. ``point_size`` is a share of the frame height; ``arm_opacity`` the arms'
    light (0: no arms). More than ``crowd`` echoes share their light (each dimmed by
    sqrt(crowd / count)), so a packed star glows in colour instead of burning white."""

    type: ClassVar[str] = "star_lattice"
    Life: ClassVar = frozenset({"glow"})  # colour and place mean something: only glow breathes
    Style: ClassVar = {
        "colormap": ("colormap", "spectrum"),
        "glow": ("glow", "soft"),
        "stroke": ("stroke", "hairline"),
    }
    Presets: ClassVar = {
        "prism": {"style": {"colormap": "spectrum", "glow": "strong"}},
        "plain": {"params": {"afterglow": 0.6}, "style": {"glow": "none"}},
        # the star's first look (2026-11-25): saturated colours, no white heat or halo
        "vivid": {
            "params": {"flash_style": "vivid"},
            "style": {"colormap": "spectrum", "glow": "strong"},
        },
    }

    class Params(ParamsBase):
        loops: int = Field(0, ge=0, le=nlm.MAX_LOOPS)  # K; 0: one per delay (3 without)
        delays: tuple[float, ...] = ()  # one per loop, > 0; () for the default
        directions: tuple[Vec3, ...] = ()  # one arm direction per loop; () for the fan
        fan: Literal["grid", "cap", "sphere"] = "grid"
        spacing: float = Field(1.0, gt=0)  # world length of an arm (of the mean delay)
        by_delay: bool = True  # arm lengths in proportion to the delays
        origin: Vec3 = (0.0, 0.0, 0.0)  # where the first sound (n = 0) is
        max_time: float = Field(6.0, ge=0)  # echoes arriving by this time are shown
        max_order: int = Field(nlm.MAX_ORDER, ge=0, le=nlm.MAX_ORDER)  # laps in all, at most
        max_points: int = Field(20_000, ge=1, le=MAX_DRAWN)
        arms: float = Field(float(nlm.MAX_LOOPS), ge=0, le=nlm.MAX_LOOPS)  # loops shown
        time: float = 1e9  # seconds since the sound; an echo lights when this reaches it
        rise: float = Field(0.05, ge=0)  # seconds an echo takes to light
        decay: float = Field(0.6, gt=0)  # seconds its flash takes to settle
        afterglow: float = Field(0.35, ge=0, le=1)  # its light after the flash
        swell: float = Field(1.2, ge=0)  # a flashing echo grows by this share of its size
        white: float = Field(0.6, ge=0, le=1)  # how white-hot the flash runs
        # "shared": the flash every lattice of lights shares (white heat, halo); "vivid": the
        # star's own, brighter and bigger in its own colour, so a packed star stays saturated
        flash_style: Literal["shared", "vivid"] = "shared"
        lap_gain: float = Field(1.0, gt=0, le=1)  # light kept per lap round a loop
        color_by: Literal["arrival", "order", "arm", "value"] = "arrival"
        arm_values: tuple[float, ...] = ()  # color_by "value": one number per loop
        hue: float = 0.0  # where on the palette the colours start
        hue_span: float = Field(0.85, ge=0, le=1)  # share of the palette the values run over
        point_size: float = Field(0.005, gt=0, le=0.05)  # radius, share of the frame height
        arm_opacity: float = Field(0.3, ge=0, le=1)
        crowd: float = Field(300.0, gt=0)  # echoes at full light; more share it (see draw)

        @model_validator(mode="after")
        def _per_loop(self):
            k = self.k
            for name in ("delays", "directions", "arm_values"):
                got = getattr(self, name)
                if got and len(got) != k:
                    raise ValueError(f"{k} loops but {len(got)} {name}")
            if any(d <= 0 for d in self.delays):
                raise ValueError("delays must be positive")
            if any(math.hypot(*v) == 0 for v in self.directions):
                raise ValueError("directions must not be zero")
            if self.color_by == "value" and not self.arm_values:
                raise ValueError('color_by "value" needs arm_values, one per loop')
            return self

        @property
        def k(self) -> int:
            """The number of loops: ``loops``, or one per delay, or 3."""
            return self.loops or len(self.delays) or DEFAULT_LOOPS

    # ------------------------------------------------------------ the numbers

    def _delays(self, p) -> tuple[float, ...]:
        return tuple(p.delays) or nlm.default_delays(p.k)

    def _arms(self, p) -> np.ndarray:
        u = np.asarray(p.directions, dtype=np.float64) if p.directions else nlm.star_directions(
            p.k, p.fan)  # fmt: skip
        return nlm.arm_vectors(u, self._delays(p), p.spacing, p.by_delay)

    def lattice(self, state: SceneState) -> dict:
        """``ndlattice_math.enumerate_lattice`` for these params (cached there)."""
        p = self.me(state).params
        return nlm.enumerate_lattice(self._delays(p), p.max_time, p.max_order, p.max_points)

    def _values(self, p, lat) -> np.ndarray:
        v = nlm.values(p.color_by, lat["index"], lat["arrival"], p.max_time, p.arm_values)
        if p.color_by == "value" and len(v):
            lo, hi = float(v.min()), float(v.max())
            v = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
        return v

    def visible(self, state: SceneState) -> dict[str, np.ndarray]:
        """What is drawn at this state, for the echoes that show (weight and light above 0):
        ``points`` (M, 3), ``arrival``, ``values`` (0-1), ``light`` (0-1, the alpha),
        ``burst`` (1 at the top of the flash, falling to 0) and ``flash`` (the burst dimmed as
        the light is, by arms, laps and the crowd: what is drawn). Pure in the state."""
        p = self.me(state).params
        lat = self.lattice(state)
        w = nlm.arm_weights(p.k, p.arms)
        weight = nlm.point_weights(lat["index"], w)
        light, burst = nlm.flash(p.time - lat["arrival"], p.rise, p.decay, p.afterglow)
        dim = weight
        if p.lap_gain < 1.0:
            dim = dim * p.lap_gain ** lat["order"]
        # a crowd shares the light: past ``crowd`` echoes (counted by their weights, so the
        # share moves smoothly as arms fade in) each one is dimmed by sqrt(crowd / count),
        # so a packed star glows in its colours instead of burning white
        dim = dim * min(1.0, math.sqrt(p.crowd / max(float(weight.sum()), 1e-9)))
        light = light * dim
        keep = light > 0
        idx = lat["index"][keep]
        return {"points": nlm.project(idx, self._arms(p), w, p.origin),
                "arrival": lat["arrival"][keep], "values": self._values(p, lat)[keep],
                "light": light[keep], "burst": burst[keep],
                "flash": (burst * dim)[keep]}  # fmt: skip

    def _tips(self, p, lat, w) -> np.ndarray:
        """(K, 3) the end of each arm: its farthest lap among the echoes, as shown."""
        laps = lat["index"].max(axis=0).astype(np.float64) if len(lat["index"]) else np.zeros(
            p.k)  # fmt: skip
        v = self._arms(p) * nlm.smoothstep(w)[:, None]
        return np.asarray(p.origin, dtype=np.float64) + v * np.maximum(laps, 1.0)[:, None]

    # ------------------------------------------------------------ the contract

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        lat = self.lattice(state)
        w = nlm.arm_weights(p.k, p.arms)
        pts = nlm.project(lat["index"], self._arms(p), w, p.origin)
        return Bounds.of_points(np.vstack([pts, self._tips(p, lat, w), [p.origin]]),
                                pad=p.spacing * 0.1)  # fmt: skip

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        lat = self.lattice(state)
        w = nlm.arm_weights(p.k, p.arms)
        vis = self.visible(state)
        return {"origin": np.asarray(p.origin, dtype=np.float64),
                "tips": self._tips(p, lat, w)[w > 0],
                "points": vis["points"][:SHOWN_FEATURES]}  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        lat = self.lattice(state)
        w = nlm.arm_weights(p.k, p.arms)
        weight = nlm.point_weights(lat["index"], w)
        shown = weight > 0
        heard = shown & (lat["arrival"] <= p.time)
        return {
            "kind": "star_lattice", "loops": p.k,
            "delays": [round(d, 6) for d in self._delays(p)],
            "arms": round(min(p.arms, p.k), 6), "arms_shown": int((w > 0).sum()),
            "points": len(lat["arrival"]), "shown": int(shown.sum()),
            "heard": int(heard.sum()), "time": round(min(p.time, 1e9), 6),
            "max_time": p.max_time, "complete_until": round(lat["complete_until"], 6),
            "truncated": bool(lat["truncated"]), "color_by": p.color_by,
        }  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        lut = me.style["colormap"]
        glow = float(me.style.get("glow", 1.0))
        vis = self.visible(state)
        if len(vis["points"]):
            colors = lut_colors(lut, p.hue + p.hue_span * vis["values"])
            colors[:, 3] = vis["light"]
            radius = p.point_size * np.asarray(cam.units_per_height(vis["points"]))
            if p.flash_style == "vivid":
                colors[:, :3] *= (1.0 + VIVID_BRIGHT * vis["burst"])[:, None]
                colors[:, 3] *= p.opacity
                disc_batch(ctx, cam, vis["points"], radius * (1.0 + VIVID_GROW * vis["burst"]),
                           colors=colors, glow=glow)  # fmt: skip
            else:
                flash_points(ctx, cam, vis["points"], radius, colors=colors,
                             flash=vis["flash"], glow=glow, opacity=p.opacity, swell=p.swell,
                             white=p.white)  # fmt: skip
        if p.arm_opacity > 0:
            lat = self.lattice(state)
            w = nlm.arm_weights(p.k, p.arms)
            on = np.flatnonzero(w > 0)
            if not len(on):
                return
            o = np.asarray(p.origin, dtype=np.float64)
            tips = self._tips(p, lat, w)[on]
            segs = np.stack([np.broadcast_to(o, tips.shape), tips], axis=1)  # (A, 2, 3)
            width = me.style["stroke"] * np.asarray(
                cam.units_per_height(segs.reshape(-1, 3))).reshape(-1, 2)  # fmt: skip
            hue = lut_colors(lut, p.hue + p.hue_span * on / p.k)
            hue[:, 3] = p.arm_opacity * w[on]
            polylines(ctx, cam, segs, width, color=(1.0, 1.0, 1.0, p.opacity), glow=1.0,
                      colors=np.repeat(hue[:, None], 2, axis=1))  # fmt: skip
