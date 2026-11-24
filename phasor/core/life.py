"""Living motion: nothing on screen sits perfectly still.

A smooth, deterministic function of (object id, seed, channel, t): the same frame always
renders the same, so ``state_at(t)`` stays a pure function of (document, t) (contract 3).
The noise is a few slow sine waves whose frequencies and phases come from
``rng(object_id, seed)``, never from a clock.

What moves (amplitudes live in the theme's ``[life]``; the scene's ``life`` is ``off``,
``subtle`` or ``lively``):

- types that declare ``"wander"`` in ``Life`` (dots) drift a fraction of their radius ``r``;
- glowing objects breathe: their glow swells and fades a little (strokes shimmer faster);
- types that declare ``"hue"`` (color fields) turn their colors a few percent of the wheel (a
  field's geometry never moves: a pole stays at its computed place);
- ``"float"`` (equations) drift their ``pos`` slowly by ``[life] float`` world units, times
  their ``drift`` (a look sets it: ``prism_field`` lets equations over a field drift less);
- ``"twinkle"`` (point clouds): the ``twinkle`` amount is scaled by the level (0 when
  ``off`` or ``exact``); the type flickers each point's light by it, a pure function of
  (id, seed, t).

Each type says what may happen to it (``ObjectBase.Life``, glow only by default): guessing
from parameter names once made polygons, clocks and acoustic sources drift.

Meaningful geometry stays put: objects tagged ``exact`` do not move, and a label's baked
slot is measured against the wiggling anchor, so labels follow.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from typing import Any

from phasor.core.rng import rng

LEVELS = ("off", "subtle", "lively")
EXACT = "exact"  # tag: this object never moves by life


def scale(level: str, theme_life: Mapping[str, float]) -> float:
    """Amplitude multiplier for a scene ``life`` level."""
    if level not in LEVELS:
        raise ValueError(f"life must be one of {LEVELS}, got {level!r}")
    return {"off": 0.0, "subtle": float(theme_life.get("subtle", 0.5)), "lively": 1.0}[level]


@functools.cache
def _waves(object_id: str, seed: int, channel: str) -> tuple[tuple[float, float], ...]:
    g = rng(f"{object_id}/{channel}", seed)
    # two slow waves; relative frequencies near 1 and 1.7, random phases
    return tuple(
        (float(f), float(g.uniform(0, 2 * math.pi)))
        for f in (g.uniform(0.8, 1.2), g.uniform(1.5, 2.1))
    )


def noise(object_id: str, seed: int, channel: str, t: float, period: float) -> float:
    """Smooth noise in [-1, 1] with a typical period ``period`` seconds."""
    (f1, p1), (f2, p2) = _waves(object_id, seed, channel)
    w = 2 * math.pi * t / max(period, 1e-6)
    return (math.sin(w * f1 + p1) + 0.6 * math.sin(w * f2 + p2)) / 1.6


def adjust(
    object_id: str,
    tags: tuple[str, ...],
    fields: set[str],
    values: dict[str, Any],
    style: Mapping[str, Any],
    *,
    seed: int,
    amp: float,
    theme_life: Mapping[str, float],
    t: float,
    channels: frozenset[str] = frozenset({"glow"}),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply life to one object's resolved parameter ``values`` and ``style``; returns the
    new (values, style). Pure. ``amp`` 0 changes nothing."""
    twinkle = "twinkle" in channels and "twinkle" in fields
    if amp <= 0 or EXACT in tags:
        if twinkle:  # still: no flicker either
            values = {**values, "twinkle": 0.0}
        return values, dict(style)
    th = theme_life
    values, style = dict(values), dict(style)
    if twinkle:  # a point cloud's points flicker (drawn by the type from (id, seed, t))
        values["twinkle"] = values["twinkle"] * amp
    if (
        "wander" in channels and {"pos", "r"} <= fields
    ):  # a point drifts a fraction of its radius, in the camera plane
        d = amp * th["point"] * values["r"]
        x, y, z = values["pos"]
        values["pos"] = (
            x + d * noise(object_id, seed, "x", t, th["point_period"]),
            y + d * noise(object_id, seed, "y", t, th["point_period"]),
            z,
        )
    if "float" in channels and "pos" in fields:  # an equation drifts slowly, a little
        # ``drift`` (if the type has it) is the share of the theme's float this one takes
        d = amp * float(th.get("float", 0.06)) * float(values.get("drift", 1.0))
        period = float(th.get("float_period", 9.0))

        def offset(oid: str) -> tuple[float, float]:
            return (d * noise(oid, seed, "fx", t, period),
                    d * noise(oid, seed, "fy", t, period * 1.3))  # fmt: skip

        dx, dy = offset(object_id)
        src, m = values.get("from_id"), float(values.get("morph", 1.0))
        if src and m < 1.0:  # mid-transform: from where the source was drifting to its own
            sx, sy = offset(src)
            dx, dy = sx + (dx - sx) * m, sy + (dy - sy) * m
        for key in ("pos", "from_pos") if src else ("pos",):
            x, y, z = values[key]
            values[key] = (x + dx, y + dy, z)
    if (
        "hue" in channels and "hue" in fields
    ):  # a color field turns its colors by a few percent of the wheel
        values["hue"] = values["hue"] + amp * th["field"] * noise(
            object_id, seed, "hue", t, th["field_period"]
        )
    glow = style.get("glow")
    if (
        "glow" in channels and isinstance(glow, float) and glow > 1.0
    ):  # light above plain paint breathes
        curve = "points" in fields
        a, period = (
            (th["shimmer"], th["shimmer_period"]) if curve else (th["glow"], th["glow_period"])
        )
        k = 1.0 + amp * a * noise(object_id, seed, "glow", t, period)
        style["glow"] = 1.0 + (glow - 1.0) * k  # only the emitted part swells
    return values, style
