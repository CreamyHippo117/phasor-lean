"""A lattice of points that flash as a surface sweeps through it.

    from phasor import ac, expr, sweep

    # every echo of a clap is a mirror copy of the room: the image sources form a crystal,
    # and each one flashes when the sphere of time-since-the-clap (round the listener)
    # reaches it, which is the moment its echo is heard
    s.add(ac.Room3D(id="room", size=(8, 5, 3)))
    s.add(sweep.SweepLattice(id="crystal", size=(8, 5, 3), source=(2.2, 2, 1.4), order=58,
                             center=(6, 3.4, 1.4), time=expr("t - 1"), speed=12, extent=60))
    s.add(sweep.SweepFront(id="front", lattice="crystal"))
    s.at(8).to("crystal.flat", 1.0, duration=3)  # the sphere opens out into a plane

The points come from a rule (``mirror_box``: the image sources of a shoebox room up to
``order`` reflections; ``grid``: integer steps along three vectors, the delays of feedback
loops) or are given (``points``, with optional ``values``). The sweep surface is a sphere
round ``center`` (``flat`` 0), the plane through ``center`` with normal ``normal`` (``flat``
1) or the conic between them, and it has travelled ``speed * time``; the arrival distance of
each point is analytic (``sweep_math.arrival``). Everything is a pure function of the
parameters at ``t``: animating ``flat`` or ``time`` simply moves the front.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import ac_math
from phasor.domains import sweep_math as sm
from phasor.domains.strokes import flash_points, lut_colors

MAX_ORDER = ac_math.MAX_ORDER  # mirror_box: about 2 million images (ac_math.MAX_IMAGES)
MAX_GRID = 2_000_000  # grid: points at most
LIGHT_FLOOR = 1.0 / 512  # points dimmer than this are not drawn


@functools.lru_cache(maxsize=16)
def lattice_points(rule: str, key: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(positions (N, 3), values (N,)) of a lattice, cached per rule and parameters and
    read-only."""
    if rule == "mirror_box":
        pos, value = sm.mirror_box(*key)
    elif rule == "grid":
        pos, value = sm.grid(*key)
    else:
        points, values = key
        pos = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        value = np.asarray(values, dtype=np.float64) if values else np.zeros(len(pos))
    pos, value = np.ascontiguousarray(pos), np.ascontiguousarray(value)
    pos.flags.writeable = False
    value.flags.writeable = False
    return pos, value


def _vec3_positive(v):
    if min(v) <= 0:
        raise ValueError("size must be positive in every direction")
    return v


@register
class SweepLattice(ObjectBase):
    """Points that flash as a surface sweeps them. Each point's light is ``before`` until
    the front reaches it, then rises over ``attack`` seconds, swells (``swell``: size,
    ``white``: white heat) and decays with time constant ``decay`` to the afterglow
    ``after``. Color is the ``colormap`` role at ``hue + hue_step * v`` (a wheel), where v is
    the point's value (``color_by="value"``: reflections for ``mirror_box``, steps for
    ``grid``, ``values`` for given points) or its arrival distance (``color_by="arrival"``);
    ``fade`` dims each unit of value (an echo losing energy at every reflection). Points are
    drawn as small glowing discs of radius ``r`` (world units), never smaller than
    ``min_size`` of the frame height. Only points within ``extent`` of the centre (0: all),
    in view, and at most ``max_points`` (the ones nearest the front in time) are drawn."""

    type: ClassVar[str] = "sweep_lattice"
    Life: ClassVar = frozenset({"glow"})  # the points are where the echoes are: they never move
    Style: ClassVar = {"colormap": ("colormap", "spectrum"), "glow": ("glow", "strong")}
    Presets: ClassVar = {
        "echo": {},  # the defaults: a crystal of echoes, colored by reflections
        # feedback loops: the plane sweeping a staircase of delays
        "stairs": {"params": {"rule": "grid", "flat": 1.0, "normal": (1.0, 1.0, 0.0),
                              "center": (0.0, 0.0, 0.0), "hue_step": 0.08}},
    }  # fmt: skip

    class Params(ParamsBase):
        rule: Literal["mirror_box", "grid", "points"] = "mirror_box"
        # mirror_box: a shoebox room with its corner at the origin and a source in it
        size: Vec3 = (8.0, 5.0, 3.0)
        source: Vec3 = (2.2, 2.0, 1.4)
        order: int = Field(8, ge=0, le=MAX_ORDER)
        # grid: origin + i a + j b + k c, 0 <= i < counts[0] ...
        basis: tuple[Vec3, Vec3, Vec3] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        counts: tuple[int, int, int] = (8, 8, 1)
        origin: Vec3 = (0.0, 0.0, 0.0)
        # points: given directly, each with an optional value
        points: tuple[Vec3, ...] = ()
        values: tuple[float, ...] = ()
        # the sweep surface
        center: Vec3 = (6.0, 3.4, 1.4)  # the sphere's centre, a point on the start plane
        normal: Vec3 = (0.0, 0.0, 1.0)  # the plane's normal (and where the conic opens)
        flat: float = Field(0.0, ge=0, le=1)  # 0 sphere, 1 plane, between: the conic
        time: float = 0.0  # seconds since the front left the centre; negative: not yet
        speed: float = Field(1.0, gt=0)  # world units the front travels per second
        extent: float = Field(0.0, ge=0)  # only points this far from the centre; 0: all
        # the flash
        attack: float = Field(0.03, ge=0)  # seconds to light up
        decay: float = Field(0.4, gt=0)  # seconds: time constant of the flash
        before: float = Field(0.06, ge=0, le=1)  # light of points not yet reached; 0: hidden
        after: float = Field(0.18, ge=0, le=1)  # the afterglow
        swell: float = Field(1.2, ge=0)  # a flashing point grows by this share of its size
        white: float = Field(0.6, ge=0, le=1)  # how white-hot the flash runs
        fade: float = Field(1.0, gt=0, le=1)  # light kept per unit of value (per reflection)
        r: float = Field(0.08, gt=0)  # point radius, world units
        min_size: float = Field(0.0025, ge=0)  # smallest radius, share of the frame height
        color_by: Literal["value", "arrival"] = "value"
        hue_step: float = 0.05  # share of the colormap per unit of value (or of distance)
        hue: float = 0.0  # turns the colormap
        max_points: int = Field(60_000, ge=1, le=2_000_000)

        @field_validator("size")
        @classmethod
        def _size(cls, v):
            return _vec3_positive(v)

        @field_validator("normal")
        @classmethod
        def _normal(cls, v):
            if not any(v):
                raise ValueError("the normal must not be zero")
            return v

        @field_validator("counts")
        @classmethod
        def _counts(cls, v):
            if min(v) < 1 or v[0] * v[1] * v[2] > MAX_GRID:
                raise ValueError(f"counts must be 1 or more, {MAX_GRID:,} points at most")
            return v

        @model_validator(mode="after")
        def _values(self):
            if self.values and len(self.values) != len(self.points):
                raise ValueError(f"{len(self.points)} points but {len(self.values)} values")
            return self

    # ------------------------------------------------------------ data

    def lattice(self, state: SceneState) -> tuple[np.ndarray, np.ndarray]:
        """Every point of the lattice and its value (cached, read-only)."""
        p = self.me(state).params
        if p.rule == "mirror_box":
            key: tuple = (tuple(p.size), tuple(p.source), p.order)
        elif p.rule == "grid":
            key = (tuple(map(tuple, p.basis)), tuple(p.counts), tuple(p.origin))
        else:
            key = (tuple(map(tuple, p.points)), tuple(p.values))
        return lattice_points(p.rule, key)

    def reach(self, state: SceneState) -> float:
        """How far the front has travelled (negative before it leaves)."""
        p = self.me(state).params
        return p.speed * p.time

    def sweep(self, state: SceneState) -> dict[str, Any]:
        """The front's geometry: centre, normal, flat, reach and the extent (the radius of
        the ball the lattice is drawn in: ``extent``, or every point's)."""
        p = self.me(state).params
        pos, _ = self.lattice(state)
        c = np.asarray(p.center, dtype=np.float64)
        extent = p.extent or (float(np.sqrt(((pos - c) ** 2).sum(axis=1).max())) if len(pos)
                              else 1.0)  # fmt: skip
        return {"center": c, "normal": sm.unit(p.normal), "flat": p.flat,
                "reach": self.reach(state), "extent": max(extent, 1e-9)}  # fmt: skip

    def _arrivals(self, state: SceneState):
        """(positions, values, arrival distances) of the points inside the extent."""
        p = self.me(state).params
        pos, value = self.lattice(state)
        s = sm.arrival(pos, p.center, p.normal, p.flat)
        if p.extent > 0:
            d2 = ((pos - np.asarray(p.center)) ** 2).sum(axis=1)
            keep = d2 <= p.extent * p.extent
            pos, value, s = pos[keep], value[keep], s[keep]
        return pos, value, s

    def _light(self, state: SceneState, value, s):
        p = self.me(state).params
        age = np.where(s >= 0, p.time - s / p.speed, -1.0)  # behind the start: never
        light, flash = sm.envelope(age, attack=p.attack, decay=p.decay, before=p.before,
                                   after=p.after)  # fmt: skip
        if p.fade < 1:
            light = light * p.fade ** np.maximum(value, 0.0)
        return light, flash, age

    def shown(self, state: SceneState) -> dict[str, np.ndarray]:
        """What is drawn at this time: ``pos``, ``value``, ``arrival`` (distance), ``light``
        and ``flash`` of every point lit, in view and within the count, far to near."""
        me = self.me(state)
        p = me.params
        pos, value, s = self._arrivals(state)
        light, flash, age = self._light(state, value, s)
        ok = light * p.opacity >= LIGHT_FLOOR
        if ok.any():
            cam = state.camera_for(me)
            clip = np.c_[pos, np.ones(len(pos))] @ cam.view_proj.T
            w = clip[:, 3]
            margin = 1.05 * np.maximum(np.abs(w), 1e-9)
            ok &= (w > 0) & (np.abs(clip[:, 0]) <= margin) & (np.abs(clip[:, 1]) <= margin)
        idx = np.flatnonzero(ok)
        if len(idx) > p.max_points:  # the ones nearest the front in time
            near = np.argsort(np.abs(age[idx]), kind="stable")[: p.max_points]
            idx = np.sort(idx[near])
        if len(idx):
            depth = state.camera_for(me).depth(pos[idx])
            idx = idx[np.argsort(-depth, kind="stable")]  # far to near: blending, depth test
        return {"pos": pos[idx], "value": value[idx], "arrival": s[idx], "light": light[idx],
                "flash": flash[idx]}  # fmt: skip

    # ------------------------------------------------------------ contract

    def world_bounds(self, state: SceneState) -> Bounds:
        pos, _, _ = self._arrivals(state)
        if not len(pos):
            return Bounds.of_points([self.me(state).params.center])
        return Bounds.of_points(pos, pad=self.me(state).params.r)

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        pos, _, s = self._arrivals(state)
        out = {"center": np.asarray(p.center, dtype=np.float64)}
        if p.rule == "mirror_box":
            out["source"] = np.asarray(p.source, dtype=np.float64)
        if len(pos):
            out["points"] = pos[np.argsort(np.abs(s - self.reach(state)), kind="stable")[:200]]
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        pos, value, s = self._arrivals(state)
        reach = self.reach(state)
        _, flash, _ = self._light(state, value, s)
        reached = (s >= 0) & (s <= reach)
        ahead = s[(s > reach)]
        out: dict[str, Any] = {
            "kind": "sweep_lattice", "backdrop": True, "rule": p.rule, "points": len(pos),
            "reach": round(max(reach, 0.0), 6), "flat": round(p.flat, 6),
            "reached": int(reached.sum()), "flashing": int((flash > 0.05).sum()),
            "rate": round(sm.rate(s, max(reach, 0.0), p.speed, 1.0), 3),  # flashes per second
        }  # fmt: skip
        if len(ahead):
            out["next_arrival"] = round(float(ahead.min() / p.speed - p.time + state.t), 6)
        if p.rule == "mirror_box":
            out.update(order=p.order, deepest_reached=int(value[reached].max(initial=0)))
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        vis = self.shown(state)
        if not len(vis["pos"]):
            return
        cam = state.camera_for(me)
        where = vis["arrival"] if p.color_by == "arrival" else vis["value"]
        colors = lut_colors(me.style["colormap"], p.hue + p.hue_step * where)
        colors[:, 3] = np.minimum(vis["light"], 1.0)
        size = np.maximum(p.r, p.min_size * np.asarray(cam.units_per_height(vis["pos"])))
        flash_points(ctx, cam, vis["pos"], size, colors=colors, flash=vis["flash"],
                     glow=float(me.style["glow"]), opacity=p.opacity, swell=p.swell,
                     white=p.white)  # fmt: skip


@register
class SweepFront(ObjectBase):
    """The sweep surface of a ``SweepLattice`` (named by ``lattice``), drawn as a soft
    glowing shell: see-through where it faces the camera and bright at its rim, where the
    eye looks along it (a Fresnel-like ``rim`` power), so a sphere reads as a glowing ring
    with a faint veil inside it and a plane seen edge-on as a line of light. It is cut to
    the lattice's extent and fades out at an open edge. ``material`` (a surface role) sets
    how strong it is; glass keeps it from hiding anything (it is never solid paint)."""

    type: ClassVar[str] = "sweep_front"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),
        "glow": ("glow", "soft"),
        "material": ("surface", "glass"),
    }

    class Params(ParamsBase):
        lattice: str = Field("lattice", min_length=1)  # the id of a SweepLattice
        rim: float = Field(4.0, gt=0)  # how sharply the light gathers at the rim
        veil: float = Field(0.02, ge=0, le=1)  # light where it faces the camera
        rings: int = Field(sm.FRONT_RINGS, ge=4, le=512)
        sectors: int = Field(sm.FRONT_SECTORS, ge=8, le=1024)

    def _sweep(self, state: SceneState) -> dict[str, Any] | None:
        p = self.me(state).params
        if p.lattice not in state or not isinstance(state[p.lattice].impl, SweepLattice):
            return None
        return state[p.lattice].impl.sweep(state)

    def mesh(self, state: SceneState):
        """``sweep_math.front_mesh`` for the lattice at this time, or None."""
        sw = self._sweep(state)
        if sw is None:
            return None
        p = self.me(state).params
        return sm.front_mesh(sw["center"], sw["normal"], sw["flat"], sw["reach"], sw["extent"],
                             p.rings, p.sectors)  # fmt: skip

    def world_bounds(self, state: SceneState) -> Bounds:
        sw = self._sweep(state)
        if sw is None:
            return Bounds.of_points([(0.0, 0.0, 0.0)])
        return Bounds.of_points([sw["center"] - sw["extent"], sw["center"] + sw["extent"]])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        sw = self._sweep(state)
        if sw is None:
            return {}
        return {"center": sw["center"], "apex": sw["center"] + max(sw["reach"], 0) * sw["normal"]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        sw = self._sweep(state)
        if sw is None:
            return {"kind": "sweep_front", "lattice_missing": p.lattice}
        b = sw["flat"]
        shape = "sphere" if b == 0 else "plane" if b == 1 else (
            "ellipsoid" if b < 0.5 else "paraboloid" if b == 0.5 else "hyperboloid")  # fmt: skip
        apex = sw["center"] + max(sw["reach"], 0.0) * sw["normal"]
        return {"kind": "sweep_front", "shape": shape, "flat": round(b, 6),
                "reach": round(max(sw["reach"], 0.0), 6), "extent": round(sw["extent"], 6),
                "apex": [round(float(v), 6) for v in apex],
                "visible": self.mesh(state) is not None}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        mesh = self.mesh(state)
        if mesh is None:
            return
        me = self.me(state)
        p = me.params
        pos, normals, tris, edge = mesh
        cam = state.camera_for(me)
        if cam.perspective:
            view = cam.eye - pos
            view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
        else:
            view = -np.asarray(cam.forward, dtype=np.float64)[None]
        facing = np.abs((normals * view).sum(axis=1))
        light = (p.veil + (1.0 - p.veil) * (1.0 - facing) ** p.rim) * edge
        colors = np.empty((len(pos), 4))
        colors[:, :3] = np.asarray(me.style["color"][:3])
        colors[:, 3] = light
        # emissive: unlit light of its own, so it glows and blooms and never turns into a
        # lit, shaded ball (glass is lit and nearly opaque face-on: see SweepFront's docs)
        ctx.surface(pos, tris, normals=normals, colors=colors, material="emissive",
                    opacity=p.opacity * min(float(me.style["material"]), 0.99),
                    glow=float(me.style["glow"]))  # fmt: skip
