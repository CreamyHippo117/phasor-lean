"""PDE objects: fields, profiles and surfaces of the solvers in
``pde_math``.

    from phasor import pde

    s.add(pde.Profile(id="rod", kind="heat", time=expr("t / 8"), exact=1.0))
    s.add(pde.Field(id="plate", kind="heat", mode=(0, 0), time=expr("t / 20")))
    s.add(pde.Surface(id="drum", kind="wave", mode=(2, 1), time=expr("t / 4")))

``time`` is simulation time; every frame is a pure function of the parameters, computed by a
checkpointed ``Simulation`` so scrubbing is cheap. A solver step that would be unstable is a
validation error.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field as PField
from pydantic import model_validator

from phasor.core.color import ramp
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.simulation import runner_for
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import pde_math as pm
from phasor.domains.heatmap import PRISM, HeatParams, heat_pixels, lut_colour, wheel_colour
from phasor.domains.strokes import disc, polyline, polylines, with_alpha

CHECKPOINT_EVERY = 100
MAX_STEPS = 400_000  # a frame further into the simulation than this is refused


@functools.lru_cache(maxsize=32)
def _sim(kind: str, key: tuple):
    a = dict(key)
    if kind == "heat1d":
        return pm.Heat1D(k=a["diffusivity"], length=a["length"], n=a["n"], modes=a["modes"],
                         boundary=a["boundary"], method=a["method"])  # fmt: skip
    if kind == "heat3d":
        return pm.Heat3D(k=a["diffusivity"], size=a["cube"], n=a["n"], mode=a["mode"])
    if kind == "wave1d":
        return pm.Wave1D(
            c=a["speed"], length=a["length"], n=a["n"], mode=a["modes"][0] if a["modes"] else 0
        )
    if kind == "advection1d":
        return pm.Advection1D(c=a["speed"], length=a["length"], n=a["n"])
    if kind == "schrodinger1d":
        return pm.Schrodinger1D(length=a["length"], n=a["n"], potential=a["potential"], dt=a["dt"])
    if kind == "heat2d":
        return pm.Heat2D(k=a["diffusivity"], size=a["size"], n=a["n"], mode=a["mode"])
    if kind == "wave2d":
        return pm.Wave2D(c=a["speed"], size=a["size"], n=a["n"], mode=a["mode"])
    if kind == "reaction":
        return pm.GrayScott(f=a["feed"], k=a["kill"], n=a["n"], seed=a["seed"])
    raise ValueError(kind)


def _state(sim, time: float):
    runner = runner_for(sim, every=CHECKPOINT_EVERY)
    if runner.step_of(time) > MAX_STEPS:
        raise ValueError(
            f"time {time:g} is {runner.step_of(time):,} steps in (limit {MAX_STEPS:,})"
        )
    # Between two solver steps the state is blended linearly, so a field whose step is longer
    # than a frame still moves every frame (a coarse wave stepped 5 frames at a time looked
    # like 6 fps). Still pure: it depends only on the two steps either side of ``time``.
    f = max(time, 0.0) / sim.dt
    n0 = int(np.floor(f))
    w = f - n0
    a = runner.state_at(n0 * sim.dt)
    if w < 1e-9:
        return a
    b = runner.state_at((n0 + 1) * sim.dt)
    out = {}
    for k, v in a.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.inexact) and k in b:
            m = (1 - w) * v + w * b[k]
            m.setflags(write=False)
            out[k] = m
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- Profile (1D)


@register
class Profile(ObjectBase):
    """A one-dimensional solution as a curve u(x, t): a heating rod, a plucked or standing
    string, a pulse carried along (advection), or a quantum wave packet's density. The exact
    solution, where one exists, is drawn in a second color (``exact`` is its opacity)."""

    type: ClassVar[str] = "pde_profile"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "exact": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {"glowing": {"style": {"glow": "soft"}}}

    class Params(ParamsBase):
        kind: Literal["heat", "wave", "advection", "schrodinger"] = "heat"
        time: float = PField(0.0, ge=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # lower-left corner of the plot
        extent: tuple[float, float] = (6.0, 2.5)
        length: float = PField(1.0, gt=0)
        n: int = PField(100, ge=8, le=2000)
        diffusivity: float = PField(1.0, gt=0)
        speed: float = PField(1.0, gt=0)
        modes: tuple[float, ...] = (1.0,)  # heat: amplitudes of sin(m pi x / L); wave: (m,)
        potential: Literal["free", "harmonic", "barrier"] = "free"
        dt: float = PField(0.01, gt=0)  # quantum solver step
        exact: float = PField(1.0, ge=0, le=1)
        ymax: float = PField(1.2, gt=0)  # the y range shown is [-ymax, ymax] (0 to ymax: density)
        # heat only: ends held at 0 (dirichlet) or insulated (neumann); solver explicit or
        # spectral (exact per mode, any step). The ends are drawn: a pin at 0 or a wall.
        boundary: Literal["dirichlet", "neumann"] = "dirichlet"
        method: Literal["explicit", "spectral"] = "explicit"

        @model_validator(mode="after")
        def _stable(self):
            _sim(self.kind + "1d", _key(self))  # an unstable step raises here
            return self

    def _key_sim(self, p):
        return _sim(p.kind + "1d", _key(p))

    def _data(self, state: SceneState):
        p = self.me(state).params
        sim = self._key_sim(p)
        s = _state(sim, p.time)
        x = sim.x
        if p.kind == "schrodinger":
            u = sim.density(s)
            ex = None
        else:
            u = s["u"]
            ex = sim.exact(p.time)
        return x, u, ex

    def _xy(self, state, p, x, y, density: bool):
        w, h = p.extent
        sx = p.pos[0] + (x - x[0]) / max(x[-1] - x[0], 1e-12) * w
        ysc = (y / p.ymax) * (h if density else h / 2)
        base = p.pos[1] + (0.0 if density else h / 2)
        return np.stack([sx, base + ysc, np.full_like(sx, p.pos[2])], axis=1)

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        return Bounds((x, y, z), (x + p.extent[0], y + p.extent[1], z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        x, u, _ = self._data(state)
        return {"curve": self._xy(state, p, x, u, p.kind == "schrodinger")}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        sim = self._key_sim(p)
        x, u, ex = self._data(state)
        out: dict[str, Any] = {
            "kind": "pde_profile",
            "equation": p.kind,
            "time": round(p.time, 6),
            "max": round(float(u.max()), 6),
            "min": round(float(u.min()), 6),
            "dt": sim.dt,
            "steps": _runner_steps(sim, p.time),
        }
        if ex is not None:
            out["max_error_vs_exact"] = round(float(np.abs(u - ex).max()), 6)
        if p.kind == "schrodinger":
            out["norm"] = round(float(u.sum() * sim.dx), 9)
        if p.kind == "heat":
            out["total_heat"] = round(float(np.trapezoid(u, x)), 6)
            out["boundary"] = _BOUNDARY_TEXT[p.boundary]
            out["method"] = p.method
        elif p.kind in ("wave", "schrodinger"):
            out["boundary"] = _BOUNDARY_TEXT["dirichlet"]
        else:
            out["boundary"] = "periodic: what leaves one end enters the other"
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        x, u, ex = self._data(state)
        dens = p.kind == "schrodinger"
        w = me.style["stroke"] * float(cam.units_per_height())
        x0, y0, z0 = p.pos
        axis_y = y0 if dens else y0 + p.extent[1] / 2
        polyline(
            ctx,
            cam,
            [(x0, axis_y, z0), (x0 + p.extent[0], axis_y, z0)],
            w * 0.7,
            color=with_alpha(me.style["exact"], 0.5 * p.opacity),
        )
        if ex is not None and p.exact > 0:
            polyline(
                ctx,
                cam,
                self._xy(state, p, x, ex, dens),
                w * 0.8,
                color=with_alpha(me.style["exact"], p.opacity * p.exact),
            )
        polyline(
            ctx,
            cam,
            self._xy(state, p, x, u, dens),
            w,
            color=with_alpha(me.style["color"], p.opacity),
            glow=me.style["glow"],
        )
        if p.kind != "advection":  # the boundary conditions, visible at both ends
            muted = with_alpha(me.style["exact"], p.opacity)
            neumann = p.kind == "heat" and p.boundary == "neumann"
            for xe in (x0, x0 + p.extent[0]):
                if neumann:  # an insulating wall: nothing crosses it
                    polyline(ctx, cam, [(xe, y0, z0), (xe, y0 + p.extent[1], z0)], w * 2.2,
                             color=muted)  # fmt: skip
                else:  # a pin holding u at 0
                    disc(ctx, cam, (xe, axis_y, z0), w * 2.2, color=muted)


def _key(p) -> tuple:
    names = (
        "diffusivity",
        "speed",
        "length",
        "n",
        "modes",
        "potential",
        "dt",
        "size",
        "mode",
        "feed",
        "kill",
        "seed",
        "boundary",
        "method",
        "cube",
    )
    return tuple(sorted((k, getattr(p, k)) for k in names if hasattr(p, k)))


_BOUNDARY_TEXT = {
    "dirichlet": "u = 0 at both ends (held)",
    "neumann": "no flux at both ends (insulated)",
}


def _runner_steps(sim, t: float) -> int:
    return runner_for(sim, every=CHECKPOINT_EVERY).step_of(t)


# ---------------------------------------------------------------- Field and Surface (2D)


class _Plate(ObjectBase):
    """What the 2D objects share: the solver, its state at ``time`` and a value scale."""

    def _sim_of(self, p):
        kind = {"heat": "heat2d", "wave": "wave2d", "reaction": "reaction"}[p.kind]
        return _sim(kind, _key(p))

    def _grid(self, state: SceneState):
        p = self.me(state).params
        sim = self._sim_of(p)
        s = _state(sim, p.time)
        u = s["v"] if p.kind == "reaction" else s["u"]
        return sim, np.asarray(u, dtype=np.float64)

    def _size(self, p) -> tuple[float, float]:
        return p.size

    def _stats(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        sim, u = self._grid(state)
        out = {
            "equation": p.kind,
            "time": round(p.time, 6),
            "max": round(float(u.max()), 6),
            "min": round(float(u.min()), 6),
            "grid": list(u.shape),
            "dt": sim.dt,
            "steps": _runner_steps(sim, p.time),
        }
        ex = sim.exact(p.time) if hasattr(sim, "exact") else None
        if ex is not None:
            out["max_error_vs_exact"] = round(float(np.abs(u - ex).max()), 6)
        if p.kind == "heat":
            out["total_heat"] = round(float(u.sum() * sim.dx**2), 6)
        return out


def _heat(v: np.ndarray, colormap, size, p, glow: float, state, *, signed=None):
    """``ctx.field`` pixels for values ``v`` (0..1, at grid nodes) on a ``size`` rectangle,
    plain or in the prism look (``HeatParams``, a ``glow`` role above 1). ``signed``:
    (positive, negative) colours for a wave, whose 0.5 is rest: with glow both directions
    glow, each in its own colour, or (``p.hue_cycles``) in the hue of their level."""
    colour = lut_colour(ramp(colormap, np.float32), "linear" if p.smooth else "nearest")
    weight = (lambda x: x) if glow > 1 else None
    if signed is not None and glow > 1:
        pos, neg = (np.asarray(c[:3], dtype=np.float64) for c in signed)
        weight = lambda x: np.abs(2 * x - 1)  # noqa: E731
        colour = lambda x: np.where((x >= 0.5)[..., None], pos, neg)  # noqa: E731
    if p.hue_cycles > 0:
        colour = wheel_colour(colormap, p)
    return heat_pixels(
        v, size, colour, how="smooth" if p.smooth else "linear", weight=weight,
        background=state.background, **p.drawing(),
    )  # fmt: skip


@register
class Field(_Plate):
    """A 2D solution as a color plane: a plate heating up, a drum vibrating, a reaction-
    diffusion pattern growing. Heat and reaction map ``u / scale`` onto the colormap; a wave
    maps its sign too (zero in the middle). A backdrop: labels may sit on it."""

    type: ClassVar[str] = "pde_field"
    Style: ClassVar = {"colormap": ("colormap", "ember"), "glow": ("glow", "none"),
                       "positive": ("emphasis", "highlight"),
                       "negative": ("emphasis", "secondary")}  # fmt: skip
    Life: ClassVar = frozenset({"hue", "glow"})
    Presets: ClassVar = {
        "heat": {"style": {"colormap": "ember"}},
        "drum": {"style": {"colormap": "ocean"}},
        "pattern": {"style": {"colormap": "sequential"}},
        "prism": PRISM,  # glowing out of the dark in spectrum hues (heatmap.PRISM)
    }

    class Params(HeatParams):
        kind: Literal["heat", "wave", "reaction"] = "heat"
        time: float = PField(0.0, ge=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        size: tuple[float, float] = (6.0, 6.0)
        n: int = PField(48, ge=8, le=400)
        diffusivity: float = PField(0.1, gt=0)
        speed: float = PField(1.0, gt=0)
        mode: tuple[int, int] = (1, 1)  # (m, n); heat with (0, 0): a hot spot
        feed: float = PField(0.037, gt=0)
        kill: float = PField(0.06, gt=0)
        seed: int = 0
        scale: float = PField(1.0, gt=0)

        @model_validator(mode="after")
        def _stable(self):
            kind = {"heat": "heat2d", "wave": "wave2d", "reaction": "reaction"}[self.kind]
            _sim(kind, _key(self))
            return self

    def _rect(self, p):
        c = np.asarray(p.pos, dtype=np.float64) - np.array([p.size[0] / 2, p.size[1] / 2, 0.0])
        return c, np.array([p.size[0], 0.0, 0.0]), np.array([0.0, p.size[1], 0.0])

    def world_bounds(self, state: SceneState) -> Bounds:
        c, r, u = self._rect(self.me(state).params)
        return Bounds.of_points([c, c + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": np.asarray(self.me(state).params.pos, dtype=np.float64)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        return {"kind": "pde_field", "backdrop": True, **self._stats(state)}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        _, u = self._grid(state)
        v = np.clip(u / p.scale, 0, 1) if p.kind != "wave" else 0.5 + 0.5 * np.tanh(u / p.scale)
        corner, right, up = self._rect(p)
        glow = float(me.style.get("glow", 1.0))
        signed = (me.style["positive"], me.style["negative"]) if p.kind == "wave" else None
        pixels = _heat(v, me.style["colormap"], p.size, p, glow, state, signed=signed)
        ctx.field(corner, right, up, pixels=pixels, opacity=p.opacity, glow=glow)


@register
class Surface(_Plate):
    """A 2D solution as a lit 3D surface: the height is the value (times ``height``). Give
    the scene an orbit camera. In the prism look (``hue_cycles`` > 0) the surface is dark
    where the field is quiet and takes the spectrum's hue of its value, and ``lines`` grid
    lines each way are drawn on it as glowing strokes in the hue of the value under them."""

    type: ClassVar[str] = "pde_surface"
    Style: ClassVar = {"colormap": ("colormap", "ember"), "material": ("surface", "solid"),
                       "glow": ("glow", "none"), "stroke": ("stroke", "hairline")}  # fmt: skip
    Life: ClassVar = frozenset({"hue", "glow"})
    Presets: ClassVar = {
        # the prism look: a dark landscape in spectrum hues under a net of glowing lines
        "prism": {"params": {"hue_cycles": 1.6, "contrast": 1.5, "lines": 24},
                  "style": {"colormap": "spectrum", "glow": "strong", "material": "glass"}},
    }  # fmt: skip

    class Params(ParamsBase):
        kind: Literal["heat", "wave", "reaction"] = "heat"
        time: float = PField(0.0, ge=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre of the base
        size: tuple[float, float] = (6.0, 6.0)
        n: int = PField(32, ge=8, le=200)
        diffusivity: float = PField(0.1, gt=0)
        speed: float = PField(1.0, gt=0)
        mode: tuple[int, int] = (1, 1)
        feed: float = PField(0.037, gt=0)
        kill: float = PField(0.06, gt=0)
        seed: int = 0
        scale: float = PField(1.0, gt=0)
        height: float = PField(2.0, gt=0)  # world height of a value of ``scale``
        # > 0 (prism): colour from the colormap as a wheel, this many turns over the values
        hue_cycles: float = PField(0.0, ge=0, le=8)
        hue: float = 0.0  # with hue_cycles: turns the colours; life adds a little
        contrast: float = PField(1.0, ge=0.25, le=8)  # with hue_cycles: brightness = strength^this
        lines: int = PField(0, ge=0, le=100)  # with hue_cycles: glowing grid lines each way

        @model_validator(mode="after")
        def _stable(self):
            kind = {"heat": "heat2d", "wave": "wave2d", "reaction": "reaction"}[self.kind]
            _sim(kind, _key(self))
            return self

    def _mesh(self, state: SceneState):
        p = self.me(state).params
        _, u = self._grid(state)
        nx, ny = u.shape
        xs = p.pos[0] + (np.linspace(0, 1, nx) - 0.5) * p.size[0]
        ys = p.pos[1] + (np.linspace(0, 1, ny) - 0.5) * p.size[1]
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = p.pos[2] + u / p.scale * p.height
        return np.stack([X, Y, Z], axis=-1), u

    def world_bounds(self, state: SceneState) -> Bounds:
        pts, _ = self._mesh(state)
        return Bounds.of_points(pts.reshape(-1, 3))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        pts, u = self._mesh(state)
        i = np.unravel_index(int(np.argmax(u)), u.shape)
        return {"peak": pts[i], "center": pts.reshape(-1, 3).mean(axis=0)}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        return {"kind": "pde_surface", **self._stats(state)}

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        pts, u = self._mesh(state)
        nx, ny = u.shape
        du, dv = np.gradient(pts, axis=0), np.gradient(pts, axis=1)
        normals = np.cross(du, dv)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
        lut = ramp(me.style["colormap"], np.float64)
        v = np.clip(u / p.scale, 0, 1) if p.kind != "wave" else 0.5 + 0.5 * np.tanh(u / p.scale)
        rgb = lut[np.clip((v * (len(lut) - 1)).astype(int), 0, len(lut) - 1)]
        if p.hue_cycles > 0:  # prism: dark where quiet, the spectrum's hue of the value
            hue = wheel_colour(me.style["colormap"], p)(v)
            strength = (np.abs(2 * v - 1) if p.kind == "wave" else v) ** p.contrast
            rgb = hue * strength[..., None]
            self._net(ctx, state, pts, normals, hue * (0.3 + 0.7 * strength[..., None]))
        colors = np.concatenate([rgb, np.ones((*rgb.shape[:-1], 1))], axis=-1)
        k = (np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
        tris = np.vstack([np.c_[k, k + ny, k + 1], np.c_[k + 1, k + ny, k + ny + 1]])
        uv = np.stack(np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij"), -1)
        ctx.surface(
            pts.reshape(-1, 3),
            tris,
            normals=normals.reshape(-1, 3),
            colors=colors.reshape(-1, 4),
            uv=uv.reshape(-1, 2),
            material=me.roles["material"],
            opacity=me.params.opacity * float(me.style["material"]),
        )

    def _net(self, ctx: DrawContext, state: SceneState, pts, normals, rgb) -> None:
        """The prism look's ``lines`` grid lines each way, as glowing strokes coloured
        ``rgb`` per grid point, lifted a hair off the surface so they never sink into it."""
        me = self.me(state)
        p = me.params
        if p.lines <= 0:
            return
        cam = state.camera_for(me)
        lifted = pts + normals * (0.004 * max(p.size))
        rgba = np.concatenate([rgb, np.full((*rgb.shape[:-1], 1), p.opacity)], axis=-1)
        glow = float(me.style["glow"])
        for axis in (0, 1):  # lines of constant x, then of constant y
            grid, cols = np.moveaxis(lifted, axis, 0), np.moveaxis(rgba, axis, 0)
            step = max(1, (len(grid) - 1) // p.lines)
            rows = grid[::step]
            w = me.style["stroke"] * cam.units_per_height(rows.reshape(-1, 3))
            polylines(ctx, cam, rows, w.reshape(rows.shape[:2]), color=(1, 1, 1, p.opacity),
                      glow=glow, colors=cols[::step])  # fmt: skip


# ---------------------------------------------------------------- Modes (separation of variables)


@register
class Modes(ObjectBase):
    """Separation of variables for a heating rod: the profile u(x, t) written as a sum of
    modes, each drawn faintly in its own color and decaying at its own rate exp(-k w^2 t)
    (high modes die first, so the profile smooths), with their partial sum drawn bright."""

    type: ClassVar[str] = "pde_modes"
    Style: ClassVar = {
        "color": ("emphasis", "highlight"),
        "colormap": ("colormap", "sequential"),
        "axis": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
    }

    class Params(ParamsBase):
        time: float = PField(0.0, ge=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # lower-left corner of the plot
        extent: tuple[float, float] = (6.0, 2.5)
        length: float = PField(1.0, gt=0)
        diffusivity: float = PField(1.0, gt=0)
        modes: tuple[float, ...] = ()  # sine amplitudes; empty: a bump, decomposed numerically
        count: int = PField(7, ge=1, le=40)  # modes drawn and summed
        boundary: Literal["dirichlet", "neumann"] = "dirichlet"
        components: float = PField(0.6, ge=0, le=1)  # opacity of the single modes
        ymax: float = PField(1.2, gt=0)

    def _setup(self, p):
        sim = pm.Heat1D(k=p.diffusivity, length=p.length, n=400, modes=p.modes,
                        boundary=p.boundary)  # fmt: skip
        u0 = sim.profile(0.0)
        cosine = p.boundary == "neumann"
        coef0 = (pm.cosine_modes if cosine else pm.sine_modes)(sim.x, u0, p.length, p.count)
        first = 0 if cosine else 1
        m = np.arange(first, first + len(coef0))
        decay = np.exp(-p.diffusivity * (m * np.pi / p.length) ** 2 * p.time)
        return sim, coef0 * decay, m, cosine

    def _xy(self, p, x, y):
        w, h = p.extent
        sx = p.pos[0] + x / p.length * w
        return np.stack([sx, p.pos[1] + h / 2 + y / p.ymax * h / 2, np.full_like(sx, p.pos[2])], 1)

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        x, y, z = p.pos
        return Bounds((x, y, z), (x + p.extent[0], y + p.extent[1], z))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        sim, c, _m, cos = self._setup(p)
        return {"sum": self._xy(p, sim.x, pm.mode_sum(sim.x, c, p.length, cosine=cos))}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        sim, c, m, cos = self._setup(p)
        truth = sim.exact(p.time)
        out: dict[str, Any] = {
            "kind": "pde_modes",
            "time": round(p.time, 6),
            "basis": "cos(m pi x / L)" if cos else "sin(m pi x / L)",
            "modes": [int(k) for k in m],
            "amplitudes": [round(float(a), 6) for a in c],
            "decay_rates": [round(p.diffusivity * (k * np.pi / p.length) ** 2, 6) for k in m],
        }
        if truth is not None:
            err = np.abs(pm.mode_sum(sim.x, c, p.length, cosine=cos) - truth).max()
            out["partial_sum_error"] = round(float(err), 6)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        sim, c, m, cos = self._setup(p)
        x = sim.x
        w = me.style["stroke"] * float(cam.units_per_height())
        x0, y0, z0 = p.pos
        mid = y0 + p.extent[1] / 2
        polyline(ctx, cam, [(x0, mid, z0), (x0 + p.extent[0], mid, z0)], w * 0.7,
                 color=with_alpha(me.style["axis"], 0.5 * p.opacity))  # fmt: skip
        lut = np.asarray(me.style["colormap"], dtype=np.float64)
        if p.components > 0:
            for i, (k, a) in enumerate(zip(m, c, strict=True)):
                if abs(a) < 1e-6:
                    continue
                shape = (
                    np.cos(k * np.pi * x / p.length) if cos else np.sin(k * np.pi * x / p.length)
                )
                rgb = lut[int((0.25 + 0.7 * i / max(len(m) - 1, 1)) * (len(lut) - 1))]
                polyline(ctx, cam, self._xy(p, x, a * shape), w * 0.6,
                         color=(*rgb, p.opacity * p.components))  # fmt: skip
        total = pm.mode_sum(x, c, p.length, cosine=cos)
        polyline(ctx, cam, self._xy(p, x, total), w, color=with_alpha(me.style["color"], p.opacity),
                 glow=me.style["glow"])  # fmt: skip


# ---------------------------------------------------------------- Slice (3D)


@register
class Slice(ObjectBase):
    """Inside a heating cube: the plane ``axis`` = ``at`` (0..1) of a 3D heat solution, drawn
    as a color plane where it sits in the cube, inside the cube's outline. Animate ``at`` to
    sweep through the volume; give the scene an orbit camera to see it in 3D."""

    type: ClassVar[str] = "pde_slice"
    Style: ClassVar = {
        "colormap": ("colormap", "ember"),
        "edge": ("emphasis", "muted"),
        "stroke": ("stroke", "hairline"),
        "glow": ("glow", "none"),
    }
    Presets: ClassVar = {
        "prism": PRISM,  # glowing out of the dark in spectrum hues (heatmap.PRISM)
    }
    Life: ClassVar = frozenset({"hue", "glow"})

    class Params(HeatParams):
        time: float = PField(0.0, ge=0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre of the cube
        side: float = PField(4.0, gt=0)  # world size of the cube
        cube: float = PField(1.0, gt=0)  # the cube's size in the equation
        n: int = PField(24, ge=4, le=64)
        diffusivity: float = PField(0.1, gt=0)
        mode: tuple[int, int, int] = (0, 0, 0)  # (0, 0, 0): a hot spot
        axis: Literal["x", "y", "z"] = "z"
        at: float = PField(0.5, ge=0, le=1)
        scale: float = PField(0.0, ge=0)  # value shown at full color; 0: the cube's hottest now

        @model_validator(mode="after")
        def _stable(self):
            _sim("heat3d", _key(self))
            return self

    def _u(self, p):
        sim = _sim("heat3d", _key(p))
        return sim, np.asarray(_state(sim, p.time)["u"], dtype=np.float64)

    def _plane(self, p):
        """(corner, right, up) of the slice in world space."""
        k = "xyz".index(p.axis)
        a, b = [i for i in range(3) if i != k]
        e = np.eye(3) * p.side
        corner = np.asarray(p.pos, dtype=np.float64) - p.side / 2 + e[k] * p.at
        return corner, e[a], e[b]

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c = np.asarray(p.pos, dtype=np.float64)
        return Bounds(tuple(c - p.side / 2), tuple(c + p.side / 2))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        corner, r, u = self._plane(self.me(state).params)
        return {
            "slice_center": corner + (r + u) / 2,
            "center": np.asarray(self.me(state).params.pos),
        }

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        sim, u = self._u(p)
        sl = pm.slice3d(u, p.axis, p.at)
        out: dict[str, Any] = {
            "kind": "pde_slice",
            "equation": "heat",
            "time": round(p.time, 6),
            "plane": f"{p.axis} = {p.at:g} of the cube",
            "slice_max": round(float(sl.max()), 6),
            "max": round(float(u.max()), 6),
            "total_heat": round(float(u.sum() * sim.dx**3), 6),
            "grid": list(u.shape),
            "steps": _runner_steps(sim, p.time),
        }
        ex = sim.exact(p.time)
        if ex is not None:
            out["max_error_vs_exact"] = round(float(np.abs(u - ex).max()), 6)
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        _, u = self._u(p)
        top = p.scale or max(float(u.max()), 1e-12)
        v = np.clip(pm.slice3d(u, p.axis, p.at) / top, 0, 1)
        corner, r, up = self._plane(p)
        glow = float(me.style.get("glow", 1.0))
        ctx.field(corner, r, up, pixels=_heat(v, me.style["colormap"], (p.side, p.side), p,
                                              glow, state),
                  opacity=p.opacity, glow=glow)  # fmt: skip
        c = np.asarray(p.pos, dtype=np.float64) - p.side / 2
        w = me.style["stroke"] * float(cam.units_per_height(np.asarray(p.pos)))
        e = np.eye(3) * p.side
        col = with_alpha(me.style["edge"], p.opacity * 0.7)
        for i in range(3):  # the twelve edges of the cube
            j, k = [a for a in range(3) if a != i]
            for s1 in (0, 1):
                for s2 in (0, 1):
                    a = c + s1 * e[j] + s2 * e[k]
                    polyline(ctx, cam, [a, a + e[i]], w, color=col)
