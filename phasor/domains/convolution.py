"""Live convolution: two signals and the output they make.

    from phasor import conv

    s.add(conv.Convolution(id="c", f="box", g="decay", progress=0.0))
    s.at("go").sweep("c.progress", 1.0, show=False)

The top panel shows the signal ``f`` (one hue), the kernel ``g`` flipped and slid to the shift
tau (another), and under their curves the product f(s) g(tau - s), shaded where they overlap.
The bottom panel shares the time axis: the area of that product is the output (f * g)(tau),
drawn point by point as ``progress`` 0 -> 1 slides the kernel across, with a beacon at the
point being made. A signal is a name (``box``, ``triangle``, ``decay``, ``ramp``, ``gauss``,
``wave``) or a formula in ``t``, zero outside its support. ``pos`` is the picture's centre and
``extent`` its size in world units. Numerics are in ``convolution_math``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field, model_validator

from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains import convolution_math as cm
from phasor.domains.strokes import disc_batch, polyline, segment_batch, with_alpha

_CACHE: dict = {}
_LIMIT = 64
_PAD = 0.04  # share of the time axis left empty at each end
_TOP = (0.25, 0.20)  # the signals' panel: baseline and half-height, as shares of the height
_LOW = (-0.28, 0.17)  # the output's panel


def _memo(key, make):
    if key not in _CACHE:
        if len(_CACHE) >= _LIMIT:
            _CACHE.clear()
        _CACHE[key] = make()
    return _CACHE[key]


def _parse(spec: str, support) -> cm.Signal:
    return cm.Signal.parse(spec, None if tuple(support) == (0.0, 0.0) else tuple(support))


def _signals(p) -> tuple[cm.Signal, cm.Signal]:
    key = ("sig", p.f, p.g, p.f_support, p.g_support)
    return _memo(key, lambda: (_parse(p.f, p.f_support), _parse(p.g, p.g_support)))


def _curve(p):
    """The whole output and the scales that keep the picture still while the kernel slides."""

    def make():
        f, g = _signals(p)
        tau, out = cm.convolution(f, g)
        lo, hi = cm.shift_range(f, g)
        s = np.linspace(*f.support, 200)
        u = np.linspace(*g.support, 200)
        prod = np.abs(f(s)[None, :] * g(tau[:, None] - s[None, :])).max()
        top = max(float(np.abs(f(s)).max()), float(np.abs(g(u)).max()), float(prod), 1e-9)
        return {
            "tau": tau,
            "out": out,
            "top": top,
            "low": max(float(np.abs(out).max()), 1e-9),
            "x": (min(f.support[0], lo, f.support[0] + g.support[0] - g.support[1]),
                  max(f.support[1], hi, f.support[1] + g.support[1] - g.support[0])),
        }  # fmt: skip

    return _memo(("curve", p.f, p.g, p.f_support, p.g_support), make)


def _trace(sig: cm.Signal) -> np.ndarray:
    """Where to sample a signal to draw it: its support, with the vertical sides of a jump."""
    lo, hi = sig.support
    d = 0.003 * (hi - lo)
    return cm.edge_samples(sig.support, lo - d, hi + d, 240)


def _shift(p) -> float:
    f, g = _signals(p)
    lo, hi = cm.shift_range(f, g)
    return lo + p.progress * (hi - lo)


class _Frame:
    """The mapping from (time, value) to the object's world box."""

    def __init__(self, p, c):
        self.c = np.asarray(p.pos, dtype=np.float64)
        self.W, self.H = p.extent
        lo, hi = c["x"]
        pad = (hi - lo) * _PAD
        self.x0, self.span = lo - pad, (hi - lo) + 2 * pad
        self.top, self.low = c["top"], c["low"]

    def x(self, t):
        return (
            self.c[0]
            + (np.asarray(t, dtype=np.float64) - self.x0) / self.span * self.W
            - self.W / 2
        )

    def y_top(self, v):
        return self.c[1] + self.H * (_TOP[0] + np.asarray(v, dtype=np.float64) / self.top * _TOP[1])

    def y_low(self, v):
        return self.c[1] + self.H * (_LOW[0] + np.asarray(v, dtype=np.float64) / self.low * _LOW[1])

    def pts(self, t, y):
        t = np.asarray(t, dtype=np.float64)
        return np.c_[self.x(t), y, np.full(t.shape, self.c[2])]


def _lut_at(lut, u):
    from phasor.domains.geometry_math import sample_ramp

    return sample_ramp(np.asarray(lut, dtype=np.float64), u)


@register
class Convolution(ObjectBase):
    """A signal ``f`` and a kernel ``g`` flipped and sliding along it, their product shaded under
    the curves, and the area of that product drawn as the output, point by point, as
    ``progress`` runs 0 -> 1 over every shift at which they can overlap. Presets: ``plain``
    (the output only as it is made), ``ghost`` (also the whole output, faint, ahead of it)."""

    type: ClassVar[str] = "convolution"
    Style: ClassVar = {
        "color": ("emphasis", "secondary"),  # f
        "other": ("emphasis", "highlight"),  # g, flipped
        "product": ("emphasis", "danger"),  # where they multiply
        "axis": ("emphasis", "muted"),
        "stroke": ("stroke", "regular"),
        "glow": ("glow", "none"),
        "colormap": ("colormap", "spectrum"),  # the output, hue along the shift
    }
    Presets: ClassVar = {
        "plain": {"params": {"ghost": False}, "style": {}},
        "ghost": {"params": {"ghost": True}, "style": {}},
    }  # fmt: skip

    class Params(ParamsBase):
        f: str = "box"  # a name or a formula in t
        g: str = "decay"
        f_support: tuple[float, float] = (0.0, 0.0)  # where f is not zero; (0, 0): the shape's own
        g_support: tuple[float, float] = (0.0, 0.0)
        pos: Vec3 = (0.0, 0.0, 0.0)  # the picture's centre
        extent: tuple[float, float] = (11.0, 6.4)  # its width and height
        progress: float = Field(0.5, ge=0, le=1)  # the kernel's shift, over its whole range
        ghost: bool = True  # the whole output, faint, ahead of the traced part
        shade: float = Field(1.0, ge=0, le=1)  # how strongly the product is filled
        hue_cycles: float = Field(1.0, ge=0)  # colormap cycles along the output

        @model_validator(mode="after")
        def _signals_parse(self):
            try:
                _parse(self.f, self.f_support)
                _parse(self.g, self.g_support)
            except cm.FunctionError as e:
                raise ValueError(str(e)) from None
            return self

    def world_bounds(self, state: SceneState) -> Bounds:
        p = self.me(state).params
        c = np.asarray(p.pos, dtype=np.float64)
        half = np.array([p.extent[0] / 2, p.extent[1] / 2, 0.0])
        return Bounds(tuple(c - half), tuple(c + half))

    def _point(self, p) -> tuple[float, float, np.ndarray]:
        """The shift, the output there, and where the beacon sits in the world."""
        f, g = _signals(p)
        tau = _shift(p)
        area = float(cm.product_at(f, g, tau)["area"])
        fr = _Frame(p, _curve(p))
        return tau, area, np.array([fr.x(tau), float(fr.y_low(area)), p.pos[2]])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        p = self.me(state).params
        fr = _Frame(p, _curve(p))
        tau, _, beacon = self._point(p)
        g = _signals(p)[1]
        mid = float(fr.x(tau - (g.support[0] + g.support[1]) / 2))
        return {"center": np.asarray(p.pos, dtype=np.float64),
                "kernel": np.array([mid, fr.y_top(0.0), p.pos[2]]),
                "output": beacon}  # fmt: skip

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        f, g = _signals(p)
        tau, area, _ = self._point(p)
        lo, hi = cm.shift_range(f, g)
        at = cm.product_at(f, g, tau)
        return {
            "kind": "convolution",
            "f": p.f,
            "g": p.g,
            "shift": round(tau, 6),
            "shift_range": [round(lo, 6), round(hi, 6)],
            "output": round(area, 6),
            "peak": round(float(np.abs(_curve(p)["out"]).max()), 6),
            "overlapping": int(np.count_nonzero(at["product"])) >= 2,
            "progress": round(p.progress, 6),
        }

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        cam = state.camera_for(me)
        unit = float(cam.units_per_height(np.asarray(p.pos, dtype=np.float64)))
        w = me.style["stroke"] * unit
        glow = me.style["glow"]
        alpha = p.opacity
        f, g = _signals(p)
        c = _curve(p)
        fr = _Frame(p, c)
        tau = _shift(p)
        at = cm.product_at(f, g, tau)

        # baselines: the time axis under each panel, faint
        lo_t, hi_t = fr.x0, fr.x0 + fr.span
        for base in (fr.y_top(0.0), fr.y_low(0.0)):
            polyline(ctx, cam, fr.pts([lo_t, hi_t], [base, base]), w * 0.5,
                     color=with_alpha(me.style["axis"], alpha * 0.5), glow=1.0)  # fmt: skip

        # f, and the flipped kernel g(tau - s) slid to tau
        s = _trace(f)
        polyline(ctx, cam, fr.pts(s, fr.y_top(f(s))), w,
                 color=with_alpha(me.style["color"], alpha), glow=glow)  # fmt: skip
        u = _trace(g)  # g(u), drawn at tau - u
        polyline(ctx, cam, fr.pts(tau - u, fr.y_top(g(u))), w,
                 color=with_alpha(me.style["other"], alpha), glow=glow)  # fmt: skip

        # the product, filled down to the baseline, brightest where it is largest
        sg, pr = at["s"], at["product"]
        step = max(1, len(sg) // 140)
        sg, pr = sg[::step], pr[::step]
        live = pr != 0.0
        if p.shade > 0 and live.any():
            base = fr.y_top(0.0)
            a = fr.pts(sg[live], np.full(live.sum(), base))
            b = fr.pts(sg[live], fr.y_top(pr[live]))
            col = np.array(with_alpha(me.style["product"], alpha * 0.55 * p.shade))
            col = np.tile(col, (live.sum(), 1))
            col[:, 3] *= 0.35 + 0.65 * np.abs(pr[live]) / max(float(np.abs(pr).max()), 1e-12)
            pitch = (fr.x(sg[1]) - fr.x(sg[0])) / unit * 1.05 if len(sg) > 1 else w
            segment_batch(ctx, cam, a, b, pitch, colors=col, glow=1.0 + (glow - 1.0) * 0.4)
            full = fr.pts(sg, fr.y_top(pr))
            polyline(ctx, cam, full, w * 0.8, color=with_alpha(me.style["product"], alpha),
                     glow=glow)  # fmt: skip

        # the output, made so far, hue running along the shift; the whole of it ahead, faint
        tau_all, out = c["tau"], c["out"]
        if p.ghost:
            polyline(ctx, cam, fr.pts(tau_all, fr.y_low(out)), w * 0.6,
                     color=with_alpha(me.style["axis"], alpha * 0.45), glow=1.0)  # fmt: skip
        done = tau_all <= tau
        t_done = np.append(tau_all[done], tau)
        o_done = np.append(out[done], at["area"])
        if len(t_done) >= 2:
            lut = np.asarray(me.style["colormap"], dtype=np.float64)
            span = max(tau_all[-1] - tau_all[0], 1e-12)
            hue = ((t_done - tau_all[0]) / span * p.hue_cycles) % 1.0
            rgb = _lut_at(np.vstack([lut, lut[:1]]), hue)
            cols = np.c_[rgb, np.full(len(t_done), alpha)]
            polyline(
                ctx,
                cam,
                fr.pts(t_done, fr.y_low(o_done)),
                w * 1.2,
                color=with_alpha(me.style["color"], alpha),
                glow=glow,
                colors=cols,
            )

        # the guide from the kernel's position down to the point being made, and the beacon
        top_y, low_y = float(fr.y_top(0.0)), float(fr.y_low(at["area"]))
        x_at = float(fr.x(tau))
        polyline(ctx, cam, np.array([[x_at, top_y, p.pos[2]],
                                     [x_at, low_y, p.pos[2]]]), w * 0.5,
                 color=with_alpha(me.style["axis"], alpha * 0.55), glow=1.0)  # fmt: skip
        disc_batch(
            ctx,
            cam,
            np.array([[x_at, low_y, p.pos[2]]]),
            unit * 0.014,
            colors=with_alpha(me.style["other"], alpha),
            glow=glow * (1.2 if glow > 1.0 else 1.0),
        )
