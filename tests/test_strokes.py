"""``strokes.polyline`` builds the same triangles however fast it is.

The stroke builder was most of what is left of a frame's ``draw`` after step 3a. It was
rewritten for speed; ``_polyline_before`` is the version it replaced, kept here as the
oracle: every output array must be bit-identical, for every kind of input.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from phasor.domains import strokes


def _hue_colors_before(pts, lut, cycles, shift=0.0, total=None):
    pts = np.asarray(pts, dtype=np.float64)
    lut = np.asarray(lut, dtype=np.float64)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = max(float(along[-1] if total is None else total), 1e-12)
    g = ((along / total * cycles + shift) % 1.0) * len(lut)
    k0 = np.floor(g).astype(int) % len(lut)
    f = (g - np.floor(g))[:, None]
    rgb = lut[k0] * (1 - f) + lut[(k0 + 1) % len(lut)] * f
    return np.c_[rgb, np.ones(len(pts))]


def _polyline_before(ctx, cam, pts, width, *, color, glow=1.0, colors=None):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return
    theme_hue = getattr(ctx, "stroke_hue", None)
    if colors is None and theme_hue is not None and glow > 1.0:
        lut, cycles, shift = theme_hue
        colors = _hue_colors_before(pts, lut, cycles, shift)
    width = np.broadcast_to(np.asarray(width, dtype=np.float64), (len(pts),))
    flat = np.c_[pts @ cam.right, pts @ cam.up]
    seg = np.diff(flat, axis=0)
    norm = np.linalg.norm(seg, axis=1, keepdims=True)
    seg = seg / np.where(norm == 0, 1, norm)
    normals = np.c_[-seg[:, 1], seg[:, 0]]
    n_prev = np.vstack([normals[:1], normals])
    n_next = np.vstack([normals, normals[-1:]])
    miter = n_prev + n_next
    miter /= np.maximum(np.linalg.norm(miter, axis=1, keepdims=True), 1e-9)
    scale = 1 / np.maximum((miter * n_prev).sum(1), 0.25)
    offset = (miter * scale[:, None]) * (width[:, None] / 2)
    off3 = np.outer(offset[:, 0], cam.right) + np.outer(offset[:, 1], cam.up)
    positions = np.empty((2 * len(pts), 3))
    positions[0::2], positions[1::2] = pts + off3, pts - off3
    k = 2 * np.arange(len(pts) - 1)
    indices = np.vstack([np.c_[k, k + 1, k + 2], np.c_[k + 1, k + 3, k + 2]])
    per_vertex = None
    if colors is not None:
        per_vertex = np.repeat(np.asarray(colors, dtype=np.float64), 2, axis=0)
    ctx.triangles(positions, indices, color=color, glow=glow, colors=per_vertex)


class _Ctx:
    def __init__(self, stroke_hue=None):
        self.stroke_hue = stroke_hue
        self.calls = []

    def triangles(self, positions, indices=None, *, color, glow=1.0, colors=None):
        self.calls.append((positions, indices, color, glow, colors))


def _cameras():
    flat = SimpleNamespace(right=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))
    a, b = np.deg2rad(37.0), np.deg2rad(-21.0)
    right = np.array([np.cos(a), np.sin(a), 0.0])
    up = np.array([-np.sin(a) * np.sin(b), np.cos(a) * np.sin(b), np.cos(b)])
    return [flat, SimpleNamespace(right=right, up=up)]


def _cases():
    rng = np.random.default_rng(11)
    lut = rng.random((64, 3))
    walk = np.cumsum(rng.normal(size=(300, 3)), axis=0)
    yield "two points", [(0, 0, 0), (1, 0.5, 0)], 0.05, {}
    yield "random walk", walk, 0.1, {}
    yield "per-point width", walk, rng.random(300) * 0.2, {}
    yield "colors", walk, 0.1, {"colors": rng.random((300, 4))}
    yield "theme hue", walk, 0.1, {"glow": 2.0, "_hue": (lut, 1.5, 0.25)}
    yield "theme hue, plain glow", walk, 0.1, {"glow": 1.0, "_hue": (lut, 1.5, 0.25)}
    rep = np.repeat(walk[:20], 3, axis=0)  # zero-length segments
    yield "repeated points", rep, 0.1, {}
    yield "hairpin", [(0, 0, 0), (1, 0, 0), (0, 1e-6, 0), (1, 2e-6, 0)], 0.1, {}
    yield "one point", [(0, 0, 0)], 0.1, {}
    yield "int points", [(0, 0, 0), (2, 1, 0), (3, 3, 1)], 1, {}
    yield "list width", walk[:5], [0.1, 0.2, 0.3, 0.2, 0.1], {}


@pytest.mark.parametrize("cam", _cameras(), ids=["2d", "3d"])
@pytest.mark.parametrize(
    ("name", "pts", "width", "kw"), list(_cases()), ids=lambda v: v if isinstance(v, str) else ""
)
def test_polyline_is_bit_identical_to_the_version_it_replaced(cam, name, pts, width, kw):
    kw = dict(kw)
    hue = kw.pop("_hue", None)
    want, got = _Ctx(hue), _Ctx(hue)
    _polyline_before(want, cam, pts, width, color=(1, 0, 0, 1), **kw)
    strokes.polyline(got, cam, pts, width, color=(1, 0, 0, 1), **kw)
    assert len(got.calls) == len(want.calls)
    for g, w in zip(got.calls, want.calls, strict=True):
        for a, b in zip(g, w, strict=True):
            if isinstance(b, np.ndarray):
                assert isinstance(a, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                assert np.array_equal(a, b)
            else:
                assert a == b


def test_hue_colors_is_bit_identical_to_the_version_it_replaced():
    rng = np.random.default_rng(3)
    pts, lut = np.cumsum(rng.normal(size=(200, 3)), axis=0), rng.random((32, 3))
    for cycles, shift, total in ((1.0, 0.0, None), (2.5, 0.3, 40.0), (0.0, 0.9, None)):
        got = strokes.hue_colors(pts, lut, cycles, shift, total)
        want = _hue_colors_before(pts, lut, cycles, shift, total)
        assert got.dtype == want.dtype and np.array_equal(got, want)


def test_polyline_still_rejects_a_width_per_point_of_the_wrong_length():
    with pytest.raises(ValueError):
        strokes.polyline(_Ctx(), _cameras()[0], [(0, 0, 0), (1, 0, 0), (2, 0, 0)], [0.1, 0.2],
                         color=(1, 1, 1, 1))  # fmt: skip


@pytest.mark.parametrize("cam", _cameras(), ids=["2d", "3d"])
@pytest.mark.parametrize("with_colors", [False, True])
def test_many_strokes_at_once_are_the_strokes_one_by_one(cam, with_colors):
    """``polylines`` (S strokes of N points, one draw) gives exactly the vertices, colors and
    triangles of S ``polyline`` calls concatenated: what FlowParticles draws in one go."""
    rng = np.random.default_rng(5)
    for s, n in ((1, 2), (7, 8), (145, 8), (3, 50)):
        pts = np.cumsum(rng.normal(size=(s, n, 3)), axis=1)
        pts[0, 3:5] = pts[0, 2] if n > 5 else pts[0, 0]  # a zero-length segment
        width = rng.random((s, n)) * 0.1
        colors = rng.random((s, n, 4)) if with_colors else None
        one = _Ctx()
        for k in range(s):
            strokes.polyline(one, cam, pts[k], width[k], color=(1, 1, 1, 0.5), glow=1.5,
                             colors=None if colors is None else colors[k])  # fmt: skip
        many = _Ctx()
        strokes.polylines(many, cam, pts, width, color=(1, 1, 1, 0.5), glow=1.5, colors=colors)
        ((pos, idx, color, glow, cols),) = many.calls
        assert color == (1, 1, 1, 0.5) and glow == 1.5
        offsets = np.cumsum([0] + [len(c[0]) for c in one.calls[:-1]])
        assert np.array_equal(pos, np.concatenate([c[0] for c in one.calls]))
        shifted = [c[1] + o for c, o in zip(one.calls, offsets, strict=True)]
        assert np.array_equal(idx, np.concatenate(shifted))
        if colors is None:
            assert cols is None
        else:
            assert np.array_equal(cols, np.concatenate([c[4] for c in one.calls]))


def test_polylines_with_nothing_to_draw_draws_nothing():
    ctx = _Ctx()
    strokes.polylines(ctx, _cameras()[0], np.zeros((0, 8, 3)), 0.1, color=(1, 1, 1, 1))
    strokes.polylines(ctx, _cameras()[0], np.zeros((4, 1, 3)), 0.1, color=(1, 1, 1, 1))
    assert ctx.calls == []


def _flow_particles_one_by_one(self, ctx, state, pass_):
    """FlowParticles.draw before batching: one stroke and one draw per particle."""
    from phasor.domains import geometry_math as gm
    from phasor.domains.geometry import ramp

    me = self.me(state)
    p = me.params
    if p.opacity <= 0:
        return
    cam = state.camera_for(me)
    paths, phase, strength = self._paths(me.id, p)
    seg, alpha = gm.particles(paths, phase, state.t, p.lifetime, max(p.tail, 0.02))
    top = float(np.nanmax(strength)) if np.isfinite(strength).any() else 1.0
    rgb = gm.sample_ramp(ramp(me.style["colormap"]), gm.normalize01(strength, 0.0, top))
    for pts, a, c in zip(seg, alpha, rgb, strict=True):
        if a.max() <= 0.01 or np.linalg.norm(pts[-1] - pts[0]) < 1e-9:
            continue
        cols = np.empty((len(pts), 4))
        cols[:, :3], cols[:, 3] = c, a * p.opacity
        strokes.polyline(ctx, cam, pts, me.style["stroke"] * cam.units_per_height(pts),
                         color=(*c, p.opacity), glow=me.style["glow"], colors=cols)  # fmt: skip


@pytest.mark.gl
@pytest.mark.parametrize("perspective", [False, True], ids=["2d", "3d"])
def test_flow_particles_in_one_draw_look_exactly_as_one_draw_each(perspective, monkeypatch):
    from phasor import Scene, geo
    from phasor.domains.geometry import FlowParticles
    from phasor.render.passes import GLRenderer
    from tests.conftest import gl_available

    if not gl_available():
        pytest.skip("needs OpenGL")
    s = Scene("flow_batch")
    s.add(geo.FlowParticles(id="swirl", fp="-y", fq="x", fr="0.2", count=300, tail=0.3,
                            style={"glow": "strong"}))  # fmt: skip
    if perspective:  # widths then vary along each stroke with depth
        s.use_camera("orbit")
        s.set("camera.distance", 12.0)
        s.set("camera.elevation", 30.0)
    s.marker("end", 4.0)
    e, size = s.engine, (320, 180)

    def frames():
        r = GLRenderer()  # fresh, so nothing is cached between the two versions
        out = []
        for t in (0.7, 2.1, 3.3):
            st = e.state_at(t)
            out += [c.copy() for c in r.color_and_emission_raw(st, size)]
            out += [r.id_buffer(st, size), r.solo_mask(st, size, "swirl")]
        return out

    batched = frames()
    assert all((ids != 0).sum() > 500 for ids in batched[2::4])  # particles on screen
    monkeypatch.setattr(FlowParticles, "draw", _flow_particles_one_by_one)
    for got, want in zip(batched, frames(), strict=True):
        assert np.array_equal(got, want)
