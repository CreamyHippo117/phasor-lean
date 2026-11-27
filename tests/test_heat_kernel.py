"""Heat maps are drawn by the ``heat_shade`` kernel, and the pictures
are the ones the NumPy code drew. The old colouring of each type is written out here as the
plain functions it used to pass to ``heat_pixels``; a ``Recipe`` must give the same bits
through the reference, and the compiled kernel must agree to far below one colour step."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor import Scene, ac
from phasor.core.color import ramp
from phasor.domains.heatmap import (
    PRISM,
    HeatParams,
    Recipe,
    field_pixels,
    heat_pixels,
    lut_colour,
    shade_values,
    wheel_colour,
    wheel_recipe,
)
from phasor.render import gpu
from phasor.render.gpu import Tile
from phasor.render.gpu.registry import KernelError, get
from tests import cuda_emu
from tests.test_kernels import CASES

needs_compiler = pytest.mark.skipif(cuda_emu.compiler() is None, reason="no C++ compiler")
GEN = np.random.default_rng(5)
X, Y = np.linspace(0, 1, 31), np.linspace(0, 1, 19)
GRID = np.clip(0.5 + 0.45 * np.sin(6 * X)[:, None] * np.cos(5 * Y)[None, :]
               + 0.05 * GEN.standard_normal((31, 19)), 0, 1)  # fmt: skip
CMAP = GEN.random((256, 4))
BG = np.array([0.02, 0.03, 0.06, 1.0])
POS, NEG = (1.0, 0.55, 0.2, 1.0), (0.2, 0.5, 1.0, 1.0)
SIZE = (4.0, 3.0)
VIEWS = [(96, 72, (2.0, 1.5), 3.0), (80, 50, (1.1, 2.2), 1.3), (64, 48, (3.9, 0.1), 2.0)]
P_PLAIN = HeatParams()
P_PRISM = HeatParams(**PRISM["params"], hue=0.37)


@pytest.fixture(params=["reference", pytest.param("cpu", marks=needs_compiler)])
def backend(request, monkeypatch):
    """Draw heat maps on one backend: the reference must give the old bits exactly."""
    real = gpu.run

    def run(name, tile, **kw):
        return real(name, tile, **{**kw, "backend": request.param})

    monkeypatch.setattr(gpu, "run", run)
    return request.param


def _same(new: np.ndarray, old: np.ndarray, backend: str) -> None:
    assert new.shape == old.shape and new.dtype == old.dtype == np.float32
    if backend == "reference":
        np.testing.assert_array_equal(new, old)
    else:  # the compiled kernel: only the maths library's last bit may differ
        assert np.abs(new.astype(np.float64) - old).max() <= 1e-6


def _both(old, new, backend):
    for view in VIEWS:
        _same(new(*view), old(*view), backend)


# ---------------------------------------------------------------- each type's old colouring


@pytest.mark.parametrize("p", [P_PLAIN, P_PRISM, HeatParams(smooth=True)], ids=["plain", "prism", "smooth"])  # fmt: skip
@pytest.mark.parametrize("glow", [1.0, 2.0])
@pytest.mark.parametrize("signed", [False, True])
def test_a_pde_field_is_the_picture_it_was(p, glow, signed, backend):
    from phasor.domains.pde import _heat

    class State:
        background = BG

    colour = lut_colour(ramp(CMAP[:, :3], np.float32), "linear" if p.smooth else "nearest")
    weight = (lambda x: x) if glow > 1 else None
    if signed and glow > 1:
        pos, neg = (np.asarray(c[:3], dtype=np.float64) for c in (POS, NEG))
        weight = lambda x: np.abs(2 * x - 1)  # noqa: E731
        colour = lambda x: np.where((x >= 0.5)[..., None], pos, neg)  # noqa: E731
    if p.hue_cycles > 0:
        colour = wheel_colour(CMAP[:, :3], p)
    old = heat_pixels(GRID, SIZE, colour, how="smooth" if p.smooth else "linear", weight=weight,
                      background=BG, **p.drawing())  # fmt: skip
    new = _heat(GRID, CMAP[:, :3], SIZE, p, glow, State, signed=(POS, NEG) if signed else None)
    _both(old, new, backend)


@pytest.mark.parametrize("p", [P_PLAIN, P_PRISM], ids=["plain", "prism"])
@pytest.mark.parametrize("glow", [1.0, 2.0])
def test_a_pressure_slice_is_the_picture_it_was(p, glow, backend):
    slab, gain = (GRID - 0.5).astype(np.float32), 1.7
    pos, neg, bg = np.asarray(POS[:3]), np.asarray(NEG[:3]), np.asarray(BG[:3])

    def colour(val):
        level = np.tanh(gain * 8.0 * val)
        tint = np.where(level[..., None] >= 0, pos, neg)
        return bg + (tint - bg) * np.abs(level)[..., None]

    level = lambda v: 0.5 + 0.5 * np.tanh(gain * 8.0 * v)  # noqa: E731
    if p.hue_cycles > 0:
        wheel = wheel_colour(CMAP, p)
        colour = lambda v: wheel(level(v))  # noqa: E731
    how = "smooth" if p.smooth else "linear"
    old = heat_pixels(slab, SIZE, colour, cells=True, how=how, level=level,
                      weight=(lambda v: np.abs(np.tanh(gain * 8.0 * v))) if glow > 1 else None,
                      background=BG, **p.drawing())  # fmt: skip
    way = {"transform": "tanh", "gain": gain * 8.0, "level": "signed",
           "weight": "abs" if glow > 1 else None}  # fmt: skip
    recipe = (wheel_recipe(CMAP, p, **way) if p.hue_cycles > 0 else
              Recipe(colour="sign", positive=POS[:3], negative=NEG[:3], mix=True, **way))  # fmt: skip
    new = heat_pixels(slab, SIZE, recipe, cells=True, how=how, background=BG, **p.drawing())
    _both(old, new, backend)


@pytest.mark.parametrize("p", [P_PLAIN, P_PRISM], ids=["plain", "prism"])
@pytest.mark.parametrize("glow", [1.0, 2.0])
def test_a_spectrogram_is_the_picture_it_was(p, glow, backend):
    mag = GRID * 1.1 - 0.05  # a little outside 0..1: the square root clips it
    lut = ramp(CMAP[:, :3], np.float32)
    paint = lut_colour(lut, "linear" if p.smooth else "nearest")
    wheel = wheel_colour(CMAP, p)

    def up_hue(v, u, h):
        return np.broadcast_to(wheel(h)[:, None], (*v.shape, 3))

    how = "smooth" if p.smooth else "nearest"
    root = lambda v: np.sqrt(np.clip(v, 0, 1))  # noqa: E731
    old = heat_pixels(mag, SIZE, lambda v: paint(root(v)), cells=True, how=how, level=root,
                      weight=root if glow > 1 else None, background=BG, **p.drawing(),
                      colour_at=up_hue if p.hue_cycles > 0 else None)  # fmt: skip
    way = {"transform": "sqrt", "weight": "value" if glow > 1 else None}
    recipe = (wheel_recipe(CMAP, p, hue_of="height", **way) if p.hue_cycles > 0 else
              Recipe(lut=lut, blend=p.smooth, **way))  # fmt: skip
    new = heat_pixels(mag, SIZE, recipe, cells=True, how=how, background=BG, **p.drawing())
    _both(old, new, backend)


def _old_wave_pixels(impl, state):
    """``waves._WaveField.pixels_for`` before step 9c."""
    from phasor.domains.waves import CORE, FRONT

    me = impl.me(state)
    p = me.params
    corner, _, _ = impl._rect(p)
    table = impl._table(p)
    lam, k, wt = impl._wave(p)
    glow = float(me.style.get("glow", 1.0))
    g = p.gain * math.sqrt(lam) / max(float(np.abs(table[:, 2]).sum()), 1e-9)
    pos, neg = np.asarray(me.style["positive"][:3]), np.asarray(me.style["negative"][:3])
    bg = np.asarray(state.background[:3], dtype=np.float64)
    wheel = wheel_colour(me.style["colormap"], p) if p.hue_cycles > 0 else None
    share = p.blend if p.mode == "field" else 1.0 - p.blend

    def level(s):
        return (1.0 - share) * (0.5 + 0.5 * s) + share * s

    def pixels(pw, ph, center, height):
        raw = gpu.render("wave_field", pw, ph, sources=table, wavenumber=k, omega_t=wt,
                         center=(corner[0] + center[0], corner[1] + center[1]),
                         height=height, core=CORE * lam, edge=FRONT * lam)  # fmt: skip
        field = raw[..., 0].astype(np.float64)
        envelope = np.sqrt(2.0 * np.maximum(raw[..., 1].astype(np.float64), 0.0))
        s = np.tanh(g * field)
        if share > 0:
            s = (1.0 - share) * s + share * np.tanh(g * envelope) ** 2
        if wheel is not None:
            rgb = wheel(1.0 - level(s))
        else:
            tint = np.where(s[..., None] >= 0, pos, neg)
            rgb = bg + (tint - bg) * np.abs(s)[..., None]
        half_w = height * pw / ph / 2
        xs = center[0] + (np.arange(pw) + 0.5) / pw * 2 * half_w - half_w
        ys = center[1] + (0.5 - (np.arange(ph) + 0.5) / ph) * height
        return shade_values(s, rgb, xs / p.size[0], ys / p.size[1], p.size, level=level,
                            isolines=p.isolines, weight=np.abs if glow > 1 else None,
                            background=state.background, ambient=p.ambient, fade=p.fade,
                            contrast=p.contrast)  # fmt: skip

    return pixels


@pytest.mark.parametrize("params", [
    {},
    {"preset": "prism"},
    {"preset": "prism", "mode": "intensity"},
    {"preset": "prism", "blend": 0.4, "time": 0.8},
    {"mode": "intensity", "blend": 0.25},
], ids=["plain", "prism", "intensity", "blended", "plain blended"])  # fmt: skip
def test_a_wave_picture_is_the_picture_it_was(params, backend):
    s = Scene("w", passes=None)
    s.add(ac.Interference(id="w", size=(8, 4.5), **params))
    state = s.engine.state_at(0.3)
    impl = state["w"].impl
    old, new = _old_wave_pixels(impl, state), impl.pixels_for(state)
    for view in [(96, 54, (4.0, 2.25), 4.5), (80, 50, (2.6, 1.4), 1.9)]:
        _same(new(*view), old(*view), backend)


# ---------------------------------------------------------------- the kernel itself


@needs_compiler
@pytest.mark.parametrize(("tile", "params"), CASES["heat_shade"], ids=lambda v: str(v)[:24])
def test_the_compiled_kernel_is_the_reference_to_the_last_bits(tile, params):
    got = gpu.run("heat_shade", tile, backend="cpu", **params)
    want = get("heat_shade").reference(tile, **params)
    assert np.abs(got.astype(np.float64) - want).max() <= 1e-6


@pytest.mark.gpu
@pytest.mark.parametrize(("tile", "params"), CASES["heat_shade"], ids=lambda v: str(v)[:24])
def test_the_card_draws_the_reference_within_a_colour_step(tile, params):
    from phasor.render.gpu.bridge import to_host

    got = to_host(gpu.run("heat_shade", tile, backend="cuda", **params)).astype(np.float64)
    err = np.abs(got - get("heat_shade").reference(tile, **params))
    # the card's tanh, exp and pow are not libm's: the last bits, except where a pixel sits
    # exactly on a switch (a sign, a table entry, the edge of a contour line)
    assert (err > 1e-5).mean() <= 1e-3 and np.median(err) <= 1e-7, float(err.max())


@pytest.mark.parametrize("which", [2, 3, 6], ids=["grid", "cells", "image"])
def test_tiles_stitch_into_exactly_the_full_picture(which, backend):
    _, params = CASES["heat_shade"][which]
    w, h = 64, 48
    whole = np.asarray(gpu.run("heat_shade", Tile.full(w, h), **params))
    pieces = np.empty_like(whole)
    tiles = gpu.plan_tiles(w, h, 16, 16 * 20 * 11)  # bands and columns
    assert len(tiles) > 6
    for t in tiles:
        pieces[t.y0 : t.y0 + t.h, t.x0 : t.x0 + t.w] = gpu.run("heat_shade", t, **params)
    np.testing.assert_array_equal(pieces, whole)


def test_a_field_made_per_pixel_is_asked_for_each_tile_and_its_neighbours(monkeypatch):
    image = CASES["heat_shade"][6][1]["image"]  # (48, 64, 4)
    recipe = CASES["heat_shade"][6][1]["recipe"]
    asked = []

    def image_of(tile, center, height):
        asked.append(tile)
        return image[tile.y0 : tile.y0 + tile.h, tile.x0 : tile.x0 + tile.w]

    px = field_pixels(image_of, SIZE, recipe, isolines=8, fade=0.1, background=BG)
    whole = px(64, 48, (2.0, 1.5), 3.0)
    assert asked == [Tile.full(64, 48)]
    small = gpu.plan_tiles(64, 48, 16, 16 * 64 * 10)
    monkeypatch.setattr(gpu, "plan_tiles", lambda w, h, bpp: small)
    asked.clear()
    np.testing.assert_array_equal(px(64, 48, (2.0, 1.5), 3.0), whole)
    assert len(asked) == len(small) > 2
    assert asked[1].y0 == small[1].y0 - 1 and asked[1].h == small[1].h + 2  # a row each side


def test_a_recipe_says_what_it_cannot_draw():
    with pytest.raises(KernelError, match="transform must be one of"):
        Recipe(lut=CMAP, transform="log")
    with pytest.raises(KernelError, match="needs a"):
        Recipe(colour="wheel")
    with pytest.raises(KernelError, match="needs a"):
        Recipe(lut=CMAP[:1], blend=True)
    ref, view = get("heat_shade").reference, {"center": (2.0, 1.5), "height": 3.0, "size": SIZE}
    with pytest.raises(KernelError, match="a grid or an image"):
        ref(Tile.full(8, 8), recipe=Recipe(lut=CMAP), **view)
    with pytest.raises(KernelError, match="reads a wave_field image"):
        ref(Tile.full(8, 8), recipe=Recipe(lut=CMAP, transform="wave"), grid=GRID, **view)
    with pytest.raises(KernelError, match="must cover the tile"):
        ref(Tile(4, 4, 4, 4, 16, 16), recipe=Recipe(lut=CMAP), image=np.zeros((4, 4)),
            origin=(4, 4), **view)  # fmt: skip
    with pytest.raises(ValueError, match="Recipe says"):
        heat_pixels(GRID, SIZE, Recipe(lut=CMAP), weight=np.abs)


def test_a_picture_one_pixel_wide_has_no_lines(backend):
    params = {**CASES["heat_shade"][2][1]}
    out = np.asarray(gpu.run("heat_shade", Tile.full(1, 5), **params))
    assert out.shape == (5, 1, 4) and np.isfinite(out).all()
