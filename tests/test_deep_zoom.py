"""The perturbation deep zoom.

The ground truth is the slow way: iterate each pixel entirely in fixed point with 64 more bits
than the renderer uses (``fractal_math.direct_escape``). The kernel's NumPy reference, its CUDA
source on the CPU (``test_kernels.py``) and the ``cx.Fractal`` object are held to it.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from phasor.domains import fractal_math as fm
from phasor.render.gpu import Tile, get
from phasor.render.gpu.registry import plane

# A Misiurewicz point (the orbit of 0 lands on a repelling fixed point after 24 steps): a
# spiral at every scale, so a 1e-30 view is full of structure at a modest iteration count.
SPIRAL = "-0.7756837680090537974694835039347410457282 0.1364673682946901247332744096178487651978"
# The deep point of the well-known seahorse-valley zoom: near a minibrot, thousands of steps.
SEAHORSE = "-0.743643887037158704752191506114774 0.131825904205311970493132056385139"


def _deep(text, tile, center, height, max_iter, rebase=True):
    c = fm.parse_point(text)
    bits = fm.bits_for(height / tile.full_h)
    orbit = fm.reference_orbit(c, bits, max_iter)
    return c, bits, get("deep").reference(tile, center, height, orbit, max_iter, rebase)


def _truth(c, bits, tile, center, height, max_iter, idx):
    x, y = plane(tile, center, height)
    return np.array([
        fm.direct_escape((c[0] + Fraction(float(x.flat[i])), c[1] + Fraction(float(y.flat[i]))),
                         bits + 64, max_iter)
        for i in idx
    ])  # fmt: skip


# ---------------------------------------------------------------- arbitrary precision


def test_points_are_parsed_exactly():
    assert fm.parse_point("-0.75 0.1") == (Fraction(-3, 4), Fraction(1, 10))
    assert fm.parse_point(" 1e-30, -2.5E+1 ") == (Fraction(1, 10**30), Fraction(-25))
    c = fm.parse_point(SPIRAL)
    assert fm.parse_point(fm.format_point(c, 50)) == c  # 40 digits survive the round trip
    for bad in ("", "1", "1 2 3", "0.5 x", "nan 0", "1j 0"):
        with pytest.raises(ValueError, match="two decimal numbers"):
            fm.parse_point(bad)


def test_bits_follow_the_pixel_size():
    assert fm.bits_for(1e-3) == 64
    assert fm.bits_for(1e-32) >= 106 + fm.MARGIN and fm.bits_for(1e-32) % 32 == 0
    assert fm.bits_for(1e-32) > fm.bits_for(1e-20) > fm.bits_for(1e-10)


def test_reference_orbit_agrees_with_float64_where_float64_is_enough():
    c = (-0.12, 0.74)
    orbit = fm.reference_orbit((Fraction(c[0]), Fraction(c[1])), 128, 60)
    z, want = 0j, [0j]
    for _ in range(60):
        z = z * z + complex(*c)
        want.append(z)
    np.testing.assert_allclose(orbit[:, 0] + 1j * orbit[:, 1], want, atol=1e-12)
    assert fm.escapes_at(orbit, 60) is None and len(orbit) == 61


def test_reference_orbit_stops_where_it_escapes():
    orbit = fm.reference_orbit(fm.parse_point("0.5 0.5"), 64, 1000)
    k = fm.escapes_at(orbit, 1000)
    assert k == len(orbit) - 1 and 3 < k < 20
    assert (orbit[:-1] ** 2).sum(axis=1).max() <= fm.BAILOUT < (orbit[-1] ** 2).sum()


def test_direct_escape_matches_the_float64_kernel_at_ordinary_scales():
    for c in [(0.3, 0.5), (-0.75, 0.1), (-1.3, 0.05), (0.0, 0.0), (-1.0, 0.0)]:
        want = float(get("escape").reference(Tile.full(1, 1), c, 1e-3, 300, None, "f64")[0, 0])
        got = fm.direct_escape((Fraction(c[0]), Fraction(c[1])), 96, 300)
        assert got == pytest.approx(want, abs=1e-6), c


# ---------------------------------------------------------------- perturbation


def test_perturbation_equals_float64_at_shallow_zoom():
    tile = Tile(4, 2, 40, 30, 48, 36)  # a tile off the origin, not square
    _, _, got = _deep("-0.75 0.1", tile, (0.002, -0.001), 0.01, 400)
    want = get("escape").reference(tile, (-0.748, 0.099), 0.01, 400, None, "f64")
    assert ((got < 0) == (want < 0)).all()
    assert np.abs(got - want).max() < 1e-3


def test_a_1e_30_view_matches_exact_iteration():
    """The roadmap's exit demo: a 1e-30 view, every pixel checked against the slow truth."""
    tile = Tile.full(32, 18)
    for center in [(0.0, 0.0), (2e-30, 1e-30)]:  # reference at, then off, the view center
        c, bits, got = _deep(SPIRAL, tile, center, 1.8e-30, 3000)
        want = _truth(c, bits, tile, center, 1.8e-30, 3000, range(tile.n))
        assert np.abs(got.ravel() - want).max() < 1e-3, center
        assert got.min() > 1000 and got.max() - got.min() > 100  # deep, and not a flat color


def test_float64_alone_is_blind_at_1e_30():
    """Why perturbation exists: float64 sees one point where the view has 576 pixels."""
    flat = get("escape").reference(Tile.full(32, 18), (-0.7756837680090538, 0.1364673682946901),
                                   1.8e-30, 3000, None, "f64")  # fmt: skip
    assert len(np.unique(flat)) == 1
    _, _, deep = _deep(SPIRAL, Tile.full(32, 18), (0.0, 0.0), 1.8e-30, 3000)
    assert len(np.unique(deep)) > 500


def test_a_reference_that_escapes_early_still_gives_the_right_answer():
    """The reference leaves after 19 steps; pixels that outlast it rebase onto its start."""
    tile = Tile.full(24, 16)
    c, bits, got = _deep("-0.75 0.2", tile, (0.0, -0.1), 0.05, 500)
    assert fm.escapes_at(fm.reference_orbit(c, bits, 500), 500) == 19
    assert (got < 0).any() and (got > 100).any()  # pixels deep in the set and far past 19
    want = _truth(c, bits, tile, (0.0, -0.1), 0.05, 500, range(tile.n))
    assert ((got.ravel() < 0) == (want < 0)).all()
    assert np.abs(got.ravel() - want).max() < 1e-3


@pytest.mark.slow
def test_rebasing_repairs_glitches_near_a_minibrot():
    """Near a minibrot at 1e-14, thousands of steps in: without rebasing (``rebase=False``)
    pixels whose orbit passes nearer 0 than the reference does lose their precision. With it,
    every pixel agrees with the exact orbit except a few on the set's edge, where rounding the
    reference to float64 decides (an exact reference fixes those too)."""
    tile = Tile.full(48, 27)
    h, n = 1e-14, 8000
    c, bits, fixed = _deep(SEAHORSE, tile, (0.0, 0.0), h, n, rebase=True)
    *_, plain = _deep(SEAHORSE, tile, (0.0, 0.0), h, n, rebase=False)
    want = _truth(c, bits, tile, (0.0, 0.0), h, n, range(tile.n))
    wrong_fixed = (np.abs(fixed.ravel() - want) > 1e-3).mean()
    wrong_plain = (np.abs(plain.ravel() - want) > 1e-3).mean()
    assert wrong_fixed < 0.015
    assert wrong_plain > 2 * wrong_fixed


# ---------------------------------------------------------------- the object


def test_fractal_center_and_zoom_are_validated_and_reported():
    from phasor import Scene, cx

    with pytest.raises(Exception, match="two decimal numbers"):
        Scene("x", passes=None).add(cx.Fractal(id="f", center="minus one"))
    with pytest.raises(Exception, match="less than or equal to 280"):
        Scene("x", passes=None).add(cx.Fractal(id="f", zoom=400))
    s = Scene("f")
    s.add(cx.Fractal(id="f", center=SPIRAL, size=(8, 4.5), zoom=30, max_iter=3000))
    sem = s.engine.inspect(0, id="f")["objects"][0]["semantic"]
    assert fm.parse_point(sem["center"]) == fm.parse_point(SPIRAL)
    assert sem["zoom"] == 30 and sem["view_height"] == pytest.approx(4.5e-30)
    assert sem["precision"] == "perturbation"


def test_the_precision_steps_up_along_a_dive():
    from phasor import Scene, cx, expr

    s = Scene("f")
    s.add(cx.Fractal(id="f", center=SPIRAL, size=(8, 4.5), zoom=expr("t"), max_iter=3000))
    seen = [s.engine.inspect(float(t), id="f")["objects"][0]["semantic"]["precision"]
            for t in (0, 5, 6, 7, 8, 30)]  # fmt: skip
    # 4.5e-6 is under the float64 line; 4.5e-8 / 1080 rows is under 1e-10 per pixel
    assert seen == ["f32", "f32", "f64", "f64", "perturbation", "perturbation"]


def test_float64_and_perturbation_agree_where_they_hand_over():
    """No seam in a dive: just before the hand-over float64 and perturbation both agree with
    exact iteration (perturbation better: that is why it takes over here, not at 1e-13)."""
    from phasor.domains.cx import _orbit

    c = fm.parse_point(SPIRAL)
    tile, h = Tile.full(48, 27), 27 * 2 * fm.DEEP_BELOW  # pixels just above the line
    bits = fm.bits_for(h / 27)
    f64 = get("escape").reference(tile, (float(c[0]), float(c[1])), h, 3000, None, "f64")
    deep = get("deep").reference(tile, (0.0, 0.0), h, _orbit(c, bits, 3000), 3000)
    want = _truth(c, bits, tile, (0.0, 0.0), h, 3000, range(tile.n))
    assert (np.abs(f64.ravel() - want) < 0.1).mean() > 0.98
    assert (np.abs(deep.ravel() - want) < 0.1).mean() == 1.0


@pytest.mark.gl
def test_a_1e_30_frame_through_the_engine_is_the_kernels_picture():
    from phasor import Scene, cx
    from phasor.domains.cx import _orbit
    from phasor.render import gpu
    from phasor.style.theme import load_theme

    s = Scene("dive")
    s.set("camera.center", (0.0, 0.0, 0.0))
    s.set("camera.height", 4.5)
    s.add(cx.Fractal(id="f", center=SPIRAL, pos=(0, 0, 0), size=(8, 4.5), zoom=30, max_iter=3000))
    color = s.engine.pass_runner().color(s.engine.state_at(0), (96, 54))
    c = fm.parse_point(SPIRAL)
    orbit = _orbit(c, fm.bits_for(4.5e-30 / 54), 3000)
    field = gpu.render("deep", 96, 54, center=(0.0, 0.0), height=4.5e-30, orbit=orbit,
                       max_iter=3000)  # fmt: skip
    theme = load_theme("midnight_glow")
    lut = np.asarray(theme.values["colormap"]["sequential"], dtype=np.float32)
    want = gpu.run("palette", Tile.full(96, 54), field=field, lut=lut, density=0.35,
                   inside=theme.background[:3])  # fmt: skip
    from phasor.render.gpu.bridge import to_host

    want = to_host(want)  # a CuPy array on an NVIDIA machine
    close = np.abs(color[..., :3] - want[..., :3]).max(axis=-1) < 0.03
    assert close.mean() > 0.9  # (MSAA and float16 move the finest filaments)
    assert len(np.unique(np.round(color[..., :3], 2).reshape(-1, 3), axis=0)) > 50


# ---------------------------------------------------------------- the exit demo


def _demo():
    import runpy
    from pathlib import Path

    path = Path(__file__).parents[1] / "examples" / "deep_zoom.py"
    return runpy.run_path(str(path), run_name="not_main")["s"].engine


def test_exit_demo_dives_to_1e_30_through_every_precision():
    e = _demo()
    seen, heights = [], []
    for t in np.linspace(0, e.duration, 49):
        sem = e.inspect(float(t), id="set")["objects"][0]["semantic"]
        if not seen or seen[-1] != sem["precision"]:
            seen.append(sem["precision"])
        heights.append(sem["view_height"])
    assert seen == ["f32", "f64", "perturbation"]  # each switch happens once, in order
    assert all(b <= a for a, b in itertools.pairwise(heights))  # only ever deeper
    assert heights[0] == pytest.approx(4.5) and heights[-1] == pytest.approx(4.5e-30)


@pytest.mark.gl
@pytest.mark.slow
def test_exit_demo_lint_is_clean_and_matches_its_golden():
    import os

    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = _demo()
    assert e.lint() == []
    path = GOLDEN / e.pass_runner().renderer_class() / "deep_zoom.png"
    sheet = e.storyboard([0.0, 4.0, 12.0], size=(320, 180))["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.98
