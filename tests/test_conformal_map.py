"""cx.ComplexMap along a conformal path (``path="family"``): every in-between picture is
the image of the plane under a holomorphic map, so the grid rolls instead of folding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, cx
from phasor.domains.cx import _parse
from phasor.domains.cx_math import family_at, map_at, refine
from tests.conftest import gl_available

ROLL = "(exp(a*z) - 1)/a"


def exact_roll(z: np.ndarray, a: float) -> np.ndarray:
    """(e^{az} - 1)/a without 0/0: z times the series of (e^w - 1)/w, w = a z."""
    w = a * z
    return z * sum(w**k / np.prod(np.arange(2, k + 2, dtype=float)) for k in range(30))


def _grid(extent=(4.0, 2 * np.pi), n=41):
    x, y = np.meshgrid(np.linspace(-extent[0] / 2, extent[0] / 2, n),
                       np.linspace(-extent[1] / 2, extent[1] / 2, n))  # fmt: skip
    return x + 1j * y


def _derivatives(g, z, h=1e-5):
    gx = (g(z + h) - g(z - h)) / (2 * h)
    gy = (g(z + 1j * h) - g(z - 1j * h)) / (2 * h)
    return gx, gy


@pytest.mark.parametrize("a", [0.0, 3e-6, 1e-3, 0.3, 0.7, 1.0])
def test_the_exp_family_is_the_roll_even_at_a_zero(a):
    f = _parse(ROLL).numpy()
    z = _grid()
    assert np.allclose(family_at(f, z, a), exact_roll(z, a), rtol=1e-8, atol=1e-8)
    if a == 0.0:
        assert np.allclose(family_at(f, z, a), z, atol=1e-8)  # the identity at the start


@pytest.mark.parametrize("a", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_every_step_of_the_family_is_holomorphic_with_no_fold(a):
    f = _parse(ROLL).numpy()
    z = _grid()

    def g(q):
        return map_at(f, q, a, "family")

    gx, gy = _derivatives(g, z)
    assert np.allclose(gy, 1j * gx, rtol=1e-5, atol=1e-6)  # Cauchy-Riemann
    # grid lines cross at right angles: the images of d/dx and d/dy are orthogonal
    cos = np.abs((gx * np.conj(gy)).real) / (np.abs(gx) * np.abs(gy))
    assert cos.max() < 1e-5
    assert np.abs(gx).min() >= np.exp(-a * 2.0) * (1 - 1e-5)  # f' = e^{az}: never zero


def test_the_straight_line_path_folds_where_the_family_does_not():
    """(1 - t) z + t (e^z - 1) is holomorphic too, but its derivative vanishes at
    z = +-i pi when t = 1/2: the grid folds over there. The family never does."""
    f_lin = _parse("exp(z) - 1").numpy()
    z = _grid()

    def lin(q):
        return map_at(f_lin, q, 0.5, "linear")

    gx, _ = _derivatives(lin, z)
    assert np.abs(gx).min() < 1e-6  # a critical point inside the picture
    fold = z.flat[np.abs(gx).argmin()]
    assert fold.real == pytest.approx(0.0, abs=1e-9) and abs(fold.imag) == pytest.approx(np.pi)


def test_refinement_follows_the_curve_and_keeps_its_ends():
    f = _parse(ROLL).numpy()
    line = 0.5 + 1j * np.linspace(-9.5, 9.5, 24)  # three turns round the circle, few samples

    def g(q):
        return map_at(f, q, 1.0, "family")

    z, w = refine(g, line, g(line), tol=0.002, limit=12.0, cap=4000)
    assert z[0] == line[0] and z[-1] == line[-1] and len(z) > 4 * len(line)
    assert np.all(np.diff(z.imag) > 0)  # still in order along the line
    mid = g((z[1:] + z[:-1]) / 2)
    assert np.abs(mid - (w[1:] + w[:-1]) / 2).max() <= 0.002  # chords hug the arc
    assert np.allclose(np.abs(w + 1), np.exp(0.5))  # on the circle it should be on


def test_refinement_leaves_breaks_alone_and_stops_at_the_cap():
    f = _parse("1/z").numpy()
    line = np.linspace(-1, 1, 21) + 0j  # through the pole

    def g(q):
        return map_at(f, q, 1.0, "linear")

    z, w = refine(g, line, g(line), tol=1e-4, limit=12.0, cap=60)
    assert len(z) <= 60 and not np.isfinite(w[np.argmin(np.abs(z))])


# ---------------------------------------------------------------- the object


def _image(amount: float, **kw):
    s = Scene("roll", passes=None)
    s.add(cx.ComplexMap(id="m", preset="roll", extent=(2.0, 2 * np.pi), lines=(5, 7),
                        amount=amount, **kw))  # fmt: skip
    st = s.engine.state_at(0.0)
    return st["m"].impl, st


def test_the_roll_preset_wraps_the_imaginary_axis_round_a_circle():
    impl, st = _image(1.0)
    p = st["m"].params
    assert p.path == "family" and p.function == ROLL
    axis = [pts for fam, pts in impl._runs(p) if fam == 0][2]  # Re z = 0, the middle line
    w = axis[:, 0] + 1j * axis[:, 1]
    assert np.allclose(np.abs(w + 1), 1.0, atol=1e-9)  # e^{iy} - 1: the unit circle at -1
    half, st = _image(0.5)
    axis = [pts for fam, pts in half._runs(st["m"].params) if fam == 0][2]
    w = axis[:, 0] + 1j * axis[:, 1]
    assert np.allclose(np.abs(w + 2), 2.0, atol=1e-9)  # halfway: a circle of radius 1/a
    flat, st = _image(0.0)
    axis = [pts for fam, pts in flat._runs(st["m"].params) if fam == 0][2]
    assert np.allclose(axis[:, 0], 0.0, atol=1e-8)  # at the start: the plain axis
    sem = flat.semantics(st)
    assert sem["path"] == "family" and sem["amount"] == 0.0


def test_the_left_half_plane_lands_inside_the_unit_disc_drawn_at_one():
    """The matched-z picture, z = e^{sT}: with ``center`` -1 drawn at the origin the roll
    ends on e^s itself, and starts with the jw axis tangent to the unit circle at 1."""
    impl, st = _image(1.0, center=(-1.0, 0.0))
    p = st["m"].params
    for _, pts in impl._runs(p):
        w = pts[:, 0] + 1j * pts[:, 1]
        assert np.abs(w).max() <= 1.0 + 1e-9  # every line of Re s <= 0 lands in the disc


def test_a_constant_a_belongs_to_the_family_path():
    with pytest.raises(Exception, match="DomainColoring only"):
        Scene("x", passes=None).add(cx.ComplexMap(id="m", function=ROLL))
    with pytest.raises(Exception, match="needs the constant a"):
        Scene("y", passes=None).add(cx.ComplexMap(id="m", function="z**2", path="family"))


def test_a_rolling_map_is_a_pure_function_of_time():
    s = Scene("morph", passes=None)
    s.add(cx.ComplexMap(id="m", preset="roll", amount=0.0, extent=(2.0, 6.0)))
    s.marker("go", 0.5)
    s.at("go").morph("m.amount", 1.0)
    e = s.engine
    a, b = e.inspect(1.0, id="m"), e.inspect(1.0, id="m")
    assert a == b and a["objects"][0]["semantic"]["amount"] not in (0.0, 1.0)


# ---------------------------------------------------------------- the example


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_s_plane_roll_example_matches_its_golden():
    """The roll and the even 600x dive (``log:`` easing on the camera's height)."""
    import os
    import runpy

    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "s_plane_roll.py"),
                       run_name="x")["s"].engine  # fmt: skip
    assert e.lint(strict=True) == []
    assert e.document.timeline.tracks["camera.height"][0].ease.startswith("log:")
    sheet = e.storyboard([0.5, 2.5, 4.5, 8.0, 11.0], size=(480, 270))["image"]
    path = GOLDEN / e.pass_runner().renderer_class() / "s_plane_roll.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
