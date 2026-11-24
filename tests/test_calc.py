"""Calculus numerics against known results."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor.domains import calc_math as cm


def F(src):
    return cm.RealFunction.parse(src)


def test_functions_are_read_safely_and_evaluate_on_the_real_line():
    f = F("x**2 - 1")
    assert f(np.array([0.0, 2.0, -3.0])).tolist() == [-1.0, 3.0, 8.0]
    assert math.isnan(float(F("log(x)")(np.array(-1.0))))  # not real there: NaN, not a crash
    assert float(F("sin(pi * x)")(np.array(0.5))) == pytest.approx(1.0)
    assert F("x**3").derivative().expr == 3 * F("x**2").expr  # derivatives are exact
    with pytest.raises(cm.FunctionError, match="cannot use"):
        F("__import__('os')")
    with pytest.raises(cm.FunctionError, match="unknown name"):
        F("y + 1")  # only x here
    with pytest.raises(cm.FunctionError):
        F("x.real")


def test_riemann_sums_converge_with_the_expected_order_and_sandwich_the_integral():
    f = F("x**2")
    exact = cm.exact_integral(f, 0.0, 3.0)
    assert exact == pytest.approx(9.0)
    err = {
        r: [abs(cm.riemann_sum(f, 0, 3, n, r)["value"] - exact) for n in (10, 20)] for r in cm.RULES
    }
    assert err["left"][1] == pytest.approx(err["left"][0] / 2, rel=0.1)  # first order
    assert err["mid"][1] == pytest.approx(err["mid"][0] / 4, rel=0.1)  # second order
    assert err["trapezoid"][1] == pytest.approx(err["trapezoid"][0] / 4, rel=0.1)
    lo, hi = (cm.riemann_sum(f, 0, 3, 12, r)["value"] for r in ("lower", "upper"))
    assert lo < exact < hi  # lower and upper sums squeeze the integral
    left, right = (cm.riemann_sum(f, 0, 3, 12, r)["value"] for r in ("left", "right"))
    assert left == pytest.approx(lo) and right == pytest.approx(hi)  # x^2 increases on [0, 3]
    rects = cm.riemann_sum(f, 0, 3, 6, "mid")["rects"]
    assert rects.shape == (6, 5) and rects[0, 4] == pytest.approx(0.25**2) and rects[-1, 1] == 3.0
    with pytest.raises(ValueError, match="unknown rule"):
        cm.riemann_sum(f, 0, 1, 4, "wobbly")
    with pytest.raises(ValueError, match="n >= 1"):
        cm.riemann_sum(f, 1, 0, 4)


def test_taylor_polynomials_approximate_better_near_the_center_and_with_more_terms():
    f = F("sin(x)")
    c = cm.taylor_coefficients(f, 0.0, 7)
    assert c == pytest.approx([0, 1, 0, -1 / 6, 0, 1 / 120, 0, -1 / 5040])
    e = cm.taylor_coefficients(F("exp(x)"), 0.0, 5)
    assert e == pytest.approx([1 / math.factorial(k) for k in range(6)])
    x = np.array([0.5, 1.5])
    errs = [
        np.abs(cm.taylor_polynomial(cm.taylor_coefficients(f, 0.0, k), 0.0)(x) - f(x))
        for k in (1, 3, 7)
    ]
    assert (errs[0] > errs[1]).all() and (errs[1] > errs[2]).all()
    assert errs[2][0] < errs[2][1]  # closer to the center, better
    off = cm.taylor_coefficients(F("log(x)"), 1.0, 4)  # centered elsewhere
    assert off == pytest.approx([0, 1, -0.5, 1 / 3, -0.25])
    # the remainder obeys Lagrange's bound: |sin x - T_7(x)| <= |x|^8 / 8!
    t7 = cm.taylor_polynomial(cm.taylor_coefficients(f, 0.0, 7), 0.0)
    xs = np.linspace(-2, 2, 41)
    assert (np.abs(t7(xs) - f(xs)) <= np.abs(xs) ** 8 / math.factorial(8) + 1e-12).all()
    with pytest.raises(ValueError, match="no finite real derivative"):
        cm.taylor_coefficients(F("log(x)"), 0.0, 2)


def test_epsilon_delta_finds_the_widest_delta_that_works():
    f = F("3 * x + 1")  # |f(x) - 7| = 3 |x - 2|: delta = epsilon / 3
    assert cm.delta_for_epsilon(f, 2.0, 7.0, 0.3) == pytest.approx(0.1, abs=2e-3)
    sinc = F("sin(x) / x")  # undefined at 0, limit 1
    d = cm.delta_for_epsilon(sinc, 0.0, 1.0, 0.01)
    assert 0.2 < d < 0.3  # 1 - sin(x)/x ~ x^2 / 6 < 0.01 for |x| < 0.245
    assert (
        cm.delta_for_epsilon(sinc, 0.0, 1.0, 0.001) < d
    )  # a smaller epsilon needs a smaller delta
    assert cm.delta_for_epsilon(f, 2.0, 9.0, 0.3) == 0.0  # the wrong limit: no delta works


def test_vector_fields_divergence_curl_and_greens_theorem():
    rot = cm.VectorField.parse("-y", "x")  # a spinning field
    assert rot.divergence() == 0 and rot.curl() == 2
    src = cm.VectorField.parse("x", "y")  # flowing outward
    assert src.divergence() == 2 and src.curl() == 0
    px, py = rot(np.array([1.0, 0.0]), np.array([0.0, 2.0]))
    assert px.tolist() == [0.0, -2.0] and py.tolist() == [1.0, 0.0]
    # Green: circulation around a circle of radius r = integral of the curl (2) over its area
    assert rot.circulation(1.5) == pytest.approx(2.0 * math.pi * 1.5**2, rel=1e-6)
    assert src.circulation(1.5) == pytest.approx(0.0, abs=1e-9)
    mixed = cm.VectorField.parse("x * y", "x + y**2")
    assert mixed.evaluate(mixed.divergence(), 2.0, 3.0) == pytest.approx(3.0 + 6.0)  # y + 2y
    assert mixed.evaluate(mixed.curl(), 2.0, 3.0) == pytest.approx(1.0 - 2.0)
    assert mixed.circulation(0.5, center=(1.0, 1.0)) == pytest.approx(
        math.pi * 0.25 * mixed.evaluate(mixed.curl(), 1.0, 1.0), rel=0.03
    )  # small disc: curl * area


# ---------------------------------------------------------------- limits, flux, Green


def test_limits_are_found_symbolically():
    assert cm.limit_at(cm.RealFunction.parse("sin(x)/x"), 0.0) == pytest.approx(1.0)
    assert cm.limit_at(cm.RealFunction.parse("(x**2 - 1)/(x - 1)"), 1.0) == pytest.approx(2.0)
    assert cm.limit_at(cm.RealFunction.parse("1/x"), 0.0) is None  # the sides disagree
    assert cm.limit_at(cm.RealFunction.parse("abs(x)/x"), 0.0) is None


@pytest.mark.parametrize(
    ("p", "q", "center", "radius"),
    [("-y", "x", (0.0, 0.0), 1.5), ("x", "y", (0.5, -0.2), 1.0),
     ("x**2 - y", "x*y + sin(y)", (0.3, 0.4), 0.8), ("exp(x) * y", "y**3 - x", (-0.5, 0.0), 1.2)],
)  # fmt: skip
def test_green_and_the_divergence_theorem_hold_around_circles(p, q, center, radius):
    f = cm.VectorField.parse(p, q)
    circ = cm.loop_integral(f, center, radius, "circulation")
    flux = cm.loop_integral(f, center, radius, "flux")
    assert circ == pytest.approx(cm.disc_integral(f, f.curl(), center, radius), abs=1e-6)
    assert flux == pytest.approx(cm.disc_integral(f, f.divergence(), center, radius), abs=1e-6)
    assert circ == pytest.approx(f.circulation(radius, center), abs=1e-6)


def test_a_running_loop_integral_accumulates_along_the_curve():
    f = cm.VectorField.parse("-y", "x")  # tangent to every circle: constant |F| r per angle
    whole = cm.loop_integral(f, (0.0, 0.0), 1.0, "circulation")
    assert whole == pytest.approx(2 * math.pi)
    assert cm.loop_integral(f, (0.0, 0.0), 1.0, "circulation", upto=0.25) == pytest.approx(
        math.pi / 2
    )
    assert cm.loop_integral(f, (0.0, 0.0), 1.0, "flux") == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="flux or circulation"):
        cm.loop_integral(f, (0.0, 0.0), 1.0, "work")


@pytest.mark.gl
@pytest.mark.skipif(not __import__("tests.conftest").conftest.gl_available(),
                    reason="needs OpenGL")  # fmt: skip
def test_the_dense_field_look_matches_its_golden():
    """Many small arrows, heads in proportion, color by strength (preset "dense"), in 2D and
    on a plane in 3D."""
    import os

    from PIL import Image

    from phasor import Scene, calc, geo
    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    s = Scene("dense")
    s.add(calc.VectorField(id="f", p="sin(y) - 0.3*x", q="-sin(x) + 0.2*y",
                           extent=(14.0, 8.0), preset="dense"))  # fmt: skip
    t = Scene("dense3d")
    t.use_camera("orbit", elevation=28.0, azimuth=-60.0, distance=13.0, drift=0.0)
    t.add(geo.VectorField3D(id="f", fp="-y", fq="x", fr="0", extent=(8.0, 8.0, 0.0),
                            preset="dense"))  # fmt: skip
    pic = np.concatenate([s.engine.still(0.0, (480, 270)), t.engine.still(0.0, (480, 270))], 1)
    path = GOLDEN / s.engine.pass_runner().renderer_class() / "dense_fields.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, pic)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(pic, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
