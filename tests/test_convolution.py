"""Live convolution: the numerics against known results, the object through
the engine, its glow in the prism look and its golden storyboard."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, conv
from phasor.domains import convolution_math as cm
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def P(spec, support=None):
    return cm.Signal.parse(spec, support)


def _area(sig):
    s = np.linspace(*sig.support, 4000)
    return float(np.trapezoid(sig(s), s))


def test_box_with_box_is_a_triangle():
    f = P("box")  # 1 on [-1, 1]
    tau, out = cm.convolution(f, f)
    assert tau[0] == -2 and tau[-1] == 2
    assert out == pytest.approx(np.maximum(0, 2 - np.abs(tau)), abs=0.01)


def test_decay_with_decay_is_t_exp():
    a = 1.5
    f = P(f"exp(-{a} * t)", (0, 8))
    tau, out = cm.convolution(f, f, samples=161)
    assert out == pytest.approx(tau * np.exp(-a * tau), abs=0.01)


def test_agrees_with_the_discrete_fft_convolution():
    from phasor.domains.fourier_math import convolve

    f, g = P("triangle"), P("decay")
    dt = 0.005
    a = f(np.arange(*f.support, dt))
    b = g(np.arange(*g.support, dt))
    disc = convolve(a, b) * dt  # sample k sits at tau = lo + k dt
    lo = cm.shift_range(f, g)[0]
    tau, out = cm.convolution(f, g, samples=101)
    k = np.round((tau - lo) / dt).astype(int).clip(0, len(disc) - 1)
    assert out == pytest.approx(disc[k], abs=0.01)


def test_gauss_widths_add_in_variance():
    f = P("exp(-t**2 / 2)", (-8, 8))  # sigma 1, area sqrt(2 pi)
    tau, out = cm.convolution(f, f, samples=81)
    sigma2 = 2.0
    want = 2 * np.pi * np.exp(-(tau**2) / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
    assert out == pytest.approx(want, abs=1e-3)


def test_convolution_commutes_and_total_area_is_the_product_of_areas():
    f, g = P("triangle"), P("ramp")
    t1, o1 = cm.convolution(f, g, samples=401)
    t2, o2 = cm.convolution(g, f, samples=401)
    assert t1 == pytest.approx(t2)
    assert o1 == pytest.approx(o2, abs=0.01)
    assert np.trapezoid(o1, t1) == pytest.approx(_area(f) * _area(g), rel=0.01)


def test_the_slider_picture_agrees_with_the_output_curve():
    f, g = P("box"), P("decay")
    tau, out = cm.convolution(f, g, samples=61)
    for k in (0, 17, 30, 60):
        got = cm.product_at(f, g, tau[k])
        assert got["area"] == pytest.approx(out[k], abs=1e-12)
        assert got["product"] == pytest.approx(got["f"] * got["g"])


def test_the_kernel_is_flipped_before_it_slides():
    f, g = P("box"), P("ramp")  # ramp rises on [0, 2]
    at = cm.product_at(f, g, 1.0)
    s, gs = at["s"], at["g"]
    inside = gs != 0
    assert s[inside].min() >= 1.0 - 2.0 - 1e-9 and s[inside].max() <= 1.0 + 1e-9
    # flipped: the larger values are at the smaller s (g(1 - s) is big for s near -1)
    left = gs[(s > -0.9) & (s < -0.8)].mean()
    right = gs[(s > 0.8) & (s < 0.9)].mean()
    assert left > right


def test_no_overlap_outside_the_shift_range_and_formulas_are_safe():
    f, g = P("box"), P("decay")
    lo, hi = cm.shift_range(f, g)
    assert (lo, hi) == (-1.0, 4.0)
    assert cm.product_at(f, g, lo - 0.5)["area"] == 0.0
    assert cm.product_at(f, g, hi + 0.5)["area"] == 0.0
    with pytest.raises(cm.FunctionError):
        P("__import__('os')")
    with pytest.raises(cm.FunctionError, match="unknown name"):
        P("t + q")
    with pytest.raises(cm.FunctionError, match="support"):
        P("box", (2, 1))
    assert P("1 / t", (-1, 1))(np.array([0.0]))[0] == 0.0  # not finite -> zero


def _scene(**kw):
    s = Scene("conv_test", **kw)
    s.add(conv.Convolution(id="c", f="box", g="decay", progress=0.5))
    return s


def test_object_reports_the_shift_and_the_output():
    e = _scene().engine
    sem = e.inspect(0.5)["objects"][0]["semantic"]
    assert sem["kind"] == "convolution"
    assert sem["shift_range"] == [-1.0, 4.0]
    assert sem["shift"] == pytest.approx(1.5)
    want = cm.product_at(P("box"), P("decay"), 1.5)["area"]
    assert sem["output"] == pytest.approx(want, abs=1e-5)
    assert sem["peak"] > 0 and sem["overlapping"] is True


def test_progress_moves_the_kernel_across_the_whole_range():
    s = Scene("conv_sweep", passes=None)
    s.add(conv.Convolution(id="c", f="box", g="decay", progress=0.0))
    s.marker("go", 0.0)
    s.marker("end", 2.0)
    s.at("go").sweep("c.progress", 1.0, show=False)
    e = s.engine
    first = e.inspect(0.0)["objects"][0]["semantic"]
    last = e.inspect(e.duration)["objects"][0]["semantic"]
    assert first["shift"] == pytest.approx(-1.0) and first["overlapping"] is False
    assert last["shift"] == pytest.approx(4.0) and last["output"] == pytest.approx(0.0, abs=1e-3)


def test_bad_signals_are_rejected_when_added():
    s = Scene("conv_bad")
    with pytest.raises(Exception, match="cannot use|unknown"):
        s.add(conv.Convolution(id="a", f="__import__('os')"))
    with pytest.raises(Exception, match="support"):
        s.add(conv.Convolution(id="b", g="decay", g_support=(3, 1)))


@needs_gl
def test_it_draws_and_glows_in_the_prism_look():
    def emit(**kw):
        s = Scene("conv_glow", **kw)
        s.add(conv.Convolution(id="c", f="box", g="decay", progress=0.5))
        r = s.engine.pass_runner()
        return r.color_and_emission(s.engine.state_at(0.0), (320, 180))

    pic, light = emit()
    assert pic[..., :3].max() > 0.0 and light[..., :3].max() == 0.0  # plain: paint only
    again, _ = emit()
    assert np.array_equal(pic, again)
    pic, light = emit(look="prism")
    assert light[..., :3].max() > 0.2


@needs_gl
@pytest.mark.slow
def test_convolution_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "convolution.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([1.7, 2.3, 3.6], size=(320, 180))["image"]
    _golden("convolution", sheet, e.pass_runner().renderer_class())
