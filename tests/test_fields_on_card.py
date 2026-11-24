"""Kernel-drawn objects do their per-pixel and per-point work
where the kernel left it (on the card with CUDA), and the pictures are the same bits as the
host path they replace."""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Scene, cloud, cx
from phasor.domains.cloud import _colors
from phasor.render import gpu
from tests.test_cloud import _pixels, _scene


def _old_host_mean(image: np.ndarray, s: int) -> np.ndarray:
    """The host average ``gpu.render`` and ``cx.Fractal`` did before step 9a."""
    h, w = image.shape[0] // s, image.shape[1] // s
    shape = (h, s, w, s, *image.shape[2:])
    return image.reshape(shape).mean(axis=(1, 3), dtype=np.float64).astype(image.dtype)


def _old_colors(v: np.ndarray, hue: float, lut: np.ndarray) -> np.ndarray:
    """``cloud._colors`` before step 9b (``% 1.0``)."""
    x = ((v + hue) % 1.0) * len(lut)
    k0 = np.minimum(x.astype(np.int64), len(lut) - 1)
    t = (x - k0)[:, None].astype(np.float32)
    return lut[k0] * (1 - t) + lut[(k0 + 1) % len(lut)] * t


def _values(n: int = 200_000) -> np.ndarray:
    gen = np.random.default_rng(3)
    v = gen.random(n)
    v[:6] = [0.0, 1.0 - 2**-53, 0.5, 1e-300, 0.25, 0.75]  # edges of the wheel
    return v


LUT = np.random.default_rng(4).random((256, 3)).astype(np.float32)


@pytest.mark.parametrize("s", [1, 2, 3, 4])
def test_box_mean_is_the_old_host_mean(s):
    img = (np.random.default_rng(s).random((9 * s, 7 * s, 4)) * 5).astype(np.float32)
    np.testing.assert_array_equal(gpu.box_mean(img, s), _old_host_mean(img, s))


@pytest.mark.gpu
@pytest.mark.parametrize("s", [2, 3, 4])
def test_box_mean_on_the_card_is_the_same_bits(s):
    cp = pytest.importorskip("cupy")
    img = np.random.default_rng(s).random((108 * s, 192 * s, 4)).astype(np.float32)
    img[:4] *= 1e-30  # tiny values, where an order-dependent sum would show first
    img[4:8] *= 1e6
    on_card = cp.asnumpy(gpu.box_mean(cp.asarray(img), s))
    np.testing.assert_array_equal(on_card, _old_host_mean(img, s))


@pytest.mark.parametrize("hue", [0.0, 0.3, -0.7, 2.25, -3.1])
def test_colors_floor_is_the_old_modulo(hue):
    v = _values()
    np.testing.assert_array_equal(_colors(v, hue, LUT), _old_colors(v, hue, LUT))


@pytest.mark.gpu
@pytest.mark.parametrize("hue", [0.0, 0.3, -0.7, 2.25])
def test_colors_on_the_card_are_the_same_bits(hue):
    cp = pytest.importorskip("cupy")
    v = _values()
    np.testing.assert_array_equal(cp.asnumpy(_colors(cp.asarray(v), hue, LUT)),
                                  _old_colors(v, hue, LUT))  # fmt: skip


def _host_only(monkeypatch):
    monkeypatch.setattr(cloud, "card_points", lambda *a, **k: None)


@pytest.mark.gpu
@pytest.mark.parametrize("params", [
    {"source": "attractor", "kind": "clifford", "count": 60_000, "seed": 2},
    {"source": "roots", "degree": 10, "twinkle": 0.5, "hue": 0.4},
    {"source": "roots", "degree": 9, "pos": (0.3, -0.2, 0.0), "scale": 1.3, "white": 0.5},
])  # fmt: skip
def test_a_cloud_drawn_on_the_card_is_the_host_picture(params, monkeypatch):
    on_card, _, _ = _pixels(_scene(size=(4.5, 2.6), **params), 192, 108)
    _host_only(monkeypatch)
    on_host, _, _ = _pixels(_scene(size=(4.5, 2.6), **params), 192, 108)
    assert on_card[..., 3].max() > 0
    np.testing.assert_array_equal(on_card, on_host)


def _buddha_pixels(zoom: float, pw: int = 160, ph: int = 90) -> np.ndarray:
    s = Scene("b", passes=None)
    s.add(cx.Fractal(id="b", kind="buddhabrot", size=(8, 4.5), center="-0.45 0.1", zoom=zoom,
                     max_iter=100, count=40_000))  # fmt: skip
    st = s.engine.state_at(0.0)
    calls = []

    class Ctx:
        def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0):
            calls.append(pixels)

    st["b"].impl.draw(Ctx(), st, "color")
    return calls[0](pw, ph, (4.0, 2.25), 4.5)


@pytest.mark.gpu
@pytest.mark.parametrize("zoom", [0.0, 0.4])
def test_a_buddhabrot_cut_on_the_card_is_the_host_picture(zoom, monkeypatch):
    on_card = _buddha_pixels(zoom)
    _host_only(monkeypatch)
    on_host = _buddha_pixels(zoom)
    assert on_card[..., 3].max() > 0
    np.testing.assert_array_equal(on_card, on_host)


@pytest.mark.gpu
def test_buddhabrot_semantics_do_not_change_with_the_card():
    s = Scene("b", passes=None)
    s.add(cx.Fractal(id="b", kind="buddhabrot", size=(8, 4.5), max_iter=100, count=40_000))
    st = s.engine.state_at(0.0)
    host = st["b"].impl.orbits(st)
    card = st["b"].impl.orbits(st, card=True)
    np.testing.assert_array_equal(cloud.array_module(card[0]).asnumpy(card[0]), host[0])
    np.testing.assert_array_equal(cloud.array_module(card[1]).asnumpy(card[1]), host[1])
