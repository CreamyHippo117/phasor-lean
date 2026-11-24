"""Motion-blur samples summed on the card.

``Engine.frame_blurred`` reads each sample as float16, sends it to the card once and adds it
there in float32 with the ``accumulate`` kernel; the sum goes straight into ``finish``. The
old path widened on the host and summed in float64. These tests hold the new path to the
old one's bytes (within one 8-bit code value) and to determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Dot, Scene
from phasor.engine import blur_times
from phasor.render import gpu
from phasor.render.gpu import Tile
from phasor.render.gpu.bridge import to_host
from phasor.render.post import finish
from tests.conftest import gl_available

needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
SIZE = (320, 180)
FPS = 24.0


def _scene(bright: bool) -> Scene:
    """A dot crossing the frame: a strong glow (light far above the knee), or dim paint
    (values near black, where sRGB steps are steepest and rounding shows first)."""
    s = Scene("blur_sum_bright" if bright else "blur_sum_dark")
    s.set("camera.height", 4)
    style = {"color": "highlight", "glow": "strong"} if bright else {"color": "muted"}
    s.add(Dot(id="d", pos=(-2, 0), r=0.3, opacity=1.0 if bright else 0.15, style=style))
    s.at(0).to("d.pos", (2, 0.5), duration=2.0, ease="linear")
    return s


def _host_sum(e, t: float, samples: int) -> np.ndarray:
    """The pre-step-2 path: widen on the host, average in float64, finish on the host."""
    runner = e.pass_runner()
    pic = glow = 0.0
    times = blur_times(t, FPS, 0.5, samples)
    for ts in times:
        p, g = runner.color_and_emission(e.state_at(ts), SIZE)
        pic, glow = pic + p.astype(np.float64), glow + g.astype(np.float64)
    pic, glow = (pic / len(times)).astype(np.float32), (glow / len(times)).astype(np.float32)
    return finish(pic, e._snap.compiled.theme, glow, backend="reference", output="srgb8")


@pytest.mark.gl
@needs_gl
@pytest.mark.parametrize("bright", [True, False], ids=["bright", "dark"])
@pytest.mark.parametrize("samples", [2, 4, 8])
def test_blur_sum_on_the_card_matches_the_host_sum(bright, samples):
    e = _scene(bright).engine
    t = 1.0
    assert e.is_moving(t - 0.5 / FPS, t + 0.5 / FPS)
    got = e.frame_blurred(t, FPS, samples=samples, size=SIZE, output="srgb8")
    want = _host_sum(e, t, samples)
    diff = np.abs(got.astype(np.int16) - want)
    assert diff.max() <= 1 and (diff > 0).mean() <= 1e-3, (int(diff.max()), (diff > 0).mean())
    assert (got != _scene_frame(e, t)).any()  # it really took the blur path


def _scene_frame(e, t: float) -> np.ndarray:
    return e.frame(t, SIZE, output="srgb8")


def test_one_sample_sum_is_the_widened_sample_exactly():
    """``frame_blurred`` never sums a single sample (it returns ``frame``), so the one-sample
    case is checked on the kernel: weight 1 on a float16 frame is its exact float32 value."""
    rng = np.random.default_rng(4)
    half = (rng.random((27, 48, 4)) * 40).astype(np.float16)
    for backend in ("reference", "auto"):
        got = to_host(gpu.run("accumulate", Tile.full(48, 27), image=half, backend=backend))
        np.testing.assert_array_equal(got, half.astype(np.float32), err_msg=backend)


def test_accumulate_leaves_its_inputs_alone():
    image = np.ones((4, 6, 4), dtype=np.float16)
    acc = np.full((4, 6, 4), 2.0, dtype=np.float32)
    out = gpu.run("accumulate", Tile.full(6, 4), image=image, weight=0.5, acc=acc,
                  backend="reference")  # fmt: skip
    assert (out == 2.5).all() and (acc == 2.0).all() and out is not acc


@pytest.mark.gl
@needs_gl
def test_a_still_moment_is_still_exactly_frame():
    e = _scene(True).engine
    for t in (2.5, 3.0):  # after the move
        assert not e.is_moving(t - 0.5 / FPS, t + 0.5 / FPS)
        assert np.array_equal(e.frame_blurred(t, FPS, size=SIZE), e.frame(t, SIZE))
        assert np.array_equal(
            e.frame_blurred(t, FPS, size=SIZE, output="srgb8"), _scene_frame(e, t)
        )


@pytest.mark.gpu
@pytest.mark.gl
@needs_gl
def test_same_blurred_frame_twice_is_identical():
    e = _scene(True).engine
    first = e.frame_blurred(1.0, FPS, samples=6, size=SIZE, output="srgb8")
    _scene_frame(e, 0.3)  # something else in between
    assert np.array_equal(first, e.frame_blurred(1.0, FPS, samples=6, size=SIZE, output="srgb8"))
    linear = e.frame_blurred(1.0, FPS, samples=6, size=SIZE)
    assert np.array_equal(linear, e.frame_blurred(1.0, FPS, samples=6, size=SIZE))


def _old_frame(e, t: float, output: str):
    """The pre-change sharp path: widen on the host, finish from host arrays."""
    pic, glow = e.pass_runner().color_and_emission(e.state_at(t), SIZE)
    return finish(pic, e._snap.compiled.theme, glow, output=output)


@pytest.mark.gl
@needs_gl
def test_a_sharp_frame_through_the_card_is_the_host_path_exactly():
    """``frame`` (every frame where only life moves: drift, breathing glow, hue) widens on
    the card now too; a weight-1 sum is exact, so the pictures are bit-identical."""
    for bright in (True, False):
        e = _scene(bright).engine
        for t in (0.0, 1.0, 3.0):
            for output in ("linear", "srgb8"):
                assert np.array_equal(e.frame(t, SIZE, output=output), _old_frame(e, t, output))
