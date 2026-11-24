"""The encoder runs on its own thread.

``render.encode.BackgroundWriter`` wraps any writer: ``write`` hands the frame to a thread
and returns, so frame k is encoded while frame k+1 renders. Frames reach the real writer
in order, all of them, and an encoder error reaches the caller.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from phasor.render.encode import BackgroundWriter, EncodeError
from tests.conftest import gl_available


class _Slow:
    """A writer that takes its time and records what it got, and on which thread."""

    def __init__(self, delay=0.002, fail_at=None):
        self.path, self.encoder = "x", "fake"
        self.frames, self.threads, self.closed = [], set(), False
        self.delay, self.fail_at = delay, fail_at

    def write(self, frame):
        time.sleep(self.delay)
        if self.fail_at is not None and len(self.frames) == self.fail_at:
            raise EncodeError("disk full")
        self.frames.append(frame)
        self.threads.add(threading.get_ident())

    def close(self):
        self.closed = True


def _frames(n):
    return [np.full((4, 6, 3), k, dtype=np.uint8) for k in range(n)]


def test_every_frame_arrives_in_order_on_another_thread():
    inner = _Slow()
    w = BackgroundWriter(inner)
    assert (w.path, w.encoder) == ("x", "fake")
    frames = _frames(40)
    for f in frames:
        w.write(f)
    w.close()
    assert inner.closed and len(inner.frames) == 40
    assert all(a is b for a, b in zip(inner.frames, frames, strict=True))
    assert threading.get_ident() not in inner.threads


def test_write_does_not_wait_for_the_encoder():
    inner = _Slow(delay=0.05)
    w = BackgroundWriter(inner, depth=4)
    t0 = time.perf_counter()
    for f in _frames(3):
        w.write(f)
    assert time.perf_counter() - t0 < 0.05  # handed over, not encoded yet
    w.close()
    assert len(inner.frames) == 3


def test_an_encoder_error_reaches_the_caller():
    w = BackgroundWriter(_Slow(fail_at=2))
    with pytest.raises(EncodeError, match="disk full"):
        for f in _frames(50):
            w.write(f)
        w.close()


def test_an_error_after_the_last_write_reaches_close():
    w = BackgroundWriter(_Slow(fail_at=2))
    for f in _frames(3):
        w.write(f)
    with pytest.raises(EncodeError, match="disk full"):
        w.close()


def test_abort_stops_without_writing_the_rest():
    inner = _Slow(delay=0.02)
    w = BackgroundWriter(inner, depth=8)
    for f in _frames(8):
        w.write(f)
    w.abort()
    assert len(inner.frames) < 8 and inner.closed


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_rendered_png_sequence_is_the_engine_frames(tmp_path):
    from PIL import Image

    from phasor import Dot, Scene
    from phasor.render.job import render_file

    s = Scene("encode_thread")
    s.add(Dot(id="d", pos=(-2.0, 0.0)))
    s.at(0).to("d.pos", (2.0, 0.0, 0.0), duration=0.5)
    s.marker("end", 0.5)
    e = s.engine
    r = render_file(e, tmp_path / "seq", preset="preview", fps=12)
    files = sorted((tmp_path / "seq").glob("frame_*.png"))
    assert len(files) == r["frames"] == 6
    for k, f in enumerate(files):
        want = e.frame(k / 12, (960, 540), output="srgb8")
        assert np.array_equal(np.asarray(Image.open(f)), want), k
