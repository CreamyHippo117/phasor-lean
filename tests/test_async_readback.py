"""Asynchronous readback of the color pass.

``GLRenderer.start_color_raw`` draws a frame and starts copying its two targets into pixel
pack buffers, returning at once; ``collect()`` on the result waits for the copy and gives
the arrays. ``Engine`` starts every motion-blur sample of a frame before collecting any, so
the card copies one sample while the next is being built. Only the timing changes: the
pixels must be exactly those of the synchronous read.
"""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Dot, Line, Scene, Text
from tests.conftest import gl_available

pytestmark = [pytest.mark.gl, pytest.mark.skipif(not gl_available(), reason="needs OpenGL")]


def _engine():
    s = Scene("async_read")
    s.set("camera.height", 4)
    xs = np.linspace(-3, 3, 200)
    s.add(Line(id="wave", points=[(x, np.sin(2 * x), 0.0) for x in xs], style={"glow": "strong"}))
    s.add(Dot(id="d", pos=(-2.0, -1.0)))
    s.add(Text(id="t", text="async", pos=(0.0, 1.5)))
    s.at(0).to("d.pos", (2.0, 1.0, 0.0), duration=2.0)
    return s.engine


def test_reads_started_together_and_collected_in_any_order_equal_synchronous_reads():
    from phasor.render.passes import GLRenderer

    e, r = _engine(), GLRenderer()
    times, sizes = (0.2, 0.9, 1.6), [(320, 180), (160, 90), (320, 180)]  # sizes change too
    want = [tuple(a.copy() for a in r.color_and_emission_raw(e.state_at(t), s))
            for t, s in zip(times, sizes, strict=True)]  # fmt: skip
    pending = [r.start_color_raw(e.state_at(t), s) for t, s in zip(times, sizes, strict=True)]
    for k in (2, 0, 1):
        got = pending[k].collect()
        assert all(np.array_equal(g, w) for g, w in zip(got, want[k], strict=True)), k
        assert got[0].dtype == np.float16 and got[0].shape == (*sizes[k][::-1], 4)


def test_a_read_is_collected_once():
    from phasor.render.passes import GLRenderer

    e, r = _engine(), GLRenderer()
    p = r.start_color_raw(e.state_at(0.5), (64, 36))
    p.collect()
    with pytest.raises(RuntimeError):
        p.collect()


def test_pack_buffers_are_reused_after_collection(monkeypatch):
    from phasor.render.passes import GLRenderer

    e, r = _engine(), GLRenderer()
    size = (320, 180)
    for p in [r.start_color_raw(e.state_at(t), size) for t in (0.1, 0.2, 0.3, 0.4)]:
        p.collect()  # four in flight: the pool holds four pairs now
    made = []
    real = r.ctx.buffer
    monkeypatch.setattr(r.ctx, "buffer", lambda *a, **k: made.append(1) or real(*a, **k))
    for _ in range(3):  # later frames reuse them
        for p in [r.start_color_raw(e.state_at(t), size) for t in (0.5, 0.6, 0.7, 0.8)]:
            p.collect()
    assert made == []


def test_blurred_and_sharp_frames_are_unchanged_by_the_asynchronous_read():
    """``Engine`` frames through the asynchronous path equal frames from synchronous reads."""
    from phasor.engine import blur_times
    from phasor.render.post import finish

    e = _engine()
    r = e.pass_runner()
    size = (320, 180)
    for t in (0.5, 1.0):
        times = blur_times(t, 24.0, 0.5, 4)
        pic = glow = 0.0
        for ts in times:
            p, g = r.color_and_emission_raw(e.state_at(ts), size)
            pic, glow = pic + 0.25 * p.astype(np.float32), glow + 0.25 * g.astype(np.float32)
        want = finish(np.ascontiguousarray(pic[::-1], dtype=np.float32), e._snap.compiled.theme,
                      np.ascontiguousarray(glow[::-1], dtype=np.float32), output="srgb8")  # fmt: skip
        got = e.frame_blurred(t, 24.0, size=size, output="srgb8")
        assert np.abs(got.astype(np.int16) - want).max() <= 1
        sharp = r.color_and_emission_raw(e.state_at(t), size)
        want_sharp = finish(np.ascontiguousarray(sharp[0][::-1], dtype=np.float32),
                            e._snap.compiled.theme,
                            np.ascontiguousarray(sharp[1][::-1], dtype=np.float32))  # fmt: skip
        assert np.array_equal(e.frame(t, size), want_sharp)
