"""GL buffers are reused across draws.

``GLRenderer`` keeps one vertex array per program and vertex layout over buffers that grow
and are refilled for each draw, instead of creating and releasing them per draw command.
Nothing about the picture may change: a renderer that has drawn other scenes before (bigger
and smaller meshes left in its buffers) draws exactly what a fresh one draws, in every pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Dot, Line, Scene, Text, cx
from tests.conftest import gl_available

pytestmark = [pytest.mark.gl, pytest.mark.skipif(not gl_available(), reason="needs OpenGL")]
SIZE = (320, 180)


def _scene(n: int):
    """Strokes, dots, a hue-along curve (per-vertex colors), text and a kernel-drawn plane:
    every vertex layout but the lit surface. ``n`` sets how many vertices the curves have."""
    s = Scene(f"arena_{n}")
    s.set("camera.height", 4)
    s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(3.0, 1.7), pos=(1.5, 1.0)))
    xs = np.linspace(-3.5, 3.5, n)
    s.add(Line(id="wave", points=[(x, 0.8 * np.sin(3 * x), 0.0) for x in xs],
               style={"glow": "strong"}))  # fmt: skip
    s.add(Line(id="rainbow", points=[(x, -1.2 + 0.3 * np.cos(5 * x), 0.0) for x in xs],
               hue_along=1.0))  # fmt: skip
    s.add(Dot(id="d", pos=(-2.0, -1.5)))
    s.add(Text(id="t", text=f"n = {n}", pos=(-2.5, 1.6)))
    s.at(0).to("d.pos", (2.0, -1.5, 0.0), duration=2.0)
    return s.engine


def _everything(r, e, t: float) -> list[np.ndarray]:
    s = e.state_at(t)
    color, emission = r.color_and_emission_raw(s, SIZE)
    return [color.copy(), emission.copy(), r.id_buffer(s, SIZE), r.solo_mask(s, SIZE, "wave")]


def _same(a: list, b: list) -> None:
    for name, x, y in zip(("color", "emission", "id", "solo"), a, b, strict=True):
        assert np.array_equal(x, y), name


def test_a_warm_renderer_draws_what_a_fresh_one_draws():
    from phasor.render.passes import GLRenderer

    big, small = _scene(4000), _scene(40)
    warm = GLRenderer()
    _everything(warm, big, 1.0)  # leaves big buffers behind
    for e in (small, big, small):  # shrink, grow, shrink again
        for t in (0.5, 1.5):
            _same(_everything(warm, e, t), _everything(GLRenderer(), e, t))


def test_a_warm_renderer_creates_no_buffers_or_vertex_arrays(monkeypatch):
    from phasor.render.passes import GLRenderer

    e = _scene(400)
    r = GLRenderer()
    _everything(r, e, 1.0)  # programs, targets, text and arena buffers made here
    made = []
    for name in ("buffer", "vertex_array"):
        real = getattr(r.ctx, name)

        def counting(*a, _real=real, _name=name, **kw):
            made.append(_name)
            return _real(*a, **kw)

        monkeypatch.setattr(r.ctx, name, counting)
    _everything(r, e, 1.25)  # a different moment: new strokes, same layouts
    assert made == []
