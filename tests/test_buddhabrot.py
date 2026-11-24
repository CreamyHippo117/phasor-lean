"""The fractal cloud mode: ``cx.Fractal(kind=
"buddhabrot")`` draws escaping orbits with step 3's ``splat`` kernel and
``cloud_math.buddhabrot``. The plain kinds are unchanged: their goldens (``deep_zoom``,
``prism_fractal``) and ``tests/test_deep_zoom.py`` still hold."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, cx
from phasor.core.rng import rng
from phasor.domains import cloud_math as clm
from phasor.render.gpu import Tile, get
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden"
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
splat_ref = get("splat").reference


def _scene(look="none", **kw):
    s = Scene("bb", look=look, passes=None)
    s.set("camera.center", (0.0, 0.0, 0.0))
    s.set("camera.height", 4.5)
    kw = {"pos": (0, 0, 0), "size": (8, 4.5), "center": "-0.45 0", "zoom": 0.2,
          "max_iter": 200, "count": 60_000, **kw}  # fmt: skip
    s.add(cx.Fractal(id="cloud", kind="buddhabrot", **kw))
    return s.engine


def _orbits(e, t=0.0):
    st = e.state_at(t)
    return st["cloud"].impl.orbits(st)


def _old_buddhabrot(n, max_iter, gen, min_iter=8):
    """The first version of ``cloud_math.buddhabrot`` (every orbit of the batch stepped to
    the end), with the batch it now uses: the specification the faster one must match."""
    pts, vals, have = [], [], 0
    while have < n:
        r = 2.0 * np.sqrt(gen.random(clm.BUDDHA_BATCH))
        c = r * np.exp(2j * np.pi * gen.random(clm.BUDDHA_BATCH))
        q = (c.real - 0.25) ** 2 + c.imag**2
        inside = (q * (q + (c.real - 0.25)) <= 0.25 * c.imag**2) | (
            (c.real + 1) ** 2 + c.imag**2 <= 1 / 16
        )
        c = c[~inside]
        z = np.zeros_like(c)
        orbit = np.empty((max_iter, len(c)), dtype=np.complex128)
        escaped = np.full(len(c), -1)
        alive = np.ones(len(c), dtype=bool)
        for k in range(max_iter):
            z = np.where(alive, z * z + c, z)
            orbit[k] = z
            gone = alive & (z.real * z.real + z.imag * z.imag > 4.0)
            escaped[gone] = k + 1
            alive &= ~gone
        keep = escaped >= min_iter
        e = escaped[keep]
        mask = np.arange(max_iter)[:, None] < e[None, :]
        pts.append(orbit[:, keep].T[mask.T])
        vals.append(np.repeat(np.log(e) / np.log(max_iter), e))
        have += int(e.sum())
    return np.concatenate(pts)[:n], np.minimum(np.concatenate(vals)[:n], 0.999999)


# ---------------------------------------------------------------- the orbits


def test_the_fast_orbits_are_the_plain_iteration_bit_for_bit():
    for n, m in ((40_000, 60), (90_000, 300)):
        z, v = clm.buddhabrot(n, m, rng("spec", 2))
        z0, v0 = _old_buddhabrot(n, m, rng("spec", 2))
        np.testing.assert_array_equal(z, z0)
        np.testing.assert_array_equal(v, v0)


def test_the_same_seed_draws_the_same_cloud_and_another_seed_does_not():
    a, b, c = _scene(seed=3), _scene(seed=3), _scene(seed=4)
    (pa, va), (pb, vb), (pc, _) = _orbits(a), _orbits(b), _orbits(c)
    np.testing.assert_array_equal(pa, pb)
    np.testing.assert_array_equal(va, vb)
    assert not (len(pa) == len(pc) and np.array_equal(pa, pc))
    lut = np.eye(3, dtype=np.float32)
    pic = [splat_ref(Tile.full(96, 54), p, lut[(v * 2.999).astype(int)], (4.0, 2.25), 4.5)
           for p, v in ((pa, va), (pb, vb))]  # fmt: skip
    np.testing.assert_array_equal(*pic)


def test_orbits_land_where_the_buddhabrot_is():
    """Light is conserved (each point adds its weight), the cloud is symmetric under
    conjugation, and nearly all of it sits in |z| <= 2 round the set."""
    e = _scene(count=200_000)
    pts, v = _orbits(e)
    sem = e.inspect(0.0, id="cloud")["objects"][0]["semantic"]
    assert sem["set"] == "buddhabrot" and sem["points"] == 200_000
    assert sem["in_rectangle"] == len(pts) and 0.9 * 200_000 < len(pts) <= 200_000
    assert v.min() >= np.log(8) / np.log(200) - 1e-12 and v.max() < 1
    # back in the plane: z = (q - half size) * k + c
    k = 10.0**-0.2
    z = (pts[:, 0] - 4.0) * k - 0.45 + 1j * (pts[:, 1] - 2.25) * k
    assert (np.abs(z) <= 2).mean() > 0.9
    top, bottom = (z.imag > 0).sum(), (z.imag < 0).sum()
    assert abs(top - bottom) / (top + bottom) < 0.03
    # every point's light lands in the picture: the raw density sums to the points (less
    # the share of footprints cut off at the frame)
    raw = splat_ref(Tile.full(160, 90), pts, np.ones(3, dtype=np.float32), (4.0, 2.25), 4.5,
                    curve=False)  # fmt: skip
    assert raw[..., 3].sum() == pytest.approx(len(pts), rel=0.01)
    # the far left of the frame (Re z < -2.2) is all but dark: only the odd escaping step
    assert raw[:, :8, 3].mean() < 0.05 * raw[..., 3].mean()


def test_the_view_follows_center_and_zoom():
    wide, close = _scene(), _scene(center="-0.1 0.7", zoom=1.0)
    assert len(_orbits(close)[0]) < 0.2 * len(_orbits(wide)[0])
    sem = close.inspect(0.0, id="cloud")["objects"][0]["semantic"]
    assert sem["zoom"] == 1.0 and sem["view_height"] == pytest.approx(0.45)


def test_max_iter_is_capped_for_the_cloud_and_not_for_the_other_kinds():
    from phasor.core.issues import DocumentError

    s = Scene("cap", passes=None)
    with pytest.raises((DocumentError, ValueError), match="5000"):
        s.add(cx.Fractal(id="a", kind="buddhabrot", max_iter=6000))
    s.add(cx.Fractal(id="b", max_iter=20_000))  # mandelbrot: as before


def test_the_prism_look_lights_the_cloud_white_hot_and_glowing():
    s = Scene("look", look="prism", passes=None)
    s.add(cx.Fractal(id="c", kind="buddhabrot"))
    o = s.engine.document.object("c")
    assert o.params["exposure"] == pytest.approx(0.06) and o.params["white"] == 0.8
    assert o.style["glow"] == "soft"
    plain = cx.Fractal(id="p", kind="buddhabrot").to_spec()
    assert "white" not in plain.params  # the plain default: no white-hot core


def test_the_field_is_light_on_the_background():
    e = _scene(count=100_000)
    st = e.state_at(0.0)
    calls = []

    class Ctx:
        def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0):
            calls.append((pixels, glow))

    st["cloud"].impl.draw(Ctx(), st, "color")
    ((pixels, _glow),) = calls
    img = pixels(160, 90, (4.0, 2.25), 4.5)
    bg = np.asarray(st.background[:3], dtype=np.float32)
    empty = img[..., 3] == 0
    np.testing.assert_allclose(img[empty][:, :3], np.broadcast_to(bg, (int(empty.sum()), 3)))
    a = img[..., 3]
    assert a.max() > 0.3 and a[:, :5].mean() < 0.1 * a.max()  # a lit body, dim far left


@needs_gl
def test_the_prism_cloud_emits_light_and_the_plain_one_does_not():
    def emission(look):
        s = Scene("e", look=look)
        s.set("camera.center", (0.0, 0.0, 0.0))
        s.set("camera.height", 4.5)
        s.add(cx.Fractal(id="f", kind="buddhabrot", pos=(0, 0, 0), size=(8, 4.5),
                         center="-0.45 0", zoom=0.2, max_iter=200, count=100_000))  # fmt: skip
        r = s.engine.pass_runner()
        return r.color_and_emission(s.engine.state_at(0.0), (160, 90))[1][..., :3]

    assert emission("none").max() == 0.0
    assert emission("prism").max() > 0.05


@needs_gl
@pytest.mark.slow
def test_buddhabrot_golden_storyboard():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    e = runpy.run_path(str(ROOT / "examples" / "buddhabrot.py"), run_name="x")["s"].engine
    assert {i.code for i in e.lint(strict=True)} <= {"raw_motion"}  # a slow camera drift
    assert [i for i in e.lint() if i.severity == "error"] == []
    sheet = e.storyboard([1.5, 6.0, 10.5], size=(320, 180))["image"]
    klass = e.pass_runner().renderer_class()
    path = GOLDEN / klass / "buddhabrot.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {klass} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.98
