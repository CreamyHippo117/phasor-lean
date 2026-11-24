"""Point clouds: the ``splat`` kernel, the point sources and ``PointCloud``.

The kernel itself is also checked three ways (reference, CPU emulator, real GPU) through
``CASES`` in ``tests/test_kernels.py``."""

from __future__ import annotations

import os
import runpy
import time
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, cloud
from phasor.core.rng import rng
from phasor.domains import cloud_math as clm
from phasor.render import gpu
from phasor.render.gpu import Tile, get
from phasor.render.gpu.kernels.splat import splat_reach
from tests import cuda_emu
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden"
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
needs_compiler = pytest.mark.skipif(cuda_emu.compiler() is None, reason="no C++ compiler")
splat_ref = get("splat").reference


def _cloud(n=500, seed=1):
    g = np.random.default_rng(seed)
    return g.normal(0.0, 0.5, (n, 2)), g.random((n, 3)).astype(np.float32), g.random(n)


def _naive(tile, points, colors, center, height, weights, sigma):
    """The footprint written out point by point with ``np.add.at``: the specification."""
    acc = np.zeros((tile.h, tile.w, 4))
    s = height / tile.full_h
    r = splat_reach(sigma)
    for (x, y), c, w in zip(points, colors, weights, strict=True):
        fx = (x - center[0]) / s + tile.full_w / 2 - 0.5 - tile.x0
        fy = (center[1] - y) / s + tile.full_h / 2 - 0.5 - tile.y0
        px = np.floor(fx + 0.5).astype(int) + np.arange(-r, r + 1)
        py = np.floor(fy + 0.5).astype(int) + np.arange(-r, r + 1)
        g = np.exp(-0.5 * ((py[:, None] - fy) ** 2 + (px[None, :] - fx) ** 2) / sigma**2)
        g *= w / g.sum()
        ok_y, ok_x = (py >= 0) & (py < tile.h), (px >= 0) & (px < tile.w)
        if not (ok_y.any() and ok_x.any()):
            continue
        yy, xx = np.meshgrid(py[ok_y], px[ok_x], indexing="ij")
        gg = g[np.ix_(ok_y, ok_x)]
        for k in range(3):
            np.add.at(acc[..., k], (yy, xx), gg * c[k])
        np.add.at(acc[..., 3], (yy, xx), gg)
    return acc


# ---------------------------------------------------------------- the kernel


@pytest.mark.parametrize("tile", [Tile.full(40, 30), Tile(6, 4, 22, 17, 40, 30)])
@pytest.mark.parametrize("sigma", [0.7, 1.0, 1.6])
def test_reference_matches_np_add_at_point_by_point(tile, sigma):
    pts, rgb, w = _cloud(400)
    want = _naive(tile, pts, rgb, (0.1, -0.05), 2.5, w, sigma)
    got = splat_ref(tile, pts, rgb, (0.1, -0.05), 2.5, weights=w, sigma=sigma, curve=False)
    np.testing.assert_allclose(got, want, atol=2e-5)
    toned = splat_ref(tile, pts, rgb, (0.1, -0.05), 2.5, weights=w, sigma=sigma, gain=2.0)
    d = want[..., 3]
    a = 1 - np.exp(-2.0 * d)
    np.testing.assert_allclose(toned[..., 3], a, atol=2e-5)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(d[..., None] > 0, want[..., :3] / d[..., None], 0)
    np.testing.assert_allclose(toned[..., :3], mean * a[..., None], atol=2e-5)


def _backends():
    out = [pytest.param("reference", id="reference")]
    out.append(pytest.param("emulator", id="emulator", marks=needs_compiler))
    out.append(pytest.param("cuda", id="cuda", marks=pytest.mark.gpu))
    return out


def _splat(backend, tile, **kw):
    from phasor.render.gpu.bridge import to_host

    if backend == "emulator":
        return get("splat").launch(cuda_emu.EmuGpu(), tile, **kw)
    return to_host(gpu.run("splat", tile, backend=backend, **kw))


@pytest.mark.parametrize("backend", _backends())
def test_light_is_conserved_before_the_tone_curve(backend):
    """Every point well inside the frame gives all of its weight, and its colour times its
    weight, to the image: nothing is lost or made at the footprint's edges."""
    g = np.random.default_rng(4)
    pts = g.uniform(-1.0, 1.0, (200, 2))  # the view is 4 tall: all well inside
    rgb = g.random((200, 3)).astype(np.float32)
    w = g.random(200).astype(np.float32)
    out = _splat(backend, Tile.full(48, 32), points=pts, colors=rgb, center=(0.0, 0.0),
                 height=4.0, weights=w, sigma=1.3, curve=False)  # fmt: skip
    assert out[..., 3].sum() == pytest.approx(w.sum(), rel=1e-5)
    np.testing.assert_allclose(out[..., :3].sum((0, 1)), (rgb * w[:, None]).sum(0), rtol=1e-5)


@pytest.mark.parametrize("backend", _backends())
def test_a_single_point_lands_on_its_pixel(backend):
    """A point at the centre of pixel (col 13, row 5) of a 32 x 24 view lights that pixel
    most, symmetrically; y is up in the plane and row 0 is the top."""
    tile = Tile.full(32, 24)
    height = 2.4  # 0.1 per pixel
    x = (13 + 0.5 - 16) * 0.1
    y = (12 - (5 + 0.5)) * 0.1
    out = _splat(backend, tile, points=[[x, y]], colors=[[1.0, 0.5, 0.25]], center=(0.0, 0.0),
                 height=height, curve=False)  # fmt: skip
    d = out[..., 3]
    assert np.unravel_index(int(np.argmax(d)), d.shape) == (5, 13)
    assert d[5, 12] == pytest.approx(d[5, 14], rel=1e-5)
    assert d[4, 13] == pytest.approx(d[6, 13], rel=1e-5)
    assert d.sum() == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_allclose(out[5, 13, :3] / d[5, 13], [1.0, 0.5, 0.25], rtol=1e-5)


def test_dense_light_saturates_softly_and_alpha_is_the_glow_weight():
    pts = np.zeros((10_000, 2))
    out = splat_ref(Tile.full(9, 9), pts, [[0.2, 0.4, 1.0]], (0.0, 0.0), 0.9, gain=1.0)
    assert 0.99 < out[4, 4, 3] <= 1.0  # a pile of points: the curve's ceiling, not more
    np.testing.assert_allclose(out[4, 4, :3], np.array([0.2, 0.4, 1.0]) * out[4, 4, 3], rtol=1e-5)
    assert (out[..., 3] <= 1.0).all() and (out[..., :3] <= 1.0).all()
    empty = splat_ref(Tile.full(4, 4), np.zeros((0, 2)), np.zeros((0, 3)), (0, 0), 1.0)
    assert not empty.any()


def test_tiles_stitch_into_exactly_the_full_image():
    pts, rgb, w = _cloud(3000)
    kw = {"points": pts, "colors": rgb, "weights": w, "center": (0.0, 0.0), "height": 3.0,
          "backend": "reference"}  # fmt: skip
    whole = gpu.render("splat", 64, 48, **kw)
    pieces = gpu.render("splat", 64, 48, budget=16 * 64 * 7, **kw)
    np.testing.assert_allclose(pieces, whole, atol=1e-6)


def test_bad_input_is_refused():
    from phasor.render.gpu import KernelError

    with pytest.raises(KernelError, match="sigma"):
        splat_ref(Tile.full(4, 4), [[0, 0]], [[1, 1, 1]], (0, 0), 1.0, sigma=5.0)
    with pytest.raises(KernelError, match="colours"):
        splat_ref(Tile.full(4, 4), [[0, 0], [1, 1]], [[1, 1, 1]] * 3, (0, 0), 1.0)


@pytest.mark.gpu
def test_a_million_points_at_1080p_on_the_gpu_in_well_under_a_second():
    """Measured on an RTX 4070 Ti (2026-11-22): about 20 ms for the whole call, host arrays
    in and the image back on the host. Other work on the card makes it slower, never 50x."""
    if gpu.default_gpu() is None:
        pytest.skip("no CUDA device")
    pts, rgb, _ = _cloud(1_000_000, seed=2)
    kw = {"points": pts, "colors": rgb, "center": (0.0, 0.0), "height": 3.0, "gain": 5.0,
          "backend": "cuda"}  # fmt: skip
    gpu.render("splat", 1920, 1080, **kw)  # compiles
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = gpu.render("splat", 1920, 1080, **kw)
        times.append(time.perf_counter() - t0)
    print(f"splat: 1e6 points at 1920x1080 in {min(times) * 1000:.1f} ms")
    assert out.shape == (1080, 1920, 4) and out[..., 3].max() > 0.5
    assert min(times) < 0.5


# ---------------------------------------------------------------- the point sources


def test_roots_are_roots_of_their_polynomials():
    z, v = clm.polynomial_roots(8, (-1.0, 1.0))
    assert len(z) == 2**8 * 8 and v.min() >= 0 and v.max() < 1
    # rebuild each polynomial from its index: digit d of the index picks a_d
    for i in (0, 1, 77, 255):
        roots = z[i * 8 : (i + 1) * 8]
        a = [(-1.0, 1.0)[(i >> d) & 1] for d in range(8)]  # a_0 .. a_7
        coeffs = [1.0, *a[::-1]]  # highest first
        assert np.abs(np.polyval(coeffs, roots)).max() < 1e-9
        assert v[i * 8] == pytest.approx(i / 256)


def test_roots_of_pm1_polynomials_lie_in_the_annulus_and_samples_are_seeded():
    z, _ = clm.polynomial_roots(10)
    r = np.abs(z)
    assert (r > 0.5 - 1e-9).all() and (r < 2 + 1e-9).all()  # 1/2 < |z| < 2 for +-1 coefficients
    a, _ = clm.polynomial_roots(14, count=300, gen=rng("c", 3))
    b, _ = clm.polynomial_roots(14, count=300, gen=rng("c", 3))
    c, _ = clm.polynomial_roots(14, count=300, gen=rng("c", 4))
    assert len(a) == 300 * 14
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    with pytest.raises(ValueError, match="random generator"):
        clm.polynomial_roots(14, count=300)


@pytest.mark.parametrize("kind", ["clifford", "dejong", "lorenz"])
def test_attractors_fill_the_unit_square_and_are_seeded(kind):
    z, v = clm.attractor(kind, 20_000, rng("a", 0))
    assert len(z) == 20_000 and np.isfinite(z).all()
    assert np.abs(z.real).max() == pytest.approx(1.0) or np.abs(z.imag).max() == pytest.approx(1.0)
    assert np.abs(z.real).max() <= 1 + 1e-12 and np.abs(z.imag).max() <= 1 + 1e-12
    assert v.min() >= 0 and v.max() < 1
    z2, _ = clm.attractor(kind, 20_000, rng("a", 0))
    np.testing.assert_array_equal(z, z2)
    with pytest.raises(ValueError):
        clm.attractor(kind, 10, rng("a"), constants=(1.0,))


def test_buddhabrot_points_are_orbits_that_escape():
    z, v = clm.buddhabrot(30_000, 100, rng("b", 0))
    assert len(z) == 30_000 and v.min() >= 0 and v.max() < 1
    # every point is z_k of an orbit that had not yet escaped (|z| <= 2) or the escaping step
    assert np.abs(z).max() < 2 + 4 + 2  # |z^2 + c| < 4 + 2 for |z| <= 2, |c| < 2
    assert (np.abs(z) <= 2).mean() > 0.8
    assert np.abs(z.imag.mean()) < 0.05  # symmetric under conjugation, on average


# ---------------------------------------------------------------- the object


def _scene(**kw):
    s = Scene("pc", passes=None)
    s.set("camera.height", 4.5)
    s.add(cloud.PointCloud(id="pts", **kw))
    return s.engine


def test_point_cloud_semantics_and_points_given_directly():
    e = _scene(source="points", points=((0.0, 0.0), (1.0, 0.5), (9.0, 9.0)), values=(0, .5, .9),
               size=(4.0, 2.0))  # fmt: skip
    sem = e.inspect(0.0, id="pts")["objects"][0]["semantic"]
    assert sem["kind"] == "point_cloud" and sem["backdrop"] is True
    assert sem["points"] == 3 and sem["in_rectangle"] == 2  # (9, 9) is off the rectangle
    st = e.state_at(0.0)
    pts, v = st["pts"].impl.cloud(st)
    np.testing.assert_allclose(pts[1], (1.0, 0.5))
    np.testing.assert_allclose(v, (0, 0.5, 0.9))
    e2 = _scene(source="points", points=((1.0, 0.5),), pos=(1.0, -1.0, 0.0), scale=2.0)
    st2 = e2.state_at(0.0)
    np.testing.assert_allclose(st2["pts"].impl.cloud(st2)[0][0], (3.0, 0.0))


def test_point_cloud_params_are_checked():
    e = Scene("bad", passes=None).engine
    for params, msg in [({"source": "points", "points": [[0, 0]], "values": [0, 1]}, "values"),
                        ({"coefficients": [0, 0]}, "nonzero"),
                        ({"source": "attractor", "kind": "lorenz", "constants": [1, 2]}, "constants"),
                        ({"point_size": 9}, "point_size")]:  # fmt: skip
        r = e.apply({"op": "add_object", "object": {"id": "c", "type": "point_cloud",
                                                    "params": params}}, origin="script")  # fmt: skip
        assert not r.ok and msg in " ".join(i.message for i in r.errors), params


def test_the_same_seed_gives_the_same_points_and_picture():
    e1 = _scene(source="attractor", kind="clifford", count=5000, seed=7)
    e2 = _scene(source="attractor", kind="clifford", count=5000, seed=7)
    e3 = _scene(source="attractor", kind="clifford", count=5000, seed=8)

    def cloud_and_picture(e):
        st = e.state_at(0.0)
        pts, v = st["pts"].impl.cloud(st)
        lut = np.ones((4, 3), dtype=np.float32)
        img = splat_ref(Tile.full(64, 36), pts, lut[(v * 3).astype(int)], (0.0, 0.0), 4.5)
        return pts, img

    (p1, i1), (p2, i2), (p3, _) = map(cloud_and_picture, (e1, e2, e3))
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(i1, i2)
    assert not np.array_equal(p1, p3)


def test_presets_and_the_prism_look_set_the_palette_and_glow():
    s = Scene("look", look="prism", passes=None)
    s.add(cloud.PointCloud(id="pts"))
    obj = s.engine.document.objects[0]
    assert obj.style["glow"] == "strong" and obj.style["colormap"] == "spectrum"
    assert obj.params["exposure"] == pytest.approx(0.06)
    s2 = Scene("look2", look="prism", palette="gold", passes=None)
    s2.add(cloud.PointCloud(id="pts"))
    assert s2.engine.document.objects[0].style["colormap"] == "gold"
    plain = cloud.PointCloud(id="p", preset="plain").to_spec()
    assert plain.style["glow"] == "none"


def _pixels(e, pw=96, ph=54):
    """The field pixels the renderer would ask for, for the whole rectangle."""
    st = e.state_at(0.0)
    calls = []

    class Ctx:
        def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0):
            calls.append((pixels, glow))

    st["pts"].impl.draw(Ctx(), st, "color")
    ((pixels, glow),) = calls
    p = st["pts"].params
    return pixels(pw, ph, (p.size[0] / 2, p.size[1] / 2), p.size[1]), glow, st


def test_the_cloud_is_light_on_the_scene_background():
    e = _scene(degree=8, size=(4.5, 2.6))
    img, _, st = _pixels(e)
    bg = np.asarray(st.background[:3], dtype=np.float32)
    assert img.shape == (54, 96, 4)
    empty = img[..., 3] == 0
    assert empty.any() and (~empty).any()
    np.testing.assert_allclose(img[empty][:, :3], np.broadcast_to(bg, (int(empty.sum()), 3)))
    # light lands where the roots are: none in the middle (|z| < 1/2), some on the ring
    assert img[27, 48, 3] == 0 and img[..., 3].max() > 0.3


def test_brightness_does_not_depend_on_resolution():
    """``exposure`` is measured against the mean density at the current pixel size, so a
    small preview and a large render light the cloud the same."""
    e = _scene(degree=12, size=(4.5, 2.6), point_size=1.0)
    small, _, _ = _pixels(e, 192, 108)
    large, _, _ = _pixels(e, 384, 216)
    down = large.reshape(108, 2, 192, 2, 4).mean(axis=(1, 3))
    assert small[..., 3].mean() == pytest.approx(down[..., 3].mean(), rel=0.25)


@needs_gl
@pytest.mark.slow
def test_point_cloud_golden_storyboard():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    e = runpy.run_path(str(ROOT / "examples" / "point_cloud.py"), run_name="x")["s"].engine
    assert [i for i in e.lint() if i.severity == "error"] == []
    sheet = e.storyboard([1.5, 6.0, 9.0, 12.0], size=(320, 180))["image"]
    klass = e.pass_runner().renderer_class()
    path = GOLDEN / klass / "point_cloud.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {klass} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.98


# ---------------------------------------------------------------- motion, twinkle, white


def _matched_shift(a: np.ndarray, b: np.ndarray, degree: int) -> np.ndarray:
    """Per root, how far it moved: the roots of each polynomial matched one to one (the
    solver gives them in no set order) so the total move is least."""
    from scipy.optimize import linear_sum_assignment

    out = []
    for k in range(len(a) // degree):
        x, y = a[k * degree : (k + 1) * degree], b[k * degree : (k + 1) * degree]
        cost = np.abs(x[:, None] - y[None, :])
        r, c = linear_sum_assignment(cost)
        out.append(cost[r, c])
    return np.concatenate(out)


def test_twist_moves_every_root_continuously_and_keeps_the_sample():
    deg = 8
    z0, v0 = clm.polynomial_roots(deg, count=60, gen=rng("t", 1), twist=0.02)
    z1, v1 = clm.polynomial_roots(deg, count=60, gen=rng("t", 1), twist=0.0205)
    np.testing.assert_array_equal(v0, v1)  # the same polynomials, the same colours
    step = _matched_shift(z0, z1, deg)
    assert step.max() < 0.05 and step.mean() > 1e-4  # they glide: small steps, but they move
    # each root is a root of its twisted polynomial: a_d from (-1, e^(2 pi i twist))
    w = np.exp(2j * np.pi * 0.02)
    all_z, _ = clm.polynomial_roots(deg, twist=0.02)
    for i in (0, 5, 200):
        a = [(-1.0, w)[(i >> d) & 1] for d in range(deg)]
        assert np.abs(np.polyval([1.0, *a[::-1]], all_z[i * deg : (i + 1) * deg])).max() < 1e-9
    # no twist and a whole turn: the classic real cloud
    plain, _ = clm.polynomial_roots(deg)
    turned, _ = clm.polynomial_roots(deg, twist=1.0)
    np.testing.assert_array_equal(plain, turned)


def test_roots_solve_the_same_on_one_thread_and_many(monkeypatch):
    many, _ = clm.polynomial_roots(12, twist=0.03)
    monkeypatch.setattr(clm, "ROOT_THREADS", 1)
    one, _ = clm.polynomial_roots(12, twist=0.03)
    np.testing.assert_array_equal(many, one)


def test_a_tweened_twist_glides_the_cloud_and_state_stays_pure():
    s = Scene("tw", passes=None)
    s.add(cloud.PointCloud(id="pts", degree=8, size=(4.5, 2.6)))
    s.at(0.0).to("pts.twist", 0.1, duration=2.0)
    e = s.engine

    def pts(t):
        st = e.state_at(t)
        return st["pts"].impl.cloud(st)[0]

    a, b = pts(1.0), pts(1.0 + 1 / 30)
    np.testing.assert_array_equal(a, pts(1.0))  # the same t, the same points
    za, zb = a[:, 0] + 1j * a[:, 1], b[:, 0] + 1j * b[:, 1]
    step = _matched_shift(za, zb, 8)
    assert step.mean() > 1e-4 and step.max() < 0.08  # a frame apart: every root a small step
    sem = e.inspect(1.0, id="pts")["objects"][0]["semantic"]
    assert 0 < sem["twist"] < 0.1


def _scene_life(life, tags=(), **kw):
    s = Scene("lf", passes=None, life=life)
    s.set("camera.height", 4.5)
    s.add(cloud.PointCloud(id="pts", degree=8, size=(4.5, 2.6), twinkle=0.8, tags=tags, **kw))
    return s.engine


def _pixels_at(e, t, pw=96, ph=54, zoom=1.0, at=(0.0, 0.0)):
    st = e.state_at(t)
    calls = []

    class Ctx:
        def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0):
            calls.append(pixels)

    st["pts"].impl.draw(Ctx(), st, "color")
    p = st["pts"].params
    center = (p.size[0] / 2 + at[0], p.size[1] / 2 + at[1])  # quad coordinates
    return calls[0](pw, ph, center, p.size[1] / zoom)


def test_twinkle_flickers_points_purely_and_only_when_alive():
    e = _scene_life("lively")
    a, b = _pixels_at(e, 1.0), _pixels_at(e, 2.0)
    np.testing.assert_array_equal(a, _pixels_at(e, 1.0))  # a pure function of t
    assert np.abs(a - b).max() > 0.01  # it moves
    assert a[..., 3].mean() == pytest.approx(b[..., 3].mean(), rel=0.15)  # not a pulse
    assert e.state_at(1.0)["pts"].params.twinkle == pytest.approx(0.8)
    for still in (_scene_life("off"), _scene_life("lively", tags=("exact",))):
        assert still.state_at(1.0)["pts"].params.twinkle == 0.0
        np.testing.assert_array_equal(_pixels_at(still, 1.0), _pixels_at(still, 2.0))


def test_white_appears_only_where_the_light_is_dense():
    g = np.random.default_rng(5)
    pile = np.zeros((4000, 2))  # one dense spot in the middle
    sparse = g.uniform(-1.5, 1.5, (300, 2))  # and a thin scatter round it
    pts = np.r_[pile, sparse]
    red = np.tile([[1.0, 0.05, 0.05]], (len(pts), 1)).astype(np.float32)
    kw = {"center": (0.0, 0.0), "height": 3.0, "gain": 2.0}
    plain = splat_ref(Tile.full(40, 40), pts, red, **kw)
    white = splat_ref(Tile.full(40, 40), pts, red, white=0.8, **kw)
    a = white[..., 3]
    np.testing.assert_array_equal(a, plain[..., 3])  # white changes colour, never the light
    assert (white[..., :3] <= a[..., None] + 1e-6).all()  # rgb stays within a: no flare
    core = np.unravel_index(int(np.argmax(a)), a.shape)
    assert white[core][1] / a[core] > 0.7  # the dense core runs white-hot
    thin = (a > 0) & (a < 0.3)
    assert thin.any()
    assert (white[thin][:, 1] / white[thin][:, 3]).max() < 0.1  # thin light keeps its colour


def test_zooming_in_neither_washes_out_nor_whitens_the_cloud():
    """The camera fall of the example: light per point grows only as zoom^ZOOM_LIGHT, so
    the close-up is no brighter than the whole cloud and its white-hot share fades."""
    e = _scene_life("off", white=0.8, exposure=0.06)

    def white_share(img):
        a = img[..., 3]
        lit = a > 0.02
        return float((img[..., :3].min(-1)[lit] / a[lit]).mean()), float(a[lit].mean())

    whole_white, whole_a = white_share(_pixels_at(e, 0.0, 192, 108))
    close_white, close_a = white_share(_pixels_at(e, 0.0, 192, 108, zoom=8.0, at=(0.0, 1.0)))
    assert close_a <= whole_a * 1.25 and close_a < 0.6
    assert close_white <= whole_white
