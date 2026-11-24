"""CUDA on a real NVIDIA card.

Every test here needs a CUDA device and skips itself without one, except the fallback tests
at the end, which run everywhere. ``test_kernels.py`` already holds each kernel to its NumPy
reference on the GPU; this file covers what that does not: whole frames, the hand-off to
OpenGL, repeatability, precision of a deep zoom, tiling, speed and memory over a long render.

    uv run pytest tests/test_gpu.py                       # on the NVIDIA machine
    PHASOR_LONG_RENDER=2000 uv run pytest tests/test_gpu.py -k memory   # the full 2,000 frames
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from phasor.render import gpu
from phasor.render.gpu import Tile, get, registry
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
CASES = __import__("tests.test_kernels", fromlist=["ALL_CASES"]).ALL_CASES

needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _host(a):
    from phasor.render.gpu.bridge import to_host

    return to_host(a)


# ---------------------------------------------------------------- the device is really used


@pytest.mark.gpu
def test_auto_picks_cuda_and_the_answer_lives_on_the_card():
    import cupy

    assert gpu.default_gpu() is not None
    params = {"center": (-0.6, 0.0), "height": 2.6, "max_iter": 64}
    out = gpu.run("escape", Tile.full(64, 36), **params)
    assert isinstance(out, cupy.ndarray)
    from tests.test_kernels import _close

    _close("escape", _host(out), get("escape").reference(Tile.full(64, 36), **params),
           strict=False)  # fmt: skip


# ---------------------------------------------------------------- the same frame twice


@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tile", "params"), CASES, ids=lambda v: str(v)[:24])
def test_the_same_kernel_twice_gives_identical_bits(name, tile, params):
    a = _host(gpu.run(name, tile, backend="cuda", **params)).copy()
    b = _host(gpu.run(name, tile, backend="cuda", **params))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------- tiles and supersampling


@pytest.mark.gpu
@pytest.mark.parametrize("supersample", [1, 2])
def test_a_tiled_render_on_the_card_equals_one_tile(supersample):
    f = gpu.ComplexFunction.parse("1/(z**2 + 1)")
    kw = {"function": f, "center": (0.0, 0.0), "height": 4.5, "backend": "cuda",
          "supersample": supersample}  # fmt: skip
    whole = gpu.render("domain", 320, 180, budget=1 << 34, **kw)
    small = 320 * 7 * 16 * supersample**2  # about seven rows per tile: many tiles
    tiled = gpu.render("domain", 320, 180, budget=small, **kw)
    np.testing.assert_array_equal(whole, tiled)


@pytest.mark.gpu
@pytest.mark.slow
def test_a_4k_frame_fits_in_the_vram_budget():
    from phasor.render.gpu.budget import free_pools, free_vram

    before = free_vram()
    out = gpu.render("escape", 3840, 2160, center=(-0.6, 0.0), height=2.6, max_iter=500,
                     backend="cuda")  # fmt: skip
    assert out.shape == (2160, 3840) and (out < 0).any() and (out > 0).any()
    free_pools()
    assert free_vram() >= before * 0.9  # nothing big stays behind


# ---------------------------------------------------------------- deep zoom precision


@pytest.mark.gpu
def test_a_1e_30_view_on_the_card_matches_exact_iteration():
    from tests.test_deep_zoom import SPIRAL, _deep, _truth

    tile = Tile.full(12, 7)
    for center in [(0.0, 0.0), (3e-31, -2e-31)]:
        c, bits, ref = _deep(SPIRAL, tile, center, 1.8e-30, 3000)
        from phasor.domains import fractal_math as fm

        orbit = fm.reference_orbit(c, bits, 3000)
        got = _host(gpu.run("deep", tile, backend="cuda", center=center, height=1.8e-30,
                            orbit=orbit, max_iter=3000, rebase=True))  # fmt: skip
        want = _truth(c, bits, tile, center, 1.8e-30, 3000, range(tile.n))
        assert np.abs(got.ravel() - want).max() < 1e-3, center
        np.testing.assert_allclose(got, ref, atol=1e-3)


@pytest.mark.gpu
def test_float64_on_the_card_is_real_float64():
    """Some cards run doubles slowly, none may run them as floats: at a 1e-12 view a float32
    plane is one flat color, a float64 one is not."""
    params = {"center": (-0.7756837680090538, 0.1364673682946901), "height": 1e-11,
              "max_iter": 3000}  # fmt: skip
    tile = Tile.full(48, 27)
    f32 = _host(gpu.run("escape", tile, backend="cuda", precision="f32", **params))
    f64 = _host(gpu.run("escape", tile, backend="cuda", precision="f64", **params))
    assert len(np.unique(f32)) < 10 < 500 < len(np.unique(f64))
    want = get("escape").reference(tile, precision="f64", **params)
    assert (np.abs(f64 - want) < 1e-2).mean() > 0.98


# ---------------------------------------------------------------- whole frames


def _plane_scene():
    from phasor import Scene, Text, cx

    s = Scene(title="gpu frame")
    s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(8, 4.5)))
    s.add(cx.Fractal(id="set", pos=(2.4, -1.2), size=(3.2, 1.8), max_iter=200))
    s.add(Text(id="title", text="$f(z) = 1/(z^2+1)$", pos=(-2.5, 1.8)))
    return s.engine


@pytest.mark.gpu
@needs_gl
def test_a_whole_frame_on_the_card_matches_the_numpy_frame(monkeypatch):
    from tests.ssim import ssim

    on_card = _plane_scene().still(0.0, (640, 360))
    monkeypatch.setattr(registry, "default_gpu", lambda: None)
    on_cpu = _plane_scene().still(0.0, (640, 360))
    assert ssim(on_card, on_cpu) >= 0.995
    assert np.abs(on_card.astype(int) - on_cpu.astype(int)).mean() < 1.0


@pytest.mark.gpu
@needs_gl
def test_the_same_frame_twice_on_the_card_is_identical():
    a = _plane_scene().frame(0.0, (480, 270))
    b = _plane_scene().frame(0.0, (480, 270))
    np.testing.assert_array_equal(a, b)


@pytest.mark.gpu
@needs_gl
def test_cuda_frames_reach_opengl_unchanged():
    import cupy

    from phasor.render.glcontext import on_gl_thread
    from phasor.render.gpu.bridge import CopyBridge
    from phasor.render.passes import default_renderer

    ctx = default_renderer().ctx
    bridge = CopyBridge(ctx)

    def through(frame):
        with ctx:
            tex = bridge.upload(cupy.asarray(frame))
            return tex.size, bridge.read_back()

    rng = np.random.default_rng(1)
    for w, h in [(64, 36), (333, 101), (64, 36)]:  # an odd size, and back (texture reuse)
        frame = rng.random((h, w, 4), dtype=np.float32) * 4  # HDR values above 1 survive
        size, back = on_gl_thread(through, frame)
        assert size == (w, h)
        np.testing.assert_allclose(back, frame, rtol=1e-3, atol=1e-3)  # float16 on the way


# ---------------------------------------------------------------- speed


def _ms(fn, repeats=3):
    fn()  # compile and warm up
    import cupy

    times = []
    for _ in range(repeats):
        cupy.cuda.Device().synchronize()
        t0 = time.perf_counter()
        fn()
        cupy.cuda.Device().synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("name", ["escape", "domain", "gauss"])
def test_the_card_is_faster_than_numpy(name):
    from phasor.render.bench import _cases

    w, h = 1920, 1080
    params = _cases(w, h)[name]
    tile = Tile.full(w, h)
    card = _ms(lambda: gpu.run(name, tile, backend="cuda", **params))
    cpu = _ms(lambda: gpu.run(name, tile, backend="reference", **params), repeats=1)
    print(f"{name}: card {card * 1000:.1f} ms, numpy {cpu * 1000:.1f} ms, x{cpu / card:.0f}")
    assert card < cpu


@pytest.mark.gpu
def test_bench_runs_on_the_card():
    from phasor.render.bench import run_bench

    r = run_bench(320, 180, repeats=1)
    assert r["backend"] == "cuda" and r["peak_vram_bytes"] > 0
    assert set(r["kernels"]) >= set(gpu.registered())


# ---------------------------------------------------------------- memory over a long render


@pytest.mark.gpu
@needs_gl
@pytest.mark.slow
def test_memory_stays_flat_over_a_long_render():
    """B6: VRAM and host memory flat over a long render (300 frames by default; the full
    2,000 with PHASOR_LONG_RENDER=2000). Every frame differs, so no cache can hide a leak.
    Half the frames are motion-blurred."""
    import cupy

    from phasor import Scene, cx
    from phasor.render.gpu.budget import free_pools, free_vram

    frames = int(os.environ.get("PHASOR_LONG_RENDER", "300"))
    s = Scene(title="long render")
    d = s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(8, 4.5)))
    s.add(cx.Fractal(id="set", pos=(2.4, -1.2), size=(3.2, 1.8), max_iter=200))
    s.at(0.0).to(f"{d.id}.hue", 1.0, duration=frames / 30)
    e = s.engine

    def sample():
        free_pools()
        return free_vram(), _rss()

    vram, rss = [], []
    for k in range(frames):
        if k % 2:  # every other frame motion-blurred: the blur sum on the card must not leak
            e.frame_blurred(k / 30, 30, size=(480, 270), output="srgb8")
        else:
            e.frame(k / 30, (480, 270))
        if k in (frames // 10, frames - 1):
            v, r = sample()
            vram.append(v)
            rss.append(r)
    assert cupy.get_default_memory_pool().used_bytes() < 64 * 1024**2
    assert vram[1] >= vram[0] - 64 * 1024**2, f"VRAM fell by {(vram[0] - vram[1]) >> 20} MB"
    if rss[0]:
        assert rss[1] <= rss[0] + 200 * 1024**2, f"host memory grew {(rss[1] - rss[0]) >> 20} MB"


def _rss() -> int | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        return None


# ---------------------------------------------------------------- no card, no driver (runs everywhere)


def _child(code: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=300, cwd=ROOT, env={**os.environ, **env})  # fmt: skip


FALLBACK = """
import numpy as np
from phasor.render import gpu
assert gpu.default_gpu() is None, "a GPU was found"
f = gpu.ComplexFunction.parse("1/(z**2 + 1)")
out = gpu.render("domain", 64, 36, function=f, center=(0, 0), height=4.5)
assert out.shape == (36, 64, 4) and np.isfinite(out).all()
from phasor.render.doctor import run_doctor
r = run_doctor()
assert r.tier in ("B", "C"), r.tier
print("ok", r.tier)
"""


def test_hidden_card_falls_back_to_numpy():
    r = _child(FALLBACK, CUDA_VISIBLE_DEVICES="-1")  # Windows drops a variable set to ""
    assert r.returncode == 0 and r.stdout.startswith("ok"), r.stderr[-2000:]


def test_broken_cupy_falls_back_to_numpy():
    """A missing or broken driver makes ``import cupy`` or its first call fail."""
    code = "import sys; sys.modules['cupy'] = None\n" + FALLBACK
    r = _child(code)
    assert r.returncode == 0 and r.stdout.startswith("ok"), r.stderr[-2000:]
