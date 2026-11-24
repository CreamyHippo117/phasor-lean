"""The kernels' CUDA source compiled for every CPU core (``render/cpu``).

Every kernel case in ``tests/test_kernels.py`` runs on the CPU backend and is compared with
its NumPy reference (as strictly as the serial emulator), and must give the same bits at 1,
2 and all threads. Skipped cleanly where no g++ or clang++ is found (Windows CI)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from phasor.render import gpu
from phasor.render.cpu import backend as cpu
from phasor.render.gpu import KernelError, Tile
from phasor.render.gpu.registry import active_backend, get
from tests.test_kernels import ALL_CASES, CASES, _close

needs_cpu = pytest.mark.skipif(cpu.find_compiler() is None, reason="no g++ or clang++")
THREADS = sorted({1, 2, cpu.cores()})


@needs_cpu
@pytest.mark.parametrize(("name", "tile", "params"), ALL_CASES, ids=lambda v: str(v)[:24])
def test_every_kernel_on_the_cpu_matches_its_reference_at_any_thread_count(name, tile, params):
    want = get(name).reference(tile, **params)
    got = [gpu.run(name, tile, backend="cpu", threads=n, **params) for n in THREADS]
    assert all(isinstance(g, np.ndarray) for g in got)  # host arrays: nothing to copy out
    _close(name, got[0], want, strict=True)
    for g in got[1:]:  # bit for bit: no result depends on how many threads ran it
        np.testing.assert_array_equal(g, got[0], err_msg=name)


@needs_cpu
def test_splat_sums_are_the_same_bits_however_the_threads_interleave():
    """Many points on few pixels: the fixed-point atomic sums meet constantly."""
    rng = np.random.default_rng(11)
    pts = rng.normal(0.0, 0.05, (60_000, 2))
    params = {"points": pts, "colors": rng.random((60_000, 3)).astype(np.float32),
              "center": (0.0, 0.0), "height": 1.0, "sigma": 1.2, "curve": False}  # fmt: skip
    tile = Tile.full(48, 32)
    first = gpu.run("splat", tile, backend="cpu", threads=cpu.cores(), **params)
    for n in (*THREADS, cpu.cores()):
        np.testing.assert_array_equal(gpu.run("splat", tile, backend="cpu", threads=n, **params),
                                      first)  # fmt: skip
    want = get("splat").reference(tile, **params)  # raw sums in the hundreds: relative
    np.testing.assert_allclose(first, want, rtol=1e-5, atol=1e-6)


@needs_cpu
def test_auto_runs_on_the_cpu_without_cuda_and_cpu_can_be_demanded():
    tile, params = CASES["tonemap"][0]
    if gpu.default_gpu() is None:
        assert active_backend() == "cpu"
        np.testing.assert_array_equal(gpu.run("tonemap", tile, **params),
                                      gpu.run("tonemap", tile, backend="cpu", **params))  # fmt: skip
    with pytest.raises(KernelError, match="at least 1"):
        gpu.run("tonemap", tile, backend="cpu", threads=0, **params)


def test_without_a_compiler_auto_is_the_reference_and_cpu_is_refused(monkeypatch):
    monkeypatch.setattr(cpu, "find_compiler", lambda: None)
    tile, params = CASES["escape"][0]
    want = get("escape").reference(tile, **params)
    if gpu.default_gpu() is None:
        assert active_backend() == "reference"
        np.testing.assert_array_equal(gpu.run("escape", tile, **params), want)
    with pytest.raises(KernelError, match="no C\\+\\+ compiler"):
        gpu.run("escape", tile, backend="cpu", **params)


def test_thread_count_comes_from_set_threads_then_the_environment_then_the_cores(monkeypatch):
    monkeypatch.delenv("PHASOR_THREADS", raising=False)
    cpu.set_threads(None)
    assert cpu.threads() == cpu.cores() >= 1
    monkeypatch.setenv("PHASOR_THREADS", "3")
    assert cpu.threads() == 3
    cpu.set_threads(2)
    try:
        assert cpu.threads() == 2
    finally:
        cpu.set_threads(None)
    monkeypatch.setenv("PHASOR_THREADS", "many")
    with pytest.raises(KernelError, match="PHASOR_THREADS"):
        cpu.threads()
    with pytest.raises(KernelError):
        cpu.set_threads(0)


def test_each_entry_gets_a_parallel_wrapper_and_the_source_is_unchanged():
    """The wrapper's own names may not shadow a kernel's (a kernel parameter ``b`` once
    read the block index: found by ``phasor doctor``'s probe)."""
    from phasor.render.gpu.kernels.post import SOURCE

    tu = cpu.translation_unit(SOURCE)
    assert SOURCE in tu  # the same source string, not a copy edited for the CPU
    for entry in ("phasor_tonemap", "phasor_gauss", "phasor_bright", "phasor_resize"):
        assert f"void phasor_cpu__{entry}(long long phasor_blocks, int phasor_threads" in tu
    assert "phasor_gauss(n, w, h, axis, radius, sigma, in, out);" in tu
    with pytest.raises(KernelError, match="reserved"):
        cpu.translation_unit('extern "C" __global__ void k(int n, float* phasor_t) { }')


@needs_cpu
def test_a_float_atomic_add_does_not_compile_and_auto_falls_back(tmp_path, monkeypatch):
    """Float atomics would make sums depend on thread order: refused on the CPU."""
    monkeypatch.setenv("PHASOR_CPU_CACHE", str(tmp_path))
    host = cpu.default_cpu()
    src = ('extern "C" __global__ void phasor_t(int n, float* a)\n'
           "{ int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) atomicAdd(a, 1.0f); }")  # fmt: skip
    with pytest.raises(cpu.CompileError, match="does not compile"):
        host.compile(src, "phasor_t")


@needs_cpu
def test_built_kernels_are_cached_on_disk_by_source_and_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_CPU_CACHE", str(tmp_path))
    host = cpu.default_cpu()
    src = ('extern "C" __global__ void phasor_axpb(int n, float a, const float* x, float* y)\n'
           "{ int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a * x[i] + 1.0f; }")  # fmt: skip
    fn = host.compile(src, "phasor_axpb")
    libs = sorted(tmp_path.glob(f"*{cpu.LIB_SUFFIX}"))
    assert len(libs) == 1 and not list(tmp_path.glob("*.cpp"))  # no build leftovers
    x = np.arange(1000, dtype=np.float32)
    y = np.zeros_like(x)
    host.launch(fn, 1000, (np.int32(1000), np.float32(2.0), x, y))
    np.testing.assert_array_equal(y, 2 * x + 1)
    cpu._LIBS.clear()  # a new process: loads the file, compiles nothing
    calls = []
    monkeypatch.setattr(cpu.subprocess, "run", lambda *a, **k: calls.append(a))
    host.compile(src, "phasor_axpb")
    assert calls == [] and sorted(tmp_path.glob(f"*{cpu.LIB_SUFFIX}")) == libs


@needs_cpu
def test_launch_refuses_what_the_card_would_refuse():
    host = cpu.default_cpu()
    src = ('extern "C" __global__ void phasor_cp(int n, const float* x, float* y)\n'
           "{ int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = x[i]; }")  # fmt: skip
    fn = host.compile(src, "phasor_cp")
    x = np.arange(20, dtype=np.float32)
    with pytest.raises(KernelError, match="contiguous"):
        host.launch(fn, 10, (np.int32(10), x[::2], np.zeros(10, np.float32)))
    with pytest.raises(KernelError, match="scalar"):
        host.launch(fn, 10, (10, x, np.zeros(20, np.float32)))
    assert host.cp.asarray(x[::2]).flags.c_contiguous  # asarray copies like a host->card copy
    with pytest.raises(AttributeError, match="launchers may use only"):
        host.cp.linspace  # noqa: B018
    with pytest.raises(KernelError, match="no kernel entry"):
        host.compile(src, "phasor_missing")


def test_only_g_plus_plus_and_clang_plus_plus_are_used(monkeypatch, tmp_path):
    assert cpu._kind("/usr/bin/g++-13") == "g++"
    assert cpu._kind("C:/mingw/bin/g++.exe") == "g++"
    assert cpu._kind("/usr/bin/clang++") == "clang++"
    assert cpu._kind("cl.exe") is None and cpu._kind("/usr/bin/icpx") is None
    cl = tmp_path / "cl"
    cl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cl.chmod(0o755)
    monkeypatch.setenv("PHASOR_CXX", str(cl))
    cpu.find_compiler.cache_clear()
    try:
        assert cpu.find_compiler() is None  # MSVC's name: reported, never used
    finally:
        monkeypatch.delenv("PHASOR_CXX")
        cpu.find_compiler.cache_clear()


# ---------------------------------------------------------------- doctor, workers, threads


@needs_cpu
def test_doctor_reports_the_compiler_and_threads_and_tier_b_says_cpu_kernels():
    from phasor.render.doctor import (
        TIER_TEXT,
        Check,
        DoctorReport,
        check_cpu_kernels,
        format_report,
        tier_text,
    )

    c = check_cpu_kernels()
    assert c.ok, c.detail
    assert c.data["compiler"] in cpu.SUPPORTED and c.data["version"] in c.detail
    assert c.data["threads"] == cpu.threads() and f"{cpu.threads()} threads" in c.detail
    report = DoctorReport([Check("opengl", True, ""), c], "B", None)
    assert "kernels run on the CPU" in tier_text(report) and c.data["compiler"] in tier_text(report)
    assert tier_text(report) in format_report(report)
    without = DoctorReport([Check("opengl", True, "")], "B", None)
    assert tier_text(without) == TIER_TEXT["B"]  # no compiler: NumPy, as before


def test_doctor_without_a_compiler_says_numpy_and_names_msvc(monkeypatch):
    from phasor.render.doctor import check_cpu_kernels

    monkeypatch.setattr(cpu, "find_compiler", lambda: None)
    monkeypatch.setattr(cpu, "unsupported_compilers", lambda: ["cl"])
    c = check_cpu_kernels()
    assert c.ok is None and "NumPy" in c.detail and "cl found, not supported" in c.detail


def test_without_cuda_workers_times_threads_fit_the_cores(monkeypatch):
    from phasor import scene
    from phasor.render.gpu import registry

    monkeypatch.delenv("PHASOR_THREADS", raising=False)
    monkeypatch.setattr(registry, "active_backend", lambda: "cpu")
    for cores in (1, 2, 4, 6, 16, 64):
        monkeypatch.setattr(cpu, "cores", lambda c=cores: c)
        monkeypatch.setattr(scene.os, "cpu_count", lambda c=cores: c)
        for workers in (scene.default_workers(), 1, 2, 3, 4):
            t = scene.default_threads(workers)
            assert t >= 1 and (workers * t <= cores or t == 1), (cores, workers, t)
        assert scene.default_threads(1) == cores
    monkeypatch.setenv("PHASOR_THREADS", "3")
    assert scene.default_threads(2) is None  # every process reads the variable itself
    monkeypatch.delenv("PHASOR_THREADS")
    for other in ("cuda", "reference"):  # the card, or NumPy on one core: no threads to share
        monkeypatch.setattr(registry, "active_backend", lambda o=other: o)
        assert scene.default_threads(2) is None


def test_render_takes_a_threads_option():
    from phasor.cli import build_parser

    args = build_parser().parse_args(["render", "x.py", "--workers", "2", "--threads", "3"])
    assert (args.workers, args.threads) == (2, 3)
    assert build_parser().parse_args(["render", "x.py"]).threads is None


@pytest.mark.gl
@needs_cpu
def test_exit_demo_prism_primes_on_the_cpu_is_the_reference_picture(monkeypatch):
    """Milestone 14's exit demo, in a test: a finished prism_primes frame (bloom, tone map,
    lens, sRGB bytes) through the CPU kernels against the same frame on the NumPy
    references. (Its software golden is checked in tests/test_prism.py, on ``auto``.)"""
    from tests.test_prism import _demo

    if gpu.default_gpu() is not None:
        pytest.skip("CUDA present: auto would not use the CPU backend")
    e = _demo()["s"].engine
    on_cpu = e.frame(12.0, (320, 180), output="srgb8")
    monkeypatch.setattr(cpu, "find_compiler", lambda: None)
    assert active_backend() == "reference"
    on_numpy = e.frame(12.0, (320, 180), output="srgb8")
    diff = np.abs(on_cpu.astype(np.int16) - on_numpy)
    assert diff.max() <= 1 and (diff > 0).mean() < 1e-3, (int(diff.max()), (diff > 0).mean())


@needs_cpu
def test_a_library_that_builds_but_will_not_load_falls_back_and_is_not_retried(
    tmp_path, monkeypatch
):
    """Windows: a DLL whose runtime (libstdc++, libgomp) is not found raises OSError on load.
    ``auto`` must still give the picture (the reference); ``cpu`` must say why."""
    monkeypatch.setenv("PHASOR_CPU_CACHE", str(tmp_path))
    monkeypatch.setattr(cpu, "_LIBS", {})
    monkeypatch.setattr(cpu, "_FAILED", {})
    tries = []

    def broken(path, *a, **k):
        tries.append(path)
        raise OSError("libgomp-1.dll not found")

    monkeypatch.setattr(cpu.ctypes, "CDLL", broken)
    tile, params = CASES["vignette"][0]
    from phasor.render.gpu import registry

    monkeypatch.setattr(registry, "_WARNED", set())
    with pytest.warns(RuntimeWarning, match="runs on NumPy"):
        got = gpu.run("vignette", tile, **params)
    np.testing.assert_array_equal(got, get("vignette").reference(tile, **params))
    with pytest.raises(cpu.CompileError, match="does not load"):
        gpu.run("vignette", tile, backend="cpu", **params)
    assert len(tries) == 1  # the failure is remembered: no rebuild, no reload per call


@needs_cpu
def test_a_corrupt_cached_library_is_rebuilt_once(tmp_path):
    """A truncated cache entry (power cut, disk error) must heal, not leave every later run on
    NumPy until the person deletes the cache by hand (found by the Milestone 14 audit).
    Each run is a new process: dlopen hands back an already-loaded path without reading it."""
    import subprocess
    import sys

    from phasor.core.proc import NO_WINDOW

    code = (
        "import sys; from tests.test_kernels import CASES; from phasor.render import gpu\n"
        "t, p = CASES['vignette'][0]\n"
        "import numpy as np; sys.stdout.write(gpu.run('vignette', t, backend='cpu', **p).tobytes().hex())"
    )
    env = {**os.environ, "PHASOR_CPU_CACHE": str(tmp_path)}

    def run() -> str:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env=env, cwd=Path(__file__).parent.parent, timeout=300,
                              creationflags=NO_WINDOW)  # fmt: skip
        assert done.returncode == 0, done.stderr[-800:]
        return done.stdout

    want = run()
    libs = list(tmp_path.glob(f"*{cpu.LIB_SUFFIX}"))
    assert libs, "the build should have been cached"
    for lib in libs:
        lib.write_bytes(lib.read_bytes()[:40])  # truncated: will not load
    assert run() == want  # no CompileError: it rebuilt
    assert any(lib.stat().st_size > 1000 for lib in libs)  # rebuilt, not left truncated
