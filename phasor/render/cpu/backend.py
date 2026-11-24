"""Kernels on every CPU core, from the same CUDA source.

A machine without CUDA (tier B) used to run each kernel's NumPy reference on one core. This
backend compiles the kernel's *own* CUDA C string (there is still one source per kernel, in
``render/gpu/kernels/``) with a C++ compiler found at run time, g++ or clang++, and runs its
thread indices on every core:

- ``__global__`` and ``__device__`` are defined away; ``blockIdx``/``threadIdx`` are
  per-thread globals and ``blockDim`` is the CUDA block size (``registry.BLOCK``).
- Each ``extern "C" __global__`` entry gets a generated wrapper that runs the blocks in
  parallel (OpenMP where the compiler has it, else ``std::thread``) and, in each block, the
  thread indices in order, past ``n`` too, like the card: kernels must return there.
- Integer ``atomicAdd`` is a real atomic. Integer sums do not depend on the order threads
  add in, so the splat kernel's fixed-point light is the same bits at any thread count.
  A float ``atomicAdd`` is deliberately missing: its sum would depend on thread order, so a
  kernel that uses one does not compile here (``CompileError``) and runs on its reference.
- No fused multiply-add (``-ffp-contract=off``) and no fast math, like the NumPy reference.

Built libraries are cached in the user's cache folder (``cache_dir()``), keyed by a hash of
the whole translation unit, the compiler and its version, and the flags; a second run, and
every render worker, loads them without compiling. No prebuilt binaries
ship. Windows builds a ``.dll`` with MinGW g++ or clang++: untested on Windows. MSVC
``cl`` is not supported (reported as such by ``phasor doctor``).

Threads: ``threads()`` is all the cores this process may use, or ``PHASOR_THREADS``, or what
``set_threads`` set (render workers: ``scene.parallel_frames``); ``CpuBackend(threads=n)``
and ``registry.run(..., threads=n)`` choose per call.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phasor.core.proc import NO_WINDOW
from phasor.render.gpu.registry import BLOCK, HEADER, PRECISIONS, KernelError

# libgomp's idle threads otherwise spin after each kernel, taking cores from llvmpipe and the
# Python draw step (measured: post 2x slower). libgomp reads this once, when it loads, so it
# is set here, before the OpenMP probe or any kernel library can load it.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

SUPPORTED = ("g++", "clang++")
# -O3, no fused multiply-add (the reference rounds every + and *), no fast math. Hidden
# symbols, so the kernel inlines into its wrapper's loop (only the wrappers are exported).
FLAGS = ("-shared", "-O3", "-ffp-contract=off", "-std=c++17", "-fvisibility=hidden",
         "-Werror=return-type")  # fmt: skip
LIB_SUFFIX = ".dll" if sys.platform == "win32" else ".so"

PRELUDE = (
    "#define PHASOR_CPU_BLOCK "
    + str(BLOCK)
    + "\n"
    + r"""
#include <math.h>
#include <atomic>
#include <thread>
#include <vector>
struct phasor_dim3 { unsigned x, y, z; };
static thread_local phasor_dim3 blockIdx, threadIdx;
static const phasor_dim3 blockDim = {PHASOR_CPU_BLOCK, 1, 1};
#define __global__
#define __device__
#define __forceinline__ inline
#ifdef _WIN32
#define PHASOR_CPU_EXPORT __declspec(dllexport)
#else
#define PHASOR_CPU_EXPORT __attribute__((visibility("default")))
#endif
// Integer atomics only: integer sums are the same whatever order the threads add in.
static inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) {
    return __atomic_fetch_add(a, v, __ATOMIC_RELAXED);
}
static inline unsigned atomicAdd(unsigned* a, unsigned v) {
    return __atomic_fetch_add(a, v, __ATOMIC_RELAXED);
}
static inline int atomicAdd(int* a, int v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }

// Run f(block) for every block on `threads` threads. Blocks are handed out a few at a time
// (fractal blocks differ a lot in cost); which thread runs a block never changes a result.
template <class F> static void phasor_cpu_parallel(long long blocks, int threads, F f) {
    if (threads < 1) threads = 1;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
    for (long long b = 0; b < blocks; b++) f(b);
#else
    std::atomic<long long> next(0);
    auto work = [&]() {
        for (;;) {
            long long b = next.fetch_add(4);
            if (b >= blocks) return;
            long long e = b + 4 < blocks ? b + 4 : blocks;
            for (; b < e; b++) f(b);
        }
    };
    std::vector<std::thread> pool;
    for (int k = 1; k < threads && k < blocks; k++) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
#endif
}
"""
)
_WRAPPER_NAMES = {"phasor_blocks", "phasor_threads", "phasor_block", "phasor_t"}
_ENTRY = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(([^)]*)\)')


class CompileError(KernelError):
    """The host compiler rejected a kernel source (or a float ``atomicAdd`` in it)."""


@dataclass(frozen=True)
class Compiler:
    path: str
    name: str  # "g++" or "clang++"
    version: str  # the first line of ``--version``
    openmp: bool


def _kind(path: str) -> str | None:
    base = Path(path).name.lower().removesuffix(".exe")
    if "clang++" in base:
        return "clang++"
    if "g++" in base:
        return "g++"
    return None


def cache_dir() -> Path:
    """Where built kernels live: ``PHASOR_CPU_CACHE``, else the user's cache folder."""
    if os.environ.get("PHASOR_CPU_CACHE"):
        return Path(os.environ["PHASOR_CPU_CACHE"])
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "phasor" / "cpu_kernels"


def unsupported_compilers() -> list[str]:
    """C++ compilers on PATH that this backend does not use (MSVC ``cl``)."""
    return [c for c in ("cl",) if shutil.which(c)]


@functools.cache
def find_compiler() -> Compiler | None:
    """g++ or clang++ (``PHASOR_CXX`` names one explicitly), with its version and whether it
    builds OpenMP; None when there is none (kernels then run on the NumPy reference)."""
    wanted = os.environ.get("PHASOR_CXX")
    candidates = [wanted] if wanted else list(SUPPORTED)
    for c in candidates:
        path = shutil.which(c)
        if path is None or _kind(path) is None:
            continue
        try:
            done = subprocess.run([path, "--version"], capture_output=True, text=True,
                                  timeout=30, creationflags=NO_WINDOW)  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode != 0:
            continue
        version = (done.stdout.strip().splitlines() or ["unknown"])[0]
        return Compiler(path, _kind(path) or c, version, _has_openmp(path, version))
    return None


def _has_openmp(path: str, version: str) -> bool:
    """Whether ``path`` builds and loads an OpenMP library; asked once per compiler version
    (the answer is kept next to the built kernels)."""
    d = cache_dir()
    key = hashlib.sha256(f"{path}\n{version}".encode()).hexdigest()[:16]
    note = d / f"openmp-{key}.json"
    try:
        return bool(json.loads(note.read_text(encoding="utf-8"))["openmp"])
    except (OSError, ValueError, KeyError):
        pass
    probe = '#include <omp.h>\nextern "C" int phasor_omp(void) { return omp_get_max_threads(); }\n'
    ok = False
    # (a loaded DLL cannot be deleted on Windows: leave it to the temp folder's cleanup)
    with tempfile.TemporaryDirectory(prefix="phasor_omp_", ignore_cleanup_errors=True) as tmp:
        src, lib = Path(tmp) / "probe.cpp", Path(tmp) / f"probe{LIB_SUFFIX}"
        src.write_text(probe, encoding="utf-8")
        try:
            done = subprocess.run([path, *_pic(), "-shared", "-fopenmp", str(src), "-o", str(lib)],
                                  capture_output=True, text=True, timeout=120,
                                  creationflags=NO_WINDOW)  # fmt: skip
            if done.returncode == 0:
                ok = _load(path, lib).phasor_omp() >= 1
        except (OSError, subprocess.TimeoutExpired):
            ok = False
    try:
        d.mkdir(parents=True, exist_ok=True)
        note.write_text(json.dumps({"openmp": ok, "compiler": version}), encoding="utf-8")
    except OSError:
        pass
    return ok


_DLL_DIRS: dict[str, Any] = {}  # Windows: the compiler's runtime DLLs, kept on the search path


def _load(compiler_path: str, lib: Path) -> ctypes.CDLL:
    """Load a built library. On Windows, Python does not search PATH for a DLL's own
    dependencies (libstdc++, libgomp, libwinpthread): the compiler's folder is added."""
    if sys.platform == "win32" and compiler_path not in _DLL_DIRS:
        folder = str(Path(compiler_path).parent)
        _DLL_DIRS[compiler_path] = os.add_dll_directory(folder)  # type: ignore[attr-defined]
    return ctypes.CDLL(str(lib))


def _pic() -> tuple[str, ...]:
    return () if sys.platform == "win32" else ("-fPIC",)


# ---------------------------------------------------------------- threads

_THREADS: int | None = None


def cores() -> int:
    """The cores this process may run on."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def threads() -> int:
    """Kernel threads per launch: ``set_threads``, else ``PHASOR_THREADS``, else every core."""
    if _THREADS is not None:
        return _THREADS
    env = os.environ.get("PHASOR_THREADS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            raise KernelError(f"PHASOR_THREADS must be a whole number, got {env!r}") from None
    return cores()


def set_threads(n: int | None) -> None:
    """Set this process's kernel thread count (None: back to the default)."""
    global _THREADS
    if n is not None and int(n) < 1:
        raise KernelError(f"threads must be at least 1, got {n}")
    _THREADS = None if n is None else int(n)


# ---------------------------------------------------------------- the backend


class _HostArrays:
    """The array calls launchers may use (those NumPy and CuPy share). As on the card,
    ``asarray`` gives a contiguous array: C reads the bytes in order."""

    float32, float64, int32, uint8, uint64 = np.float32, np.float64, np.int32, np.uint8, np.uint64

    @staticmethod
    def asarray(a, dtype=None):
        return np.ascontiguousarray(a, dtype=dtype)

    @staticmethod
    def ascontiguousarray(a, dtype=None):
        return np.ascontiguousarray(a, dtype=dtype)

    empty = staticmethod(np.empty)
    zeros = staticmethod(np.zeros)
    empty_like = staticmethod(np.empty_like)

    def __getattr__(self, name):
        raise AttributeError(f"launchers may use only asarray, ascontiguousarray, empty, zeros, "
                             f"empty_like and dtypes from cp, not cp.{name}")  # fmt: skip


_LIBS: dict[str, ctypes.CDLL] = {}  # loaded libraries, by key: shared by every CpuBackend
_FAILED: dict[str, str] = {}  # keys that did not build or load in this process: not retried
_SCALARS = {np.int32: ctypes.c_int, np.float32: ctypes.c_float, np.float64: ctypes.c_double}
_ARRAYS = (np.float32, np.float64, np.uint8, np.uint64, np.int32)


def translation_unit(source: str) -> str:
    """The C++ the compiler sees: the prelude, ``complex.cuh``, the kernel source unchanged,
    then a parallel wrapper ``phasor_cpu__<entry>`` per ``extern "C" __global__`` entry."""
    wrappers = []
    for name, params in _ENTRY.findall(source):
        params = " ".join(params.split())
        names = [re.findall(r"\w+", p)[-1] for p in params.split(",") if p.strip()]
        if set(names) & _WRAPPER_NAMES:
            raise KernelError(f"{name}: parameter names {sorted(_WRAPPER_NAMES)} are reserved")
        wrappers.append(
            f'extern "C" PHASOR_CPU_EXPORT void phasor_cpu__{name}('
            "long long phasor_blocks, int phasor_threads"
            f"{', ' + params if params else ''}) {{\n"
            "    phasor_cpu_parallel(phasor_blocks, phasor_threads,\n"
            "                        [&](long long phasor_block) {\n"
            "        blockIdx.x = (unsigned)phasor_block;\n"
            "        for (unsigned phasor_t = 0; phasor_t < PHASOR_CPU_BLOCK; phasor_t++) {\n"
            f"            threadIdx.x = phasor_t; {name}({', '.join(names)});\n"
            "        }\n    });\n}\n"
        )
    return PRELUDE + HEADER + source + "\n" + "".join(wrappers)


class CpuBackend:
    """What a launcher may use (``registry.Gpu``'s interface), on the host's cores."""

    cp = _HostArrays()

    def __init__(self, compiler: Compiler, threads: int | None = None) -> None:
        if threads is not None and int(threads) < 1:
            raise KernelError(f"threads must be at least 1, got {threads}")
        self.compiler = compiler
        self._threads = threads

    @property
    def threads(self) -> int:
        return self._threads if self._threads is not None else threads()

    def flags(self, precision: str) -> tuple[str, ...]:
        omp = ("-fopenmp",) if self.compiler.openmp else ("-pthread",)
        return (*_pic(), *FLAGS, *omp, *PRECISIONS[precision])

    def compile(self, source: str, entry: str, precision: str = "f32") -> Any:
        if f"void {entry}" not in source:
            raise KernelError(f"no kernel entry {entry!r} in the source")
        return getattr(self._library(source, precision), f"phasor_cpu__{entry}")

    def _library(self, source: str, precision: str) -> ctypes.CDLL:
        code = translation_unit(source)
        flags = self.flags(precision)
        c = self.compiler
        key = hashlib.sha256("\n".join([code, c.path, c.version, *flags]).encode()).hexdigest()[:24]
        if key in _LIBS:
            return _LIBS[key]
        if key in _FAILED:
            raise CompileError(_FAILED[key])
        try:
            _LIBS[key] = self._build_and_load(key, code, flags)
        except CompileError as e:
            _FAILED[key] = str(e)
            raise
        return _LIBS[key]

    def _build_and_load(self, key: str, code: str, flags: tuple[str, ...]) -> ctypes.CDLL:
        c = self.compiler
        d = cache_dir()
        lib = d / f"{key}{LIB_SUFFIX}"
        for attempt in (0, 1):
            cached = lib.exists()
            if not cached:
                self._compile(key, code, flags, d, lib)
            try:
                return _load(c.path, lib)
            except OSError as e:
                if cached and attempt == 0:  # a truncated or corrupt cache entry: rebuild it once
                    lib.unlink(missing_ok=True)
                    continue
                # built but cannot load (a missing runtime DLL): use the reference
                raise CompileError(f"built kernel library does not load: {e}") from e
        raise AssertionError("unreachable")  # pragma: no cover

    def _compile(self, key: str, code: str, flags: tuple[str, ...], d: Path, lib: Path) -> None:
        c = self.compiler
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f"{key}-", suffix=LIB_SUFFIX, dir=d)
        os.close(fd)
        tmp = Path(tmp_name)
        src = tmp.with_suffix(".cpp")
        src.write_text(code, encoding="utf-8")
        try:
            try:
                done = subprocess.run([c.path, *flags, str(src), "-o", str(tmp)],
                                      capture_output=True, text=True, timeout=600,
                                      creationflags=NO_WINDOW)  # fmt: skip
            except (OSError, subprocess.TimeoutExpired) as e:
                raise CompileError(f"{c.name} did not run: {e}") from e
            if done.returncode != 0:
                raise CompileError(f"kernel does not compile with {c.name}:\n{done.stderr}")
            try:
                os.replace(tmp, lib)  # atomic: a render worker never loads half a file
            except OSError:  # another process won (Windows: and has it loaded)
                if not lib.exists():
                    raise
        finally:
            src.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)

    def launch(self, compiled: Any, n: int, args: tuple) -> None:
        """Run thread indices 0..n-1 (whole blocks, like the card) on ``threads`` cores."""
        cargs: list[Any] = []
        for a in args:
            if isinstance(a, np.ndarray):
                if not a.flags.c_contiguous or a.dtype.type not in _ARRAYS:
                    raise KernelError(f"kernel array must be C-contiguous {_ARRAYS}, got "
                                      f"{a.dtype} contiguous={a.flags.c_contiguous}")  # fmt: skip
                cargs.append(ctypes.c_void_p(a.ctypes.data))
            elif type(a) in _SCALARS:
                cargs.append(_SCALARS[type(a)](a))
            else:
                raise KernelError(f"kernel scalar {a!r} must be a NumPy int32/float32/float64")
        compiled.restype = None
        compiled(ctypes.c_longlong(-(-int(n) // BLOCK)), ctypes.c_int(self.threads), *cargs)


@functools.cache
def _default(compiler: Compiler | None) -> CpuBackend | None:
    return None if compiler is None else CpuBackend(compiler)


def default_cpu() -> CpuBackend | None:
    """The CPU backend, or None when no supported C++ compiler is found."""
    return _default(find_compiler())
