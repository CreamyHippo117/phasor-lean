"""Run phasor's CUDA kernels on the CPU, for machines without an NVIDIA GPU.

The kernel sources are plain C++ plus a few CUDA words. This compiles the *same source
string* with the host C++ compiler, with ``__global__``/``__device__`` defined away and
``blockIdx``/``threadIdx``/``blockDim`` provided as globals, then calls the kernel once per
thread index. ``EmuGpu`` stands in for ``registry.Gpu``, so the real launcher code runs too:
argument order, types and the output buffer are all exercised.

What this does not cover: NVRTC accepting the source, CuPy's argument passing, and GPU
floating point (the card compiles without fused multiply-add, like this emulator,
but its exp, log and sin are not libm's). Those need the ``gpu``-marked tests on real hardware.
"""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from phasor.render.gpu.registry import BLOCK, HEADER, PRECISIONS

PRELUDE = """
#include <math.h>
struct emu_dim3 { unsigned x, y, z; };
static emu_dim3 blockIdx, threadIdx, blockDim;
#define __global__
#define __device__
#define __forceinline__ inline
extern "C" void emu_set(unsigned b, unsigned t, unsigned d) {
    blockIdx.x = b; threadIdx.x = t; blockDim.x = d;
}
// Threads run one after another here, so an atomic add is a plain add (returns the old value).
static inline float atomicAdd(float* a, float v) { float old = *a; *a = old + v; return old; }
static inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) {
    unsigned long long old = *a; *a = old + v; return old;
}
"""
_CTYPES = {np.int32: ctypes.c_int, np.float32: ctypes.c_float, np.float64: ctypes.c_double}
_DIR = Path(tempfile.mkdtemp(prefix="phasor_emu_"))
_LIBS: dict[str, ctypes.CDLL] = {}  # compiled kernels, by source hash


def compiler() -> str | None:
    return shutil.which("g++") or shutil.which("clang++")


class DeviceArray(np.ndarray):
    """An array "on the card". CuPy refuses host (NumPy) arrays in ``ascontiguousarray`` and
    in kernel arguments; this stand-in refuses them too, so a launcher that forgets
    ``cp.asarray`` fails here and not only on an NVIDIA machine."""


class _StrictCupy:
    """The part of CuPy launchers may use, with CuPy's rule that host arrays must be copied
    to the device with ``asarray`` first."""

    float32, float64, int32, uint8 = np.float32, np.float64, np.int32, np.uint8
    uint64 = np.uint64

    @staticmethod
    def asarray(a, dtype=None):
        return np.array(a, dtype=dtype, copy=True).view(DeviceArray)

    @staticmethod
    def ascontiguousarray(a, dtype=None):
        if not isinstance(a, DeviceArray):
            raise TypeError(f"Argument 'a' has incorrect type (expected a device array, got "
                            f"{type(a).__module__}.{type(a).__name__})")  # fmt: skip
        return np.ascontiguousarray(a, dtype=dtype).view(DeviceArray)

    @staticmethod
    def empty(shape, dtype=np.float32):
        return np.empty(shape, dtype=dtype).view(DeviceArray)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype).view(DeviceArray)

    @staticmethod
    def empty_like(a, dtype=None):
        return np.empty_like(a, dtype=dtype).view(DeviceArray)

    def __getattr__(self, name):
        raise AttributeError(f"launchers may use only asarray, ascontiguousarray, empty, zeros, "
                             f"empty_like and dtypes from cp, not cp.{name}")  # fmt: skip


class EmuGpu:
    cp = _StrictCupy()

    def __init__(self) -> None:
        # One cache for every instance: a loaded library cannot be overwritten on Windows,
        # so a second EmuGpu must reuse it rather than link the same file again.
        self._libs = _LIBS

    def compile(self, source: str, entry: str, precision: str = "f32"):
        code = PRELUDE + HEADER + source
        key = hashlib.sha1((code + precision).encode()).hexdigest()[:16]
        if key not in self._libs:
            src, lib = _DIR / f"{key}.cpp", _DIR / f"{key}.so"
            src.write_text(code, encoding="utf-8")
            # -ffp-contract=off: no fused multiply-add, like the NumPy reference.
            cmd = [compiler(), "-shared", "-fPIC", "-O1", "-ffp-contract=off", "-Wall",
                   "-Werror=return-type", *PRECISIONS[precision], str(src), "-o", str(lib)]  # fmt: skip
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"kernel does not compile:\n{done.stderr}")
            self._libs[key] = ctypes.CDLL(str(lib))
        lib = self._libs[key]
        return lib.emu_set, getattr(lib, entry)

    def launch(self, compiled, n: int, args: tuple) -> None:
        emu_set, fn = compiled
        cargs, argtypes = [], []
        for a in args:
            if isinstance(a, np.ndarray):
                if not isinstance(a, DeviceArray):
                    raise TypeError("a host (NumPy) array passed to a kernel: cp.asarray it")
                assert a.flags.c_contiguous and a.dtype in (
                    np.float32,
                    np.float64,
                    np.uint8,
                    np.uint64,
                ), (
                    a.dtype,
                    a.flags,
                )
                cargs.append(a.ctypes.data_as(ctypes.c_void_p))
                argtypes.append(ctypes.c_void_p)
            else:
                assert type(a) in _CTYPES, f"scalar {a!r} must be a NumPy int32/float32/float64"
                cargs.append(_CTYPES[type(a)](a))
                argtypes.append(_CTYPES[type(a)])
        fn.argtypes, fn.restype = argtypes, None
        for block in range(-(-n // BLOCK)):
            for thread in range(BLOCK):  # includes indices past n: the kernel must return
                emu_set(block, thread, BLOCK)
                fn(*cargs)
