"""The CPU kernel backend: the kernels' CUDA source, compiled by g++ or
clang++ at run time, on every core. See ``backend.py``."""

from phasor.render.cpu.backend import (
    CompileError,
    Compiler,
    CpuBackend,
    cache_dir,
    cores,
    default_cpu,
    find_compiler,
    set_threads,
    threads,
)

__all__ = [
    "CompileError", "Compiler", "CpuBackend", "cache_dir", "cores", "default_cpu",
    "find_compiler", "set_threads", "threads",
]  # fmt: skip
