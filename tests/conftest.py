"""Shared fixtures. Tests marked ``gl`` or ``gpu`` skip themselves when the hardware is absent,
so the same suite runs on tier A, B and C machines (and in CI)."""

from __future__ import annotations

import functools
import os

import pytest

# New scenes default to the prism look; the suite runs plain so every picture and position it
# checks stays what it was written against. tests/test_beauty.py checks the default itself.
os.environ.setdefault("PHASOR_DEFAULT_LOOK", "none")


@functools.cache
def gl_available() -> bool:
    try:
        from phasor.render.glcontext import create_standalone_context

        create_standalone_context().release()
        return True
    except Exception:
        return False


@functools.cache
def cuda_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    skip_gl = pytest.mark.skip(reason="no OpenGL 4.3 context on this machine")
    skip_gpu = pytest.mark.skip(reason="no CUDA device on this machine")
    for item in items:
        if "gl" in item.keywords and not gl_available():
            item.add_marker(skip_gl)
        if "gpu" in item.keywords and not cuda_available():
            item.add_marker(skip_gpu)
