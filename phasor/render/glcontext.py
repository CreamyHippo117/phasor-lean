"""The single way phasor creates a headless OpenGL context.

Windows and desktop Linux use the platform default (WGL / GLX). Headless Linux
(CI, cloud containers) has no X display, so we fall back to EGL, which works with
Mesa's llvmpipe software renderer. Nothing else in phasor may call
``moderngl.create_context`` directly.

GL calls go to whichever context is *current*, and creating a context makes it current.
So every piece of GL work must run inside ``with ctx:`` on the GL thread
(``on_gl_thread``); the renderer does this for you.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import moderngl

T = TypeVar("T")
REQUIRED_GL = 430  # OpenGL 4.3: compute shaders, SSBOs, integer textures


class GLUnavailableError(RuntimeError):
    """No OpenGL 4.3 context could be created on this machine."""


def _backends() -> list[str | None]:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return ["egl", None]
    if sys.platform.startswith("linux"):
        return [None, "egl"]
    return [None]


def create_standalone_context(require: int = REQUIRED_GL) -> moderngl.Context:
    """Create an offscreen context with at least GL ``require`` (e.g. 430).

    Raises GLUnavailableError listing every backend that was tried.
    """
    try:
        import moderngl
    except ImportError as e:  # pragma: no cover - depends on install
        raise GLUnavailableError("moderngl is not installed (pip install phasor[gl])") from e

    errors: list[str] = []
    for backend in _backends():
        kwargs: dict[str, object] = {"standalone": True, "require": require}
        if backend:
            kwargs["backend"] = backend
        try:
            return moderngl.create_context(**kwargs)
        except Exception as e:
            errors.append(f"{backend or 'default'}: {e}")
    raise GLUnavailableError("; ".join(errors))


_local = threading.local()


def _mark_gl_thread() -> None:
    _local.gl = True


@functools.cache
def _gl_thread() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(1, thread_name_prefix="phasor-gl", initializer=_mark_gl_thread)


def on_gl_thread(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` on phasor's one OpenGL thread and return its result (or raise its error).

    A context is current in one thread at a time, and the shared context stays current in
    the thread that made it. So every use of it happens on that thread: the editor drawing
    and an AI connection linting from another thread take turns there. Mesa and WGL both
    refuse a context current elsewhere ("cannot create buffer", found on Windows
    2026-11-21). Called from the GL thread itself, ``fn`` simply runs."""
    if getattr(_local, "gl", False):
        return fn(*args, **kwargs)
    return _gl_thread().submit(fn, *args, **kwargs).result()


@functools.cache
def shared_context() -> moderngl.Context:
    """The process-wide offscreen context, made on (and used only on) the GL thread; see
    ``on_gl_thread``. Renderers share it, so GL objects are compiled once and there is
    never a question of which context owns them."""
    return on_gl_thread(create_standalone_context)
