"""The CUDA to OpenGL bridge.

One interface, ``FrameBridge.upload(array) -> moderngl.Texture``, and nothing outside this
module may move pixels between CUDA and GL. ``CopyBridge`` goes through host memory
(``cupy.asnumpy`` then ``texture.write``): about 66 MB for a 4K RGBA16F frame, which is
fine for offline rendering. An ``InteropBridge`` that stays on the GPU is a later
optimization and must pass the same tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import moderngl


class FrameBridge(Protocol):
    def upload(self, array: Any) -> moderngl.Texture: ...


def to_host(array: Any) -> np.ndarray:
    """A NumPy view or copy of a NumPy or CuPy array. The only GPU-to-host transfer."""
    if isinstance(array, np.ndarray):
        return array
    import cupy

    return cupy.asnumpy(array)


class CopyBridge:
    """Uploads linear RGBA frames, shape (h, w, 4), row 0 = top, as RGBA16F textures.

    The texture is reused while the size stays the same. GL stores rows bottom-up, so the
    image is flipped on the way in; sampling with v = 0 at the bottom shows it upright.

    Call it with the context current (inside ``with ctx:``), like all GL work. It does not
    enter the context itself: leaving a nested ``with ctx:`` would make no context current
    for the caller that is still drawing.
    """

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._texture: moderngl.Texture | None = None

    def upload(self, array: Any) -> moderngl.Texture:
        if array.ndim != 3 or array.shape[2] != 4:
            raise ValueError(f"expected an (h, w, 4) RGBA frame, got shape {tuple(array.shape)}")
        host = np.ascontiguousarray(to_host(array)[::-1], dtype=np.float16)
        h, w = host.shape[:2]
        if self._texture is None or self._texture.size != (w, h):
            if self._texture is not None:
                self._texture.release()
            self._texture = self.ctx.texture((w, h), 4, dtype="f2")
        self._texture.write(host.tobytes())
        return self._texture

    def read_back(self) -> np.ndarray:
        """The last uploaded frame as (h, w, 4) float32, row 0 = top. For tests."""
        if self._texture is None:
            raise RuntimeError("nothing has been uploaded")
        raw = self._texture.read()
        w, h = self._texture.size
        return np.frombuffer(raw, dtype=np.float16).reshape(h, w, 4)[::-1].astype(np.float32)
