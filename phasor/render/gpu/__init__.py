"""GPU math for the final renderer: the kernel registry, tiles and VRAM budget, the
expression code generator, and the bridge that hands frames to OpenGL.

    from phasor.render import gpu
    field = gpu.render("escape", 3840, 2160, center=(-0.5, 0), height=2.5)

Importing this package loads neither CuPy nor ModernGL.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from phasor.render.gpu.budget import Tile, budget_bytes, plan_tiles
from phasor.render.gpu.codegen import ComplexFunction, FunctionError
from phasor.render.gpu.registry import KernelError, default_gpu, get, kernel, registered, run

__all__ = [
    "ComplexFunction", "FunctionError", "KernelError", "Tile", "box_mean", "default_gpu", "get",
    "kernel", "plan_tiles", "registered", "render", "run",
]  # fmt: skip


def box_mean(image: Any, s: int) -> Any:
    """``image`` (h * s, w * s, ...) averaged over s x s blocks to (h, w, ...), on the device
    that holds it (a NumPy or CuPy array, the same calls in both): the mean of each block in
    float64, returned in the image's dtype. The same bits on either device: the block sums
    are of at most s * s float32 values, and each result is rounded once."""
    if s == 1:
        return image
    h, w = image.shape[0] // s, image.shape[1] // s
    shape = (h, s, w, s, *image.shape[2:])
    return image.reshape(shape).mean(axis=(1, 3), dtype=np.float64).astype(image.dtype)


def render(
    name: str,
    width: int,
    height_px: int,
    *,
    supersample: int = 1,
    backend: str = "auto",
    budget: int | None = None,
    **params: Any,
) -> np.ndarray:
    """Run a per-pixel kernel over a whole image, tile by tile, and stitch the result on
    the host. ``supersample=s`` renders each tile s times larger and averages it down, so
    only one enlarged tile is ever in memory.

    Averaging is right for color kernels (``domain``); a field such as ``escape`` should be
    supersampled after it has been colored.
    """
    from phasor.render.gpu.bridge import to_host

    if supersample < 1:
        raise ValueError("supersample must be at least 1")
    out: np.ndarray | None = None
    per_pixel = 16 * supersample * supersample  # RGBA float32 at the enlarged size
    budget = budget_bytes() if budget is None else budget
    for tile in plan_tiles(width, height_px, per_pixel, budget):
        # averaged where the kernel left it (on the card for CUDA), so only the output size
        # crosses to the host
        part = to_host(box_mean(run(name, tile.scaled(supersample), backend=backend, **params),
                                supersample))  # fmt: skip
        if out is None:
            out = np.empty((height_px, width, *part.shape[2:]), dtype=part.dtype)
        out[tile.y0 : tile.y0 + tile.h, tile.x0 : tile.x0 + tile.w] = part
    assert out is not None
    return out
