"""The final look: bloom from HDR light, then the tone map, once.

    emission (or light above threshold) -> halve, halve, ... -> blur each level
                     -> enlarge and sum -> [fringe] -> tonemap(frame + strength * bloom)
                     -> [vignette] -> display range

``fringe`` and ``vignette`` (theme ``[bloom]``, both optional, 0 = off) are the lens: warm
and cool rims on the glow, darker corners. Paint is never fringed.

The tone curve is identity below the theme's knee, so paint colors look exactly as the
theme defines them; only glow is compressed.

``render.output.encode_srgb8`` then encodes sRGB, also once (contract 4). Every step is
a registered kernel, so this runs on CUDA when there is a device and on the NumPy
references otherwise, with the same result.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from phasor.core.timing import stage
from phasor.render.gpu import run
from phasor.render.gpu.bridge import to_host
from phasor.render.gpu.budget import Tile
from phasor.style.theme import Theme

BLUR_SIGMA = 1.5  # pixels, at each pyramid level
MIN_LEVEL_PX = 4


def _whole(image: Any) -> Tile:
    h, w = image.shape[:2]
    return Tile.full(w, h)


def bloom(image: Any, threshold: float, levels: int, backend: str = "auto") -> Any:
    """The halo of ``image`` (the emission target) above ``threshold``: a sum of blurred
    copies at 1/2, 1/4, ... resolution, so the halo is wide but costs little."""
    h, w = image.shape[:2]
    light = run("bright", _whole(image), image=image, threshold=threshold, backend=backend)
    total = None
    level = light
    for _ in range(int(levels)):
        lw, lh = max(level.shape[1] // 2, 1), max(level.shape[0] // 2, 1)
        if min(lw, lh) < MIN_LEVEL_PX:
            break
        level = run("resize", Tile.full(lw, lh), image=level, backend=backend)
        blurred = run("gauss", Tile.full(lw, lh), image=level, sigma=BLUR_SIGMA, axis=0,
                      backend=backend)  # fmt: skip
        blurred = run("gauss", Tile.full(lw, lh), image=blurred, sigma=BLUR_SIGMA, axis=1,
                      backend=backend)  # fmt: skip
        up = run("resize", Tile.full(w, h), image=blurred, backend=backend)
        total = up if total is None else total + up
    return light if total is None else total


def card_sync(backend: str = "auto") -> None:
    """Wait for the card, so a stopwatch charges its work to the stage that launched it."""
    if backend == "reference":
        return
    from phasor.render.gpu.registry import default_gpu

    g = default_gpu()
    if g is not None:
        g.cp.cuda.Device().synchronize()


def finish(
    image: np.ndarray,
    theme: Theme,
    emission: np.ndarray | None = None,
    *,
    backend: str = "auto",
    output: str = "linear",
) -> np.ndarray:
    """Linear HDR RGBA (h, w, 4) float32 -> linear display RGBA in 0..1, as NumPy; or,
    with ``output="srgb8"``, the finished (h, w, 3) sRGB bytes, encoded where the frame
    already is (on the card when there is one).

    The bloom comes from ``emission`` (the light glow roles add, from the renderer's second
    target), so plain paint never blooms and a thin glowing curve blooms in proportion to
    its light at any resolution. Without ``emission`` the light above the threshold in
    ``image`` itself is used. ``image`` and ``emission`` may already be on the card (CuPy
    arrays, as ``Engine.frame_blurred`` passes them); they are then not copied out."""
    if output not in ("linear", "srgb8"):
        raise ValueError(f"output must be 'linear' or 'srgb8', got {output!r}")
    with stage("post", sync=lambda: card_sync(backend)):
        out = _finish(image, theme, emission, backend=backend)
    if output == "srgb8":
        with stage("srgb", sync=lambda: card_sync(backend)):
            out = run("srgb8", _whole(out), image=out, backend=backend)
    return to_host(out)


def _contiguous(a: Any) -> Any:
    """Host arrays become contiguous float32; device arrays stay where they are (each
    kernel launcher makes its own input contiguous on the card)."""
    return np.ascontiguousarray(a, dtype=np.float32) if isinstance(a, np.ndarray) else a


def _finish(
    image: np.ndarray,
    theme: Theme,
    emission: np.ndarray | None = None,
    *,
    backend: str = "auto",
) -> np.ndarray:
    image = _contiguous(image)
    b = theme.bloom
    source, threshold = image, b["threshold"]
    if emission is not None:
        source, threshold = _contiguous(emission), 0.0
    glow = bloom(source, threshold, int(b["levels"]), backend) if b["strength"] > 0 else None
    if glow is not None and b.get("fringe", 0.0) > 0:  # warm and cool rims on the glow only
        glow = run("fringe", _whole(image), image=glow, amount=b["fringe"], backend=backend)
    out = run("tonemap", _whole(image), image=image, exposure=b["exposure"], bloom=glow,
              bloom_strength=b["strength"] / max(int(b["levels"]), 1), knee=b["knee"],
              backend=backend)  # fmt: skip
    if b.get("vignette", 0.0) > 0:
        out = run("vignette", _whole(image), image=out, strength=b["vignette"], backend=backend)
    return out  # still on the card when there is one; ``finish`` copies it out once
