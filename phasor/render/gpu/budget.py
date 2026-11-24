"""Tiles and the VRAM budget.

Kernels always take a tile rectangle; the caller stitches. A plan keeps the working memory
of one tile under the budget, which by default is 60% of free VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass

VRAM_FRACTION = 0.60
HOST_BUDGET = 512 * 1024**2  # working memory per tile when there is no GPU to ask


@dataclass(frozen=True)
class Tile:
    """A rectangle of a ``full_w`` x ``full_h`` image. Pixel (0, 0) is the top-left."""

    x0: int
    y0: int
    w: int
    h: int
    full_w: int
    full_h: int

    def __post_init__(self) -> None:
        ok = self.w > 0 and self.h > 0 and self.x0 >= 0 and self.y0 >= 0
        if not (ok and self.x0 + self.w <= self.full_w and self.y0 + self.h <= self.full_h):
            raise ValueError(f"tile {self} does not fit its image")

    @classmethod
    def full(cls, width: int, height: int) -> Tile:
        return cls(0, 0, width, height, width, height)

    @property
    def n(self) -> int:
        return self.w * self.h

    def scaled(self, s: int) -> Tile:
        """The same region of the image rendered ``s`` times larger (supersampling)."""
        return Tile(self.x0 * s, self.y0 * s, self.w * s, self.h * s,
                    self.full_w * s, self.full_h * s)  # fmt: skip


def free_vram() -> int | None:
    """Free bytes on the current CUDA device, or None when there is none."""
    try:
        import cupy

        free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return None


_SHARE = 1.0  # this process's share of the card (render workers split it, ADR 0003)


def set_share(share: float) -> None:
    """Use only ``share`` (0 to 1] of the free memory: each of N render workers on one card
    gets 1/N, so together they plan tiles for the card they actually share."""
    global _SHARE
    if not 0 < share <= 1:
        raise ValueError(f"share must be in (0, 1], got {share}")
    _SHARE = share


def budget_bytes() -> int:
    free = free_vram()
    return HOST_BUDGET if free is None else int(free * VRAM_FRACTION * _SHARE)


def plan_tiles(
    width: int, height: int, bytes_per_pixel: int, budget: int | None = None
) -> list[Tile]:
    """Split an image into full-width bands, each within the budget, top to bottom.

    Bands are split into columns too only when a single row is over budget. Tiles are as
    equal as possible, so no sliver tile is left at the edge.
    """
    if width <= 0 or height <= 0 or bytes_per_pixel <= 0:
        raise ValueError("width, height and bytes_per_pixel must be positive")
    budget = budget_bytes() if budget is None else budget
    pixels = max(budget // bytes_per_pixel, 1)
    cols = -(-width // pixels) if pixels < width else 1
    tile_w = -(-width // cols)
    rows_fit = max(pixels // tile_w, 1)
    bands = -(-height // rows_fit)
    tile_h = -(-height // bands)
    tiles = []
    for y0 in range(0, height, tile_h):
        for x0 in range(0, width, tile_w):
            tiles.append(Tile(x0, y0, min(tile_w, width - x0), min(tile_h, height - y0),
                              width, height))  # fmt: skip
    return tiles


def free_pools() -> None:
    """Return CuPy's cached blocks to the driver (between scenes, and every N frames)."""
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
