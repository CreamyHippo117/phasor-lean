"""Heat maps: a grid of values drawn as colour through ``ctx.field``, shared by every type
that shows one (``pde.Field``, ``pde.Slice``, ``acoustics.PressureSlice``,
``fourier.Spectrogram``).

The plain picture is what those types always drew. The prism look adds,
each optional:

- ``smooth``: Catmull-Rom (bicubic) sampling between grid values instead of straight-line
  blending, so a coarse grid reads as a smooth field;
- ``isolines``: that many contour levels drawn as thin light about a pixel wide at any zoom
  (the distance to a level is measured in pixels through the local gradient);
- a glow weight per pixel (alpha), so with the object's ``glow`` role the strong parts of the
  field shine and haze. See ``DrawContext.field``;
- ``hue_cycles``: colour from the colormap read as a wheel (the prism ``spectrum``), its hue
  running ``hue_cycles`` times over the value range the way it runs along a prism stroke, so
  each isoline has the hue of its level; brightness comes from the glow weight alone;
- ``fade``: the field melts into the background over that share of the panel's shorter side
  at its edges (no hard rectangle), glow weight included, so the haze has no edge either.

Every type that draws a heat map has these parameters (``HeatParams``); their defaults draw
what the plain pictures always drew.

The pixels are made by the ``heat_shade`` kernel (``render/gpu/kernels/heat.py``, which also
holds the maths as its NumPy reference): a type says how its values become level, colour
and glow weight with a ``Recipe``, and the whole picture is one pass on the card, or on
every CPU core. ``heat_pixels`` still takes plain functions (a scene's own type that needs
something a recipe cannot say); that path is NumPy on one core.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import Field

from phasor.core.objects import ParamsBase
from phasor.render.gpu.kernels.heat import (
    AMBIENT,
    Recipe,
    edge_window,
    isoline_weight,
    lut_colour,
    neighbourhood,
    sample,
    shade_values,
)

__all__ = [
    "AMBIENT", "PRISM", "HeatParams", "Recipe", "edge_window", "field_pixels", "heat_pixels",
    "isoline_weight", "lut_colour", "sample", "shade_values", "wheel_colour", "wheel_recipe",
]  # fmt: skip

Pixels = Callable[[int, int, tuple[float, float], float], np.ndarray]


class HeatParams(ParamsBase):
    """The heat-map parameters every type that draws one shares (the prism preset sets them)."""

    smooth: bool = False  # bicubic between grid values (the prism look); else as before
    isolines: int = Field(0, ge=0, le=64)  # contour levels drawn as thin light
    # > 0: colour from the colormap as a wheel, this many turns over the value range (prism)
    hue_cycles: float = Field(0.0, ge=0, le=8)
    hue: float = 0.0  # with hue_cycles: turns the colours (share of a turn); life adds a little
    fade: float = Field(0.0, ge=0, le=0.5)  # edges melt into the background over this share
    ambient: float = Field(AMBIENT, ge=0, le=1)  # with glow: the colour a quiet pixel keeps
    # with glow: the weight is raised to this power, so > 1 keeps the light to the strongest
    # parts and lets the rest fall dark (structure glowing out of darkness)
    contrast: float = Field(1.0, ge=0.25, le=8)

    def drawing(self) -> dict:
        """The ``heat_pixels`` keywords these parameters set."""
        return {"isolines": self.isolines, "ambient": self.ambient, "fade": self.fade,
                "contrast": self.contrast}  # fmt: skip


# The prism preset of every heat-map type: smooth, dark where the field is
# quiet and glowing where strong, in hues of the spectrum that run with the value, contour
# lines of light in the hue of their level, edges melting into the background. The prism
# looks repeat these values in their tables.
PRISM: dict = {
    "params": {
        "smooth": True,
        "isolines": 8,
        "hue_cycles": 1.6,
        "fade": 0.1,
        "ambient": 0.0,
        "contrast": 2.0,
    },
    "style": {"glow": "soft", "colormap": "spectrum"},
}


def heat_pixels(
    grid: np.ndarray,
    size: tuple[float, float],
    colour: Callable[[np.ndarray], np.ndarray] | Recipe,
    *,
    cells: bool = False,
    how: str = "linear",
    level: Callable[[np.ndarray], np.ndarray] | None = None,
    isolines: int = 0,
    weight: Callable[[np.ndarray], np.ndarray] | None = None,
    line_glow: float = 0.7,
    background: np.ndarray | None = None,
    ambient: float = AMBIENT,
    fade: float = 0.0,
    contrast: float = 1.0,
    colour_at: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Pixels:
    """``ctx.field`` pixels for ``grid[x, y]`` spanning a ``size`` rectangle.

    ``cells``: grid values sit at cell centres (``x / w * nx - 0.5``), else at nodes from edge
    to edge (``x / w * (nx - 1)``). ``colour``: a ``Recipe`` (level, colour and weight all
    come from it, and the ``heat_shade`` kernel draws the picture), or a function
    ``colour(value) -> (ph, pw, 3)`` linear RGB, or
    ``colour_at(value, u, v)`` where ``u`` (pw,) and ``v`` (ph,) are the pixels' places across
    and up the rectangle (0..1). ``level`` maps a value to 0..1 for the isolines (default: the
    value). ``weight(value)`` is the glow weight 0..1 (default: none, alpha 1 as before): with
    one, colour scales from ``ambient`` at weight 0 to full at 1, from ``background`` (the
    scene's, so a quiet field melts into it), the weight first raised to ``contrast``; the
    isolines add to the weight and brighten.
    ``fade``: colour and weight fall to the background over that share of the shorter side at
    the rectangle's edges."""
    if isinstance(colour, Recipe):
        if level is not None or weight is not None or colour_at is not None:
            raise ValueError("a Recipe says the level, the weight and the colour itself")
        return _kernel_pixels(colour, size, grid=grid, cells=cells, how=how, isolines=isolines,
                              line_glow=line_glow, background=background, ambient=ambient,
                              fade=fade, contrast=contrast)  # fmt: skip
    nx, ny = grid.shape
    sx, sy = size

    def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
        half_w = height * pw / ph / 2
        xs = center[0] + (np.arange(pw) + 0.5) / pw * 2 * half_w - half_w
        ys = center[1] + (0.5 - (np.arange(ph) + 0.5) / ph) * height  # row 0 = top
        if cells:
            g, h = xs / sx * nx - 0.5, ys / sy * ny - 0.5
        else:
            g, h = xs / sx * (nx - 1), ys / sy * (ny - 1)
        val = sample(grid, g, h, how)
        if colour_at is None:
            rgb = np.asarray(colour(val), dtype=np.float64)
        else:
            rgb = np.asarray(colour_at(val, xs / sx, ys / sy), dtype=np.float64)
        return shade_values(val, rgb, xs / sx, ys / sy, size, level=level, isolines=isolines,
                            weight=weight, line_glow=line_glow, background=background,
                            ambient=ambient, fade=fade, contrast=contrast)  # fmt: skip

    return pixels


def field_pixels(
    image_of: Callable[[Any], Any],
    size: tuple[float, float],
    recipe: Recipe,
    *,
    isolines: int = 0,
    line_glow: float = 0.7,
    background: np.ndarray | None = None,
    ambient: float = AMBIENT,
    fade: float = 0.0,
    contrast: float = 1.0,
) -> Pixels:
    """``ctx.field`` pixels of a field another kernel computes per pixel (``waves``), shaded
    as ``heat_pixels`` shades a grid. ``image_of(tile, center, height)`` returns that field
    for a ``Tile`` of the view, as the kernel left it (on the card with CUDA): it is coloured
    there and only the finished pixels come back."""
    return _kernel_pixels(recipe, size, image_of=image_of, isolines=isolines,
                          line_glow=line_glow, background=background, ambient=ambient,
                          fade=fade, contrast=contrast)  # fmt: skip


def _kernel_pixels(recipe: Recipe, size, *, grid=None, image_of=None, **look) -> Pixels:
    def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
        from phasor.render import gpu
        from phasor.render.gpu.bridge import to_host

        out = np.empty((ph, pw, 4), dtype=np.float32)
        for tile in gpu.plan_tiles(pw, ph, 64):
            source: dict[str, Any] = {"grid": grid}
            if image_of is not None:  # the tile and its neighbours (the isolines read them)
                x0, y0, x1, y1 = neighbourhood(tile)
                around = gpu.Tile(x0, y0, x1 - x0, y1 - y0, pw, ph)
                source = {"image": image_of(around, center, height), "origin": (x0, y0)}
            part = gpu.run("heat_shade", tile, center=center, height=height, size=size,
                           recipe=recipe, **source, **look)  # fmt: skip
            out[tile.y0 : tile.y0 + tile.h, tile.x0 : tile.x0 + tile.w] = to_host(part)
        return out

    return pixels


def wheel_colour(colormap, p: HeatParams) -> Callable[[np.ndarray], np.ndarray]:
    """The prism colouring of ``p`` (``hue_cycles`` > 0): a value's place 0..1 (its level) to
    a colour read round ``colormap`` as a wheel, ``hue_cycles`` turns over the range."""
    wheel = lut_colour(np.asarray(colormap, dtype=np.float64)[:, :3], "wrap")
    return lambda x: wheel(p.hue_cycles * x + p.hue)


def wheel_recipe(colormap, p: HeatParams, **how) -> Recipe:
    """The prism colouring of ``p`` as a ``Recipe`` (``wheel_colour`` for the kernel);
    ``how``: the rest of the recipe (transform, level, weight, ``hue_of``)."""
    return Recipe(colour="wheel", lut=colormap, hue_cycles=p.hue_cycles, hue=p.hue, **how)
