"""Screen-space geometry for layout.

Everything the layout stages do happens in one space: **screen heights**, x from 0 to the
aspect ratio, y from 0 (bottom) to 1 (top). World-layer and overlay objects meet there, a
gap looks the same at any zoom, and the same code will serve perspective cameras later.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from phasor.core.spatial import Camera
from phasor.core.state import ObjectState, SceneState


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    @classmethod
    def around(cls, center, w: float, h: float) -> Box:
        cx, cy = center
        return cls(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def overlap(self, other: Box) -> float:
        """Area shared with ``other``."""
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return w * h if w > 0 and h > 0 else 0.0

    def nearest_point(self, p) -> tuple[float, float]:
        return (min(max(p[0], self.x0), self.x1), min(max(p[1], self.y0), self.y1))


class View:
    """Maps one camera's world space to screen heights and back, near the world point
    ``at``. For a perspective camera, sizes and shifts depend on depth: a View measures them
    at ``at``'s depth (the object being placed), so a gap looks the same near or far."""

    def __init__(self, cam: Camera, aspect: float, at=None) -> None:
        self.cam, self.aspect = cam, aspect
        if at is None or not cam.perspective:
            base = cam.eye + cam.forward
        else:
            base = cam.eye + cam.forward * max(float(cam.depth(at)[0]), 1e-6)
        o = cam.project([base])[0]
        # Screen-height change per world unit along the camera's right and up vectors.
        self._sx = (cam.project([base + cam.right])[0][0] - o[0]) * aspect / 2
        self._sy = (cam.project([base + cam.up])[0][1] - o[1]) / 2

    def point(self, world) -> tuple[float, float]:
        ndc = self.cam.project([world])[0]
        return (float((ndc[0] + 1) / 2 * self.aspect), float((ndc[1] + 1) / 2))

    def box(self, corners: np.ndarray) -> Box:
        ndc = self.cam.project(corners)
        x = (ndc[:, 0] + 1) / 2 * self.aspect
        y = (ndc[:, 1] + 1) / 2
        return Box(float(x.min()), float(y.min()), float(x.max()), float(y.max()))

    def shift(self, dx: float, dy: float) -> np.ndarray:
        """The world vector that moves a point by (dx, dy) screen heights."""
        return (dx / self._sx) * self.cam.right + (dy / self._sy) * self.cam.up

    @property
    def units_per_height(self) -> float:
        """Layer units (world units, or overlay units) per screen height."""
        return abs(1.0 / self._sy)


def occludes(obj: ObjectState) -> bool:
    """Whether the object hides what is behind it. Translucent objects do not, and neither
    do glass surfaces (material opacity under one half). The one rule every path uses: the
    id pass, the inspector, the analytic oracle and the camera solver."""
    material = obj.style.get("material", 1.0)
    return obj.params.opacity * (material if isinstance(material, float) else 1.0) >= 0.5


def is_backdrop(state: SceneState, obj: ObjectState) -> bool:
    """A backdrop (a domain-colored plane, a fractal) is something labels are meant to sit
    on. Layout does not avoid it; the linter judges readability on it by contrast."""
    return bool(obj.impl.semantics(state).get("backdrop", False))


def view_of(state: SceneState, obj: ObjectState) -> View:
    cam = state.camera_for(obj)
    at = obj.impl.world_bounds(state).center if cam.perspective else None
    return View(cam, state.aspect, at)


def screen_box(state: SceneState, obj: ObjectState) -> Box:
    return view_of(state, obj).box(obj.impl.world_bounds(state).corners())


def frame(state: SceneState) -> Box:
    return Box(0.0, 0.0, state.aspect, 1.0)


class Coverage:
    """How much of a screen box is covered by other objects, from a boolean mask
    (row 0 = top) such as the id pass. Uses a summed-area table, so a query is O(1)."""

    def __init__(self, mask: np.ndarray, aspect: float) -> None:
        self.h, self.w = mask.shape
        self.aspect = aspect
        # int32 holds any count of pixels a frame has (up to 2**31); half the memory traffic
        # of int64 for the same exact integers
        self._sat = np.zeros((self.h + 1, self.w + 1), dtype=np.int32)
        m = mask.astype(np.int32)
        np.cumsum(m, axis=0, out=m)
        np.cumsum(m, axis=1, out=m)
        self._sat[1:, 1:] = m

    def fraction(self, box: Box) -> float:
        x0 = int(np.floor(box.x0 / self.aspect * self.w))
        x1 = int(np.ceil(box.x1 / self.aspect * self.w))
        y0 = int(np.floor((1 - box.y1) * self.h))
        y1 = int(np.ceil((1 - box.y0) * self.h))
        total = max(x1 - x0, 1) * max(y1 - y0, 1)
        x0, x1 = min(max(x0, 0), self.w), min(max(x1, 0), self.w)
        y0, y1 = min(max(y0, 0), self.h), min(max(y1, 0), self.h)
        s = self._sat
        return float(s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]) / total


def footprint(state: SceneState, obj: ObjectState, size: tuple[int, int]) -> np.ndarray:
    """(h, w) bool, row 0 = top: the pixels an object covers, worked out without a renderer.

    An object that exposes a ``curve`` feature (a line) covers a thin stripe along it, a
    ``point`` covers the disc inside its box, and anything else covers its bounding box.
    Used when there is no OpenGL (tier C).
    """
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    cam = state.camera_for(obj)
    curve = obj.impl.features(state).get("curve")
    if curve is not None and len(np.atleast_2d(curve)) >= 2:
        px = cam.to_screen(np.atleast_2d(curve), w, h)
        for a, b in itertools.pairwise(px):
            n = int(max(abs(b - a).max(), 1)) * 2 + 1
            pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    x = np.round(pts[:, 0]).astype(int) + dx
                    y = np.round(pts[:, 1]).astype(int) + dy
                    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                    mask[y[ok], x[ok]] = True
        return mask
    px = cam.to_screen(obj.impl.world_bounds(state).corners(), w, h)
    (x0, y0), (x1, y1) = px.min(0), px.max(0)
    if obj.impl.semantics(state).get("kind") == "point":  # a disc, not its square box
        yy, xx = np.mgrid[0:h, 0:w] + 0.5
        rx, ry = max((x1 - x0) / 2, 1e-9), max((y1 - y0) / 2, 1e-9)
        return ((xx - (x0 + x1) / 2) / rx) ** 2 + ((yy - (y0 + y1) / 2) / ry) ** 2 <= 1.0
    x0, y0 = int(np.floor(x0)), int(np.floor(y0))
    x1, y1 = int(np.ceil(x1)), int(np.ceil(y1))
    mask[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = True
    return mask
