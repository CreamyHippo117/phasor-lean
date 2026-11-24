"""The one spatial pipeline (contract 4).

World space is right-handed with z up; the complex plane is x = Re, y = Im. A 2D scene is
an orthographic camera looking down -z at the plane z = 0. Screen space is pixels with the
origin at the top-left, plus normalized 0-1 coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from phasor.core.values import Vec3

ORTHO_DEPTH = 100.0  # the ortho camera sits this far above its center; depth range is 2x this
REGIONS = (
    ("upper-left", "upper-center", "upper-right"),
    ("middle-left", "center", "middle-right"),
    ("lower-left", "lower-center", "lower-right"),
)


class OrthoCameraParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    center: Vec3 = (0.0, 0.0, 0.0)
    height: float = Field(8.0, gt=0)  # world units visible from top to bottom of the frame


class OrbitCameraParams(BaseModel):
    """A perspective camera on a sphere around ``target`` (z up). Angles in degrees:
    azimuth 0 looks from +x toward the target, 90 from +y; elevation 90 looks straight down."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    target: Vec3 = (0.0, 0.0, 0.0)
    azimuth: float = -60.0
    elevation: float = Field(30.0, ge=-89.0, le=89.0)
    distance: float = Field(10.0, gt=0)
    fov: float = Field(35.0, gt=1.0, lt=150.0)  # vertical field of view
    # Slow sway in azimuth, degrees either way.
    # Negative (the default): the theme's [camera] drift; 0 holds the camera still.
    drift: float = -1.0


def _frozen(a) -> np.ndarray:
    a = np.array(a, dtype=np.float64)
    a.setflags(write=False)
    return a


def _homogeneous(points) -> np.ndarray:
    """(N, 4): the points with a 1 appended. Same values as ``np.c_[p, ones]``, without its
    per-call overhead (the layout bake projects hundreds of thousands of tiny arrays)."""
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    h = np.empty((len(p), 4))
    h[:, :3] = p
    h[:, 3] = 1.0
    return h


@dataclass(frozen=True, eq=False)
class Camera:
    """A resolved camera: matrices plus the basis vectors objects need to face the viewer."""

    kind: str
    view_proj: np.ndarray  # 4x4, world -> clip (row-major; multiply as M @ [x, y, z, 1])
    eye: np.ndarray
    right: np.ndarray
    up: np.ndarray  # the world direction that appears as "up" on screen
    forward: np.ndarray
    height: float = field(default=1.0)  # ortho: world units per frame height
    tan_half_fov: float = 0.0  # perspective only: tan(vertical fov / 2)

    @property
    def perspective(self) -> bool:
        return self.kind == "perspective"

    def units_per_height(self, point=None):
        """World units spanned by the full frame height at ``point``. Constant for ortho;
        for perspective it grows with depth (at the target when ``point`` is None).
        ``point`` may be (3,) or (N, 3); the result is a float or an (N,) array."""
        if not self.perspective:
            if point is not None and np.ndim(point) == 2:
                return np.full(len(point), self.height)
            return self.height
        if point is None:
            return 2.0 * self.height * self.tan_half_fov  # height holds the target distance
        d = np.maximum(self.depth(point), 1e-6) * 2.0 * self.tan_half_fov
        return d if np.ndim(point) == 2 else float(d[0])

    def clip_w(self, points) -> np.ndarray:
        """The homogeneous w of each point: positive in front of the camera."""
        return _homogeneous(points) @ self.view_proj[3]

    def project(self, points) -> np.ndarray:
        """World points (N, 3) -> NDC (N, 3); x, y in [-1, 1] inside the frame, y up."""
        clip = _homogeneous(points) @ self.view_proj.T
        return clip[:, :3] / clip[:, 3:4]

    def to_screen(self, points, width: int, height: int) -> np.ndarray:
        """World points (N, 3) -> pixels (N, 2), origin top-left."""
        ndc = self.project(points)
        return np.c_[(ndc[:, 0] + 1) * 0.5 * width, (1 - ndc[:, 1]) * 0.5 * height]

    def from_screen(self, pixels, width: int, height: int, z: float = 0.0) -> np.ndarray:
        """Pixels (N, 2), origin top-left -> the world points (N, 3) under them on the plane
        at height ``z`` (in 2D, the plane itself). The inverse of ``to_screen`` there."""
        px = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
        ndc = np.c_[2 * px[:, 0] / width - 1, 1 - 2 * px[:, 1] / height]
        inv = np.linalg.inv(self.view_proj)
        near = np.c_[ndc, -np.ones(len(ndc)), np.ones(len(ndc))] @ inv.T
        far = np.c_[ndc, np.ones(len(ndc)), np.ones(len(ndc))] @ inv.T
        a, b = near[:, :3] / near[:, 3:4], far[:, :3] / far[:, 3:4]
        dz = b[:, 2] - a[:, 2]
        k = np.where(np.abs(dz) < 1e-12, 0.0, (z - a[:, 2]) / np.where(dz == 0, 1, dz))
        out = a + k[:, None] * (b - a)
        out[:, 2] = z
        return out

    def depth(self, points) -> np.ndarray:
        """Distance from the camera along its viewing direction."""
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return (p - self.eye) @ self.forward


def ortho_camera(params: OrthoCameraParams, aspect: float) -> Camera:
    cx, cy, cz = params.center
    half_h = params.height / 2
    half_w = half_h * aspect
    m = np.array(
        [
            [1 / half_w, 0, 0, -cx / half_w],
            [0, 1 / half_h, 0, -cy / half_h],
            [0, 0, -1 / ORTHO_DEPTH, cz / ORTHO_DEPTH],  # nearer (higher z) -> smaller NDC z
            [0, 0, 0, 1],
        ]
    )
    return Camera(
        kind="ortho",
        view_proj=_frozen(m),
        eye=_frozen([cx, cy, cz + ORTHO_DEPTH]),
        right=_frozen([1, 0, 0]),
        up=_frozen([0, 1, 0]),
        forward=_frozen([0, 0, -1]),
        height=params.height,
    )


def orbit_eye(params: OrbitCameraParams, azimuth: float | None = None) -> np.ndarray:
    az = np.radians(params.azimuth if azimuth is None else azimuth)
    el = np.radians(params.elevation)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.asarray(params.target, dtype=np.float64) + params.distance * d


NEAR, FAR = 0.02, 50.0  # near and far planes as multiples of the camera distance


def perspective_camera(params: OrbitCameraParams, aspect: float, sway: float = 0.0) -> Camera:
    """``sway``: extra azimuth in degrees (the drift, already evaluated at t)."""
    target = np.asarray(params.target, dtype=np.float64)
    eye = orbit_eye(params, params.azimuth + sway)
    forward = (target - eye) / np.linalg.norm(target - eye)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    t = float(np.tan(np.radians(params.fov) / 2))
    n, f = NEAR * params.distance, FAR * params.distance
    proj = np.array(
        [
            [1 / (t * aspect), 0, 0, 0],
            [0, 1 / t, 0, 0],
            [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
            [0, 0, -1, 0],
        ]
    )
    return Camera(
        kind="perspective",
        view_proj=_frozen(proj @ view),
        eye=_frozen(eye),
        right=_frozen(right),
        up=_frozen(up),
        forward=_frozen(forward),
        height=params.distance,
        tan_half_fov=t,
    )


def overlay_camera(aspect: float) -> Camera:
    """The screen-space overlay layer: units are frame heights, origin top-left, y down.

    So x runs 0..aspect and y runs 0..1, and sizes mean the same as in world layers.
    """
    m = np.array(
        [
            [2 / aspect, 0, 0, -1],
            [0, -2, 0, 1],
            [0, 0, -1 / ORTHO_DEPTH, 0],
            [0, 0, 0, 1],
        ]
    )
    return Camera(
        kind="overlay",
        view_proj=_frozen(m),
        eye=_frozen([0, 0, ORTHO_DEPTH]),
        right=_frozen([1, 0, 0]),
        up=_frozen([0, -1, 0]),
        forward=_frozen([0, 0, -1]),
        height=1.0,
    )


def region_name(nx: float, ny: float) -> str:
    """3x3 grid name for a normalized screen point (clamped into the frame)."""
    col = min(2, max(0, int(nx * 3)))
    row = min(2, max(0, int(ny * 3)))
    return REGIONS[row][col]
