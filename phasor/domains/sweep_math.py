"""A surface sweeping a lattice of points (``domains/sweep.py``): pure NumPy, no drawing.

The sweep surface is one family with a blend ``flat`` (0-1) between a sphere and a plane.
A point ``p`` is reached when the front has travelled its *arrival distance*

    s(p) = (1 - flat) |p - c| + flat (p - c) . n        (c the centre, n the unit normal)

so ``flat`` 0 is a sphere of radius s round ``c``, ``flat`` 1 the plane (p - c) . n = s, and
in between the level sets are conics of revolution with their focus at ``c`` (eccentricity
flat / (1 - flat): an ellipsoid, a paraboloid at one half, then one sheet of a hyperboloid).
Every member passes through the same apex ``c + s n``, so animating ``flat`` from 0 to 1
opens the sphere out into the plane tangent to it there. Arrival is linear in ``flat``.

Rules that make lattices: ``mirror_box`` (the image sources of a shoebox room, from
``ac_math.image_sources``: each echo of a clap is a mirror copy of the room) and ``grid``
(integer steps along three vectors: the delays of feedback loops). ``envelope`` is the
flash of a point as the front passes it (shared by every lattice of lights: ``nd.StarLattice``
feeds it its own arrival times, and ``strokes.flash_points`` draws the result), ``rate``
how many points the front reaches per second, ``front_mesh`` the surface itself as
triangles.
"""

from __future__ import annotations

import numpy as np

from phasor.domains import ac_math

FRONT_RINGS = 48  # rings from the apex to the edge of the front's mesh
FRONT_SECTORS = 96  # points round each ring
EDGE_FADE = 0.12  # share of an open front, at its edge, over which it fades out


def unit(normal) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(n))
    if length == 0.0:
        raise ValueError("the normal must not be zero")
    return n / length


def arrival(points, center, normal, flat: float) -> np.ndarray:
    """(N,) the distance the front has travelled when it reaches each point (see above).
    Negative for points behind a front that is not a sphere: those are never reached."""
    x = np.asarray(points, dtype=np.float64).reshape(-1, 3) - np.asarray(center, dtype=np.float64)
    out = x @ unit(normal) * flat if flat > 0 else np.zeros(len(x))
    if flat < 1:
        out = out + (1.0 - flat) * np.sqrt(np.einsum("ij,ij->i", x, x))
    return out


def mirror_box(size, source, order: int) -> tuple[np.ndarray, np.ndarray]:
    """The image sources of ``source`` in a shoebox room with its corner at the origin, up to
    ``order`` reflections: (positions (N, 3), reflections (N,)). The source is first."""
    im = ac_math.image_sources(size, source, order, 0.0)
    return im["pos"], im["reflections"].astype(np.float64)


def grid(basis, counts, origin) -> tuple[np.ndarray, np.ndarray]:
    """Points ``origin + i a + j b + k c`` for 0 <= i < counts[0] (and so on), with the
    number of steps i + j + k as their value, in order of steps."""
    a = np.asarray(basis, dtype=np.float64).reshape(3, 3)
    idx = np.indices(tuple(int(c) for c in counts)).reshape(3, -1).T  # (N, 3)
    steps = idx.sum(axis=1)
    order = np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0], steps))
    idx, steps = idx[order], steps[order]
    return np.asarray(origin, dtype=np.float64) + idx @ a, steps.astype(np.float64)


def smoothstep(x) -> np.ndarray:
    """3x^2 - 2x^3 of x clipped to [0, 1]: a ramp with flat ends."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def envelope(age, *, attack: float, decay: float, before: float, after: float):
    """(light, flash) of points ``age`` seconds after the front reached them (negative: not
    yet). Light is ``before`` until the front arrives, rises over ``attack`` seconds to 1,
    then falls as exp(-age / decay) to the afterglow ``after``. ``flash`` is the part above
    the afterglow, 0-1, which the drawing turns into size and white heat."""
    age = np.asarray(age, dtype=np.float64)
    rise = np.ones_like(age) if attack <= 0 else smoothstep(age / attack)
    rise = np.where(age >= 0, rise, 0.0)
    e = np.exp(-np.maximum(age, 0.0) / decay)
    flash = rise * e
    light = before + rise * (after + (1.0 - after) * e - before)
    return light, flash


def rate(s, reach: float, speed: float, window: float = 1.0) -> float:
    """Points reached per second over the next ``window`` seconds: those whose arrival
    distance is in [reach, reach + speed * window)."""
    s = np.asarray(s, dtype=np.float64)
    hits = (s >= reach) & (s < reach + speed * window)
    return float(hits.sum()) / window


def _cap(flat: float, reach: float, extent: float) -> float | None:
    """The polar angle (from the normal) at which the front leaves the ball of radius
    ``extent`` round the centre: pi when all of it is inside, None when none of it is."""
    if reach <= 0:
        return None
    if flat <= 0:
        return np.pi if reach <= extent else None
    cos = (reach / extent - (1.0 - flat)) / flat
    if cos > 1.0:
        return None
    return float(np.pi if cos <= -1.0 else np.arccos(cos))


def front_mesh(center, normal, flat: float, reach: float, extent: float,
               rings: int = FRONT_RINGS, sectors: int = FRONT_SECTORS):  # fmt: skip
    """The front at ``reach`` as triangles, cut to the ball of radius ``extent`` round the
    centre: (positions (N, 3), unit normals (N, 3) pointing the way it travels, triangles
    (M, 3), edge (N,) a fade weight that is 1 inside and falls to 0 at an open edge), or None
    when no part of it is inside. Rings are spaced evenly along the surface (not in angle),
    so a nearly flat front is as finely meshed at its edge as at its apex."""
    top = _cap(flat, reach, extent)
    if top is None:
        return None
    c, n = np.asarray(center, dtype=np.float64), unit(normal)
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    # the meridian, densely, then resampled at even arc length
    th = np.linspace(0.0, top, 2048)
    rho = reach / np.maximum((1.0 - flat) + flat * np.cos(th), 1e-12)
    rho = np.minimum(rho, extent)
    mx, my = rho * np.cos(th), rho * np.sin(th)
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(mx), np.diff(my)))]
    share = np.linspace(0.0, 1.0, rings + 1)
    th = np.interp(share * arc[-1], arc, th)
    rho = np.minimum(reach / np.maximum((1.0 - flat) + flat * np.cos(th), 1e-12), extent)
    phi = np.linspace(0.0, 2 * np.pi, sectors, endpoint=False)
    sin_t, cos_t = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    around = (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)[None]  # (1, S, 3)
    u = cos_t * n + sin_t * around  # (R + 1, S, 3) unit directions from the centre
    pos = c + rho[:, None, None] * u
    grad = (1.0 - flat) * u + flat * n
    grad /= np.maximum(np.linalg.norm(grad, axis=2, keepdims=True), 1e-12)
    closed = top >= np.pi - 1e-12
    edge = np.ones(rings + 1) if closed else smoothstep((1.0 - share) / EDGE_FADE)
    edge = np.repeat(edge, sectors)
    k = np.arange(rings)[:, None] * sectors
    j = np.arange(sectors)[None]
    a, b = k + j, k + (j + 1) % sectors
    first = np.stack([a, b, a + sectors], -1)
    second = np.stack([b, b + sectors, a + sectors], -1)
    tris = np.concatenate([first, second], axis=0).reshape(-1, 3)
    return pos.reshape(-1, 3), grad.reshape(-1, 3), tris, edge
