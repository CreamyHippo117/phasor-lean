"""Mirror unfolding: a ray bouncing in a room drawn as a straight line through mirror rooms.

A ray that reflects off a wall carries on straight into the mirror copy of the room across
that wall. Copy the room across every wall the ray reaches and the zig-zag becomes one
straight line through a row of rooms (the image-source picture of echoes). This module
works that out, pure NumPy and no drawing:

- ``box_unfold``: the walls a straight ray crosses, in order, through the lattice of mirror
  copies of a box (in 2D, a rectangle: a ray with no z never meets the floor or ceiling);
  for each, where and when the ray arrives and the hinge the copy turns about.
- ``hinge_matrix`` / ``chain_matrix``: a copy is the room carried through a chain of hinge
  turns (``Hinge``: a line and an angle). The last turn may be part done (``turn`` 0-1):
  that is the page turning.
- ``fold``, ``image_point``, ``zigzag``: from the unfolded frame back to the real room
  (the reflections the ray really makes) and out again (image sources).
- ``Polytope`` (``polygon``, ``polyhedron``): any convex room as its walls (planes), for a
  billiard in a polygon, a kaleidoscope or a convex mesh room; ``room_unfold`` is
  ``box_unfold`` for it, its copies named by the walls crossed (``word_name``: ``a_c_b``),
  ``copy_matrix`` the true mirror copy for a sequence of walls and ``mirror_chain`` its
  hinges (a kaleidoscope drawn with ``HingedCopy``).

A hinge is a turn about a line: a copy swings over like a page, so a renderer can draw it
as the room moved rigidly. For a flat room (2D) half a
turn about the wall's edge is exactly the reflection across it. For a box (3D) the hinge
runs through the middle of the wall: half a turn lands the box exactly on its mirror copy
(a box is symmetric about its middle), though what is inside comes out turned rather than
mirrored. The same holds for any convex room that is symmetric about a plane through the
wall's middle line, square to the wall. For one that is not (most convex polyhedra), a turn
can never land on the mirror image (a mirror image has the other handedness), so the hinge
carries a ``mirror``: the copy is first mirrored in that plane (on top of the room it
leaves, where it appears) and then turned; a half turn after that mirror is exactly the
reflection in the wall, so the copy lands on the true mirror room.

The net of a cube and a geodesic on it fit the same pieces (quarter turns about the edges
of its faces).

Layer: ``core``, so that both the ``unfold`` verb (``time``) and the ``HingedCopy`` object
(``domains``) can use it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EPS = 1e-9  # relative: crossings closer than this (times the room's size) happen together
MAX_FLIPS = 64
# Cell names: the direction of each axis, positive then negative (east/west, north/south,
# up/down: x right, y up the screen in 2D, z up).
AXIS_WORDS = (("e", "w"), ("n", "s"), ("u", "d"))
WALLS = (("west", "east"), ("south", "north"), ("floor", "ceiling"))  # (lo, hi) per axis


@dataclass(frozen=True)
class Hinge:
    """A turn of ``degrees`` about the line through ``point`` along ``axis`` (a unit vector;
    the turn is counterclockwise looking down the axis, the right-hand rule). With
    ``mirror`` (a normal), the copy is first mirrored in the plane through ``point`` with
    that normal, then turned: a half turn after it is a reflection (see the module notes)."""

    point: tuple[float, float, float]
    axis: tuple[float, float, float]
    degrees: float = 180.0
    mirror: tuple[float, float, float] | None = None

    def as_params(self) -> tuple:
        """(point, axis, degrees), and the mirror's normal fourth when there is one."""
        if self.mirror is None:
            return (self.point, self.axis, self.degrees)
        return (self.point, self.axis, self.degrees, self.mirror)


@dataclass(frozen=True)
class Crossing:
    """The ray crossing from one copy of the room into the next."""

    distance: float  # along the ray from its start, in world units
    axis: int  # 0, 1, 2: the wall's normal axis
    side: int  # +1: the copy's high wall (east, north, ceiling); -1: its low wall
    cell_from: tuple[int, int, int]  # lattice index of the copy the ray leaves
    cell_to: tuple[int, int, int]
    point: tuple[float, float, float]  # where it crosses, unfolded (on the straight line)
    wall_point: tuple[float, float, float]  # the same point in the real room: the reflection
    wall: str  # the real wall it reflects off: west, east, south, north, floor, ceiling
    hinge: Hinge  # the turn that carries copy ``cell_from`` onto copy ``cell_to``

    @property
    def name(self) -> str:
        """The name of the copy the ray enters (``cell_name``)."""
        return cell_name(self.cell_to)


@dataclass(frozen=True)
class WallCrossing:
    """The ray crossing a wall of a convex room (``room_unfold``) into the next copy."""

    distance: float  # along the ray from its start, in world units
    wall: str  # the real wall it reflects off (``wall_name``: a, b, c, ...)
    index: int  # that wall's number in the room, from 0
    word: tuple[int, ...]  # the walls crossed so far, this one last: the copy entered
    point: tuple[float, float, float]  # where it crosses, unfolded (on the straight line)
    wall_point: tuple[float, float, float]  # the same point in the real room: the reflection
    hinge: Hinge  # the turn that carries the copy left onto the copy entered

    @property
    def name(self) -> str:
        """The name of the copy the ray enters (``word_name``)."""
        return word_name(self.word)


@dataclass(frozen=True)
class Unfolding:
    start: tuple[float, float, float]
    direction: tuple[float, float, float]  # unit
    crossings: tuple[Crossing, ...] | tuple[WallCrossing, ...]
    end: float  # distance to where the ray leaves the last copy (its far wall)
    real_end: tuple[float, float, float] | None = None  # that point in the real room (convex)

    @property
    def end_point(self) -> np.ndarray:
        return np.asarray(self.start) + self.end * np.asarray(self.direction)

    def chains(self) -> list[tuple[Hinge, ...]]:
        """Each copy's chain of hinges, first crossing first (copy k: the first k + 1)."""
        out: list[tuple[Hinge, ...]] = []
        for c in self.crossings:
            out.append((*(out[-1] if out else ()), c.hinge))
        return out


def _box(lo, hi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(hi < lo):
        raise ValueError("a box needs lo <= hi, three numbers each")
    return lo, hi, hi - lo


def hinge_axis(axis: int, side: int) -> np.ndarray:
    """The direction of the hinge for a wall with normal ``axis`` on ``side``: along the wall
    and level (the y axis for an x wall, x for a y wall), signed so that half way through
    the turn the copy stands straight up out of the wall (towards +z, towards the viewer
    in 2D). The floor and ceiling turn about the x axis and swing towards -y."""
    u = np.zeros(3)
    if axis == 0:
        u[1] = side
    elif axis == 1:
        u[0] = -side
    else:
        u[0] = -side
    return u


def box_unfold(lo, hi, start, direction, flips: int) -> Unfolding:
    """The first ``flips`` walls a ray from ``start`` along ``direction`` crosses through
    the lattice of mirror copies of the box [lo, hi]. A box flat along an axis (a
    rectangle at z = c) is 2D: the ray must not move along that axis. When the ray meets
    a corner (two walls at once) both count, in axis order, at the same distance."""
    lo, hi, size = _box(lo, hi)
    p0 = np.asarray(start, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-12:
        raise ValueError("the ray needs a direction")
    d = d / np.linalg.norm(d)
    scale = float(size.max())
    if scale <= 0:
        raise ValueError("the room has no size")
    tol = EPS * scale
    moving = np.abs(d) > 1e-12
    if np.any(moving & (size <= tol)):
        raise ValueError("the ray leaves the plane of a flat room; give it no z")
    if np.any(p0 < lo - tol) or np.any(p0 > hi + tol):
        raise ValueError("the ray must start inside the room")
    if not 0 <= flips <= MAX_FLIPS:
        raise ValueError(f"flips must be 0 to {MAX_FLIPS}")
    cell = np.zeros(3, dtype=np.int64)

    def exits(c: np.ndarray) -> np.ndarray:
        """Distance to each axis's exit wall of copy ``c`` (inf where the ray is still)."""
        out = np.full(3, np.inf)
        wall = lo + (c + (d > 0)) * size
        out[moving] = (wall[moving] - p0[moving]) / d[moving]
        return out

    crossings: list[Crossing] = []
    while len(crossings) < flips:
        t = exits(cell)
        tmin = float(t.min())
        for ax in (int(a) for a in np.flatnonzero(t <= tmin + tol)):
            if len(crossings) == flips:
                break
            side = 1 if d[ax] > 0 else -1
            nxt = cell.copy()
            nxt[ax] += side
            q = p0 + tmin * d
            pivot = lo + (cell + 0.5) * size  # the middle of the copy being left
            pivot[ax] = lo[ax] + (cell[ax] + (side > 0)) * size[ax]  # on the wall
            hinge = Hinge(_tup(pivot), _tup(hinge_axis(ax, side)), 180.0)
            real = fold(lo, hi, q[None])[0]
            # Which real wall: the copy's wall folded back; a copy with an odd index on this
            # axis is mirrored, so its high wall is the room's low one.
            real_side = side if cell[ax] % 2 == 0 else -side
            crossings.append(Crossing(
                distance=tmin, axis=ax, side=side, cell_from=_itup(cell), cell_to=_itup(nxt),
                point=_tup(q), wall_point=_tup(real), wall=WALLS[ax][real_side > 0],
                hinge=hinge,
            ))  # fmt: skip
            cell = nxt
    end = float(exits(cell).min()) if moving.any() else 0.0
    return Unfolding(_tup(p0), _tup(d), tuple(crossings), end)


def _tup(v) -> tuple[float, float, float]:
    return tuple(float(x) for x in v)  # type: ignore[return-value]


def _itup(v) -> tuple[int, int, int]:
    return tuple(int(x) for x in v)  # type: ignore[return-value]


def hinge_matrix(point, axis, degrees: float) -> np.ndarray:
    """4x4: the turn of ``degrees`` about the line through ``point`` along ``axis``."""
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    a = np.radians(degrees)
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    r = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)  # Rodrigues
    if abs(degrees % 180) < 1e-12:  # exact half and whole turns: no round-off
        r = np.round(r, 12)
    p = np.asarray(point, dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = p - r @ p
    return m


def reflection_matrix(point, normal) -> np.ndarray:
    """4x4: the reflection in the plane through ``point`` with ``normal``."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    p = np.asarray(point, dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] -= 2.0 * np.outer(n, n)
    m[:3, 3] = 2.0 * (n @ p) * n
    return m


def chain_matrix(hinges: Sequence, turn: float = 1.0) -> np.ndarray:
    """4x4: the hinges applied in order (the first one first), the last only ``turn`` of
    the way (0-1). ``hinges``: ``Hinge`` or (point, axis, degrees[, mirror]) tuples; a
    hinge with a mirror mirrors first (whatever the turn), then turns."""
    m = np.eye(4)
    n = len(hinges)
    for k, h in enumerate(hinges):
        point, axis, degrees, *mirror = h.as_params() if isinstance(h, Hinge) else h
        share = float(np.clip(turn, 0.0, 1.0)) if k == n - 1 else 1.0
        if mirror and mirror[0] is not None:
            m = reflection_matrix(point, mirror[0]) @ m
        m = hinge_matrix(point, axis, degrees * share) @ m
    return m


def apply(m: np.ndarray, points) -> np.ndarray:
    """Points (..., 3) moved by the 4x4 ``m``."""
    p = np.asarray(points, dtype=np.float64)
    return p @ m[:3, :3].T + m[:3, 3]


def fold(lo, hi, points) -> np.ndarray:
    """Points of the unfolded frame (N, 3) back in the real room: each axis folded like a
    mirror at every wall (a triangle wave). A flat axis stays as it is."""
    lo, hi, size = _box(lo, hi)
    p = np.array(points, dtype=np.float64, copy=True)
    for ax in range(3):
        if size[ax] <= 0:
            continue
        u = np.mod(p[..., ax] - lo[ax], 2 * size[ax])
        p[..., ax] = lo[ax] + np.where(u <= size[ax], u, 2 * size[ax] - u)
    return p


def image_point(lo, hi, point, cell) -> np.ndarray:
    """The image of ``point`` (in the room) in the mirror copy with lattice index ``cell``:
    an image source when ``point`` is the source."""
    lo, hi, size = _box(lo, hi)
    q = np.asarray(point, dtype=np.float64)
    c = np.asarray(cell, dtype=np.int64)
    odd = c % 2 == 1
    return np.where(odd, lo + (c + 1) * size - (q - lo), q + c * size)


def zigzag(u: Unfolding, lo=None, hi=None) -> np.ndarray:
    """The path the ray really takes in the room: (K + 2, 3), from the start through every
    reflection to where the straight line leaves the last copy, folded back. A box's
    unfolding needs the box (``lo``, ``hi``); a convex room's (``room_unfold``) does not."""
    if u.real_end is not None:
        return np.asarray([u.start, *(c.wall_point for c in u.crossings), u.real_end])
    pts = [u.start, *(c.point for c in u.crossings), tuple(u.end_point)]
    return fold(lo, hi, np.asarray(pts))


def cell_name(cell) -> str:
    """A copy's name from its place in the lattice: ``e2_n1`` is two rooms east and one
    north; the room itself is ``home``."""
    parts = [f"{AXIS_WORDS[ax][c < 0]}{abs(c)}" for ax, c in enumerate(cell) if c]
    return "_".join(parts) or "home"


# ---------------------------------------------------------------- convex rooms


@dataclass(frozen=True, eq=False)
class Polytope:
    """A convex room as its walls: the room is where ``normals @ x <= offsets`` for every
    wall (outward unit normals). ``flat``: a polygon in the plane z = ``vertices[0, 2]``,
    each wall an edge standing straight up out of that plane (a ray in it has no z)."""

    normals: np.ndarray  # (W, 3)
    offsets: np.ndarray  # (W,)
    centers: np.ndarray  # (W, 3) the middle of each wall (a flat room: of each edge)
    vertices: np.ndarray  # (V, 3) its corners
    flat: bool

    @property
    def scale(self) -> float:
        v = self.vertices
        return float(np.max(v.max(axis=0) - v.min(axis=0)))

    @property
    def names(self) -> tuple[str, ...]:
        """The walls' names, in order: a, b, c, ..."""
        return tuple(wall_name(k) for k in range(len(self.offsets)))

    def contains(self, point, tol: float | None = None) -> bool:
        tol = EPS * self.scale if tol is None else tol
        p = np.asarray(point, dtype=np.float64)
        if self.flat and abs(p[2] - self.vertices[0, 2]) > tol:
            return False
        return bool(np.all(self.normals @ p <= self.offsets + tol))


def wall_name(k: int) -> str:
    """A wall's name from its number (from 0): ``a`` to ``z``, then ``aa``, ``ab``, ...
    (the way spreadsheet columns go)."""
    if k < 0:
        raise ValueError("a wall number is 0 or more")
    out = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        out = chr(ord("a") + r) + out
    return out


def word_name(word: Sequence[int]) -> str:
    """A copy's name from the walls crossed to reach it, in order: ``a_c_b`` is the room
    mirrored in wall a, then in (the copy's) wall c, then b; the room itself is ``home``."""
    return "_".join(wall_name(int(k)) for k in word) or "home"


def _frozen(*arrays) -> None:
    for a in arrays:
        a.setflags(write=False)


def _turns(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each edge (to the next corner) and the turn (cross product) at its end."""
    e = np.roll(p, -1, axis=0) - p
    f = np.roll(e, -1, axis=0)
    return e, e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]


def polygon(points) -> Polytope:
    """A convex polygon room from its corners in order, either way round (a closing point
    equal to the first is dropped; so are corners where the outline goes straight on).
    Wall k runs from corner k to corner k + 1 (named ``a``, ``b``, ...). The corners share
    one z; the room is flat (2D)."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] not in (2, 3):
        raise ValueError("a polygon needs points (x, y) or (x, y, z)")
    if p.shape[1] == 2:
        p = np.c_[p, np.zeros(len(p))]
    scale = float(np.max(np.ptp(p, axis=0))) if len(p) else 0.0
    tol = EPS * max(scale, 1e-300)
    if np.ptp(p[:, 2]) > tol:
        raise ValueError("a polygon room lies in one plane z = c; its corners share one z")
    nxt = np.roll(p, -1, axis=0)  # a corner repeated (the closing one) counts once
    p = p[[k for k in range(len(p)) if np.linalg.norm(p[k] - nxt[k]) > tol]]
    while len(p) >= 3:  # corners where the outline goes straight on are no corners
        _, turn = _turns(np.roll(p, 1, axis=0))  # the turn at each corner
        straight = np.abs(turn) <= tol * scale
        if not straight.any():
            break
        p = p[~straight]
    if len(p) < 3:
        raise ValueError("a polygon room needs three corners or more, not all in a line")
    e, turn = _turns(p)
    angle = np.arctan2(turn, np.einsum("ij,ij->i", e, np.roll(e, -1, axis=0)))
    if not (np.all(turn > 0) or np.all(turn < 0)) or abs(angle.sum()) > 2 * np.pi + 1e-6:
        raise ValueError("the polygon is not convex: a ray can leave a non-convex mirror copy")
    sign = 1.0 if turn[0] > 0 else -1.0  # counterclockwise: the outside is to the right
    n = np.c_[e[:, 1], -e[:, 0], np.zeros(len(e))] * sign
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    d = np.einsum("ij,ij->i", n, p)
    centers = p + e / 2
    _frozen(n, d, centers, p)
    return Polytope(n, d, centers, p, True)


def polyhedron(vertices, triangles) -> Polytope:
    """A convex room from a closed triangle mesh (a box's or any ``.obj``): its walls are
    the planes its triangles lie in, numbered in the order the mesh first uses them (wall
    ``a`` is the plane of the first triangle). A mesh that is not convex is refused."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(triangles, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or not len(f):
        raise ValueError("a mesh needs vertices (N, 3) and triangles (M, 3)")
    corners = v[np.unique(f)]
    scale = float(np.max(np.ptp(corners, axis=0)))
    if scale <= 0:
        raise ValueError("the room has no size")
    tol = 1e-6 * scale
    inner = corners.mean(axis=0)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    groups: list[list[int]] = []
    normals: list[np.ndarray] = []
    offsets: list[float] = []
    for t in np.flatnonzero(area > 1e-12 * scale * scale):
        n = cross[t] / area[t]
        d = float(n @ tri[t, 0])
        if n @ inner > d:  # face the outside
            n, d = -n, -d
        for g, (gn, gd) in enumerate(zip(normals, offsets, strict=True)):
            if np.linalg.norm(gn - n) < 1e-6 and abs(gd - d) < tol:
                groups[g].append(int(t))
                break
        else:
            groups.append([int(t)])
            normals.append(n)
            offsets.append(d)
    if len(normals) < 4:
        raise ValueError("the mesh does not close round a room")
    n_arr, d_arr = np.array(normals), np.array(offsets)
    if np.any(corners @ n_arr.T - d_arr > tol):
        raise ValueError("the mesh is not convex: a ray can leave a non-convex mirror copy")
    centers = np.array([(tri[g].mean(axis=1) * area[g, None]).sum(axis=0) / area[g].sum()
                        for g in groups])  # fmt: skip
    _frozen(n_arr, d_arr, centers, corners)
    return Polytope(n_arr, d_arr, centers, corners, False)


def _wall_in(room: Polytope, m: np.ndarray, k: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Wall ``k`` of the copy ``m`` (a 4x4) of the room: its normal, offset and middle."""
    r, t = m[:3, :3], m[:3, 3]
    n = r @ room.normals[k]
    return n, float(room.offsets[k] + n @ t), r @ room.centers[k] + t


def _same_points(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return bool(np.all(dist.min(axis=1) <= tol) and np.all(dist.min(axis=0) <= tol))


def _wall_hinge(room: Polytope, m: np.ndarray, k: int) -> Hinge:
    """The hinge that carries the copy ``m`` over its wall ``k`` onto the copy across it:
    half a turn about a line through the wall's middle (a flat room: about its edge)."""
    n, _, c = _wall_in(room, m, k)
    level = np.cross([0.0, 0.0, 1.0], n)  # along the wall and level: lifts the copy up (+z)
    if np.linalg.norm(level) < 1e-9:  # a floor or a ceiling: about x, as in a box
        level = np.array([-np.sign(n[2]), 0.0, 0.0])
    level /= np.linalg.norm(level)
    if room.flat:  # half a turn about the edge is the mirror in the plane: exact
        return Hinge(_tup(c), _tup(level), 180.0)
    corners = apply(m, room.vertices)
    tol = 1e-6 * room.scale
    for axis in (level, np.cross(n, level)):
        square = np.cross(axis, n)  # the plane through the line, square to the wall
        if _same_points(apply(reflection_matrix(c, square), corners), corners, tol):
            return Hinge(_tup(c), _tup(axis), 180.0)  # symmetric about it: a turn will do
    return Hinge(_tup(c), _tup(level), 180.0, _tup(np.cross(level, n)))


def _reflect(room: Polytope, m: np.ndarray, k: int) -> np.ndarray:
    n, d, _ = _wall_in(room, m, k)
    return reflection_matrix(d * n, n) @ m


def copy_matrix(room: Polytope, word: Sequence[int]) -> np.ndarray:
    """4x4: the true mirror copy of the room reached by reflecting in its wall ``word[0]``,
    then in the copy's wall ``word[1]``, ... (a reflection for an odd number of walls)."""
    m = np.eye(4)
    for k in word:
        m = _reflect(room, m, int(k))
    return m


def mirror_chain(room: Polytope, word: Sequence[int]) -> tuple[Hinge, ...]:
    """The hinges (``HingedCopy``'s ``hinges``) that carry the room onto the mirror copy
    ``copy_matrix(room, word)``, one half turn per wall: a kaleidoscope, copy by copy."""
    m = np.eye(4)
    out = []
    for k in word:
        out.append(_wall_hinge(room, m, int(k)))
        m = _reflect(room, m, int(k))
    return tuple(out)


def _side_steps(d: np.ndarray, flat: bool) -> tuple[np.ndarray, np.ndarray]:
    """Two directions square to the ray and to each other: a corner is passed as a ray
    moved a hair along the first (then a hair less along the second) would pass it."""
    # flat: to the left of the ray, in the plane; else square to it and its least axis
    other = np.eye(3)[int(np.argmin(np.abs(d)))]
    s1 = np.cross([0.0, 0.0, 1.0], d) if flat else np.cross(d, other)
    s1 /= np.linalg.norm(s1)
    return s1, np.cross(d, s1)


def room_unfold(room: Polytope, start, direction, flips: int) -> Unfolding:
    """``box_unfold`` for any convex room: the first ``flips`` walls a ray from ``start``
    along ``direction`` crosses, each into the mirror copy of the copy it leaves, across
    that wall. Where it meets a corner (or an edge, in 3D) it passes as a ray a hair to its
    left would: one wall after the other at the same distance. For a corner of 180/n
    degrees (a box's, a kaleidoscope's) every nearby ray lands in the same copy; at other
    corners the two sides part, and this is the left one."""
    p0 = np.asarray(start, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-12:
        raise ValueError("the ray needs a direction")
    d = d / np.linalg.norm(d)
    if room.flat and abs(d[2]) > 1e-12:
        raise ValueError("the ray leaves the plane of a flat room; give it no z")
    if not room.contains(p0):
        raise ValueError("the ray must start inside the room")
    if not 0 <= flips <= MAX_FLIPS:
        raise ValueError(f"flips must be 0 to {MAX_FLIPS}")
    tol = EPS * room.scale
    s1, s2 = _side_steps(d, room.flat)

    def later(a, b) -> bool:
        """``a`` after ``b``: by distance, then (at a corner) as the moved rays pass."""
        for x, y, eps in ((a[0], b[0], tol), (a[1], b[1], 1e-9), (a[2], b[2], 1e-9)):
            if abs(x - y) > eps:
                return x > y
        return False

    def next_wall(m: np.ndarray, now) -> tuple[int, tuple[float, float, float]]:
        """The wall of copy ``m`` the ray leaves by after ``now``, and when."""
        r, t = m[:3, :3], m[:3, 3]
        n = room.normals @ r.T
        off = room.offsets + n @ t
        nd = n @ d
        best, key = -1, None
        for k in np.flatnonzero(nd > 1e-12):
            cand = (float((off[k] - n[k] @ p0) / nd[k]), float(-(n[k] @ s1) / nd[k]),
                    float(-(n[k] @ s2) / nd[k]))  # fmt: skip
            if later(cand, now) and (key is None or later(key, cand)):
                best, key = int(k), cand
        if key is None:
            raise ValueError("the ray never leaves the room")
        return best, key

    m = np.eye(4)
    now: tuple[float, float, float] = (0.0, -np.inf, -np.inf)
    word: list[int] = []
    crossings: list[WallCrossing] = []
    while len(crossings) < flips:
        k, now = next_wall(m, now)
        q = p0 + now[0] * d
        hinge = _wall_hinge(room, m, k)
        real = apply(np.linalg.inv(m), q)
        m = _reflect(room, m, k)
        word.append(k)
        crossings.append(WallCrossing(
            distance=now[0], wall=wall_name(k), index=k, word=tuple(word), point=_tup(q),
            wall_point=_tup(real), hinge=hinge,
        ))  # fmt: skip
    end = next_wall(m, now)[1][0]
    real_end = apply(np.linalg.inv(m), p0 + end * d)
    return Unfolding(_tup(p0), _tup(d), tuple(crossings), end, _tup(real_end))
