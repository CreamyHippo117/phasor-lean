"""Nets and geodesics of convex polyhedra: the surface laid flat, a shortest path straight.

Cut a convex polyhedron along some of its edges and it opens out flat: a net. Which faces
stay joined is a spanning tree of the faces (a face hangs on its parent by their shared
edge); each face reaches the plane of the root face through the chain of hinge turns from
it up the tree, the same ``Hinge`` chains as the mirror rooms (``core/unfold_math.py``),
every turn ``amount`` of the way (0: the closed solid, 1: the flat net). A turn is about
the shared edge by the angle between the two faces' outward normals (a quarter turn on a
cube), so the faces always keep their outside on the same side.

A shortest path on the surface (a geodesic) crosses a run of faces, each at most once and
never through a vertex; laid flat along that run it is a straight line. ``geodesic`` tries
every run, unfolding it into the first face's plane, keeps the straight lines that really
pass through each shared edge (a window of directions that narrows edge by edge), and
returns the shortest: the spider and the fly.

Pure NumPy, no drawing; ``domains/net.py`` draws it.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phasor.core import unfold_math
from phasor.core.unfold_math import Hinge

MAX_FACES = 64
TOL = 1e-9  # relative to the solid's size

# A box's faces: the axis of the outward normal and its sign, in the order they are stored.
BOX_FACES = (("bottom", 2, -1), ("top", 2, 1), ("south", 1, -1), ("north", 1, 1),
             ("west", 0, -1), ("east", 0, 1))  # fmt: skip


@dataclass(frozen=True)
class Solid:
    """A closed convex polyhedron: ``faces`` index ``vertices``, counterclockwise seen from
    outside; ``names`` name the faces."""

    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    names: tuple[str, ...]

    @functools.cached_property
    def v(self) -> np.ndarray:
        return np.asarray(self.vertices, dtype=np.float64)

    @functools.cached_property
    def size(self) -> float:
        return float(np.ptp(self.v, axis=0).max())

    @functools.cached_property
    def normals(self) -> np.ndarray:
        out = []
        for f in self.faces:
            p = self.v[list(f)]
            n = np.zeros(3)
            for a, b in zip(p, np.roll(p, -1, axis=0), strict=True):  # Newell's method
                n += np.cross(a, b)
            out.append(n / np.linalg.norm(n))
        return np.asarray(out)

    @functools.cached_property
    def edges(self) -> dict[tuple[int, int], tuple[int, int]]:
        """(face, neighbour) -> the shared edge (i, j), counterclockwise round ``face``."""
        owner: dict[tuple[int, int], int] = {}
        for k, f in enumerate(self.faces):
            for i, j in zip(f, f[1:] + f[:1], strict=True):
                owner[(i, j)] = k
        out = {}
        for (i, j), k in owner.items():
            out[(k, owner[(j, i)])] = (i, j)
        return out

    def neighbours(self, face: int) -> list[int]:
        return sorted(g for (f, g) in self.edges if f == face)

    def index(self, name: str) -> int:
        if name not in self.names:
            raise ValueError(f"no face {name!r}; faces are {list(self.names)}")
        return self.names.index(name)

    def faces_at(self, point) -> list[int]:
        """The faces a point on the surface lies on (more than one on an edge or corner)."""
        q = np.asarray(point, dtype=np.float64)
        tol = 1e-6 * self.size
        out = []
        for k, f in enumerate(self.faces):
            p, n = self.v[list(f)], self.normals[k]
            if abs((q - p[0]) @ n) > tol:
                continue
            nxt = np.roll(p, -1, axis=0)
            if all(np.cross(b - a, q - a) @ n >= -tol * np.linalg.norm(b - a)
                   for a, b in zip(p, nxt, strict=True)):  # fmt: skip
                out.append(k)
        return out


def box(size=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)) -> Solid:
    """A box with its corner at ``origin``; faces ``bottom``, ``top`` (z), ``south``,
    ``north`` (y), ``west``, ``east`` (x)."""
    o, s = np.asarray(origin, dtype=np.float64), np.asarray(size, dtype=np.float64)
    if s.shape != (3,) or np.any(s <= 0):
        raise ValueError("a box needs three positive sizes")
    corners = [tuple(float(x) for x in o + s * (i & 1, i >> 1 & 1, i >> 2 & 1)) for i in range(8)]
    faces = []
    for _, ax, sign in BOX_FACES:
        u, w = (ax + 1) % 3, (ax + 2) % 3
        bit = 1 << ax if sign > 0 else 0
        quad = [bit, bit | 1 << u, bit | 1 << u | 1 << w, bit | 1 << w]  # ccw about +axis
        faces.append(tuple(quad if sign > 0 else quad[::-1]))
    return Solid(tuple(corners), tuple(faces), tuple(n for n, _, _ in BOX_FACES))


def solid(vertices, faces, names: Sequence[str] = ()) -> Solid:
    """A convex polyhedron from its vertices and faces (vertex indices, any winding: each
    face is turned to run counterclockwise seen from outside). Refuses a surface that is not
    closed, a face that is not flat and convex, or a solid that is not convex."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4:
        raise ValueError("a polyhedron needs at least four vertices (x, y, z)")
    fs = [tuple(int(i) for i in f) for f in faces]
    if not 4 <= len(fs) <= MAX_FACES:
        raise ValueError(f"a polyhedron needs 4 to {MAX_FACES} faces")
    if any(len(f) < 3 or len(set(f)) != len(f) or min(f) < 0 or max(f) >= len(v) for f in fs):
        raise ValueError("each face needs three or more distinct vertex indices")
    names = tuple(names) or tuple(f"f{k}" for k in range(len(fs)))
    if len(names) != len(fs) or len(set(names)) != len(names):
        raise ValueError("give one distinct name per face")
    centre = v.mean(axis=0)
    size = float(np.ptp(v, axis=0).max())
    out = []
    for f in fs:
        p = v[list(f)]
        n = np.zeros(3)
        for a, b in zip(p, np.roll(p, -1, axis=0), strict=True):
            n += np.cross(a, b)
        if np.linalg.norm(n) < 1e-12 * size**2:
            raise ValueError(f"face {f} has no area")
        n /= np.linalg.norm(n)
        if (p.mean(axis=0) - centre) @ n < 0:
            f, n = f[::-1], -n
            p = v[list(f)]
        if np.max(np.abs((p - p[0]) @ n)) > 1e-6 * size:
            raise ValueError(f"face {f} is not flat")
        turns = np.cross(np.roll(p, -1, axis=0) - p, np.roll(p, -2, axis=0) - p) @ n
        if np.any(turns < -1e-9 * size**2):
            raise ValueError(f"face {f} is not convex")
        if np.any((v - p[0]) @ n > 1e-6 * size):
            raise ValueError("the polyhedron is not convex")
        out.append(f)
    s = Solid(tuple(tuple(float(x) for x in r) for r in v), tuple(out), names)
    directed = [(i, j) for f in out for i, j in zip(f, f[1:] + f[:1], strict=True)]
    if len(set(directed)) != len(directed) or any((j, i) not in set(directed)
                                                  for i, j in directed):  # fmt: skip
        raise ValueError("the faces do not close up: every edge must join exactly two faces")
    return s


# ---------------------------------------------------------------- hinges and nets


def hinge(s: Solid, face: int, onto: int) -> Hinge:
    """The turn about their shared edge that lays ``face`` flat beside ``onto``, outside
    up: by the angle between their outward normals (180 minus the dihedral angle)."""
    if (face, onto) not in s.edges:
        raise ValueError(f"faces {s.names[face]!r} and {s.names[onto]!r} share no edge")
    i, j = s.edges[(face, onto)]
    a, b = s.normals[face], s.normals[onto]
    axis = np.cross(a, b)
    angle = float(np.degrees(np.arctan2(np.linalg.norm(axis), a @ b)))
    if np.linalg.norm(axis) < 1e-12:  # already flat: any turn of 0 about the edge
        axis = s.v[j] - s.v[i]
    axis = axis / np.linalg.norm(axis)
    return Hinge(_tup(s.v[i]), _tup(axis), angle)


def spanning_tree(s: Solid, root: int, keep: Sequence[int] = ()) -> dict[int, int]:
    """face -> the face it hangs on, every face but ``root``. ``keep``: a run of faces (each
    sharing an edge with the next) that stays joined, so that a path across it lies
    straight on the net. The rest join breadth first, lowest index first."""
    keep = list(keep)
    for f, g in itertools.pairwise(keep):
        if (f, g) not in s.edges:
            raise ValueError(f"faces {s.names[f]!r} and {s.names[g]!r} share no edge")
    if len(set(keep)) != len(keep):
        raise ValueError("a run of faces visits each face once")
    links: dict[int, set[int]] = {k: set() for k in range(len(s.faces))}
    for f, g in itertools.pairwise(keep):
        links[f].add(g)
        links[g].add(f)
    seen = set(keep) or {root}
    queue = list(keep) or [root]
    while queue:
        f = queue.pop(0)
        for g in s.neighbours(f):
            if g not in seen:
                seen.add(g)
                links[f].add(g)
                links[g].add(f)
                queue.append(g)
    parent: dict[int, int] = {}
    order, done = [root], {root}
    while order:
        f = order.pop(0)
        for g in sorted(links[f]):
            if g not in done:
                done.add(g)
                parent[g] = f
                order.append(g)
    return parent


def check_tree(s: Solid, root: int, parent: dict[int, int]) -> None:
    """Refuse a tree that does not join every face to ``root`` along shared edges."""
    for f in range(len(s.faces)):
        seen, g = set(), f
        while g != root:
            if g in seen or g not in parent:
                raise ValueError(f"face {s.names[f]!r} does not hang on {s.names[root]!r}")
            if (g, parent[g]) not in s.edges:
                raise ValueError(f"{s.names[g]!r} and {s.names[parent[g]]!r} share no edge")
            seen.add(g)
            g = parent[g]


def chains(s: Solid, root: int, parent: dict[int, int]) -> list[tuple[Hinge, ...]]:
    """Each face's chain of hinges, its own turn first, up to the root (the root's: empty)."""
    out = []
    for f in range(len(s.faces)):
        chain, g = [], f
        while g != root:
            chain.append(hinge(s, g, parent[g]))
            g = parent[g]
        out.append(tuple(chain))
    return out


def net_matrices(s: Solid, root: int, parent: dict[int, int], amount: float) -> np.ndarray:
    """(F, 4, 4): where each face is ``amount`` of the way (0-1) from the solid to the net.
    Every turn of every chain goes ``amount`` of the way, so the faces open together and a
    face and its parent always meet along their shared edge."""
    a = float(np.clip(amount, 0.0, 1.0))
    return np.stack([
        unfold_math.chain_matrix([(h.point, h.axis, h.degrees * a) for h in chain])
        for chain in chains(s, root, parent)
    ])  # fmt: skip


# ---------------------------------------------------------------- geodesics


@dataclass(frozen=True)
class Geodesic:
    """A shortest path on the surface: ``faces`` crossed in order, ``points`` (K + 2, 3) on
    the solid (the start, where it crosses each shared edge, the end)."""

    length: float
    faces: tuple[int, ...]
    points: np.ndarray


def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _clip(lo, hi, p, d, ray, sign, tol) -> tuple[float, float]:
    """Narrow [lo, hi] of s in p + s d to where sign * cross(ray, p + s d) >= 0."""
    c0, c1 = sign * _cross2(ray, p), sign * _cross2(ray, d)
    if abs(c1) < 1e-300:
        return (lo, hi) if c0 >= -tol else (1.0, 0.0)
    s = -c0 / c1
    return (max(lo, s - tol), hi) if c1 > 0 else (lo, min(hi, s + tol))


def geodesic(s: Solid, a, b) -> Geodesic:
    """The shortest path on the surface from ``a`` to ``b`` (both on it). When several are
    as short (by symmetry), the first found: faces tried lowest index first."""
    pa, pb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    starts, ends = s.faces_at(pa), s.faces_at(pb)
    if not starts or not ends:
        raise ValueError("both ends of a geodesic must lie on the surface")
    tol = TOL * s.size
    best: list = [np.inf, None, None]  # length, faces, crossings (edge, s) in order
    shared = set(starts) & set(ends)
    if shared:
        return Geodesic(float(np.linalg.norm(pb - pa)), (min(shared),), np.stack([pa, pb]))

    for f0 in starts:
        n0 = s.normals[f0]
        e1 = s.v[s.faces[f0][1]] - s.v[s.faces[f0][0]]
        e1 /= np.linalg.norm(e1)
        basis = np.stack([e1, np.cross(n0, e1)])  # the plane of f0, a at the origin

        def flat(m, pts, basis=basis):
            return (unfold_math.apply(m, pts) - pa) @ basis.T

        def walk(run, m, left, right, cuts):
            """``run``: faces so far (unfolded into f0's plane by ``m``, the last one's);
            the straight lines from a through the window [left, right] (2D, on the last
            edge crossed) reach the last face."""
            f = run[-1]
            if cuts:  # how near a can come to the window bounds the length below
                d = right - left
                u = np.clip(-(left @ d) / max(d @ d, 1e-300), 0.0, 1.0)
                if np.linalg.norm(left + u * d) >= best[0] - tol:
                    return
            if f in ends:
                q = flat(m, pb[None])[0]
                inside = not cuts or (
                    _cross2(left, q) * _cross2(left, right) >= -tol * np.linalg.norm(q)
                    and _cross2(q, right) * _cross2(left, right) >= -tol * np.linalg.norm(q)
                )
                if inside and np.linalg.norm(q) < best[0] - tol:
                    best[:] = [float(np.linalg.norm(q)), tuple(run), list(cuts)]
            if len(run) == len(s.faces):
                return
            for g in s.neighbours(f):
                if g in run:
                    continue
                i, j = s.edges[(f, g)]
                p, q = flat(m, s.v[[i, j]])
                d = q - p
                if not cuts:
                    if abs(_cross2(d, -p)) <= tol * np.linalg.norm(d):
                        continue  # a lies on this edge: starting in g covers it
                    lo, hi = 0.0, 1.0
                else:
                    sign = 1.0 if _cross2(left, right) > 0 else -1.0
                    lo, hi = _clip(0.0, 1.0, p, d, left, sign, tol)
                    lo, hi = _clip(lo, hi, p, d, right, -sign, tol)
                if hi - lo <= tol / max(np.linalg.norm(d), 1e-300):
                    continue  # through a vertex at best: never shortest on a convex solid
                mg = m @ unfold_math.hinge_matrix(*hinge(s, g, f).as_params())
                walk([*run, g], mg, p + lo * d, p + hi * d, [*cuts, (f, g)])

        walk([f0], np.eye(4), None, None, [])
    if best[1] is None:
        raise ValueError("no path found between the two points")
    length, faces, cuts = best
    # Where the straight line crosses each shared edge, back on the solid.
    pts = [pa]
    f0 = faces[0]
    n0 = s.normals[f0]
    e1 = s.v[s.faces[f0][1]] - s.v[s.faces[f0][0]]
    e1 /= np.linalg.norm(e1)
    basis = np.stack([e1, np.cross(n0, e1)])
    m_seq = [np.eye(4)]
    for f, g in cuts:
        m_seq.append(m_seq[-1] @ unfold_math.hinge_matrix(*hinge(s, g, f).as_params()))
    end = (unfold_math.apply(m_seq[-1], pb[None])[0] - pa) @ basis.T
    for k, (f, g) in enumerate(cuts):
        i, j = s.edges[(f, g)]
        p, q = (unfold_math.apply(m_seq[k], s.v[[i, j]]) - pa) @ basis.T
        d = q - p
        denom = _cross2(d, end)
        u = 0.0 if abs(denom) < 1e-300 else -_cross2(p, end) / denom
        u = float(np.clip(u, 0.0, 1.0))
        pts.append(s.v[i] + u * (s.v[j] - s.v[i]))
    pts.append(pb)
    return Geodesic(length, faces, np.asarray(pts))


def straight_on_net(points_flat: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether a polyline (N, 3) is one straight segment."""
    p = np.asarray(points_flat, dtype=np.float64)
    d = p[-1] - p[0]
    length = np.linalg.norm(d)
    if length == 0:
        return True
    off = np.cross(p - p[0], d / length)
    return bool(np.max(np.linalg.norm(off, axis=1)) <= tol * max(length, 1.0))


def _tup(v) -> tuple[float, float, float]:
    return tuple(float(x) for x in v)  # type: ignore[return-value]
