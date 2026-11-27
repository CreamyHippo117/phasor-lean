"""Acoustics numerics: pure NumPy, no drawing.

- ``image_sources``: the image-source method for a shoebox room (exact for a rectangular room
  with frequency-independent walls).
- ``impulse_response``: the arrivals (time, amplitude) and the sampled IR.
- ``rt60_fit`` (Schroeder backward integration), ``sabine_rt60`` and ``eyring_rt60`` (theory).
- ``convolve`` (FFT convolution reverb).
- ``RoomFDTD``: a 3D finite-difference pressure field as a ``Simulation`` (fixed dt, seed,
  checkpointed), with the stability limit enforced.
- Rooms as triangle meshes: ``box_mesh``, ``load_obj`` (Wavefront ``.obj``), ``feature_edges``,
  ``mesh_stats``; geometric acoustics on them: ``trace`` / ``rays`` (specular ray tracing),
  ``inside`` and ``sphere_wall_arcs`` (where a spherical front meets the walls).

Units: meters, seconds, speed of sound ``C`` (343 m/s) unless said otherwise.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from phasor.domains import fourier_math

C = 343.0
MATERIALS = {  # broadband absorption coefficients (energy absorbed per reflection)
    "concrete": 0.02,
    "glass": 0.04,
    "plaster": 0.04,
    "brick": 0.05,
    "wood": 0.10,
    "carpet": 0.30,
    "curtain": 0.50,
    "foam": 0.80,
    "anechoic": 0.99,
}
MAX_IMAGES = 2_000_000
MAX_ORDER = 113  # the highest order within MAX_IMAGES (image_count(113) is 1,949,703)


def doppler_wavefronts(
    mach: float, frequency: float, time: float, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the latest emitted rings as (x centres, radii), oldest first.

    Sound speed is one distance unit per second. The source starts at x=0, travels
    along +x, and emits at integer multiples of the period starting at time zero.
    """
    phase = time * frequency
    latest = math.floor(phase)
    pulses = np.arange(max(0, latest - count + 1), latest + 1, dtype=np.float64)
    age = (phase - pulses) / frequency
    visible = age > 0
    return mach * (pulses[visible] / frequency), age[visible]


def doppler_frequency(mach: float, frequency: float, direction_cosine) -> np.ndarray:
    """Local frequency along an emitted ray, given its angle cosine to +x.

    The absolute value gives the frequency magnitude on either arrival branch,
    including crests arriving in reverse order. At a caustic the ideal frequency
    diverges. This function describes a ray that reaches a listener.
    """
    arrival_rate = np.abs(1 - mach * np.asarray(direction_cosine, dtype=np.float64))
    with np.errstate(divide="ignore"):
        return frequency / arrival_rate


def mach_cone_rays(mach: float) -> np.ndarray:
    """The two unit rays of the trailing Mach cone in the xy plane (empty for M <= 1)."""
    if mach <= 1:
        return np.empty((0, 2))
    sine = 1 / mach
    cosine = np.sqrt((1 - sine) * (1 + sine))
    return np.array([[-cosine, sine], [-cosine, -sine]])


def absorption_of(material: str, absorption: float = -1.0) -> float:
    """``absorption`` if it is given (0-1), else the material's coefficient."""
    if absorption >= 0:
        return float(min(max(absorption, 0.0), 1.0))
    if material not in MATERIALS:
        raise ValueError(f"unknown material {material!r}; materials are {sorted(MATERIALS)}")
    return MATERIALS[material]


def room_stats(size) -> tuple[float, float]:
    """(volume, total wall area) of a shoebox room."""
    x, y, z = (float(v) for v in size)
    return x * y * z, 2 * (x * y + y * z + x * z)


def sabine_rt60(size, alpha: float, c: float = C, stats=None) -> float:
    """Time for sound to decay 60 dB, Sabine's formula: 0.161 V / (S alpha). ``stats``
    (volume, area) replaces the shoebox ``size`` for a room of any shape."""
    v, s = room_stats(size) if stats is None else stats
    return 0.161 * v / (s * alpha) if alpha > 0 else math.inf


def eyring_rt60(size, alpha: float, stats=None) -> float:
    """Eyring's formula, better than Sabine when walls absorb a lot (``stats`` as above)."""
    v, s = room_stats(size) if stats is None else stats
    return 0.161 * v / (-s * math.log(1.0 - alpha)) if 0 < alpha < 1 else (
        math.inf if alpha <= 0 else 0.0)  # fmt: skip


def image_count(order: int) -> int:
    """How many image sources ``image_sources`` makes for ``order`` (exactly, without making
    them): along each axis one image has 0 reflections and two have each count above."""
    w = np.r_[1, np.full(max(order, 0), 2)].astype(np.int64)
    return int(np.convolve(np.convolve(w, w), w)[: order + 1].sum())


def image_sources(size, source, order: int, alpha: float) -> dict[str, np.ndarray]:
    """Image sources of ``source`` in a shoebox room (corner at the origin) with up to
    ``order`` wall reflections. Returns arrays: ``pos`` (N, 3), ``reflections`` (N,), and
    ``gain`` (N,) = (sqrt(1 - alpha)) ** reflections (pressure reflection factor). The
    source itself (order 0) is first. Deterministic order: by reflections, then position.
    About (4/3) order^3 images; ``MAX_IMAGES`` caps it (order 113 or so)."""
    size = np.asarray(size, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    if order < 0:
        raise ValueError("order must be 0 or more")
    total = image_count(order)
    if total > MAX_IMAGES:
        raise ValueError(f"order {order} needs {total:,} image sources "
                         f"(limit {MAX_IMAGES:,}); lower the order")  # fmt: skip
    q = np.arange(-order, order + 1)
    axes = []
    for L, s in zip(size, src, strict=True):
        pos = np.concatenate([2 * L * q + s, 2 * L * q - s])
        refl = np.concatenate([np.abs(2 * q), np.abs(2 * q - 1)])
        keep = refl <= order
        axes.append((pos[keep], refl[keep].astype(np.int16)))
    # only the combinations within the order: the reflections summed first (small integers),
    # the positions gathered for those alone
    (px, rx), (py, ry), (pz, rz) = axes
    i, j, k = np.nonzero(rx[:, None, None] + ry[None, :, None] + rz[None, None, :] <= order)
    pos = np.stack([px[i], py[j], pz[k]], axis=1)
    refl = rx[i].astype(np.int64) + ry[j] + rz[k]
    key = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0], refl))
    pos, refl = pos[key], refl[key]
    return {"pos": pos, "reflections": refl, "gain": np.sqrt(1.0 - alpha) ** refl}


def arrivals(size, source, mic, order: int, alpha: float, c: float = C) -> dict[str, np.ndarray]:
    """What reaches the listener: for every image source its ``time`` (s), ``amplitude``
    (pressure: gain / distance, 1/r spreading), ``reflections`` and ``distance``; sorted by
    time. The first entry is the direct sound."""
    im = image_sources(size, source, order, alpha)
    d = np.linalg.norm(im["pos"] - np.asarray(mic, dtype=np.float64), axis=1)
    d = np.maximum(d, 1e-3)
    idx = np.argsort(d, kind="stable")
    return {
        "time": d[idx] / c,
        "amplitude": (im["gain"] / d)[idx],
        "reflections": im["reflections"][idx],
        "distance": d[idx],
        "pos": im["pos"][idx],
    }


def impulse_response(arr: dict[str, np.ndarray], fs: int, length: float) -> np.ndarray:
    """The sampled impulse response: each arrival is split between the two nearest samples
    (linear fractional delay) so timing is not rounded to a whole sample."""
    n = math.ceil(length * fs)
    ir = np.zeros(n + 2)
    pos = arr["time"] * fs
    i = np.floor(pos).astype(np.int64)
    frac = pos - i
    ok = i < n
    np.add.at(ir, i[ok], arr["amplitude"][ok] * (1 - frac[ok]))
    np.add.at(ir, i[ok] + 1, arr["amplitude"][ok] * frac[ok])
    return ir[:n]


def schroeder_db(ir: np.ndarray) -> np.ndarray:
    """The Schroeder decay curve in dB (0 dB at the start): backward-integrated energy."""
    e = np.cumsum((ir.astype(np.float64) ** 2)[::-1])[::-1]
    e = e / max(e[0], 1e-300)
    return 10.0 * np.log10(np.maximum(e, 1e-300))


def rt60_fit(ir: np.ndarray, fs: int, lo: float = -5.0, hi: float = -25.0) -> dict[str, Any]:
    """Fit a line to the Schroeder curve between ``lo`` and ``hi`` dB (T20 by default) and
    extrapolate to 60 dB. Returns ``rt60``, the ``slope`` (dB/s), the curve and the fitted line."""
    curve = schroeder_db(ir)
    t = np.arange(len(curve)) / fs
    sel = (curve <= lo) & (curve >= hi)
    if sel.sum() < 3:
        raise ValueError("the impulse response is too short to fit a decay")
    slope, intercept = np.polyfit(t[sel], curve[sel], 1)
    if slope >= 0:
        raise ValueError("no decay found in the impulse response")
    return {"rt60": float(-60.0 / slope), "slope": float(slope), "intercept": float(intercept),
            "curve": curve, "t": t, "fit": slope * t + intercept}  # fmt: skip


def convolve(dry: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolution reverb: the dry signal as it would sound in the room (FFT convolution)."""
    return fourier_math.convolve(dry, ir)


# ---------------------------------------------------------------- FDTD pressure field


class RoomFDTD:
    """A 3D wave-equation pressure field in a shoebox room, leapfrog finite differences.

    ``p_next = 2 p - p_prev + (c dt / dx)^2 laplacian(p) + source``, rigid walls, plus a thin
    damping shell at the walls whose strength comes from ``alpha`` (a model of absorption,
    not an exact boundary condition). Stability needs ``dt <= dx / (c sqrt(3))``; asking for
    more raises ``ValueError`` (an unstable FDTD step is a validation error).
    """

    def __init__(
        self,
        size,
        source,
        *,
        dx: float = 0.2,
        c: float = C,
        alpha: float = 0.1,
        dt: float | None = None,
        frequency: float = 120.0,
        seed: int = 0,
    ) -> None:
        self.size = tuple(float(v) for v in size)
        self.source = tuple(float(v) for v in source)
        self.dx, self.c, self.alpha = float(dx), float(c), float(alpha)
        self.frequency, self.seed = float(frequency), int(seed)
        limit = self.dx / (self.c * math.sqrt(3.0))
        self.dt = 0.95 * limit if dt is None else float(dt)
        if self.dt > limit * (1 + 1e-12):
            raise ValueError(
                f"unstable: dt = {self.dt:g} s is above the limit dx / (c sqrt(3)) = {limit:g} s; "
                "use a smaller dt or a larger dx"
            )
        self.shape = tuple(max(3, round(v / self.dx)) for v in self.size)
        self.k = (self.c * self.dt / self.dx) ** 2
        # damping shell: 2 cells, strength from the absorption coefficient
        sigma = np.ones(self.shape)
        edge = 2
        for ax, n in enumerate(self.shape):
            idx = np.arange(n)
            d = np.minimum(idx, n - 1 - idx)
            shell = np.where(d < edge, 1.0 - self.alpha * (edge - d) / edge * 0.5, 1.0)
            view = [1, 1, 1]
            view[ax] = n
            sigma = sigma * shell.reshape(view)
        self.sigma = sigma
        self.cell = tuple(min(max(int(v / self.dx), 0), n - 1)
                          for v, n in zip(self.source, self.shape, strict=True))  # fmt: skip

    # the Simulation protocol (core/simulation.py)
    def params(self) -> dict[str, Any]:
        return {"size": self.size, "source": self.source, "dx": self.dx, "c": self.c,
                "alpha": self.alpha, "frequency": self.frequency, "seed": self.seed,
                "dt": self.dt}  # fmt: skip

    def initial(self) -> dict[str, np.ndarray]:
        z = np.zeros(self.shape)
        return {"p": z, "prev": z.copy(), "n": np.zeros(1)}

    def pulse(self, n: int) -> float:
        """The source: one cycle of a Gaussian-windowed sine at ``frequency`` Hz, then silence."""
        t = n * self.dt
        t0 = 1.5 / self.frequency
        return math.sin(2 * math.pi * self.frequency * (t - t0)) * math.exp(
            -(((t - t0) * self.frequency / 0.6) ** 2)
        )

    def step(self, state: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
        p, prev = state["p"], state["prev"]
        n = int(state["n"][0])
        padded = np.pad(p, 1, mode="edge")  # rigid walls: zero normal gradient
        lap = (
            padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1]
            + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
            + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2] - 6.0 * p
        )  # fmt: skip
        new = 2.0 * p - prev + self.k * lap
        new[self.cell] += self.pulse(n) * self.k * 40.0
        new = new * self.sigma
        return {"p": new, "prev": p, "n": state["n"] + 1}

    def slice_z(self, state: dict[str, np.ndarray], z: float) -> np.ndarray:
        """The pressure on a horizontal plane at height ``z`` (m), shape (nx, ny)."""
        k = min(max(int(z / self.dx), 0), self.shape[2] - 1)
        return state["p"][:, :, k]

    def at(self, state: dict[str, np.ndarray], point) -> float:
        """The pressure at a point (nearest cell)."""
        i = [min(max(int(v / self.dx), 0), n - 1) for v, n in zip(point, self.shape, strict=True)]
        return float(state["p"][tuple(i)])


# ---------------------------------------------------------------- rooms as meshes, rays

MAX_RAYS = 20_000
MAX_BOUNCES = 32
EPS = 1e-7  # ray parameter below which a hit is the surface the ray just left


def box_mesh(size, origin=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """A shoebox room as a mesh: 8 vertices and 12 triangles (two per wall)."""
    lo = np.asarray(origin, dtype=np.float64)
    hi = lo + np.asarray(size, dtype=np.float64)
    v = np.array([[(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]]
                  for i in (0, 1) for j in (0, 1) for k in (0, 1)])  # fmt: skip
    # vertex index = 4 i + 2 j + k; each wall as a quad of four corners, split in two
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=np.int64)
    return v, f


def load_obj(path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (N, 3) and triangles (M, 3) of a Wavefront ``.obj`` file. Reads ``v`` and
    ``f`` lines (``f 1 2 3``, ``1/2/3``, ``1//3``, negative indices counting from the end);
    polygons are split into fans of triangles. Everything else is ignored."""
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    with open(path, encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(f"{path}:{no}: a vertex needs x y z")
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                idx = []
                for p in parts[1:]:
                    k = int(p.split("/")[0])
                    k = k - 1 if k > 0 else len(verts) + k
                    if not 0 <= k < len(verts):
                        raise ValueError(f"{path}:{no}: face names vertex {p!r}, not in the file")
                    idx.append(k)
                if len(idx) < 3:
                    raise ValueError(f"{path}:{no}: a face needs three vertices or more")
                tris.extend((idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1))
    if not tris:
        raise ValueError(f"{path}: no faces")
    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def _normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Unit normal of every triangle (its winding decides the sign)."""
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def feature_edges(v: np.ndarray, f: np.ndarray, angle: float = 20.0) -> np.ndarray:
    """The edges worth drawing, (E, 2) vertex indices: where the two triangles sharing an edge
    meet at more than ``angle`` degrees, plus edges with only one triangle (open borders).
    The diagonals that split a flat wall into triangles are left out."""
    v, f = np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    n = _normals(v, f)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(f)), 3)
    key = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[key], owner[key]
    uniq, start, count = np.unique(e, axis=0, return_index=True, return_counts=True)
    cos = math.cos(math.radians(angle))
    keep = []
    for k, (s, c) in enumerate(zip(start, count, strict=True)):
        # a border (1) or a non-manifold edge (3+) is always drawn; a shared one at a crease
        if c != 2 or abs(float(n[owner[s]] @ n[owner[s + 1]])) < cos:
            keep.append(k)
    return uniq[keep]


def mesh_stats(v, f) -> tuple[float, float]:
    """(volume, surface area) of a closed mesh (the volume by the divergence theorem)."""
    v, f = np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    vol = abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())) / 6.0
    return vol, area


def _hits(v: np.ndarray, f: np.ndarray, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Möller-Trumbore for every ray against every triangle: the ray parameter (R, T) of
    each hit (inf where it misses)."""
    a = v[f[:, 0]]
    e1, e2 = v[f[:, 1]] - a, v[f[:, 2]] - a
    p = np.cross(dirs[:, None, :], e2[None])  # (R, T, 3)
    det = np.einsum("rtk,tk->rt", p, e1)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origins[:, None, :] - a[None]
    u = np.einsum("rtk,rtk->rt", s, p) * inv
    q = np.cross(s, e1[None])
    w = np.einsum("rk,rtk->rt", dirs, q) * inv
    t = np.einsum("rtk,tk->rt", q, e2) * inv
    tol = 1e-9
    good = ok & (u >= -tol) & (w >= -tol) & (u + w <= 1 + tol) & (t > EPS)
    return np.where(good, t, np.inf)


def trace(v, f, origins, dirs, bounces: int, alpha: float) -> dict[str, np.ndarray]:
    """Specular reflection of rays inside a mesh room. For ``R`` rays and ``B`` bounces:
    ``points`` (R, B + 1, 3) the start and every reflection point; ``length`` (R, B + 1) the
    path length to each; ``energy`` (R, B + 1) the energy left after each point, (1 -
    alpha) ** k (0 once a ray is lost); ``alive`` (R, B + 1), False once a ray has left the
    room (an open mesh); ``wall`` (R, B + 1) the triangle hit (-1 for the start)."""
    v, f = np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    o = np.array(origins, dtype=np.float64).reshape(-1, 3)
    d = np.array(dirs, dtype=np.float64).reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    if not 0 <= bounces <= MAX_BOUNCES:
        raise ValueError(f"bounces must be 0-{MAX_BOUNCES}")
    r = len(o)
    n = _normals(v, f)
    pts = np.empty((r, bounces + 1, 3))
    length = np.zeros((r, bounces + 1))
    alive = np.ones((r, bounces + 1), dtype=bool)
    wall = np.full((r, bounces + 1), -1, dtype=np.int64)
    pts[:, 0] = o
    for k in range(1, bounces + 1):
        t = _hits(v, f, o, d)
        tri = np.argmin(t, axis=1)
        tk = t[np.arange(r), tri]
        go = np.isfinite(tk) & alive[:, k - 1]
        tk = np.where(go, tk, 0.0)
        o = o + d * tk[:, None]
        nk = n[tri]
        d = np.where(go[:, None], d - 2.0 * np.einsum("rk,rk->r", d, nk)[:, None] * nk, d)
        pts[:, k] = o
        length[:, k] = length[:, k - 1] + tk
        alive[:, k] = go
        wall[:, k] = np.where(go, tri, -1)
    energy = (1.0 - alpha) ** np.arange(bounces + 1, dtype=np.float64)
    return {"points": pts, "length": length, "alive": alive, "wall": wall,
            "energy": np.where(alive, energy[None, :], 0.0)}  # fmt: skip


def sphere_directions(n: int, gen) -> np.ndarray:
    """``n`` directions spread evenly over the sphere (a Fibonacci lattice), turned by one
    random rotation drawn from ``gen`` (a generator from ``rng``), so no direction lines up
    with the walls by accident."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * k
    s = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    dirs = np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)
    q, rr = np.linalg.qr(gen.normal(size=(3, 3)))
    return dirs @ (q * np.sign(np.diag(rr))).T


def rays(room, source, n: int, bounces: int, alpha: float, seed: int = 0,
         key: str = "rays") -> dict[str, np.ndarray]:  # fmt: skip
    """``n`` rays leaving ``source`` evenly in every direction, traced through ``room`` (a
    ``(vertices, triangles)`` pair) for ``bounces`` reflections; see ``trace``. The random
    turn of the directions comes from ``rng(key, seed)`` (the object's id), so the same
    object and seed always give the same rays. Also returns ``dirs`` (n, 3)."""
    from phasor.core.rng import rng

    if not 1 <= n <= MAX_RAYS:
        raise ValueError(f"the ray count must be 1-{MAX_RAYS:,}")
    dirs = sphere_directions(n, rng(key, seed))
    src = np.broadcast_to(np.asarray(source, dtype=np.float64), (n, 3))
    out = trace(room[0], room[1], src, dirs, bounces, alpha)
    out["dirs"] = dirs
    return out


def inside(v, f, points) -> np.ndarray:
    """Which ``points`` (N, 3) are inside the closed mesh (an odd number of walls crossed by
    a ray going out in a fixed, slightly tilted direction)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    d = np.broadcast_to(np.array([0.2873, 0.3511, 0.8913]) / 1.0, pts.shape)
    d = d / np.linalg.norm(d[0])
    t = _hits(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64), pts, d)
    return (np.isfinite(t).sum(axis=1) % 2) == 1


def planes(v, f, tol: float = 1e-6) -> list[np.ndarray]:
    """The triangles grouped by the plane they lie in (a wall made of several triangles is
    one group): a list of triangle-index arrays, in a fixed order."""
    v, f = np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    n = _normals(v, f)
    lead = n[np.arange(len(n)), np.argmax(np.abs(n), axis=1)]
    n = n * np.where(lead < 0, -1.0, 1.0)[:, None]  # one sign per plane
    d = np.einsum("tk,tk->t", n, v[f[:, 0]])
    key = np.round(np.c_[n, d] / tol).astype(np.int64)
    _, group = np.unique(key, axis=0, return_inverse=True)
    group = group.ravel()
    return [np.flatnonzero(group == g) for g in range(int(group.max()) + 1)]


def _in_triangles(v, f, tris, pts, tol: float = 1e-9) -> np.ndarray:
    """Which ``pts`` (N, 3), lying in the plane of ``tris``, fall inside any of them."""
    a, b, c = v[f[tris, 0]], v[f[tris, 1]], v[f[tris, 2]]
    e0, e1 = b - a, c - a
    d00, d01, d11 = (e0 * e0).sum(1), (e0 * e1).sum(1), (e1 * e1).sum(1)
    den = d00 * d11 - d01 * d01
    w = pts[:, None, :] - a[None]
    d20, d21 = np.einsum("ntk,tk->nt", w, e0), np.einsum("ntk,tk->nt", w, e1)
    bv = (d11 * d20 - d01 * d21) / den
    bw = (d00 * d21 - d01 * d20) / den
    return ((bv >= -tol) & (bw >= -tol) & (bv + bw <= 1 + tol)).any(axis=1)


def sphere_wall_arcs(v, f, centers, radius: float, samples: int = 96,
                     groups=None) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Where spheres of ``radius`` around ``centers`` (K, 3) cut the walls of a mesh room:
    short segments (S, 2, 3) along the circle each sphere draws on each wall plane, kept
    where they lie on the wall, and for each segment the index of its sphere (S,). Segments
    that cross a wall's border are cut at the border, so the arcs end exactly at corners.
    ``groups`` is ``planes(v, f)``, passed in when it is already known."""
    v, f = np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    groups = planes(v, f) if groups is None else groups
    n_all = _normals(v, f)
    a = np.linspace(0.0, 2 * math.pi, samples + 1)
    cs, sn = np.cos(a)[None, :, None], np.sin(a)[None, :, None]
    out, owner = [], []
    for tris in groups:
        nrm = n_all[tris[0]]
        h = (centers - v[f[tris[0], 0]]) @ nrm  # signed distance of each centre to the plane
        near = np.flatnonzero(np.abs(h) < radius)
        if not len(near):
            continue
        u = np.cross(nrm, [1.0, 0.0, 0.0] if abs(nrm[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(nrm, u)
        r = np.sqrt(radius * radius - h[near] ** 2)[:, None, None]
        mid = (centers[near] - h[near, None] * nrm)[:, None, :]
        ring = mid + r * (cs * u + sn * w)  # (K, samples + 1, 3)
        ins = _in_triangles(v, f, tris, ring.reshape(-1, 3)).reshape(len(near), samples + 1)
        a0, a1 = ring[:, :-1], ring[:, 1:]
        i0, i1 = ins[:, :-1], ins[:, 1:]
        both, one = i0 & i1, i0 ^ i1
        out.append(np.stack([a0[both], a1[both]], axis=1))
        owner.append(near[np.nonzero(both)[0]])
        if one.any():  # cut at the border: bisect between the inside and the outside end
            first = i0[one][:, None]
            pin = np.where(first, a0[one], a1[one])
            lo, hi = pin.copy(), np.where(first, a1[one], a0[one])
            for _ in range(10):
                m = (lo + hi) / 2
                ok = _in_triangles(v, f, tris, m)[:, None]
                lo, hi = np.where(ok, m, lo), np.where(ok, hi, m)
            out.append(np.stack([pin, lo], axis=1))
            owner.append(near[np.nonzero(one)[0]])
    if not out:
        return np.zeros((0, 2, 3)), np.zeros(0, dtype=np.int64)
    return np.concatenate(out), np.concatenate(owner)


def ray_segments(paths: dict[str, np.ndarray], distance: float,
                 trail: float = 0.0) -> dict[str, np.ndarray]:  # fmt: skip
    """The part of every traced ray (``trace``) that sound has reached after travelling
    ``distance`` meters, as straight pieces between reflections: ``segments`` (S, 2, 3),
    ``along`` (S, 2) the path length at both ends, ``bounce`` (S,) the reflections before
    the piece and ``ray`` (S,) its ray. With ``trail`` > 0 only the last ``trail`` meters
    behind the front are kept (the sound as a moving streak rather than a whole path)."""
    pts, length, alive = paths["points"], paths["length"], paths["alive"]
    lo, hi = length[:, :-1], length[:, 1:]
    a = np.maximum(lo, distance - trail) if trail > 0 else lo
    b = np.minimum(hi, distance)
    r, k = np.nonzero(alive[:, 1:] & (b > a))
    p0, p1 = pts[r, k], pts[r, k + 1]
    unit = (p1 - p0) / np.maximum(hi[r, k] - lo[r, k], 1e-300)[:, None]
    sa, sb = a[r, k], b[r, k]
    ends = np.stack([p0 + unit * (sa - lo[r, k])[:, None], p0 + unit * (sb - lo[r, k])[:, None]],
                    axis=1)  # fmt: skip
    return {"segments": ends, "along": np.stack([sa, sb], axis=1), "bounce": k, "ray": r}


def mirror_images(v, f, groups, source) -> np.ndarray:
    """``source`` mirrored in the plane of every wall group (``planes``): (G, 3), the first
    reflection's image sources for a room of any shape."""
    v, f = np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    n = _normals(v, f)
    src = np.asarray(source, dtype=np.float64)
    out = np.empty((len(groups), 3))
    for g, tris in enumerate(groups):
        h = (src - v[f[tris[0], 0]]) @ n[tris[0]]
        out[g] = src - 2.0 * h * n[tris[0]]
    return out


def front_visible(v, f, groups, source, plane, points) -> np.ndarray:
    """In a room of any shape, which ``points`` on the walls a front really reaches.
    ``plane[i]`` is -1 for the direct front from ``source`` (seen if nothing is in the way)
    or the wall group whose image source drew point ``i`` (seen if the path from the source
    to the point reflects off that wall itself, unblocked on both legs)."""
    v, f = np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    plane = np.asarray(plane, dtype=np.int64)
    src = np.asarray(source, dtype=np.float64)
    ok = np.zeros(len(pts), dtype=bool)
    direct = plane < 0
    if direct.any():
        ok[direct] = segments_clear(v, f, np.broadcast_to(src, (int(direct.sum()), 3)), pts[direct])
    n = _normals(v, f)
    images = mirror_images(v, f, groups, source) if (~direct).any() else None
    for g in np.unique(plane[~direct]):
        sel = np.flatnonzero(plane == g)
        tris = groups[g]
        p0, nrm = v[f[tris[0], 0]], n[tris[0]]
        img = images[g]
        hi, hm = (img - p0) @ nrm, (pts[sel] - p0) @ nrm
        s = hi / np.where(hi - hm == 0, 1e-300, hi - hm)
        hit = img + s[:, None] * (pts[sel] - img)  # where the path meets the mirror wall
        good = (s > 1e-9) & (s < 1 - 1e-9) & _in_triangles(v, f, tris, hit)
        if good.any():
            idx = np.flatnonzero(good)
            leg1 = segments_clear(v, f, np.broadcast_to(src, (len(idx), 3)), hit[idx])
            leg2 = segments_clear(v, f, hit[idx], pts[sel][idx])
            good[idx] = leg1 & leg2
        ok[sel] = good
    return ok


def segments_clear(v, f, a, b, tol: float = 1e-6) -> np.ndarray:
    """Which segments from ``a`` (N, 3) to ``b`` (N, 3) cross no wall on the way (their own
    ends, which may lie on walls, do not count)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = b - a
    ln = np.linalg.norm(d, axis=1)
    unit = d / np.maximum(ln, 1e-300)[:, None]
    t = _hits(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64), a, unit)
    return ~((t > tol) & (t < ln[:, None] - tol)).any(axis=1)


# ---------------------------------------------------------------- Directivity


def array_directivity(angle, elements: int, spacing: float, frequency: float, steer: float):
    """Normalized pressure magnitude for a uniformly excited linear array along +y.

    Angles are radians from +x (broadside), spacing is in meters, and frequency is in Hz.
    """
    cycles = (spacing * frequency / C) * (np.sin(angle) - np.sin(steer))
    # Remove integer cycles before taking the sinc ratio, including at grating lobes.
    cycles = cycles - np.rint(cycles)
    return np.abs(np.sinc(elements * cycles) / np.sinc(cycles))


def cardioid_directivity(angle):
    """Normalized cardioid pressure magnitude at angles in radians from its axis."""
    return np.square(np.cos(np.asarray(angle) / 2))


_BESSEL_ANGLES = np.linspace(0, 2 * np.pi, 256, endpoint=False)


def _piston_gain(x: np.ndarray) -> np.ndarray:
    """Evaluate 2 J1(x)/x for nonnegative x using the Bessel integral and its asymptote."""
    result = np.ones_like(x)
    small = (x > 0) & (x <= 128)
    if small.any():
        y = x[small, None]
        # In the integral for J1, the cos(t) term integrates to zero. Using
        # sinc keeps 2 J1(x)/x well-conditioned as x approaches zero.
        sine = np.sin(_BESSEL_ANGLES)
        result[small] = np.mean(2 * sine**2 * np.sinc(y * sine / np.pi), axis=1)
    large = x > 128
    y = x[large]
    inv = 1 / y
    phase = y - 3 * np.pi / 4
    even = 1 + 15 * inv**2 / 128
    odd = 3 * inv / 8 - 315 * inv**3 / 3072
    result[large] = (
        2 * np.sqrt(2 * inv / np.pi) * inv * (even * np.cos(phase) - odd * np.sin(phase))
    )
    return result


def piston_directivity(angle, radius: float, frequency: float):
    """Normalized far-field pressure magnitude for a circular piston in an infinite baffle.

    Angles are radians from the piston normal, radius is in meters, and frequency is in Hz.
    """
    angle = np.asarray(angle, dtype=np.float64)
    x = 2 * np.pi * (radius * frequency / C) * np.abs(np.sin(angle))
    return np.where(np.cos(angle) >= 0, np.abs(_piston_gain(x)), 0.0)
