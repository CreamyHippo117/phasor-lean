"""Shared drawing helpers for domain objects: curves, discs and circles that keep their width
on screen, so every domain draws strokes the same way the built-in ``Line`` does."""

from __future__ import annotations

import functools

import numpy as np

from phasor.core.objects import DrawContext
from phasor.core.spatial import Camera

DISC_SEGMENTS = 48


def hue_colors(pts, lut, cycles: float, shift: float = 0.0, total: float | None = None):
    """(N, 4) colors running ``lut`` (a wrapped ramp) along a polyline by arc length:
    ``cycles`` times over ``total`` length (default: this polyline's), shifted by ``shift``."""
    pts = np.asarray(pts, dtype=np.float64)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = max(float(along[-1] if total is None else total), 1e-12)
    return lut_colors(lut, along / total * cycles + shift)


def lut_colors(lut, where):
    """(N, 4) opaque colors of ``lut`` (a wrapped ramp) at positions ``where`` (N,), each a
    share of the ramp (wrapping at 1)."""
    lut = np.asarray(lut, dtype=np.float64)
    g = (np.asarray(where, dtype=np.float64) % 1.0) * len(lut)
    k0 = np.floor(g).astype(int) % len(lut)
    f = (g - np.floor(g))[:, None]
    rgb = lut[k0] * (1 - f) + lut[(k0 + 1) % len(lut)] * f
    out = np.empty((len(rgb), 4))
    out[:, :3], out[:, 3] = rgb, 1.0
    return out


@functools.lru_cache(maxsize=256)
def _strip_indices(n: int) -> np.ndarray:
    """The triangles of a stroke of ``n`` points (vertices 2i, 2i+1 either side of point i):
    every first triangle of a quad, then every second one. Shared and read-only."""
    k = 2 * np.arange(n - 1)
    out = np.empty((2 * (n - 1), 3), dtype=k.dtype)
    out[: n - 1, 0], out[: n - 1, 1], out[: n - 1, 2] = k, k + 1, k + 2
    out[n - 1 :, 0], out[n - 1 :, 1], out[n - 1 :, 2] = k + 1, k + 3, k + 2
    out.setflags(write=False)
    return out


def _row_norm(v: np.ndarray) -> np.ndarray:
    """``np.linalg.norm(v, axis=-1)`` for rows of 2, the same arithmetic without its
    overhead (it is called thousands of times a frame)."""
    return np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1])


def polyline(ctx: DrawContext, cam: Camera, pts, width, *, color, glow: float = 1.0,
             colors=None) -> None:  # fmt: skip
    """A stroked polyline (mitered joins). ``width`` is in world units: a number, or one per
    point (perspective: wider where it is further away, so it looks constant on screen).
    ``colors`` (N, 4), one per point, runs a color along the curve. With none given, a
    glowing stroke runs the theme's ``[strokes]`` hue when the theme turns it on (the
    ``prism_everywhere`` look); plain strokes (axes, grids) keep their color."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return
    width = np.asarray(width, dtype=np.float64)
    if width.shape not in ((), (n,)):
        width = np.broadcast_to(width, (n,))  # one per point, or an error
    polylines(ctx, cam, pts[None], width[None] if width.ndim else width, color=color,
              glow=glow, colors=None if colors is None else np.asarray(colors)[None])  # fmt: skip


def polylines(ctx: DrawContext, cam: Camera, pts, width, *, color, glow: float = 1.0,
              colors=None) -> None:  # fmt: skip
    """``S`` separate strokes of ``N`` points each, ``pts`` (S, N, 3), as one draw: exactly
    the triangles of ``S`` ``polyline`` calls, in the same order (nothing joins one stroke
    to the next). ``width``: a number or (S, N); ``colors``: None or (S, N, 4). One
    ``color`` and ``glow`` for all of them (with per-point ``colors`` only its alpha, the
    opacity, is used)."""
    pts = np.asarray(pts, dtype=np.float64)
    count, n = pts.shape[0], pts.shape[1]
    if count == 0 or n < 2:
        return
    note = getattr(ctx, "note_path", None)  # the recorder (core/record.py) keeps center lines
    if note is not None:
        for p in pts:
            note(p)
    theme_hue = getattr(ctx, "stroke_hue", None)
    if colors is None and theme_hue is not None and glow > 1.0:
        lut, cycles, shift = theme_hue
        colors = np.stack([hue_colors(p, lut, cycles, shift) for p in pts])
    width = np.asarray(width, dtype=np.float64)
    if width.ndim:
        width = np.broadcast_to(width, (count, n))
    half = (width[..., None] if width.ndim else width) / 2
    right, up = np.asarray(cam.right, dtype=np.float64), np.asarray(cam.up, dtype=np.float64)
    rows = pts.reshape(-1, 3)
    flat = np.empty((count, n, 2))  # coordinates in the camera plane
    flat[..., 0], flat[..., 1] = (rows @ right).reshape(count, n), (rows @ up).reshape(count, n)
    seg = flat[:, 1:] - flat[:, :-1]
    norm = _row_norm(seg)[..., None]
    seg = seg / np.where(norm == 0, 1, norm)
    n_prev = np.empty((count, n, 2))  # each point's incoming and outgoing segment normals
    n_prev[:, 1:, 0], n_prev[:, 1:, 1] = -seg[..., 1], seg[..., 0]
    n_prev[:, 0] = n_prev[:, 1]
    n_next = np.empty((count, n, 2))
    n_next[:, :-1], n_next[:, -1] = n_prev[:, 1:], n_prev[:, -1]
    miter = n_prev + n_next
    miter /= np.maximum(_row_norm(miter)[..., None], 1e-9)
    scale = 1 / np.maximum(miter[..., 0] * n_prev[..., 0] + miter[..., 1] * n_prev[..., 1], 0.25)
    offset = (miter * scale[..., None]) * half  # miter limit 4x (the 0.25 above)
    off3 = offset[..., 0, None] * right + offset[..., 1, None] * up
    positions = np.empty((count, 2 * n, 3))
    positions[:, 0::2], positions[:, 1::2] = pts + off3, pts - off3
    indices = _strip_indices(n)
    if count > 1:  # each stroke's triangles, moved past the vertices of those before it
        indices = (indices[None] + (2 * n * np.arange(count))[:, None, None]).reshape(-1, 3)
    per_vertex = None
    if colors is not None:
        per_vertex = np.repeat(np.asarray(colors, dtype=np.float64), 2, axis=1).reshape(-1, 4)
    ctx.triangles(positions.reshape(-1, 3), indices, color=color, glow=glow, colors=per_vertex)


def segments(ctx: DrawContext, cam: Camera, pairs, width, *, color, glow: float = 1.0) -> None:
    """Many separate straight strokes: ``pairs`` is (N, 2, 3)."""
    for a, b in np.asarray(pairs, dtype=np.float64):
        w = float(np.mean(cam.units_per_height(np.array([a, b])))) * width
        polyline(ctx, cam, [a, b], w, color=color, glow=glow)


def disc(ctx: DrawContext, cam: Camera, center, radius: float, *, color, glow: float = 1.0) -> None:
    """A filled disc facing the camera."""
    a = np.linspace(0, 2 * np.pi, DISC_SEGMENTS, endpoint=False)
    c = np.asarray(center, dtype=np.float64)
    rim = c + radius * (np.outer(np.cos(a), cam.right) + np.outer(np.sin(a), cam.up))
    positions = np.vstack([c[None, :], rim])
    k = np.arange(DISC_SEGMENTS)
    ctx.triangles(positions, np.c_[np.zeros_like(k), 1 + k, 1 + (k + 1) % DISC_SEGMENTS],
                  color=color, glow=glow)  # fmt: skip


def circle(ctx: DrawContext, cam: Camera, center, radius: float, width, *, color, glow=1.0) -> None:
    """A ring facing the camera, ``width`` in world units."""
    a = np.linspace(0, 2 * np.pi, DISC_SEGMENTS + 1)
    c = np.asarray(center, dtype=np.float64)
    pts = c + radius * (np.outer(np.cos(a), cam.right) + np.outer(np.sin(a), cam.up))
    polyline(ctx, cam, pts, width, color=color, glow=glow)


def arrow(ctx: DrawContext, cam: Camera, a, b, width, color, glow: float = 1.0,
          head: float | None = None) -> None:  # fmt: skip
    """A straight arrow from ``a`` to ``b``, ``width`` in world units. The head is scaled to
    the width, or is ``head`` (a share of the arrow's length) when given."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return
    head = length * head if head is not None else min(length * 0.3, width * 7)
    unit = d / length
    side = np.cross(cam.forward, unit)
    side /= max(np.linalg.norm(side), 1e-9)
    base = b - unit * head
    polyline(ctx, cam, [a, base], width, color=color, glow=glow)
    tri = np.array([b, base + side * head * 0.45, base - side * head * 0.45])
    ctx.triangles(tri, np.array([[0, 1, 2]]), color=color, glow=glow)


LENGTH_BY = ("size", "soft")


def field_arrows(ctx: DrawContext, cam: Camera, centers, vectors, spacing: float, *, width,
                 colors, glow: float = 1.0, length_by: str = "size") -> None:  # fmt: skip
    """A field of arrows centred on ``centers`` (N, 3), pointing along ``vectors`` (N, 3),
    for grids ``spacing`` apart. ``colors``: one RGBA, or one per arrow (N, 4).

    ``length_by="size"``: length in proportion to size, the strongest 0.85 of the spacing.
    ``length_by="soft"``: lengths follow tanh of size over a typical size, so weak arrows
    still show and strong ones never reach their neighbours (the dense look: many small
    arrows, where color says how strong)."""
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    vec = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    mag = np.linalg.norm(vec, axis=1)
    ok = np.isfinite(mag) & (mag > 1e-12)
    if not ok.any():
        return
    top = float(mag[ok].max())
    if length_by == "soft":
        typical = float(np.median(mag[ok]))
        length = spacing * 0.9 * np.tanh(mag / max(1.5 * typical, 1e-12))
    else:
        length = spacing * 0.85 * mag / top
    cols = np.asarray(colors, dtype=np.float64)
    per_arrow = cols.ndim == 2
    for k in np.flatnonzero(ok):
        if length[k] < 1e-6:
            continue
        d = vec[k] / mag[k] * length[k]
        c = tuple(cols[k]) if per_arrow else tuple(cols)
        # Soft fields are many small arrows: a head in proportion to each one, so the
        # direction still reads when the arrow is a few pixels long.
        arrow(ctx, cam, pts[k] - d / 2, pts[k] + d / 2, width, c, glow,
              head=0.38 if length_by == "soft" else None)  # fmt: skip


def with_alpha(color, opacity: float):
    return (color[0], color[1], color[2], color[3] * opacity)


def segment_batch(ctx: DrawContext, cam: Camera, a, b, width, *, colors, glow: float = 1.0,
                  opacity: float = 1.0) -> None:  # fmt: skip
    """Many straight strokes in one draw call (thousands of graph edges): ``a`` and ``b``
    are (N, 3) ends, ``width`` a share of the frame height (as stroke roles are), ``colors``
    one RGBA or (N, 4) per stroke, or (N, 2, 4) per end (a hue running along each edge).
    Each stroke is a ribbon facing the camera."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if len(a) == 0:
        return
    mid = (a + b) / 2
    view = (mid - cam.eye) if cam.perspective else np.broadcast_to(cam.forward, a.shape)
    side = np.cross(b - a, view)
    n = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.where(n > 1e-12, side / np.maximum(n, 1e-12), cam.right)
    wa = (width * np.asarray(cam.units_per_height(a)) / 2).reshape(-1, 1)
    wb = (width * np.asarray(cam.units_per_height(b)) / 2).reshape(-1, 1)
    pos = np.stack([a - side * wa, a + side * wa, b + side * wb, b - side * wb], axis=1)
    k = np.arange(len(a))[:, None] * 4
    tris = np.concatenate([k + np.array([0, 1, 2]), k + np.array([0, 2, 3])], axis=0)
    cols = np.asarray(colors, dtype=np.float64)
    if cols.ndim == 1:
        per_vertex = np.broadcast_to(cols, (len(a), 4, 4))
    elif cols.ndim == 2:
        per_vertex = np.repeat(cols[:, None, :], 4, axis=1)
    else:
        per_vertex = cols[:, [0, 0, 1, 1], :]
    per_vertex = np.array(per_vertex, dtype=np.float64)
    ctx.triangles(pos.reshape(-1, 3), tris, color=(1.0, 1.0, 1.0, opacity), glow=glow,
                  colors=per_vertex.reshape(-1, 4))  # fmt: skip


def disc_batch(ctx: DrawContext, cam: Camera, centers, radius, *, colors, glow: float = 1.0,
               opacity: float = 1.0, segments: int = 12, soft: bool = False) -> None:  # fmt: skip
    """Many discs facing the camera in one draw call: ``centers`` (N, 3), ``radius`` world
    units (one, or one per disc), ``colors`` one RGBA or (N, 4), ``segments`` round each.
    ``soft``: each disc's light falls linearly from its centre to nothing at its rim (a glow
    splat rather than a flat dot)."""
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if len(c) == 0:
        return
    seg = int(segments)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    r = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(c),))[:, None, None]
    ring = (np.cos(ang)[:, None] * cam.right + np.sin(ang)[:, None] * cam.up)[None] * r
    pos = np.concatenate([c[:, None, :], c[:, None, :] + ring], axis=1)  # (N, 1 + seg, 3)
    base = (np.arange(len(c)) * (seg + 1))[:, None]
    k = np.arange(seg)
    tris = np.stack([np.zeros(seg, int), 1 + k, 1 + (k + 1) % seg], axis=1)
    tris = (base[:, None, :] + tris[None]).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.float64)
    cols = np.broadcast_to(cols if cols.ndim == 2 else cols[None], (len(c), 4))
    per_vertex = np.repeat(cols[:, None, :], seg + 1, axis=1)
    if soft:
        per_vertex[:, 1:, 3] = 0.0
    per_vertex = per_vertex.reshape(-1, 4)
    ctx.triangles(pos.reshape(-1, 3), tris, color=(1.0, 1.0, 1.0, opacity), glow=glow,
                  colors=np.ascontiguousarray(per_vertex))  # fmt: skip


FLASH_HALO = 3.0  # a flashing point's halo, in point radii
FLASH_HALO_LIGHT = 0.35  # the halo's light at the peak of the flash
REST_GLOW = 0.3  # share of the glow role flashing points keep when not flashing


def flash_points(ctx: DrawContext, cam: Camera, centers, radius, *, colors, flash=None,
                 glow: float = 1.0, opacity: float = 1.0, swell: float = 0.0, white: float = 0.0,
                 segments: int = 8) -> None:  # fmt: skip
    """Glowing points of light that may flash: the one way a lattice of lights is drawn
    (``sweep.SweepLattice``, ``nd.StarLattice``, ``ac.ImageSources``). ``centers`` (N, 3),
    ``radius`` world units (one or one per point), ``colors`` (N, 4): each point's colour,
    its alpha its light. ``flash`` (N,) 0-1 is the part of the light above the resting level
    (``sweep_math.envelope`` gives both): a flashing point grows by ``swell * flash``, runs
    white-hot by ``white * flash`` and wears a soft halo ``FLASH_HALO`` radii wide that sends
    the whole ``glow`` role to the bloom, while the points themselves keep ``REST_GLOW`` of
    it. Without ``flash`` the points are steady and shine with the whole role. Drawn far to
    near (for blending), in at most two draw calls."""
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if not len(c):
        return
    order = np.argsort(-cam.depth(c), kind="stable")
    c = c[order]
    cols = np.array(np.broadcast_to(np.asarray(colors, dtype=np.float64), (len(order), 4)))[order]
    r = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(order),))[order]
    if flash is None:
        disc_batch(ctx, cam, c, r, colors=cols, glow=glow, opacity=opacity, segments=segments)
        return
    f = np.broadcast_to(np.asarray(flash, dtype=np.float64), (len(order),))[order]
    hot = (white * f)[:, None]
    cols[:, :3] = cols[:, :3] * (1 - hot) + hot  # the flash runs white-hot
    r = r * (1.0 + swell * f)
    disc_batch(ctx, cam, c, r, colors=cols, glow=1.0 + REST_GLOW * (glow - 1.0), opacity=opacity,
               segments=segments)  # fmt: skip
    lit = f > 0.02
    if lit.any():  # a soft halo round each flashing point, all its light to the bloom
        halo = cols[lit].copy()
        halo[:, 3] = FLASH_HALO_LIGHT * f[lit]
        disc_batch(ctx, cam, c[lit], FLASH_HALO * r[lit], colors=halo, glow=glow,
                   opacity=opacity, segments=segments + 2, soft=True)  # fmt: skip
