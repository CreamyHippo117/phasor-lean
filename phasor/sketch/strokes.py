"""What a stroke is: classified by fitting shapes to its points (no learned model).

A stroke is a sequence of points in output pixels (row 0 at the top), as a pen or mouse
draws it. ``classify`` names it and returns the fitted shape:

- ``dot``: a tiny stroke (a tap or a short dab);
- ``line``: straight (every point near the chord);
- ``arrow``: a straight shaft whose end turns back sharply in a short hook (one stroke),
  or a shaft with a short V-shaped stroke at its end (two strokes, see ``group``);
- ``circle``: closed and round (algebraic circle fit, small residual);
- ``closed``: closed but not round (any loop: a contour);
- ``scribble``: back and forth over a small area (a cross-out);
- ``curve``: anything else that is open (``wavy`` when it bends both ways several times).

Small strokes drawn close together that are none of these are handwriting (``group``).
Thresholds are shares of the stroke's own size, so a sketch means the same at any scale.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DOT_PX = 9.0  # a stroke whose box diagonal is under this many pixels is a dot
CLOSE_SHARE = 0.18  # ends nearer than this share of the length: closed
LINE_SHARE = 0.06  # every point within this share of the chord: straight
CIRCLE_RMS = 0.12  # circle fit residual (rms) under this share of the radius: round
HOOK_SHARE = 0.4  # an arrow's hook is at most this share of the stroke
HOOK_MIN = 0.04  # ... and at least this share (shorter is a flick of the pen)
HOOK_TURN = 110.0  # degrees: how sharply the hook turns back
SCRIBBLE_TURNS = 3  # direction reversals that make a cross-out
HANDWRITING_GAP = 1.2  # strokes this many median heights apart belong to one word


@dataclass
class Shape:
    kind: str
    points: np.ndarray  # (N, 2) output pixels, resampled
    data: dict[str, Any] = field(default_factory=dict)  # the fit: center, radius, ends...
    strokes: tuple[int, ...] = ()  # which input strokes made it


def as_points(stroke: Any) -> np.ndarray:
    p = np.asarray(stroke, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        raise ValueError("a stroke needs at least one point")
    keep = np.concatenate([[True], np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9])
    return p[keep]


def length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def resample(p: np.ndarray, n: int = 128) -> np.ndarray:
    """``n`` points evenly spaced along the stroke."""
    if len(p) < 2:
        return np.repeat(p[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], axis=1)


def fit_circle(p: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(center, radius, rms residual) by the algebraic (Kasa) least-squares fit."""
    x, y = p[:, 0], p[:, 1]
    a = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    (c0, c1, c2), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = c0 / 2, c1 / 2
    r = math.sqrt(max(c2 + cx * cx + cy * cy, 0.0))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return np.array([cx, cy]), r, rms


def _headings(p: np.ndarray) -> np.ndarray:
    d = np.diff(p, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def _turn(a: float, b: float) -> float:
    """Signed angle from heading ``a`` to ``b``, in (-pi, pi]."""
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def _reversals(p: np.ndarray) -> int:
    """How many times the stroke doubles back on itself (turns past 120 degrees within a
    short stretch), the mark of a scribble."""
    q = resample(p, 96)
    h = _headings(q)
    count, k = 0, 0
    while k + 4 < len(h):
        if abs(_turn(h[k], h[k + 4])) > math.radians(120):
            count += 1
            k += 6
        else:
            k += 1
    return count


def _straight(p: np.ndarray) -> float:
    """Largest distance from the chord, as a share of the chord's length."""
    a, b = p[0], p[-1]
    chord = float(np.linalg.norm(b - a))
    if chord < 1e-9:
        return math.inf
    n = np.array([-(b - a)[1], (b - a)[0]]) / chord
    return float(np.abs((p - a) @ n).max()) / chord


def _hook(p: np.ndarray) -> tuple[int, float] | None:
    """Where a one-stroke arrow's shaft ends: (index into the resampled stroke, turn in
    degrees), or None. The hook is the short last part that turns sharply back."""
    q = resample(p, 128)
    total = length(q)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    h = _headings(q)
    best: tuple[int, float] | None = None
    for k in range(len(h) - 3):
        if s[k + 1] < total * (1 - HOOK_SHARE):
            continue
        turn = abs(math.degrees(_turn(h[k - 2] if k >= 2 else h[0], h[min(k + 3, len(h) - 1)])))
        if turn > HOOK_TURN and (best is None or turn > best[1]):
            best = (k + 1, turn)
    if best is None or _straight(q[: best[0] + 1]) > LINE_SHARE * 1.5:
        return None
    if total - s[best[0]] < HOOK_MIN * total:  # a flick of the pen, not a hook
        return None
    return best


def classify(stroke: Any) -> Shape:
    """Name one stroke and fit its shape (see the module)."""
    p = as_points(stroke)
    lo, hi = p.min(axis=0), p.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if diag < DOT_PX:
        return Shape("dot", p, {"at": p.mean(axis=0)})
    L = length(p)
    q = resample(p)
    gap = float(np.linalg.norm(p[-1] - p[0]))
    reversals = _reversals(p)
    if reversals >= SCRIBBLE_TURNS and 3.0 * diag < L:
        return Shape("scribble", q, {"box": (*lo, *hi), "reversals": reversals})
    if gap < CLOSE_SHARE * L and 2.0 * diag < L:
        center, r, rms = fit_circle(q)
        aspect = (hi - lo).min() / max((hi - lo).max(), 1e-9)
        if r > 0 and rms < CIRCLE_RMS * r and aspect > 0.75:
            return Shape("circle", q, {"center": center, "radius": r})
        return Shape("closed", q, {"box": (*lo, *hi)})
    hook = _hook(p)  # before "line": a short hook barely bends the chord
    if hook is not None:
        shaft = resample(p, 128)[: hook[0] + 1]
        d = shaft[-1] - shaft[0]
        tip = p[int(np.argmax((p - p[0]) @ d))]  # the tip: farthest along the shaft
        return Shape("arrow", np.concatenate([shaft, tip[None, :]]), {"start": p[0], "end": tip})
    if _straight(p) < LINE_SHARE:
        return Shape("line", q, {"start": p[0], "end": p[-1]})
    h = _headings(resample(p, 64))
    turns = np.array([_turn(a, b) for a, b in itertools.pairwise(h)])
    signs = np.sign(turns[np.abs(turns) > 0.02])
    flips = int((np.diff(signs) != 0).sum()) if len(signs) > 1 else 0
    return Shape("curve", q, {"wavy": flips >= 3 and _straight(p) < 0.5})


def _is_v(shape: Shape, end: np.ndarray, shaft: float) -> bool:
    """A short stroke shaped like an arrow head whose tip sits at ``end``."""
    p = shape.points
    if length(p) > 0.6 * shaft:
        return False
    tip = p[int(np.argmin(np.linalg.norm(p - end, axis=1)))]
    return float(np.linalg.norm(tip - end)) < 0.12 * shaft and shape.kind in ("curve", "line")


def group(strokes: list[Any]) -> list[Shape]:
    """Classify several strokes and join the ones that belong together: a line with a V at
    its end is an arrow; a cluster of small open strokes is handwriting."""
    shapes = [classify(s) for s in strokes]
    for k, sh in enumerate(shapes):
        sh.strokes = (k,)
    used: set[int] = set()
    out: list[Shape] = []
    # arrows from two strokes
    for i, sh in enumerate(shapes):
        if sh.kind != "line" or i in used:
            continue
        shaft = length(sh.points)
        for j, other in enumerate(shapes):
            if j != i and j not in used and _is_v(other, sh.data["end"], shaft):
                used |= {i, j}
                out.append(Shape("arrow", sh.points, dict(sh.data), (i, j)))
                break
    # handwriting: small open strokes near each other (letters of a word)
    rest = [k for k in range(len(shapes)) if k not in used]
    heights = [float(np.ptp(shapes[k].points[:, 1])) for k in rest]
    small_h = float(np.median(heights)) if heights else 0.0
    small = [
        k
        for k in rest
        if shapes[k].kind in ("curve", "line", "dot", "closed", "circle")
        and float(np.ptp(shapes[k].points, axis=0).max()) < 3.0 * max(small_h, DOT_PX)
    ]
    clusters = _clusters([shapes[k] for k in small], small, HANDWRITING_GAP * max(small_h, DOT_PX))
    for members in clusters:
        if len(members) < 3:
            continue
        used |= set(members)
        pts = np.concatenate([shapes[k].points for k in members])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        data = {"box": (*lo, *hi), "strokes": [shapes[k].points for k in members]}
        out.append(Shape("handwriting", pts, data, tuple(members)))
    out += [sh for k, sh in enumerate(shapes) if k not in used]
    return sorted(out, key=lambda s: s.strokes[0])


def _clusters(shapes: list[Shape], ids: list[int], gap: float) -> list[list[int]]:
    """Connected groups of shapes whose boxes are within ``gap`` pixels of each other."""
    boxes = [(s.points.min(axis=0), s.points.max(axis=0)) for s in shapes]
    parent = list(range(len(shapes)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(shapes)):
        for b in range(a + 1, len(shapes)):
            (la, ha), (lb, hb) = boxes[a], boxes[b]
            dx = max(0.0, lb[0] - ha[0], la[0] - hb[0])
            dy = max(0.0, lb[1] - ha[1], la[1] - hb[1])
            if math.hypot(dx, dy) <= gap:
                parent[find(a)] = find(b)
    out: dict[int, list[int]] = {}
    for a in range(len(shapes)):
        out.setdefault(find(a), []).append(ids[a])
    return list(out.values())
