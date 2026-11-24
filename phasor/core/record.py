"""A recording ``DrawContext`` and the contract members derived from it (ADR 0005).

A type that defines only ``Params`` and ``draw`` still has bounds, features and semantics:
``register`` gives it the ones here, which run its ``draw`` once against a ``Recorder`` and
read what it emitted. Pure NumPy, so ``core`` stays free of GPU libraries.

- ``world_bounds``: the box around every recorded vertex; for a ``field`` the quad's four
  corners (its ``pixels`` callback is never called); for text the box the text occupies.
- ``features``: ``center`` and the box's compass points (``top``, ``top_right``, ...: the
  box's sides as the camera sees them), plus ``start``, ``end`` and ``curve`` when the
  object drew exactly one stroke (through ``domains/strokes.py``).
- ``semantics``: ``{"type", "params"}``.

A recording is cached by (type, id, resolved params, resolved style, camera, scene folder).
If ``draw`` reads another object's state (``state["other"]``), those reads are remembered
and the recording is made again when what it read changes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from phasor.core import text as text_metrics
from phasor.core.objects import Bounds
from phasor.core.state import SceneState

CACHE_SIZE = 4096
# The compass points of the box, as (along the camera's right, along its up).
COMPASS = {
    "top": (0, 1), "bottom": (0, -1), "left": (-1, 0), "right": (1, 0),
    "top_left": (-1, 1), "top_right": (1, 1), "bottom_left": (-1, -1), "bottom_right": (1, -1),
}  # fmt: skip


class Recorder:
    """A ``DrawContext`` that keeps the geometry of what is drawn and draws nothing."""

    def __init__(self, right, up) -> None:
        self.right = np.asarray(right, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64)
        self.points: list[np.ndarray] = []  # (N, 3) blocks of world positions
        self.kinds: list[str] = []  # one per command: triangles | field | surface | text
        self.strokes: list[np.ndarray] = []  # center lines noted by domains/strokes.py

    def _add(self, kind: str, pts) -> None:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        self.kinds.append(kind)
        if len(pts):
            self.points.append(pts)

    def triangles(self, positions, indices=None, *, color, glow=1.0, colors=None) -> None:
        self._add("triangles", positions)

    def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0) -> None:
        # Only the quad: ``pixels`` is the expensive part and says nothing about where it is.
        c, r, u = (np.asarray(v, dtype=np.float64) for v in (corner, right, up))
        self._add("field", [c, c + r, c + u, c + r + u])

    def surface(self, positions, indices, *, normals, colors, uv=None, material="solid",
                opacity=1.0, glow=1.0) -> None:  # fmt: skip
        self._add("surface", positions)

    def text(self, text, origin, height, *, color, anchor="center", glow=1.0,
             reveal=1.0) -> None:  # fmt: skip
        corner, wv, hv = text_metrics.quad(text, origin, height, anchor, self.right, self.up)
        self._add("text", [corner, corner + wv, corner + hv, corner + wv + hv])

    def note_path(self, points) -> None:
        """Called by ``domains/strokes.py`` with the center line of each stroke it draws."""
        self.strokes.append(np.asarray(points, dtype=np.float64).reshape(-1, 3))


@dataclass(frozen=True)
class Recording:
    points: np.ndarray  # (N, 3): every recorded vertex and corner; may be empty
    kinds: tuple[str, ...]
    strokes: tuple[np.ndarray, ...]
    right: np.ndarray
    up: np.ndarray


class _Watch:
    """The state a recorded ``draw`` sees: the real one, noting which other objects it reads."""

    def __init__(self, state: SceneState, own: str) -> None:
        self._state, self._own, self.reads = state, own, set()

    def __getitem__(self, object_id: str):
        if object_id != self._own:
            self.reads.add(object_id)
        return self._state[object_id]

    def __contains__(self, object_id: object) -> bool:
        return object_id in self._state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state, name)


_CACHE: OrderedDict[tuple, tuple[dict[str, Any], Recording]] = OrderedDict()


def _camera_key(cam) -> tuple:
    return (cam.kind, cam.view_proj.tobytes(), cam.right.tobytes(), cam.up.tobytes(),
            float(cam.height), float(cam.tan_half_fov))  # fmt: skip


def _style_key(style) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in style.items()))


def _fingerprint(state: SceneState, object_id: str) -> tuple:
    o = state[object_id]
    return (o.params, _style_key(o.style), _camera_key(state.camera_for(o)))


def record(impl, state: SceneState) -> Recording:
    """What ``impl.draw`` emits for its object in ``state`` (cached)."""
    me = state[impl.id]
    cam = state.camera_for(me)
    key = (type(impl), impl.id, *_fingerprint(state, impl.id), state.base_dir)
    hit = _CACHE.get(key)
    if hit is not None:
        reads, rec = hit
        if all(i in state and _fingerprint(state, i) == fp for i, fp in reads.items()):
            _CACHE.move_to_end(key)
            return rec
    rec_ctx = Recorder(cam.right, cam.up)
    watch = _Watch(state, impl.id)
    impl.draw(rec_ctx, watch, "color")
    pts = np.vstack(rec_ctx.points) if rec_ctx.points else np.zeros((0, 3))
    rec = Recording(pts, tuple(rec_ctx.kinds), tuple(rec_ctx.strokes), rec_ctx.right, rec_ctx.up)
    reads = {i: _fingerprint(state, i) for i in sorted(watch.reads) if i in state}
    _CACHE[key] = (reads, rec)
    if len(_CACHE) > CACHE_SIZE:
        _CACHE.popitem(last=False)
    return rec


def _anchor(state: SceneState, object_id: str) -> np.ndarray:
    """Where an object that drew nothing is: its ``pos`` when it has one, else the origin."""
    pos = getattr(state[object_id].params, "pos", None)
    return np.asarray(pos if pos is not None else (0.0, 0.0, 0.0), dtype=np.float64)


def bounds_of(rec: Recording, fallback: np.ndarray) -> Bounds:
    return Bounds.of_points(rec.points if len(rec.points) else fallback)


def box_features(bounds: Bounds, right, up) -> dict[str, np.ndarray]:
    """``center`` and the compass points of a box, its sides as seen along ``right``/``up``."""
    right, up = np.asarray(right, dtype=np.float64), np.asarray(up, dtype=np.float64)
    center = bounds.center
    half = (np.asarray(bounds.hi) - np.asarray(bounds.lo)) / 2
    along_r = float(np.abs(half) @ np.abs(right))  # the box's half extent across the view
    along_u = float(np.abs(half) @ np.abs(up))
    out = {"center": center}
    for name, (dx, dy) in COMPASS.items():
        out[name] = center + dx * along_r * right + dy * along_u * up
    return out


def features_of(rec: Recording, bounds: Bounds) -> dict[str, np.ndarray]:
    out = box_features(bounds, rec.right, rec.up)
    if len(rec.strokes) == 1 and len(rec.strokes[0]) >= 2:
        curve = rec.strokes[0]
        out.update(start=curve[0], end=curve[-1], curve=curve)
    return out


def derived_world_bounds(self, state: SceneState) -> Bounds:
    """The box around everything ``draw`` emits (ADR 0005)."""
    return bounds_of(record(self, state), _anchor(state, self.id))


def derived_features(self, state: SceneState) -> dict[str, np.ndarray]:
    """``center``, the box's compass points, and ``start``/``end``/``curve`` for one stroke."""
    rec = record(self, state)
    return features_of(rec, bounds_of(rec, _anchor(state, self.id)))


def derived_semantics(self, state: SceneState) -> dict[str, Any]:
    """``{"type", "params"}``: what a type that says nothing more about itself is."""
    me = state[self.id]
    return {"type": me.type, "params": me.params.model_dump(mode="json")}


def parts(state: SceneState, object_id: str):
    """The objects a composite built, in the order it built them."""
    return [o for o in state.objects if o.parent == object_id]


def composite_world_bounds(self, state: SceneState) -> Bounds:
    """The box around every part."""
    found = parts(state, self.id)
    if not found:
        return Bounds.of_points(_anchor(state, self.id))
    boxes = [o.impl.world_bounds(state) for o in found]
    return Bounds.of_points([p for b in boxes for p in (b.lo, b.hi)])


def composite_features(self, state: SceneState) -> dict[str, np.ndarray]:
    """``center`` and the compass points of the box around every part, and each part's own
    features as ``<part>.<feature>`` (``arrow.end``)."""
    cam = state.camera_for(state[self.id])
    out = box_features(composite_world_bounds(self, state), cam.right, cam.up)
    for o in parts(state, self.id):
        local = o.id[len(self.id) + 1 :]
        for name, pts in o.impl.features(state).items():
            out[f"{local}.{name}"] = pts
    return out


def composite_semantics(self, state: SceneState) -> dict[str, Any]:
    """``{"type", "params", "parts"}``: the parts by their own names."""
    return {**derived_semantics(self, state),
            "parts": [o.id[len(self.id) + 1 :] for o in parts(state, self.id)]}  # fmt: skip


COMPOSITE = {
    "world_bounds": composite_world_bounds,
    "features": composite_features,
    "semantics": composite_semantics,
}

DERIVED = {
    "world_bounds": derived_world_bounds,
    "features": derived_features,
    "semantics": derived_semantics,
}
