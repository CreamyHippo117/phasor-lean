"""The inspector: what is on screen at time t, as JSON-ready dicts.

Ground truth comes from the ``id`` and ``solo`` passes when a PassRunner is available:

- ``visible_fraction`` = pixels where the object is the nearest surface / its solo pixels.
  Both are counted inside the frame, so this is "how much of the on-screen part is unhidden".
- ``hidden_by`` = which objects own the pixels under the solo mask, with shares.
- ``on_screen_fraction`` = share of the full silhouette that lies inside the frame, measured
  on a zoomed-out solo pass when the bounding box crosses the frame edge.

Without a runner the same keys are filled from bounding boxes where possible, occlusion
fields are ``None``, and the snapshot says ``"occlusion": "analytic"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from phasor.core.spatial import region_name
from phasor.core.state import ObjectState, SceneState
from phasor.inspect.passes import PassRunner, inspect_size
from phasor.layout.geometry import occludes

DETAILS = ("brief", "standard", "full")
BRIEF_MAX_LINES = 40
MAX_ZOOM_OUT = 8.0
RELATION_TAG = "key"  # pairwise relations are reported against objects carrying this tag
OCCLUDER_MIN_SHARE = 0.01
TOO_SMALL = "too small or thin to measure at inspector resolution"


def _r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def _box(state: SceneState, obj: ObjectState) -> dict[str, Any]:
    """Screen-space bounding box, center and depth from the object's world bounds."""
    cam = state.camera_for(obj)
    bounds = obj.impl.world_bounds(state)
    px = cam.to_screen(bounds.corners(), state.width, state.height)
    x0, y0 = px.min(0)
    x1, y1 = px.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = state.width, state.height
    ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    area = max(x1 - x0, 1e-9) * max(y1 - y0, 1e-9)
    inside = max(ix1 - ix0, 0) * max(iy1 - iy0, 0) / area
    ndc = cam.project(bounds.corners())
    return {
        "px": (x0, y0, x1, y1),
        "center": (cx, cy),
        "depth": float(cam.depth(bounds.center)[0]),
        "bbox_inside": float(inside),
        "ndc_extent": float(np.abs(ndc[:, :2]).max()),
    }


def _on_screen(state, obj, box, runner: PassRunner | None, size) -> float:
    if box["bbox_inside"] >= 1.0 - 1e-9:
        return 1.0
    if box["bbox_inside"] <= 0.0:
        return 0.0
    if runner is None:
        return box["bbox_inside"]
    k = min(box["ndc_extent"] * 1.05, MAX_ZOOM_OUT)
    mask = runner.solo_mask(state, size, obj.id, zoom_out=k)
    total = int(mask.sum())
    if total == 0:
        return box["bbox_inside"]
    h, w = mask.shape
    x0, x1 = round(w / 2 - w / (2 * k)), round(w / 2 + w / (2 * k))
    y0, y1 = round(h / 2 - h / (2 * k)), round(h / 2 + h / (2 * k))
    return float(mask[y0:y1, x0:x1].sum() / total)


def _relations(a: dict, b: dict, a_box: dict, b_box: dict, state: SceneState) -> dict[str, Any]:
    (ax, ay), (bx, by) = a_box["center"], b_box["center"]
    ax0, ay0, ax1, ay1 = a_box["px"]
    bx0, by0, bx1, by1 = b_box["px"]
    eps = 0.005 * state.height
    depth_gap = b_box["depth"] - a_box["depth"]
    return {
        "other": b["id"],
        "left_of": bool(ax < bx - eps),
        "above": bool(ay < by - eps),
        "in_front_of": None if abs(depth_gap) < 1e-9 else bool(depth_gap > 0),
        "overlaps": bool(ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1),
    }


def snapshot(state: SceneState, runner: PassRunner | None = None) -> dict[str, Any]:
    """Full-detail snapshot of every object. Use ``render_detail`` to trim it for a reader."""
    size = inspect_size(state)
    w, h = state.width, state.height
    by_index = {o.index: o.id for o in state.objects}
    boxes = {o.id: _box(state, o) for o in state.objects}
    idbuf = runner.id_buffer(state, size) if runner else None
    parts: dict[str, list[ObjectState]] = {}  # composite id -> every part under it
    for o in state.objects:
        up = o.parent
        while up is not None:
            parts.setdefault(up, []).append(o)
            up = state[up].parent

    entries: list[dict[str, Any]] = []
    for obj in state.objects:
        box = boxes[obj.id]
        x0, y0, x1, y1 = box["px"]
        cx, cy = box["center"]
        on_screen = _on_screen(state, obj, box, runner, size)
        visible: float | None = None
        hidden_by: list[dict[str, Any]] | None = None
        note: str | None = None
        if obj.params.opacity <= 0:
            visible, hidden_by = 0.0, []
        elif runner is not None and on_screen > 0:
            # A composite (ADR 0005) draws nothing itself: it is seen where its parts are.
            members = [obj, *parts.get(obj.id, ())]
            mask = runner.solo_mask(state, size, obj.id)
            for m in members[1:]:
                mask = mask | runner.solo_mask(state, size, m.id)
            solo = int(mask.sum())
            if solo > 0:
                force = frozenset(m.id for m in members if not occludes(m))
                buf = runner.id_buffer(state, size, force) if force else idbuf
                under = buf[mask]
                own = {m.index for m in members}
                visible = min(1.0, float(np.isin(under, list(own)).sum() / solo))
                counts = np.bincount(under, minlength=len(state.objects) + 1)
                hidden_by = sorted(
                    (
                        {"id": by_index[i], "share": _r(c / solo)}
                        for i, c in enumerate(counts)
                        if i != 0 and i not in own and c / solo >= OCCLUDER_MIN_SHARE
                    ),
                    key=lambda d: (-d["share"], d["id"]),
                )
            else:
                note = TOO_SMALL
        elif runner is not None:
            visible, hidden_by = 0.0, []
        feats = obj.impl.features(state)
        cam = state.camera_for(obj)
        part_of = {"part_of": obj.parent} if obj.parent is not None else {}
        entries.append(
            {
                **part_of,
                "id": obj.id,
                "type": obj.type,
                "layer": obj.layer,
                "tags": list(obj.tags),
                "semantic": obj.impl.semantics(state),
                "bbox": {
                    "px": [_r(x0, 1), _r(y0, 1), _r(x1, 1), _r(y1, 1)],
                    "norm": [_r(x0 / w), _r(y0 / h), _r(x1 / w), _r(y1 / h)],
                },
                "center": {"px": [_r(cx, 1), _r(cy, 1)], "norm": [_r(cx / w), _r(cy / h)]},
                "depth": _r(box["depth"]),
                "region": region_name(cx / w, cy / h),
                "on_screen_fraction": _r(on_screen),
                "visible_fraction": None if visible is None else _r(visible),
                "hidden_by": hidden_by,
                "note": note,
                "opacity": _r(obj.params.opacity),
                "style": dict(obj.roles),
                "features": {
                    name: {
                        "world": np.round(np.atleast_2d(pts), 4).tolist(),
                        "px": np.round(cam.to_screen(np.atleast_2d(pts), w, h), 1).tolist(),
                    }
                    for name, pts in feats.items()
                },
            }
        )
    keyed = [e for e in entries if RELATION_TAG in e["tags"]]
    for e in entries:
        e["relations"] = [
            _relations(e, k, boxes[e["id"]], boxes[k["id"]], state) for k in keyed if k is not e
        ]
    return {
        "t": _r(state.t, 4),
        "frame": {"width": w, "height": h},
        "occlusion": "pixel" if runner else "analytic",
        "objects": entries,
    }


def _brief_line(e: dict[str, Any]) -> str:
    cx, cy = e["center"]["norm"]
    vis = "?" if e["visible_fraction"] is None else f"{e['visible_fraction']:.2f}"
    line = (
        f"{e['id']} [{e['type']}] {e['region']} ({cx:.2f},{cy:.2f}) "
        f"visible {vis} on_screen {e['on_screen_fraction']:.2f}"
    )
    if e["hidden_by"]:
        line += " hidden_by " + ", ".join(f"{h['id']} {h['share']:.2f}" for h in e["hidden_by"])
    if e["note"]:
        line += f" ({e['note']})"
    return line


def render_detail(snap: dict[str, Any], detail: str = "standard", only: str | None = None) -> Any:
    """Trim a snapshot. ``brief``: one line per object, at most 40 lines. ``standard``: no
    feature geometry. ``full``: everything. ``only`` selects one object by id."""
    if detail not in DETAILS:
        raise ValueError(f"detail must be one of {DETAILS}, got {detail!r}")
    objects = snap["objects"]
    if only is not None:
        objects = [e for e in objects if e["id"] == only]
        if not objects:
            raise KeyError(only)
    if detail == "brief":
        lines = [_brief_line(e) for e in objects]
        if len(lines) > BRIEF_MAX_LINES:
            extra = len(lines) - (BRIEF_MAX_LINES - 1)
            lines = [*lines[: BRIEF_MAX_LINES - 1], f"... {extra} more objects (ask by id)"]
        return {"t": snap["t"], "occlusion": snap["occlusion"], "objects": lines}
    if detail == "standard":
        objects = [{k: v for k, v in e.items() if k != "features"} for e in objects]
    return {**snap, "objects": objects}


def hit_test(state: SceneState, runner: PassRunner | None, x: float, y: float) -> list[str]:
    """Ids under an output-pixel point, topmost first."""
    hits = set()
    for obj in state.objects:
        x0, y0, x1, y1 = _box(state, obj)["px"]
        if x0 <= x <= x1 and y0 <= y <= y1 and obj.params.opacity > 0:
            hits.add(obj.id)
    ids = [o.id for o in reversed(state.draw_order()) if o.id in hits]
    if runner is None:
        return ids
    size = inspect_size(state)
    px = min(size[0] - 1, max(0, int(x / state.width * size[0])))
    py = min(size[1] - 1, max(0, int(y / state.height * size[1])))
    ids = [i for i in ids if runner.solo_mask(state, size, i)[py, px]]
    top = int(runner.id_buffer(state, size)[py, px])
    top_id = next((o.id for o in state.objects if o.index == top), None)
    if top_id in ids:
        ids.remove(top_id)
        ids.insert(0, top_id)
    return ids
