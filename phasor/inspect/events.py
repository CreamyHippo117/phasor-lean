"""The event log: what changes on screen, and when.

The whole timeline is sampled at 10 Hz plus every marker. An event is emitted only when a
value crosses a threshold: visibility 0.9 / 0.5 / 0.1, entering or leaving the frame, a
change of screen region, a new occluder. An event fires the moment a value crosses its
threshold, and re-arms only once the value is 5% back on the other side (hysteresis), so
an object grazing an edge does not spam the log, and "hidden" means under 10% visible.
Events carry the time, the nearest marker, the ids
and suggested fixes; the ones a viewer would notice as a problem (a label or key object
hidden or leaving the frame) are ``"severity": "warning"``.

Pure over snapshots: ``log`` takes ``(t, snapshot)`` pairs from the inspector.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

HZ = 10.0
LEVELS = (0.9, 0.5, 0.1)  # visibility thresholds
DOWN = {0.9: "partly_occluded", 0.5: "occluded", 0.1: "hidden"}
HYST = 0.05
FRAME = 0.5  # on-screen share that counts as "in frame"
OCCLUDER = 0.10  # share an object must cover to be reported as a new occluder


def sample_times(duration: float, markers: Mapping[str, float], t0: float = 0.0,
                 t1: float | None = None, hz: float = HZ) -> list[float]:  # fmt: skip
    t1 = duration if t1 is None else t1
    ts = {round(t, 6) for t in np.arange(t0, t1 + 1e-9, 1.0 / hz)}
    ts.update(round(t, 6) for t in markers.values() if t0 <= t <= t1)
    ts.update({round(t0, 6), round(t1, 6)})
    return sorted(ts)


def nearest_marker(markers: Mapping[str, float], t: float) -> str | None:
    """The last marker at or before ``t`` (what was being said), else the first one."""
    before = [(m, n) for n, m in markers.items() if m <= t + 1e-9]
    if before:
        return max(before)[1]
    return min(((m, n) for n, m in markers.items()), default=(None, None))[1]


def _important(e: dict[str, Any]) -> bool:
    return e["semantic"].get("kind") == "label" or "key" in e["tags"]


class _Track:
    """Hysteresis state for one object."""

    def __init__(self) -> None:
        self.above = dict.fromkeys(LEVELS, True)
        self.in_frame: bool | None = None
        self.cell: tuple[int, int] | None = None
        self.occluders: set[str] = set()


def _cell(norm: tuple[float, float], prev: tuple[int, int] | None) -> tuple[int, int]:
    """3x3 grid cell of a normalized point, held until it is 5% of a cell past the edge."""
    x, y = norm[0] * 3, norm[1] * 3
    col, row = min(2, max(0, int(x))), min(2, max(0, int(y)))
    if prev is not None:
        pc, pr = prev
        if pc - HYST <= x <= pc + 1 + HYST:
            col = pc
        if pr - HYST <= y <= pr + 1 + HYST:
            row = pr
    return col, row


REGION = (("upper-left", "upper-center", "upper-right"),
          ("middle-left", "center", "middle-right"),
          ("lower-left", "lower-center", "lower-right"))  # fmt: skip


def log(samples: Iterable[tuple[float, dict[str, Any]]], markers: Mapping[str, float]
        ) -> list[dict[str, Any]]:  # fmt: skip
    """Events from a time-ordered series of inspector snapshots."""
    tracks: dict[str, _Track] = {}
    out: list[dict[str, Any]] = []

    def emit(t: float, event: str, e: dict[str, Any], severity: str = "info", **info: Any):
        out.append({"t": round(t, 4), "marker": nearest_marker(markers, t), "event": event,
                    "id": e["id"], "severity": severity, **info})  # fmt: skip

    for t, snap in samples:
        for e in snap["objects"]:
            if e["opacity"] <= 0.05 or e["semantic"].get("drawn", 1.0) <= 0:
                tracks.pop(e["id"], None)  # not shown: start afresh when it appears
                continue
            if e["semantic"].get("backdrop"):
                continue
            tr = tracks.setdefault(e["id"], _Track())
            warn = "warning" if _important(e) else "info"
            is_label = e["semantic"].get("kind") == "label"
            # entering and leaving the frame
            on = e["on_screen_fraction"]
            if tr.in_frame is None:
                tr.in_frame = on >= FRAME
                if not tr.in_frame:
                    emit(t, "leaves_frame", e, warn, on_screen=on,
                         fix_hints=_frame_hints(e))  # fmt: skip
            elif tr.in_frame and on < FRAME:
                tr.in_frame = False
                emit(t, "leaves_frame", e, warn, on_screen=on, fix_hints=_frame_hints(e))
            elif not tr.in_frame and on > FRAME + HYST:
                tr.in_frame = True
                emit(t, "enters_frame", e, on_screen=on)
            # region changes
            cell = _cell(tuple(e["center"]["norm"]), tr.cell)
            if tr.cell is not None and cell != tr.cell:
                emit(t, "region", e, region=REGION[cell[1]][cell[0]],
                     was=REGION[tr.cell[1]][tr.cell[0]])  # fmt: skip
            tr.cell = cell
            # visibility (pixel occlusion only; analytic snapshots report None)
            v = e["visible_fraction"]
            if v is None or not tr.in_frame:
                continue
            hidden_by = e["hidden_by"] or []
            for level in LEVELS:
                if tr.above[level] and v < level:
                    tr.above[level] = False
                    # A label partly hidden is hard to read; any other object only matters
                    # once at least half of it is hidden.
                    sev = warn if (level <= 0.5 or is_label) else "info"
                    emit(
                        t,
                        DOWN[level],
                        e,
                        sev,
                        visible=v,
                        hidden_by=hidden_by,
                        fix_hints=_occlusion_hints(e, hidden_by),
                    )
                elif not tr.above[level] and v > level + HYST:
                    tr.above[level] = True
                    emit(t, "visible_again", e, level=level, visible=v)
            now = {h["id"] for h in hidden_by if h["share"] >= OCCLUDER}
            gone = {i for i in tr.occluders if not any(
                h["id"] == i and h["share"] >= OCCLUDER - HYST for h in hidden_by)}  # fmt: skip
            for i in sorted(now - tr.occluders):
                share = next(h["share"] for h in hidden_by if h["id"] == i)
                emit(t, "new_occluder", e, occluder=i, share=share)
            tr.occluders = (tr.occluders - gone) | now
    return out


def _occlusion_hints(e: dict[str, Any], hidden_by: list[dict[str, Any]]) -> list[str]:
    i = e["id"]
    hints = [f"camera avoid_occlusion({i})"]
    if hidden_by:
        top = hidden_by[0]["id"]
        hints += [f"set_style {top} material glass", f"move {i} or {top}"]
    if e["semantic"].get("kind") == "label":
        hints.append("give the label a 'near' constraint so the placer can move it")
    return hints


def _frame_hints(e: dict[str, Any]) -> list[str]:
    i = e["id"]
    return [f"camera keep_visible({i})", f"on_screen({i})"]


def problems(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The events a viewer would see as a problem (``severity == "warning"``)."""
    return [ev for ev in events if ev["severity"] == "warning"]


def hidden_spans(samples: list[tuple[float, dict[str, Any]]], object_id: str,
                 below: float = 0.5) -> list[tuple[float, float]]:  # fmt: skip
    """Time spans where the object is shown but its visible fraction is under ``below``."""
    spans: list[tuple[float, float]] = []
    start = None
    last = None
    for t, snap in samples:
        e = next((x for x in snap["objects"] if x["id"] == object_id), None)
        v = None if e is None or e["opacity"] <= 0.05 else e["visible_fraction"]
        low = v is not None and v < below
        if low and start is None:
            start = t
        if not low and start is not None:
            spans.append((start, t))
            start = None
        last = t
    if start is not None and last is not None:
        spans.append((start, last))
    return spans
