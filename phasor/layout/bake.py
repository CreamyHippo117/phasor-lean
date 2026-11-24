"""The layout bake (contract 7).

The solver runs at every key time (markers, the ends of every transition, and a 10 Hz grid)
and writes positions into tracks. ``baked_state`` then gives the scene at any time with those
positions in place; renderers and the inspector both read it, so what is inspected is what
renders.

A baked position is stored on screen, relative to the object it is placed against. Between
key times only that offset is interpolated, so a label stays glued to a moving anchor and
keeps its gap while the camera zooms.

The bake is a pure function of (compiled document, occlusion oracle): the same inputs give
the same tracks, and it is cached per document version by the Engine.
"""

from __future__ import annotations

import bisect
import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from phasor.core.constraints import BOX_ANCHORS, Guard, Relation
from phasor.core.issues import Issue
from phasor.core.state import SceneState
from phasor.core.values import normalize
from phasor.layout import labels, solver
from phasor.layout.geometry import (
    Box,
    Coverage,
    View,
    footprint,
    frame,
    is_backdrop,
    occludes,
    screen_box,
    view_of,
)
from phasor.layout.plan import MOVE_PARAM, Plan
from phasor.time.compile import Compiled, evaluate
from phasor.time.tracks import ResolvedTrack

SAMPLE_HZ = 10.0
ANALYTIC_WIDTH = 480
MAX_TIMES_IN_ISSUE = 12


class OcclusionOracle(Protocol):
    """What the label placer needs from a renderer (ADR 0001): which pixels are covered."""

    def coverage(self, state: SceneState, exclude: frozenset[str]) -> np.ndarray:
        """(h, w) bool, row 0 = top: pixels covered by an occluding object not in ``exclude``."""
        ...

    def depth(self, state: SceneState, exclude: frozenset[str]) -> np.ndarray:
        """(h, w) float, row 0 = top: distance along the view direction to the nearest
        occluding object not in ``exclude`` (``inf``: nothing; ``-inf``: an overlay object,
        which covers everything in the world). For 3D labels, which are hidden only by what
        is in front of them."""
        ...


class AnalyticOracle:
    """No renderer: every occluding object covers its ``geometry.footprint``, at the depth
    of its center."""

    def _size(self, state: SceneState) -> tuple[int, int]:
        return (ANALYTIC_WIDTH, max(1, round(ANALYTIC_WIDTH / state.aspect)))

    def _occluders(self, state: SceneState, exclude: frozenset[str]):
        for obj in state.objects:
            if obj.id in exclude or not occludes(obj) or is_backdrop(state, obj):
                continue
            yield obj

    def coverage(self, state: SceneState, exclude: frozenset[str]) -> np.ndarray:
        size = self._size(state)
        mask = np.zeros(size[::-1], dtype=bool)
        for obj in self._occluders(state, exclude):
            mask |= footprint(state, obj, size)
        return mask

    def depth(self, state: SceneState, exclude: frozenset[str]) -> np.ndarray:
        size = self._size(state)
        out = np.full(size[::-1], np.inf)
        for obj in self._occluders(state, exclude):
            cam = state.camera_for(obj)
            d = (
                -np.inf
                if obj.layer == "overlay"
                else cam.depth(obj.impl.world_bounds(state).center)[0]
            )
            m = footprint(state, obj, size)
            out[m] = np.minimum(out[m], d)
        return out


LIFT = 0.04  # a 3D label sits this share of the way from its anchor toward the camera


def anchor_base(state: SceneState, object_id: str, ref: str | None):
    """Where a laid-out object lives in depth. In 2D: its own position. Under a perspective
    camera an object placed against ``ref`` (a label) lives at ``ref``'s depth, nudged toward
    the camera so it is not buried in what it names; layout then only slides it sideways
    in the camera plane, so it keeps that depth."""
    obj = state[object_id]
    own = state.values[f"{object_id}.{MOVE_PARAM}"]
    cam = state.camera_for(obj)
    if ref is None or not cam.perspective or state[ref].layer != obj.layer:
        return own
    center = np.asarray(state[ref].impl.world_bounds(state).center)
    return normalize(center + (cam.eye - center) * LIFT)


@dataclass(frozen=True)
class Track:
    """One baked parameter. ``keys`` are screen points (relative to ``ref``'s center when it
    is set) for a position, or plain numbers for a scalar."""

    path: str
    kind: str  # "point" | "scalar"
    ref: str | None
    keys: tuple
    # jumps[i] is True when the label changes slot between key i and key i + 1. It then
    # holds its place and jumps at key i + 1, instead of sliding across its own anchor.
    jumps: tuple[bool, ...] = ()


@dataclass(frozen=True)
class Bake:
    times: tuple[float, ...] = ()
    tracks: tuple[Track, ...] = ()
    issues: tuple[Issue, ...] = ()
    # The camera solver's result: "camera.<param>" -> value at each key time.
    camera: dict[str, tuple] = field(default_factory=dict)
    slots: dict[str, tuple[str, ...]] = field(default_factory=dict)  # label id -> slot per key
    occlusion: str = "analytic"  # "pixel" when the placer read the real id pass

    def camera_at(self, t: float) -> dict[str, object]:
        """Solved camera parameters at ``t`` (linear between key times, which are 10 Hz)."""
        if not self.camera:
            return {}
        ts = self.times
        i = min(max(bisect.bisect_right(ts, t) - 1, 0), len(ts) - 1)
        j = min(i + 1, len(ts) - 1)
        u = 0.0 if j == i else min(max((t - ts[i]) / (ts[j] - ts[i]), 0.0), 1.0)
        out: dict[str, object] = {}
        for path, vals in self.camera.items():
            a, b = np.asarray(vals[i], dtype=np.float64), np.asarray(vals[j], dtype=np.float64)
            out[path] = normalize(a + (b - a) * u)
        return out

    def values_at(self, t: float, pre: SceneState) -> dict[str, object]:
        """Solved parameter values at ``t``, given the scene as the document alone gives it."""
        if not self.tracks:
            return {}
        ts = self.times
        i = min(max(bisect.bisect_right(ts, t) - 1, 0), len(ts) - 1)
        j = min(i + 1, len(ts) - 1)
        u = 0.0 if j == i else min(max((t - ts[i]) / (ts[j] - ts[i]), 0.0), 1.0)
        out: dict[str, object] = {}
        for tr in self.tracks:
            a, b = np.asarray(tr.keys[i]), np.asarray(tr.keys[j])
            v = a + (b - a) * (0.0 if tr.jumps and j > i and tr.jumps[i] and u < 1.0 else u)
            if tr.kind == "scalar":
                out[tr.path] = float(v)
                continue
            object_id = tr.path.rpartition(".")[0]
            obj = pre[object_id]
            if tr.ref is not None:
                v = v + np.asarray(screen_box(pre, pre[tr.ref]).center)
            # A position and its leader tip both live at the label's depth (its anchor's,
            # in 3D), exactly as the bake loop placed them.
            out[tr.path] = _to_world(pre, obj, anchor_base(pre, object_id, tr.ref), v)
        return out


def _to_world(state: SceneState, obj, base, screen_point) -> tuple:
    """The world point that appears at ``screen_point``, at the depth of ``base``. The
    shift is measured at ``base`` (in 3D a screen offset means more world units further
    away), never at wherever the object's document position happens to be."""
    cam = state.camera_for(obj)
    view = View(cam, state.aspect, base if cam.perspective else None)
    bx, by = view.point(base)
    return normalize(np.asarray(base) + view.shift(screen_point[0] - bx, screen_point[1] - by))


def key_times(c: Compiled) -> tuple[float, ...]:
    """Markers, the ends of every transition, and a uniform grid."""
    end = c.duration
    ts = {0.0, end, *c.time_map.markers.values()}
    for node in c.nodes.values():
        if isinstance(node, ResolvedTrack):
            for s in node.segments:
                ts.update((s.t0, s.t1))
    n = int(np.floor(end * SAMPLE_HZ + 1e-9))
    ts.update(k / SAMPLE_HZ for k in range(n + 1))
    return tuple(sorted(round(t, 9) for t in ts if 0.0 <= t <= end))


def baked_state(c: Compiled, baked: Bake, t: float) -> SceneState:
    """The scene at ``t`` with layout applied. Pure: depends only on (document, bake, t).
    The solved camera goes in first, so labels are placed as that camera sees them."""
    cam = baked.camera_at(t)
    pre = evaluate(c, t, cam) if cam else evaluate(c, t)
    values = baked.values_at(t, pre)
    return evaluate(c, t, {**cam, **values}) if values else pre


def bake(c: Compiled, plan: Plan, oracle: OcclusionOracle | None = None) -> Bake:
    if plan.empty:
        return Bake()
    oracle = oracle or AnalyticOracle()
    lay = c.theme.layout
    times = key_times(c)
    cam_issues: tuple[Issue, ...] = ()
    solved_camera: dict[str, tuple] = {}
    if plan.goals:
        from phasor.layout import camera as camera_solver

        sol = camera_solver.solve(c, plan, times, lambda st, i: hidden_share(st, i, oracle))
        solved_camera, cam_issues = sol.values, sol.issues
    camera_keys = {k: {p: v[k] for p, v in solved_camera.items()} for k in range(len(times))}
    placed_ids = frozenset(plan.placed)
    pos = {i: f"{i}.{MOVE_PARAM}" for i in plan.moving}

    # Which object each moving object follows between key times.
    ref: dict[str, str | None] = dict.fromkeys(plan.moving)
    for _, rel in plan.relations():
        if ref.get(rel.subject) is None:
            ref[rel.subject] = rel.target

    keys: dict[str, list] = {p: [] for p in pos.values()}
    for i in sorted(plan.leaders):
        keys[f"{i}.leader"], keys[f"{i}.leader_on"] = [], []
    failures: dict[tuple[int, str], tuple[tuple[str, ...], list[float]]] = {}
    slots: dict[str, list[str]] = {i: [] for i in plan.placed}
    previous: dict[str, labels.Previous] = {}

    def fail(rule, ids, message, t):
        failures.setdefault((rule.index, message), (tuple(ids), []))[1].append(t)

    for k_time, t in enumerate(times):
        cam_now = camera_keys[k_time]
        pre = evaluate(c, t, cam_now) if cam_now else evaluate(c, t)
        if pre.camera.perspective:  # 3D: everything placed against an anchor lives at its depth
            based = {pos[i]: anchor_base(pre, i, ref[i]) for i in plan.moving if ref[i]}
            pre = evaluate(c, t, {**cam_now, **based}) if based else pre
        boxes = {o.id: screen_box(pre, o) for o in pre.objects}
        replace: dict[str, object] = {}

        def move(state, object_id, dx, dy, replace=replace):
            obj = state[object_id]
            path = pos[object_id]
            world = normalize(np.asarray(state.values[path]) + view_of(state, obj).shift(dx, dy))
            replace[path] = world

        # Stage 1: linear relations.
        items = []
        anchors: dict[tuple[str, str], tuple[float, float]] = {}
        for rule, clause in ((r, cl) for r in plan.rules for cl in r.clauses):
            if isinstance(clause, Relation) and clause.kind != "near":
                if clause.kind == "attach_to":
                    spot = screen_anchor(pre, boxes, clause.target, clause.anchor)
                    if spot is None:
                        names = anchor_names(pre, clause.target)
                        why = f"names an unknown spot {clause.anchor!r} on {clause.target!r}"
                        fail(rule, clause.ids, f"{why}; spots are {names}", t)
                        continue
                    anchors[(clause.target, clause.anchor)] = spot
                unit = view_of(pre, pre[clause.subject]).units_per_height
                gap = lay["gap"] if clause.gap is None else clause.gap / unit
                if clause.kind == "tangent_to":
                    gap = 0.0  # tangent means touching
                items.append(solver.Item(rule.index, rule.spec.priority, clause, gap))
            elif (
                isinstance(clause, Guard)
                and clause.kind == "on_screen"
                and clause.ids[0] in plan.solved
            ):
                unit = view_of(pre, pre[clause.ids[0]]).units_per_height
                margin = lay["margin"] if clause.value is None else clause.value / unit
                items.append(solver.Item(rule.index, rule.spec.priority, clause, margin))
        if items:
            centers, unsolved = solver.solve(items, boxes, plan.solved, frame(pre), anchors)
            for object_id, (cx, cy) in centers.items():
                bx, by = boxes[object_id].center
                move(pre, object_id, cx - bx, cy - by)
            for u in unsolved:
                fail(plan.rules[u.index], u.clause.ids, u.message, t)
        for object_id in plan.solved:  # a rule that could not run still leaves a key
            replace.setdefault(pos[object_id], pre.values[pos[object_id]])
        state = evaluate(c, t, {**cam_now, **replace}) if replace else pre

        # Stage 2: the label placer.
        if plan.placed:
            if replace:
                boxes = {o.id: screen_box(state, o) for o in state.objects}
            requests, rule_of = [], {}
            for rule, rel in plan.relations(("near",)):
                unit = view_of(state, state[rel.subject]).units_per_height
                b = boxes[rel.subject]
                requests.append(labels.Request(
                    rel.subject, b.w, b.h, boxes[rel.target],
                    lay["gap"] if rel.gap is None else rel.gap / unit,
                    lay["leader"] if rel.subject in plan.leaders else None,
                ))  # fmt: skip
                rule_of[rel.subject] = (rule, rel)
            # Other text is avoided as whole boxes (exact), so it is left out of the mask.
            fixed = [
                o.id for o in state.objects
                if o.id not in placed_ids and o.params.opacity > 0
                and o.impl.semantics(state).get("kind") == "label"
            ]  # fmt: skip
            fixed_labels = [boxes[i] for i in fixed]
            # Backdrops (domain planes) are meant to be written on: not obstacles.
            backdrops = {o.id for o in state.objects if is_backdrop(state, o)}
            skip = placed_ids | set(fixed) | backdrops
            if state.camera.perspective:
                # Hidden only by what is in front of the label: one mask per label depth.
                depth = oracle.depth(state, frozenset(skip))
                requests = [
                    dataclasses.replace(r, coverage=Coverage(
                        depth < _label_depth(state, r.id) * (1 - DEPTH_SLACK), state.aspect))
                    if state[r.id].layer != "overlay" else r
                    for r in requests
                ]  # fmt: skip
                shared = None
            else:
                shared = Coverage(oracle.coverage(state, skip), state.aspect)
            result = labels.place(
                requests,
                frame=frame(state),
                coverage=shared,
                obstacles=fixed_labels,
                previous=previous,
            )
            for req in requests:
                p = result[req.id]
                bx, by = boxes[req.id].center
                move(state, req.id, p.center[0] - bx, p.center[1] - by)
                ax, ay = req.anchor.center
                previous[req.id] = labels.Previous(p.slot, (p.center[0] - ax, p.center[1] - ay))
                slots[req.id].append(p.slot)
                if req.id in plan.leaders:
                    tip = req.anchor.nearest_point(p.center)
                    obj = state[req.id]
                    replace[f"{req.id}.leader"] = _to_world(
                        state, obj, state.values[pos[req.id]], tip
                    )
                    replace[f"{req.id}.leader_on"] = 1.0 if p.leader else 0.0
                if p.blocked > c.theme.lint["crowded"]:
                    rule, rel = rule_of[req.id]
                    fail(rule, rel.ids, f"has no free place: even the best slot around "
                         f"{rel.target!r} is partly covered or cut by the frame", t)  # fmt: skip

        # Record this key: screen position of each solved parameter, relative to its ref.
        for path, world in replace.items():
            object_id, _, name = path.rpartition(".")
            if name == "leader_on":
                keys[path].append(world)
                continue
            point = np.asarray(view_of(pre, pre[object_id]).point(world))
            r = ref[object_id]
            if r is not None:
                point = point - np.asarray(boxes_pre_center(pre, r))
            keys[path].append((float(point[0]), float(point[1])))

    def jumps(object_id: str) -> tuple[bool, ...]:
        chosen = slots.get(object_id, ())
        return tuple(a != b for a, b in itertools.pairwise(chosen))

    tracks = tuple(
        Track(path, "scalar" if path.endswith(".leader_on") else "point",
              None if path.endswith(".leader_on") else ref[path.rpartition(".")[0]], tuple(ks),
              jumps(path.rpartition(".")[0]))
        for path, ks in sorted(keys.items())
    )  # fmt: skip
    issues = []
    for (index, message), (ids, ts) in sorted(failures.items()):
        rule = plan.rules[index]
        shown = ", ".join(f"{x:g}" for x in ts[:MAX_TIMES_IN_ISSUE])
        more = f" and {len(ts) - MAX_TIMES_IN_ISSUE} more" if len(ts) > MAX_TIMES_IN_ISSUE else ""
        issues.append(Issue(
            "unsolved", f"constraint {rule.spec.rule!r} {message} (at t = {shown}{more} s)",
            ids=ids, path=rule.path, times=tuple(ts),
            fix_hint="lower its priority, loosen it, or make room (move or shrink a neighbor)",
        ))  # fmt: skip
    kind = "analytic" if isinstance(oracle, AnalyticOracle) else "pixel"
    return Bake(times, tracks, (*cam_issues, *issues),
                slots={k: tuple(v) for k, v in slots.items()}, occlusion=kind,
                camera=solved_camera)  # fmt: skip


def hidden_share(state: SceneState, object_id: str, oracle: OcclusionOracle) -> float:
    """Share of an object hidden by what is in front of it (the camera solver's stage 2):
    pixels in its projected bounding disc (half radius) whose nearest surface, leaving out
    the object itself and all labels, is clearly nearer than the object."""
    obj = state[object_id]
    cam = state.camera_for(obj)
    b = obj.impl.world_bounds(state)
    labels_ = {o.id for o in state.objects if o.impl.semantics(state).get("kind") == "label"}
    depth = oracle.depth(state, frozenset({object_id} | labels_))
    h, w = depth.shape
    d = float(cam.depth(b.center)[0])
    if d <= 0:
        return 1.0
    ((px, py),) = cam.to_screen([b.center], w, h)
    r = max(0.5 * b.radius / (cam.units_per_height(b.center)) * h, 1.5)
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    disc = (xx - px) ** 2 + (yy - py) ** 2 <= r * r
    if not disc.any():
        return 0.0
    return float((depth[disc] < d * (1 - 2 * DEPTH_SLACK)).mean())


GAP = 0.012  # space left between an object and a free slot, in screen heights
FREE = 0.01  # a slot this little covered counts as free space


def free_slots(state: SceneState, object_id: str, oracle: OcclusionOracle | None,
               size: tuple[float, float] = (0.13, 0.042)) -> list[dict]:  # fmt: skip
    """Empty label-sized places around an object, nearest first (the placer's slots, in
    screen heights; ``size`` is (w, h)). Boxes are reported top-left normalized, like the
    inspector: ``[x0, y0, x1, y1]`` with y down."""
    oracle = oracle or AnalyticOracle()
    obj = state[object_id]
    anchor = screen_box(state, obj)
    skip = frozenset({object_id} | {o.id for o in state.objects if is_backdrop(state, o)})
    if state.camera_for(obj).perspective:
        d = float(state.camera_for(obj).depth(obj.impl.world_bounds(state).center)[0])
        cov = Coverage(oracle.depth(state, skip) < d * (1 - LIFT - DEPTH_SLACK), state.aspect)
    else:
        cov = Coverage(oracle.coverage(state, skip), state.aspect)
    req = labels.Request(object_id, size[0], size[1], anchor, GAP)
    fr = frame(state)
    out = []
    for slot, center, _ in labels.candidates(req):
        box = Box.around(center, *size)
        if box.overlap(fr) < box.area * 0.999 or cov.fraction(box) > FREE:
            continue
        a = state.aspect
        out.append({
            "slot": slot,
            "box": [round(box.x0 / a, 4), round(1 - box.y1, 4), round(box.x1 / a, 4),
                    round(1 - box.y0, 4)],
            "distance": round(float(np.hypot(center[0] - anchor.center[0],
                                             center[1] - anchor.center[1])), 4),
        })  # fmt: skip
    return sorted(out, key=lambda s: s["distance"])


DEPTH_SLACK = 0.01  # a surface this close behind or level with a label does not hide it


def _label_depth(state: SceneState, object_id: str) -> float:
    obj = state[object_id]
    return float(state.camera_for(obj).depth(state.values[f"{object_id}.{MOVE_PARAM}"])[0])


def anchor_names(state: SceneState, object_id: str) -> list[str]:
    feats = state[object_id].impl.features(state)
    return sorted({*BOX_ANCHORS, *feats})


def screen_anchor(
    state: SceneState, boxes: dict, object_id: str, name: str
) -> tuple[float, float] | None:
    """The screen point (in screen heights) of a named spot on an object: a spot on its box
    (``top``, ``left``, ``center``, ...) or one of its own feature points (an arrow's ``tip``)."""
    if name in BOX_ANCHORS:
        b = boxes[object_id]
        dx, dy = BOX_ANCHORS[name]
        cx, cy = b.center
        return (cx + dx * b.w / 2, cy + dy * b.h / 2)
    feats = state[object_id].impl.features(state)
    if name in feats:
        pts = np.atleast_2d(np.asarray(feats[name], dtype=np.float64))
        return view_of(state, state[object_id]).point(pts[0])
    return None


def boxes_pre_center(pre: SceneState, object_id: str) -> tuple[float, float]:
    return screen_box(pre, pre[object_id]).center


__all__ = ["AnalyticOracle", "Bake", "Box", "OcclusionOracle", "Track", "bake", "baked_state",
           "key_times"]  # fmt: skip
