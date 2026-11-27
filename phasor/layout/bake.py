"""The layout bake (contract 7).

The solver runs at every key time (markers, the ends of every transition, and a 10 Hz grid)
where something it places is shown, and at the key on either side of such a time, and writes
positions into tracks. While everything it places is hidden it does not run (a label shown
for ten seconds of a long 3D scene costs ten seconds of depth renders, not the scene's); what
it places, being hidden there, holds its first or last place or moves unseen between two
runs. ``baked_state`` then gives the scene at any time with those positions in place;
renderers and the inspector both read it, so what is inspected is what renders.

A baked position is stored on screen, relative to the object it is placed against. Between
key times only that offset is interpolated, so a label stays glued to a moving anchor and
keeps its gap while the camera zooms.

The bake is a pure function of (compiled document, occlusion oracle): the same inputs give
the same tracks, and it is cached per document version by the Engine. After an edit that
only changes motion from some time on (a verb, a new segment), or parameters of an object
the layout cannot see until some key, the Engine hands the previous bake in: the keys
before that time cannot change (``layout_unchanged_until``), so they are kept and the loop
resumes from there, with the placer's memory as it was. Not with camera goals: the camera
solver is global (``layout/camera.py``).
"""

from __future__ import annotations

import bisect
import dataclasses
import itertools
import math
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
    pixel_mask,
    screen_box,
    view_of,
)
from phasor.layout.plan import LEADER_PARAMS, MOVE_PARAM, Plan
from phasor.time.compile import Compiled, Const, evaluate
from phasor.time.tracks import ResolvedSegment, ResolvedTrack

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
class Resume:
    """What a later bake needs to start part-way through this one: every key time it
    considered and whether it ran there, and after each key it ran: the placer's memory,
    the occlusion buffers rendered so far, and the failures so far."""

    keyed: tuple[float, ...]
    needed: tuple[bool, ...]
    previous: tuple[dict, ...]  # per baked key
    renders: tuple[int, ...]  # per baked key, cumulative
    failures: tuple[tuple[int, str, tuple[str, ...], float], ...]  # (rule, message, ids, t)


@dataclass(frozen=True)
class Bake:
    times: tuple[float, ...] = ()  # the key times the placer ran at; tracks have one key each
    tracks: tuple[Track, ...] = ()
    issues: tuple[Issue, ...] = ()
    # The camera solver's result: "camera.<param>" -> value at each of ``camera_times``.
    camera: dict[str, tuple] = field(default_factory=dict)
    slots: dict[str, tuple[str, ...]] = field(default_factory=dict)  # label id -> slot per key
    occlusion: str = "analytic"  # "pixel" when the placer read the real id pass
    camera_times: tuple[float, ...] = ()  # every key time, when there are camera goals
    # Occlusion buffers (depth or coverage) a bake from nothing renders: its cost (lint
    # ``slow_layout``). Counted the same with or without OpenGL.
    renders: int = 0
    spans: tuple[tuple[float, float], ...] = ()  # (first, last) of each run of baked keys
    resume: Resume | None = field(default=None, compare=False, repr=False)

    def camera_at(self, t: float) -> dict[str, object]:
        """Solved camera parameters at ``t`` (linear between key times, which are 10 Hz)."""
        if not self.camera:
            return {}
        ts = self.camera_times
        i = min(max(bisect.bisect_right(ts, t) - 1, 0), len(ts) - 1)
        j = min(i + 1, len(ts) - 1)
        u = 0.0 if j == i else min(max((t - ts[i]) / (ts[j] - ts[i]), 0.0), 1.0)
        out: dict[str, object] = {}
        for path, vals in self.camera.items():
            a, b = np.asarray(vals[i], dtype=np.float64), np.asarray(vals[j], dtype=np.float64)
            out[path] = normalize(a + (b - a) * u)
        return out

    def values_at(self, t: float, pre: SceneState) -> dict[str, object]:
        """Solved parameter values at ``t``, given the scene as the document alone gives it.
        Before the first key and after the last, the first and last hold; across keys the
        placer skipped (everything it places hidden) values move as between any two keys."""
        if not self.tracks or not self.times:
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


def shown_keys(c: Compiled, plan: Plan, times: tuple[float, ...]) -> tuple[bool, ...]:
    """Which of ``times`` the placer runs at: where anything it moves is shown (opacity x
    dim above 0), and the key on either side of such a time, so every stretch in which
    something is shown has a baked key at both ends and moves exactly as before.

    Everywhere, when something shown may depend on where a moved object is while it is
    hidden (an expression or another object reads it), or its opacity is an expression
    (which may rise between two keys)."""
    ids = sorted(plan.moving)
    if not ids:
        return (False,) * len(times)
    if _always_needed(c, ids):
        return (True,) * len(times)
    return shown_times(c, ids, times)


def shown_times(c: Compiled, ids, times: tuple[float, ...]) -> tuple[bool, ...]:
    """Which of ``times`` any of ``ids`` is shown at (opacity x dim above 0), or is the key
    on either side of such a time (a flash between two keys still counts). True everywhere
    when an opacity or dim is an expression, which may rise between two keys: what cannot
    be proved hidden is shown. The placer (``shown_keys``) and the camera solver (whose
    ``avoid_occlusion`` rests while its object is hidden) both ask this."""
    n = len(times)
    nodes = [(c.nodes.get(f"{i}.opacity"), c.nodes.get(f"{i}.dim")) for i in ids]
    if any(x is not None and not isinstance(x, Const | ResolvedTrack)
           for pair in nodes for x in pair):  # fmt: skip
        return (True,) * n

    def value(node, t: float) -> float:
        if node is None:
            return 1.0
        return float(node.value_at(t) if isinstance(node, ResolvedTrack) else node.value)

    vis = [any(value(op, t) * value(dim, t) > 0 for op, dim in nodes) for t in times]
    return tuple(vis[k] or (k > 0 and vis[k - 1]) or (k + 1 < n and vis[k + 1]) for k in range(n))


def _always_needed(c: Compiled, ids: list[str]) -> bool:
    named = set(ids)
    written = {f"{i}.{p}" for i in ids for p in (MOVE_PARAM, *LEADER_PARAMS)}
    for path, node in c.nodes.items():
        owner = path.rpartition(".")[0]
        if any(d in written and d.rpartition(".")[0] != owner
               for d in getattr(node, "deps", ())):  # fmt: skip
            return True  # another object's expression reads where a moved one is
    for o in c.doc.objects:  # an object that names a moved one (a ring around a label)
        values = [*o.params.values(),
                  *(s.to for p, segs in c.doc.timeline.tracks.items()
                    if p.rpartition(".")[0] == o.id for s in segs)]  # fmt: skip
        if o.id not in named and any(isinstance(v, str) and v in named for v in values):
            return True
    return False


class _Counted:
    """The oracle, counting the buffers it renders."""

    def __init__(self, oracle: OcclusionOracle) -> None:
        self.oracle, self.n = oracle, 0

    def coverage(self, state: SceneState, exclude: frozenset[str]) -> np.ndarray:
        self.n += 1
        return self.oracle.coverage(state, exclude)

    def depth(self, state: SceneState, exclude: frozenset[str]) -> np.ndarray:
        self.n += 1
        return self.oracle.depth(state, exclude)


def _same(a, b) -> bool:
    try:
        return bool(a == b)
    except ValueError:  # arrays
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _same_segment(a: ResolvedSegment, b: ResolvedSegment) -> bool:
    return a.t0 == b.t0 and a.t1 == b.t1 and a.ease is b.ease and _same(a.to, b.to)


def _without_tracks(doc, unseen=()) -> dict:
    """The document as plain data with its tracks left out, for comparing two versions
    (a dump, not a copy: documents are only rebuilt by patches, contract 1). The params
    of the ``unseen`` objects are left out too (``layout_unchanged_until``)."""
    data = doc.model_dump(mode="json")
    data["timeline"]["tracks"] = {}
    for o in data["objects"]:
        if o["id"] in unseen:
            o["params"] = {}
    return data


def unchanged_until(old: Compiled, new: Compiled) -> float:
    """A time before which the two compiled documents give the same scene at every t:
    ``inf`` when they are the same, 0 when that cannot be shown.

    Shown only when the documents differ in nothing but their tracks (no scene-local code,
    whose modules may have changed): then every constant and expression is the same, and
    a track's value before its first changed segment starts is the same, because a track
    is evaluated from its base through its segments in order and stops at the first one
    that starts after ``t``. Expressions read values at the same ``t`` only, simulations
    run from parameters at ``t``, and life is a function of (id, seed, t), so the whole
    scene at ``t`` is the same (contract 3)."""
    return _until(old, new, {})


def _comparable(old: Compiled, new: Compiled) -> bool:
    a, b = old.doc, new.doc
    if a.meta.modules or b.meta.modules or old.missing or new.missing:
        return False
    return old.base_dir == new.base_dir and old.nodes.keys() == new.nodes.keys() and (
        old.time_map.markers == new.time_map.markers)  # fmt: skip


def _until(old: Compiled, new: Compiled, unseen: dict[str, float]) -> float:
    """``unchanged_until``, where a parameter of an ``unseen`` object (id -> a time before
    which the layout cannot see it) may also differ before that time."""
    if old is new:
        return math.inf
    if not _comparable(old, new):
        return 0.0
    if _without_tracks(old.doc, unseen) != _without_tracks(new.doc, unseen):
        return 0.0
    until = math.inf
    for path, x in new.nodes.items():
        y = old.nodes[path]
        if x is y:
            continue
        owner = path.rpartition(".")[0]
        if owner in unseen:  # nothing reads it, and its visibility is the same (checked)
            if not _same_node(x, y):
                until = min(until, unseen[owner])
            continue
        if not isinstance(x, ResolvedTrack) and not isinstance(y, ResolvedTrack):
            continue  # same document text: the same constant or expression
        tx, ty = (ResolvedTrack(n.value, ()) if isinstance(n, Const) else n for n in (x, y))
        if not isinstance(tx, ResolvedTrack) or not isinstance(ty, ResolvedTrack):
            return 0.0
        if not _same(tx.initial, ty.initial):
            return 0.0
        for sx, sy in itertools.zip_longest(tx.segments, ty.segments):
            if sx is None or sy is None or not _same_segment(sx, sy):
                until = min(until, *(s.t0 for s in (sx, sy) if s is not None))
                break
    return until


def _same_node(x, y) -> bool:
    """Whether two nodes surely give the same value at every t (False: maybe not)."""
    if isinstance(x, Const) and isinstance(y, Const):
        return _same(x.value, y.value)
    if isinstance(x, ResolvedTrack) and isinstance(y, ResolvedTrack):
        return _same(x.initial, y.initial) and len(x.segments) == len(y.segments) and all(
            _same_segment(a, b) for a, b in zip(x.segments, y.segments, strict=True))  # fmt: skip
    return False


VISIBILITY = ("opacity", "dim", "z_index")  # what decides whether, and in what order, it draws


def layout_unchanged_until(old: Compiled, new: Compiled, plan: Plan,
                           times: tuple[float, ...]) -> float:  # fmt: skip
    """A time before which the layout bake's keys (``times``) are the same for the two
    documents: ``unchanged_until``, or later when the edit also changed parameters of an
    object the layout cannot see before some key.

    The placer, at a key, reads the objects its rules name (their boxes, spots and
    depths), the occlusion buffer, and the other labels shown. An object reaches none of
    those while its opacity x dim is 0: the buffers draw only objects at least half
    opaque (``occludes``), other labels count only when shown, and its place in the draw
    order (which sets every object's depth bias) depends on its ``z_index`` and document
    index only. So a parameter of such an object may change, at any time, without
    changing a key at which it is hidden, as long as nothing else can see the change:

    - no rule names it, it is not the camera, and the scene has no composite (whose
      ``build`` reads the whole scene);
    - no other object's expression or updater reads its parameters, and no other object
      names it (a ring around it);
    - its opacity, dim and z_index are the same constants or tracks in both documents, and
      plain (not an expression, which may rise between keys); the rest of its entry
      (type, style, tags, layer, updaters) is the same.

    The keys before the first one at which it is shown in either document are then the
    same; the earliest such time over all edited objects (and motion edits) is returned.
    """
    if old is new:
        return math.inf
    if not _comparable(old, new) or plan.goals:
        return 0.0
    a = {o.id: o for o in old.doc.objects}
    b = {o.id: o for o in new.doc.objects}
    if a.keys() != b.keys():
        return 0.0
    edited = {i for i in b if a[i] != b[i]}  # its entry (params) or its tracks differ
    edited |= {p.rpartition(".")[0] for p, x in new.nodes.items()
               if (isinstance(x, ResolvedTrack) or isinstance(old.nodes[p], ResolvedTrack))
               and not _same_node(x, old.nodes[p])} & b.keys()  # fmt: skip
    unseen = {}
    for i in sorted(edited):
        h = _hidden_until(old, new, plan, a[i], b[i], times)
        if h > 0:
            unseen[i] = h
    return _until(old, new, unseen)


def _hidden_until(old: Compiled, new: Compiled, plan: Plan, spec_a, spec_b,
                  times: tuple[float, ...]) -> float:  # fmt: skip
    """The first of ``times`` at which the object is shown, when the layout cannot see it
    before (see ``layout_unchanged_until``); 0 when that cannot be shown."""
    i = spec_b.id
    named = {x for r in plan.rules for cl in r.clauses for x in cl.ids}
    if i in named or i == new.doc.timeline.camera or i == old.doc.timeline.camera:
        return 0.0
    ea, eb = (spec.model_dump(mode="json") for spec in (spec_a, spec_b))
    for e in (ea, eb):
        e["params"] = {k: v for k, v in e["params"].items() if k in VISIBILITY}
    if ea != eb:
        return 0.0
    for c in (old, new):
        if any(callable(getattr(o.impl, "build", None)) for o in c.objects):
            return 0.0  # a composite builds its parts from the whole scene
        for path, node in c.nodes.items():
            if path.rpartition(".")[0] != i and any(
                    d.rpartition(".")[0] == i for d in getattr(node, "deps", ())):  # fmt: skip
                return 0.0
        for o in c.doc.objects:
            if o.id != i and _names(o.params, i):
                return 0.0
        for path, segs in c.doc.timeline.tracks.items():
            if path.rpartition(".")[0] != i and any(_names(s.model_dump(), i) for s in segs):
                return 0.0
    nodes = []
    for name in VISIBILITY:
        x, y = new.nodes.get(f"{i}.{name}"), old.nodes.get(f"{i}.{name}")
        if (x is None) != (y is None):
            return 0.0
        if x is None:
            continue
        if not isinstance(x, Const | ResolvedTrack) or not _same_node(x, y):
            return 0.0
        if name != "z_index":
            nodes.append(x)

    def value(node, t: float) -> float:
        return float(node.value_at(t) if isinstance(node, ResolvedTrack) else node.value)

    for t in times:
        if math.prod(value(n, t) for n in nodes) != 0:  # shown (or NaN: maybe)
            return t
    return math.inf


def _names(value, object_id: str) -> bool:
    """Whether a parameter value (or anything inside it) is the object's id."""
    if isinstance(value, str):
        return value == object_id
    if isinstance(value, dict):
        return any(_names(v, object_id) for v in value.values())
    if isinstance(value, list | tuple):
        return any(_names(v, object_id) for v in value)
    return False


def _resume_point(prior, c: Compiled, plan: Plan, times, needed, kind: str):
    """(baked keys to keep, time to resume from) from the bake of an earlier document, or
    None when nothing of it can be kept."""
    old_c, old_plan, old = prior
    r = old.resume
    if r is None or old.occlusion != kind or plan.goals or old_plan != plan:
        return None
    until = layout_unchanged_until(old_c, c, plan, times)
    for k, (ta, tb) in enumerate(itertools.zip_longest(r.keyed, times)):
        if ta is None or tb is None or ta != tb or r.needed[k] != needed[k]:
            until = min(until, *(x for x in (ta, tb) if x is not None))
            break
    keep = bisect.bisect_left(old.times, until)
    return (keep, until) if keep else None


def bake(c: Compiled, plan: Plan, oracle: OcclusionOracle | None = None,
         prior: tuple[Compiled, Plan, Bake] | None = None,
         upto: float | None = None) -> Bake:  # fmt: skip
    """The layout of the compiled document. ``prior`` is (compiled, plan, bake) of an earlier
    document of the same scene: its keys before the first time the two could differ are
    kept as they are (``unchanged_until``), and the bake resumes from there.

    ``upto``: stop at the first baked key after that time. The scene at any time up to it
    is then exactly what the whole bake gives (each key depends only on the keys before
    it), and a later call resumes from it (pass it as ``prior``). Ignored with camera
    goals, which are solved over the whole timeline."""
    if plan.goals:
        upto = None
    if plan.empty:
        return Bake()
    base_oracle = oracle or AnalyticOracle()
    kind = "analytic" if isinstance(base_oracle, AnalyticOracle) else "pixel"
    oracle = _Counted(base_oracle)
    lay = c.theme.layout
    times = key_times(c)
    cam_issues: tuple[Issue, ...] = ()
    solved_camera: dict[str, tuple] = {}
    if plan.goals:
        from phasor.layout import camera as camera_solver

        watched = sorted({i for _, g in plan.goals if g.kind == "avoid_occlusion" for i in g.ids})
        sol = camera_solver.solve(c, plan, times, lambda st, i: hidden_share(st, i, oracle),
                                  {i: shown_times(c, [i], times) for i in watched})  # fmt: skip
        solved_camera, cam_issues = sol.values, sol.issues
    camera_keys = {k: {p: v[k] for p, v in solved_camera.items()} for k in range(len(times))}
    placed_ids = frozenset(plan.placed)
    pos = {i: f"{i}.{MOVE_PARAM}" for i in plan.moving}
    needed = shown_keys(c, plan, times)

    # Which object each moving object follows between key times.
    ref: dict[str, str | None] = dict.fromkeys(plan.moving)
    for _, rel in plan.relations():
        if ref.get(rel.subject) is None:
            ref[rel.subject] = rel.target

    keys: dict[str, list] = {p: [] for p in pos.values()}
    for i in sorted(plan.leaders):
        keys[f"{i}.leader"], keys[f"{i}.leader_on"] = [], []
    failures: list[tuple[int, str, tuple[str, ...], float]] = []
    slots: dict[str, list[str]] = {i: [] for i in plan.placed}
    previous: dict[str, labels.Previous] = {}
    baked: list[float] = []
    history: list[dict] = []
    renders: list[int] = []
    start, done = 0, 0  # first key index to run; renders of the kept keys

    point = _resume_point(prior, c, plan, times, needed, kind) if prior else None
    if point is not None:
        keep, until = point
        old = prior[2]
        r = old.resume
        baked, history, renders = list(old.times[:keep]), list(r.previous[:keep]), list(
            r.renders[:keep])  # fmt: skip
        for tr in old.tracks:
            keys[tr.path] = list(tr.keys[:keep])
        for i in slots:
            slots[i] = list(old.slots[i][:keep])
        failures = [f for f in r.failures if f[3] < until]
        previous, done = dict(history[-1]), renders[-1]
        start = bisect.bisect_left(times, until)

    def fail(rule, ids, message, t):
        failures.append((rule.index, message, tuple(ids), t))

    stop = len(times)
    for k_time in range(start, len(times)):
        if not needed[k_time]:
            continue
        t = times[k_time]
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
        baked.append(t)
        history.append(dict(previous))
        renders.append(done + oracle.n)
        if upto is not None and t > upto:
            stop = k_time + 1
            break

    def jumps(object_id: str) -> tuple[bool, ...]:
        chosen = slots.get(object_id, ())
        return tuple(a != b for a, b in itertools.pairwise(chosen))

    tracks = tuple(
        Track(path, "scalar" if path.endswith(".leader_on") else "point",
              None if path.endswith(".leader_on") else ref[path.rpartition(".")[0]], tuple(ks),
              jumps(path.rpartition(".")[0]))
        for path, ks in sorted(keys.items())
    )  # fmt: skip
    grouped: dict[tuple[int, str], tuple[tuple[str, ...], list[float]]] = {}
    for index, message, ids, t in failures:
        grouped.setdefault((index, message), (ids, []))[1].append(t)
    issues = []
    for (index, message), (ids, ts) in sorted(grouped.items()):
        rule = plan.rules[index]
        shown = ", ".join(f"{x:g}" for x in ts[:MAX_TIMES_IN_ISSUE])
        more = f" and {len(ts) - MAX_TIMES_IN_ISSUE} more" if len(ts) > MAX_TIMES_IN_ISSUE else ""
        issues.append(Issue(
            "unsolved", f"constraint {rule.spec.rule!r} {message} (at t = {shown}{more} s)",
            ids=ids, path=rule.path, times=tuple(ts),
            fix_hint="lower its priority, loosen it, or make room (move or shrink a neighbor)",
        ))  # fmt: skip
    times, needed = times[:stop], needed[:stop]
    spans = []
    for k, t in enumerate(times):
        if needed[k]:
            if k and needed[k - 1]:
                spans[-1] = (spans[-1][0], t)
            else:
                spans.append((t, t))
    resume = Resume(times, needed, tuple(history), tuple(renders), tuple(failures))
    return Bake(tuple(baked), tracks, (*cam_issues, *issues),
                slots={k: tuple(v) for k, v in slots.items()}, occlusion=kind,
                camera=solved_camera, camera_times=times if solved_camera else (),
                renders=done + oracle.n, spans=tuple(spans), resume=resume)  # fmt: skip


def hidden_share(state: SceneState, object_id: str, oracle: OcclusionOracle) -> float:
    """Share of an object hidden by what is in front of it (the camera solver's stage 2):
    pixels in its projected bounding disc (half radius) whose nearest surface, leaving out
    the object itself and all labels, is clearly nearer than the object.

    No buffer is rendered when the answer is known without one: the object is behind the
    camera (1), or nothing else in the scene occludes at all (0: the buffer, which holds
    only occluding objects, would be empty)."""
    obj = state[object_id]
    cam = state.camera_for(obj)
    b = obj.impl.world_bounds(state)
    d = float(cam.depth(b.center)[0])
    if d <= 0:  # behind the camera: no buffer needed to say so
        return 1.0
    labels_ = {o.id for o in state.objects if o.impl.semantics(state).get("kind") == "label"}
    skip = frozenset({object_id} | labels_)
    if not any(occludes(o) for o in state.objects if o.id not in skip):
        return 0.0  # nothing left that hides anything: the buffer would be empty (inf)
    depth = oracle.depth(state, skip)
    h, w = depth.shape
    ((px, py),) = cam.to_screen([b.center], w, h)
    r = max(0.5 * b.radius / (cam.units_per_height(b.center)) * h, 1.5)
    disc = pixel_mask((h, w), (px - r, px + r, py - r, py + r),
                      lambda xx, yy: (xx - px) ** 2 + (yy - py) ** 2 <= r * r)  # fmt: skip
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


__all__ = ["AnalyticOracle", "Bake", "Box", "OcclusionOracle", "Resume", "Track", "bake",
           "baked_state", "key_times", "layout_unchanged_until", "shown_keys", "shown_times",
           "unchanged_until"]  # fmt: skip
