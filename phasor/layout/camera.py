"""The camera solver: goals in, camera tracks out.

The document states goals (``camera keep_visible(surf, bp)``, ``center``, ``thirds``,
``avoid_occlusion``, ``min_size``); the solver chooses the camera parameters the document
leaves unset (``plan.camera_free``: azimuth, elevation, distance, target). Anything the
author set stays as written and the solver works around it. A parameter that is animated
but never set (``plan.camera_handoff``: a ``focus_pull`` or ``orbit`` at a marker) is the
solver's until its first motion starts; that motion then starts from the solved value (an
``orbit`` turns by its angle from there; anything else arrives where it was written).

- **Stage 1**: an analytic cost from projected bounding spheres, minimized with
  ``scipy.optimize`` (L-BFGS-B) from several starts at the first key and warm-started from
  the previous solution after that, with a ``smooth`` term that penalizes change.
- **Stage 2**: verify ``avoid_occlusion`` with the occlusion oracle (the real depth buffer
  when OpenGL is there) and re-solve among nearby cameras when the object is hidden.
- Solved at markers, the ends of transitions and a 1 Hz grid, joined with a monotone
  cubic (PCHIP, so it never overshoots), sampled into the bake's key times, and checked
  again at every one of them. What still fails becomes an ``unsolved`` issue with times.

Pure: depends on (compiled document, plan, oracle), like the rest of the bake.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

from phasor.core.constraints import Goal
from phasor.core.issues import Issue
from phasor.core.spatial import Camera, OrbitCameraParams, perspective_camera
from phasor.core.state import SceneState
from phasor.layout.geometry import is_backdrop, occludes
from phasor.layout.plan import Plan, Rule
from phasor.time.compile import Compiled, evaluate, sway
from phasor.time.tracks import ResolvedTrack

SOLVE_HZ = 1.0
PRIORITY = {"required": 10.0, "strong": 1.0, "medium": 0.3, "weak": 0.1}
W_VISIBLE = 100.0  # per (NDC overshoot)^2 past the margin
W_FILL = 4.0
W_GROUP_CENTER = 0.5
W_CENTER = 4.0
W_THIRDS = 4.0
W_MIN_SIZE = 50.0
W_OCCLUDE = 20.0
W_HIDDEN = 50.0  # stage 2: per (hidden share)^2 measured by the oracle
W_PRIOR = 0.05  # pull toward the camera's default view where the goals do not care
W_SPEED = 0.3  # smoothness also dislikes speed a little (per scaled unit per solve key)
HIDDEN_OK = 0.25  # stage 2 re-solves when more than this share of an object is hidden; 0.5
# left no margin under the event log's "occluded" (half covered), which measures the whole
# object rather than its middle (riemann_sqrt without a speech engine, 2026-11-21)
ELEVATION = (-80.0, 85.0)
MAX_TIMES_IN_ISSUE = 12


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius: float
    corners: np.ndarray  # (8, 3): the world bounds' corners, for framing


def _spheres(state: SceneState, ids) -> dict[str, Sphere]:
    out = {}
    for i in ids:
        b = state[i].impl.world_bounds(state)
        out[i] = Sphere(np.asarray(b.center, dtype=np.float64), float(max(b.radius, 1e-6)),
                        np.asarray(b.corners(), dtype=np.float64))  # fmt: skip
    return out


NEAR_SHARE = 0.1  # points nearer than this share of the camera distance count as behind it


def _cam_xy(cam: Camera, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """NDC x and y of points, with depth clamped to stay in front of the camera, plus a
    graded penalty for every point that is too near or behind. Clamping (not refusing)
    keeps the cost smooth, so the optimizer can always find its way back."""
    rel = np.atleast_2d(pts) - cam.eye
    d = rel @ cam.forward
    dmin = NEAR_SHARE * cam.height
    dc = np.maximum(d, dmin)
    t = cam.tan_half_fov
    p00 = float(np.linalg.norm(cam.view_proj[0, :3]))  # 1 / (tan * aspect)
    x = (rel @ cam.right) * p00 / dc
    y = (rel @ cam.up) / (dc * t)
    behind = float((np.maximum(0.0, dmin - d) / cam.height).__pow__(2).sum())
    return x, y, behind


def _extent(cam: Camera, s: Sphere) -> tuple[tuple[float, float, float, float], float]:
    """NDC box (x0, x1, y0, y1) of an object's bounds as this camera sees them, and a
    penalty for any of it behind the camera. Framing uses this: a flat surface seen
    edge-on is thin, which its bounding sphere would never tell."""
    x, y, behind = _cam_xy(cam, s.corners)
    return (float(x.min()), float(x.max()), float(y.min()), float(y.max())), behind


class _Vars:
    """The free parameters as one scaled vector (so every unknown moves about 1 per step)."""

    def __init__(self, free: tuple[str, ...], center: np.ndarray, radius: float,
                 elevation: tuple[float, float] = ELEVATION) -> None:  # fmt: skip
        self.free, self.center, self.radius = free, center, max(radius, 1e-3)
        self.elevation = elevation
        self.names: list[str] = []
        for p in free:
            self.names += ["tx", "ty", "tz"] if p == "target" else [p]

    def encode(self, p: OrbitCameraParams) -> np.ndarray:
        out = []
        for n in self.names:
            if n == "azimuth":
                out.append(p.azimuth / 45.0)
            elif n == "elevation":
                out.append(p.elevation / 45.0)
            elif n == "distance":
                out.append(math.log(p.distance / self.radius))
            else:
                k = "xyz".index(n[1])
                out.append((p.target[k] - self.center[k]) / self.radius)
        return np.array(out)

    def decode(self, x: np.ndarray, base: OrbitCameraParams) -> OrbitCameraParams:
        upd: dict = {}
        target = list(base.target)
        for n, v in zip(self.names, x, strict=True):
            if n == "azimuth":
                upd["azimuth"] = float(v * 45.0)
            elif n == "elevation":
                upd["elevation"] = float(np.clip(v * 45.0, *self.elevation))
            elif n == "distance":
                upd["distance"] = float(self.radius * math.exp(v))
            else:
                target["xyz".index(n[1])] = float(self.center["xyz".index(n[1])] + v * self.radius)
        if "target" in self.free:
            upd["target"] = tuple(target)
        return OrbitCameraParams(**{**base.model_dump(), **upd})

    def bounds(self) -> list[tuple[float | None, float | None]]:
        out: list[tuple[float | None, float | None]] = []
        for n in self.names:
            if n == "azimuth":
                out.append((None, None))
            elif n == "elevation":
                out.append((self.elevation[0] / 45.0, self.elevation[1] / 45.0))
            elif n == "distance":
                out.append((math.log(0.3), math.log(40.0)))
            else:
                out.append((-1.5, 1.5))
        return out


def _project(cam: Camera, s: Sphere) -> tuple[float, float, float, float] | None:
    """(ndc x, ndc y, radius in ndc y, depth) of a sphere, or None if behind the camera."""
    depth = float(cam.depth(s.center)[0])
    if depth <= s.radius * 0.5 or depth <= 1e-6:
        return None
    ndc = cam.project([s.center])[0]
    return float(ndc[0]), float(ndc[1]), s.radius / (depth * cam.tan_half_fov), depth


class Problem:
    """The cost of a camera at one time: every goal, weighted by its rule's priority."""

    def __init__(self, c: Compiled, plan: Plan, state: SceneState) -> None:
        self.c, self.plan, self.state = c, plan, state
        self.aspect = state.aspect
        th = c.theme.camera
        self.fill, self.margin, self.thirds_w = th["fill"], th["margin"], th["thirds"]
        self.smooth_w = th["smooth"]
        # Overlay text (a title, a caption) owns its strip of the frame: framing keeps the
        # world out of the top and bottom bands it occupies.
        self.top, self.bottom = 1.0, -1.0
        for o in state.objects:
            if o.layer != "overlay" or o.params.opacity <= 0:
                continue
            ndc = state.overlay.project(o.impl.world_bounds(state).corners())
            y0, y1 = float(ndc[:, 1].min()), float(ndc[:, 1].max())
            if y0 > 0.3:
                self.top = min(self.top, y0)
            elif y1 < -0.3:
                self.bottom = max(self.bottom, y1)
        ids = sorted({i for _, g in plan.goals for i in g.ids})
        self.spheres = _spheres(state, ids)
        self.kept = [i for _, g in plan.goals if g.kind == "keep_visible" for i in g.ids]
        occluders = [o.id for o in state.objects if o.layer != "overlay"
                     and occludes(o) and not is_backdrop(state, o)
                     and o.impl.semantics(state).get("kind") != "label"]  # fmt: skip
        self.occluders = _spheres(state, occluders)
        self.t = state.t

    def camera(self, p: OrbitCameraParams) -> Camera:
        return perspective_camera(p, self.aspect, sway(self.c.theme, p, self.t))

    def goal_cost(self, cam: Camera, rule: Rule, g: Goal) -> float:
        w = PRIORITY[rule.spec.priority]
        lx = 1.0 - 2.0 * self.margin / self.aspect
        top, bottom = self.top - 2.0 * self.margin, self.bottom + 2.0 * self.margin
        cost = 0.0
        if g.kind == "keep_visible":
            for i in g.ids:
                (x0, x1, y0, y1), behind = _extent(cam, self.spheres[i])
                cost += W_VISIBLE * 10.0 * behind
                ex = max(0.0, -lx - x0) + max(0.0, x1 - lx)
                ey = max(0.0, bottom - y0) + max(0.0, y1 - top)
                cost += W_VISIBLE * (ex * ex + ey * ey)
        elif g.kind == "center":
            x, y, behind = (float(v) if np.ndim(v) == 0 else float(v[0])
                            for v in _cam_xy(cam, self.spheres[g.ids[0]].center))  # fmt: skip
            cost += W_CENTER * (x * x + y * y) + W_VISIBLE * 10.0 * behind
        elif g.kind == "thirds":
            x, y, _ = (float(v) if np.ndim(v) == 0 else float(v[0])
                       for v in _cam_xy(cam, self.spheres[g.ids[0]].center))  # fmt: skip
            cost += W_THIRDS * min((x - a) ** 2 + (y - b) ** 2
                                   for a in (-1 / 3, 1 / 3) for b in (-1 / 3, 1 / 3))  # fmt: skip
        elif g.kind == "min_size":
            box, behind = _extent(cam, self.spheres[g.ids[0]])
            cost += W_VISIBLE * 10.0 * behind
            cost += W_MIN_SIZE * max(0.0, g.value - _size(box, self.aspect)) ** 2
        elif g.kind == "avoid_occlusion":
            for i in g.ids:
                cost += W_OCCLUDE * self._blocked(cam, i)
        return w * cost

    def _blocked(self, cam: Camera, i: str) -> float:
        """Bounding spheres between the eye and ``i`` (those that contain ``i`` are left to
        stage 2: a sphere around a whole surface says nothing about its holes)."""
        s = self.spheres[i]
        eye = np.asarray(cam.eye)
        seg = s.center - eye
        length = float(np.linalg.norm(seg))
        total = 0.0
        for j, o in self.occluders.items():
            if j == i or np.linalg.norm(o.center - s.center) < o.radius:
                continue
            u = float(np.clip((o.center - eye) @ seg / (length * length), 0.0, 1.0))
            if u >= 1.0:
                continue
            d = float(np.linalg.norm(eye + u * seg - o.center))
            over = max(0.0, o.radius + 0.5 * s.radius - d) / (o.radius + s.radius)
            total += over * over
        return total

    def group_cost(self, cam: Camera) -> float:
        """Frame the kept objects: about ``fill`` of the frame, roughly centered."""
        if not self.kept:
            return 0.0
        found = [_extent(cam, self.spheres[i]) for i in self.kept]
        boxes = [b for b, _ in found]
        penalty = W_VISIBLE * 10.0 * sum(p for _, p in found)
        x0, x1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
        y0, y1 = min(b[2] for b in boxes), max(b[3] for b in boxes)
        size = max((y1 - y0) / (self.top - self.bottom), (x1 - x0) / 2.0)
        gx, gy = (x0 + x1) / 2.0, (y0 + y1) / 2.0 - (self.top + self.bottom) / 2.0
        return penalty + W_FILL * (size - self.fill) ** 2 + W_GROUP_CENTER * (gx * gx + gy * gy)

    def cost(self, p: OrbitCameraParams) -> float:
        cam = self.camera(p)
        return self.group_cost(cam) + sum(self.goal_cost(cam, r, g) for r, g in self.plan.goals)


class _Handoff:
    """Parameters the solver owns until their first motion (``plan.camera_handoff``).

    Before the motion starts they are solved like free ones; from then on they follow the
    document's track, started from the solved value instead of the default: an ``orbit``
    keeps the offset (it turns by its angle), anything else eases it away over the first
    motion, with that motion's own easing.
    """

    def __init__(self, c: Compiled, plan: Plan) -> None:
        cam = c.doc.timeline.camera
        self.tracks: dict[str, tuple[ResolvedTrack, bool]] = {}
        for n in plan.camera_handoff:
            node = c.nodes[f"{cam}.{n}"]
            first = c.doc.timeline.tracks[f"{cam}.{n}"][0]
            self.tracks[n] = (node, getattr(first, "verb", None) == "orbit")
        self.start: dict[str, object] = {}

    def reached(self, t: float, solved: Callable[[], OrbitCameraParams]) -> None:
        """Record the solved values of parameters whose motion starts at ``t``."""
        todo = [n for n, (tr, _) in self.tracks.items()
                if n not in self.start and t >= tr.segments[0].t0 - 1e-9]  # fmt: skip
        if todo:
            p = solved()
            for n in todo:
                self.start[n] = getattr(p, n)

    def pin(self, p: OrbitCameraParams, t: float) -> OrbitCameraParams:
        upd: dict = {}
        for n, (tr, keep) in self.tracks.items():
            s = tr.segments[0]
            if t < s.t0 - 1e-9 or n not in self.start:
                continue
            start = np.asarray(self.start[n], dtype=np.float64)
            init = np.asarray(tr.initial, dtype=np.float64)
            w = (
                1.0
                if keep or t <= s.t0
                else 1.0 - (s.ease(min((t - s.t0) / max(s.t1 - s.t0, 1e-9), 1.0)))
            )
            v = np.asarray(tr.value_at(t), dtype=np.float64) + (start - init) * w
            upd[n] = tuple(float(a) for a in v) if v.ndim else float(v)
        return OrbitCameraParams(**{**p.model_dump(), **upd}) if upd else p


def _base(c: Compiled, state: SceneState) -> OrbitCameraParams:
    cam = c.doc.timeline.camera
    return OrbitCameraParams(**{n: state.values[f"{cam}.{n}"]
                                for n in OrbitCameraParams.model_fields})  # fmt: skip


def _defaults() -> dict:
    return {n: f.default for n, f in OrbitCameraParams.model_fields.items()}


def solve_times(c: Compiled) -> list[float]:
    end = c.duration
    ts = {0.0, end, *c.time_map.markers.values()}
    for node in c.nodes.values():
        if isinstance(node, ResolvedTrack):
            for s in node.segments:
                ts.update((s.t0, s.t1))
    n = math.floor(end * SOLVE_HZ + 1e-9)
    ts.update(k / SOLVE_HZ for k in range(n + 1))
    return sorted(round(t, 9) for t in ts if 0.0 <= t <= end)


def _starts(x: np.ndarray, names: list[str]) -> list[np.ndarray]:
    out = [x]
    for n, deltas in (("elevation", (0.5, -0.4)), ("azimuth", (1.0, -1.0)),
                      ("distance", (0.5, -0.4))):  # fmt: skip
        if n in names:
            k = names.index(n)
            for dv in deltas:
                y = x.copy()
                y[k] += dv
                out.append(y)
    return out


@dataclass(frozen=True)
class Solution:
    times: tuple[float, ...]
    values: dict[str, tuple]  # "camera.<param>" -> value per time
    issues: tuple[Issue, ...]


REFINE_ROUNDS = 6  # 3 was too few for riemann_sqrt with estimated narration timing (2026-11-21)


def solve(
    c: Compiled,
    plan: Plan,
    times: tuple[float, ...],
    hidden: Callable[[SceneState, str], float] | None = None,
) -> Solution:
    """Camera values for the free parameters at every one of ``times`` (the bake's keys).

    ``hidden(state, id)`` is the stage 2 check: the share of ``id`` hidden by what is in
    front of it, measured on a real (or analytic) depth buffer.
    """
    cam_id = c.doc.timeline.camera
    keys = solve_times(c)
    occl = [(r, g) for r, g in plan.goals if g.kind == "avoid_occlusion"]
    # The goals' objects over the whole timeline set the scale of the search (where the
    # target may go, how far the camera may be), so a moving object can be followed.
    allsph = [sp for t in keys for sp in Problem(c, plan, evaluate(c, t)).spheres.values()]
    center = np.mean([sp.center for sp in allsph], axis=0)
    radius = max(float(np.linalg.norm(sp.center - center)) + sp.radius for sp in allsph)
    th = c.theme.camera
    chosen = plan.camera_free + plan.camera_handoff
    vars_ = _Vars(chosen, center, radius, (th["min_elevation"], th["max_elevation"]))
    hand = _Handoff(c, plan)
    prior = vars_.encode(OrbitCameraParams(**{**_defaults(), "target": tuple(vars_.center)}))
    angles = np.array([n in ("azimuth", "elevation") for n in vars_.names], dtype=float)

    def solve_at(t: float, near: np.ndarray | None, last: np.ndarray | None = None) -> np.ndarray:
        """Best camera at ``t``; ``near`` (the neighbor's solution) anchors smoothness."""
        pre = evaluate(c, t)
        prob = Problem(c, plan, pre)
        base = _base(c, pre)
        x0 = vars_.encode(base) if near is None else near
        if not vars_.names:  # nothing to choose: the goals are only checked
            return x0

        def f(x):
            cost = prob.cost(hand.pin(vars_.decode(x, base), t)) + W_PRIOR * float(
                ((x - prior) ** 2).sum()
            )
            if near is not None:
                cost += prob.smooth_w * float(((x - near) ** 2).sum())
            if last is not None:  # angles only: the target may follow a moving object freely
                cost += prob.smooth_w * W_SPEED * float((((x - last) * angles) ** 2).sum())
            return cost

        starts = _starts(x0, vars_.names) if near is None else [x0]
        best = min((minimize(f, s0, method="L-BFGS-B", bounds=vars_.bounds()) for s0 in starts),
                   key=lambda r: r.fun)  # fmt: skip
        x = best.x
        if near is not None and _check(c, plan, t, hand.pin(vars_.decode(x, base), t), None):
            # Warm start was not enough (something moved fast): search from several starts.
            again = min((minimize(f, s0, method="L-BFGS-B", bounds=vars_.bounds())
                         for s0 in _starts(x, vars_.names)), key=lambda r: r.fun)  # fmt: skip
            if again.fun < best.fun:
                x = again.x
        if hidden is not None and occl:
            x = _stage2(c, f, x, vars_, hand.pin, base, occl, t, hidden)
        return x

    # Smoothness penalizes jerks, not motion: each key is pulled toward the straight-line
    # continuation of the two before it, so following a moving object costs nothing.
    solved: dict[float, np.ndarray] = {}
    hist: list[tuple[float, np.ndarray]] = []
    for t in keys:
        near = None
        if len(hist) == 1:
            near = hist[-1][1]
        elif len(hist) >= 2:
            (ta, xa), (tb, xb) = hist[-2], hist[-1]
            near = xb + (xb - xa) * ((t - tb) / (tb - ta)) if tb > ta else xb
        solved[t] = solve_at(t, near, hist[-1][1] if hist else None)
        hist.append((t, solved[t]))
        hand.reached(t, lambda t=t: vars_.decode(solved[t], _base(c, evaluate(c, t))))

    def curve() -> np.ndarray:
        ks = sorted(solved)
        if len(ks) == 1:
            return np.repeat(solved[ks[0]][None, :], len(times), axis=0)
        return PchipInterpolator(np.array(ks), np.array([solved[k] for k in ks]), axis=0)(
            np.array(times))  # fmt: skip

    # Verify at every key time; where a goal fails between solves (fast motion, something
    # passing in front), solve there too and join again.
    at = curve()
    for _ in range(REFINE_ROUNDS):
        bad = [
            t
            for t, x in zip(times, at, strict=True)
            if t not in solved
            and _check(c, plan, t, hand.pin(vars_.decode(x, _base(c, evaluate(c, t))), t), hidden)
        ]
        if not bad or not vars_.names:
            break
        for t, x in zip(times, at, strict=True):
            if t in bad:
                solved[t] = solve_at(t, x)
        at = curve()

    values: dict[str, list] = {f"{cam_id}.{p}": [] for p in chosen}
    failures: dict[tuple[int, str], tuple[tuple[str, ...], list[float]]] = {}
    for t, x in zip(times, at, strict=True):
        p = hand.pin(vars_.decode(x, _base(c, evaluate(c, t))), t)
        for name in chosen:
            v = getattr(p, name)
            values[f"{cam_id}.{name}"].append(tuple(v) if isinstance(v, tuple) else float(v))
        for index, why, ids in _check(c, plan, t, p, hidden):
            failures.setdefault((index, why), (ids, []))[1].append(t)
    issues = []
    for (index, why), (ids, ts) in sorted(failures.items()):
        rule = plan.rules[index]
        shown = ", ".join(f"{x:g}" for x in ts[:MAX_TIMES_IN_ISSUE])
        more = f" and {len(ts) - MAX_TIMES_IN_ISSUE} more" if len(ts) > MAX_TIMES_IN_ISSUE else ""
        issues.append(Issue(
            "unsolved", f"camera goal {rule.spec.rule!r}: {why} (at t = {shown}{more} s)",
            ids=ids, path=rule.path, times=tuple(ts),
            fix_hint="free another camera parameter (leave it unset), lower the goal's "
            "priority, or move the objects",
        ))  # fmt: skip
    return Solution(tuple(times), {k: tuple(v) for k, v in values.items()}, tuple(issues))


def _stage2(c, f, x, vars_, pin, base, occl, t, hidden) -> np.ndarray:
    """Check avoid_occlusion on the depth buffer; if hidden, try nearby cameras."""
    cam_id = c.doc.timeline.camera

    def shares(y) -> list[tuple[float, float]]:
        """(priority weight, hidden share) of every object an avoid_occlusion goal names."""
        p = pin(vars_.decode(y, base), t)
        state = evaluate(c, t, {f"{cam_id}.{n}": getattr(p, n) for n in vars_.free})
        return [(PRIORITY[r.spec.priority], hidden(state, i)) for r, g in occl for i in g.ids]

    def clear(sh) -> bool:  # the same test as ``_check``: no object hidden past HIDDEN_OK
        return all(h <= HIDDEN_OK for _, h in sh)

    def weighted(sh) -> float:
        return sum(w * h**2 for w, h in sh)

    now = shares(x)
    if clear(now):
        return x
    tried = [(f(y), shares(y), y) for y in _nearby(x, vars_.names)]
    # A nearby camera that clears the view wins outright, the smoothest of them. Found
    # 2026-11-21 (riemann_sqrt with estimated narration timing): the "anything hidden?" test
    # used priority-weighted shares, so a medium goal 86% hidden (0.3 * 0.86^2 < 0.5^2)
    # looked fine, a weak goal never triggered at all, and _check then reported it unsolved.
    ok = [(cost, y) for cost, sh, y in tried if clear(sh)]
    if ok:
        return min(ok, key=lambda cy: cy[0])[1]
    best, best_cost = x, f(x) + W_HIDDEN * weighted(now)
    for cost, sh, y in tried:
        if cost + W_HIDDEN * weighted(sh) < best_cost:
            best, best_cost = y, cost + W_HIDDEN * weighted(sh)
    return best


def _nearby(x: np.ndarray, names: list[str]) -> list[np.ndarray]:
    out = []
    for n, deltas in (("elevation", (0.15, 0.3, 0.45, -0.15, -0.3)),
                      ("azimuth", (0.3, 0.6, -0.3, -0.6))):  # fmt: skip
        if n in names:
            k = names.index(n)
            for dv in deltas:
                y = x.copy()
                y[k] += dv
                out.append(y)
    return out


def _check(c, plan, t, p, hidden) -> list[tuple[int, str, tuple[str, ...]]]:
    """Which goals fail at ``t`` with camera ``p``: (rule index, why, ids). Occlusion is
    checked only when ``hidden`` is given."""
    pre = evaluate(c, t)
    prob = Problem(c, plan, pre)
    cam = prob.camera(p)
    lx = 1.0 - 2.0 * prob.margin / prob.aspect
    top, bottom = prob.top - 2.0 * prob.margin, prob.bottom + 2.0 * prob.margin
    cam_id = c.doc.timeline.camera
    state = None
    out = []
    for rule, g in plan.goals:
        bad: list[str] = []
        why = ""
        for i in g.ids:
            box, behind = _extent(cam, prob.spheres[i])
            if g.kind == "keep_visible":
                if behind > 0 or box[0] < -lx - 0.02 or box[1] > lx + 0.02 or \
                        box[2] < bottom - 0.02 or box[3] > top + 0.02:  # fmt: skip
                    bad.append(i)
                    why = "cannot keep it all in frame"
            elif g.kind == "min_size" and (behind > 0 or _size(box, prob.aspect) < g.value * 0.9):
                bad.append(i)
                why = f"it is smaller than {g.value:g} of the frame"
            elif g.kind == "avoid_occlusion" and hidden is not None:
                if state is None:
                    state = evaluate(
                        c,
                        t,
                        {
                            f"{cam_id}.{n}": getattr(p, n)
                            for n in plan.camera_free + plan.camera_handoff
                        },
                    )
                if hidden(state, i) > HIDDEN_OK:
                    bad.append(i)
                    why = "it is hidden behind something in front of it"
        if bad:
            out.append((rule.index, why, tuple(bad)))
    return out


def _size(box, aspect: float) -> float:
    """How big an NDC box is, as a share of the frame height (its larger side)."""
    return max((box[3] - box[2]) / 2.0, (box[1] - box[0]) / 2.0 * aspect)
