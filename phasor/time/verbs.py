"""Choreography verbs: animate by intent, timed by the theme.

    s.at("poles_appear").reveal("pole_a", "pole_b")       # fade in, staggered
    s.at("loop_draw").trace("loop").then().reveal("loop.label")
    s.at("residue_glow").dim_others("loop", "loop.label")
    s.at("zoom").focus_pull("loop")

A verb expands into ordinary segments (and, where something must start hidden, a base
value), each tagged with the verb's name. The document stays plain data, and lint can tell
choreographed motion from raw ``to()`` transitions (``raw_motion`` in strict mode).

Pure: verbs read a compiled document and the scene at the anchor time, and return patch ops.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from phasor.core import unfold_math
from phasor.core.schema import ExprSpec, MarkerRef, TimeRef
from phasor.core.state import SceneState
from phasor.time.compile import Compiled, evaluate
from phasor.time.easing import LOG
from phasor.time.tracks import ResolvedTrack

FOCUS_MARGIN = 0.15  # focus_pull leaves this share of the frame around the object
ZOOM_PARAMS = ("height", "distance")  # camera scales: a verb moves them by factors (log:)


class VerbError(ValueError):
    pass


@dataclass(frozen=True)
class Motion:
    ops: tuple[dict[str, Any], ...]
    end: float  # seconds after the anchor when the last transition ends


def shift(at: TimeRef, dt: float) -> TimeRef:
    if isinstance(at, MarkerRef):
        return MarkerRef(marker=at.marker, offset=round(at.offset + dt, 6))
    return round(float(at) + dt, 6)


def _ref(at: TimeRef) -> Any:
    return at.model_dump() if isinstance(at, MarkerRef) else at


class Verbs:
    """Verb expansion for one anchor time. ``state`` is the scene at that time."""

    def __init__(self, compiled: Compiled, state: SceneState, at: TimeRef) -> None:
        self.c, self.state, self.at = compiled, state, at
        self.t = compiled.time_map.resolve(at)
        self.motion = compiled.theme.motion

    # ---------------------------------------------------------- helpers

    def _object(self, object_id: str):
        if object_id not in self.state:
            known = ", ".join(sorted(o.id for o in self.state.objects))
            raise VerbError(f"no object {object_id!r}; objects are {known}")
        return self.state[object_id]

    def _has(self, object_id: str, param: str) -> bool:
        return param in type(self._object(object_id).params).model_fields

    def _base(self, path: str) -> Any:
        target, _, name = path.rpartition(".")
        spec = self.c.doc.object(target) or self.c.doc.camera(target)
        value = spec.params.get(name) if spec else None
        if isinstance(value, ExprSpec):
            raise VerbError(f"{path} is an expression; animate what it reads instead")
        return value

    def _written(self, path: str) -> float:
        """The document's own value of ``path`` at the anchor, without the bake: a turn is
        written relative to it, so a camera the solver aims turns by the same angle."""
        return float(evaluate(self.c, self.t).values[path])

    def _starts_hidden(self, path: str) -> bool:
        """True when nothing animates ``path`` before the anchor, so a verb may set the
        base value (what the parameter is before its first transition)."""
        node = self.c.nodes.get(path)
        segs = node.segments if isinstance(node, ResolvedTrack) else ()
        return not any(s.t0 < self.t - 1e-9 for s in segs)

    def _segment(self, path: str, to: Any, verb: str, delay: float, timing: str | None = None):
        spec = self.motion[timing or verb]
        t0 = shift(self.at, delay)
        t1 = shift(self.at, delay + spec["duration"])
        seg = {"t0": _ref(t0), "t1": _ref(t1), "to": to, "ease": self._ease(path, spec["ease"]),
               "verb": verb}  # fmt: skip
        return {"op": "add_segment", "path": path, "segment": seg}, delay + spec["duration"]

    def _ease(self, path: str, ease: str) -> str:
        """A camera's zoom (``height``, ``distance``) moves by factors: ``log:`` + the curve,
        so a 600x dive looks even. Everything else eases as the theme says."""
        target, _, name = path.rpartition(".")
        if name in ZOOM_PARAMS and self.c.doc.camera(target) and not ease.startswith(LOG):
            return LOG + ease
        return ease

    def _each(self, ids: Sequence[str], verb: str, make) -> Motion:
        if not ids:
            raise VerbError(f"{verb} needs at least one object id")
        ops: list[dict[str, Any]] = []
        end = 0.0
        for k, object_id in enumerate(ids):
            self._object(object_id)
            more, done = make(object_id, k * self.motion["stagger"])
            ops += more
            end = max(end, done)
        return Motion(tuple(ops), end)

    # ---------------------------------------------------------- verbs

    def _undo_dimming_before(self, object_id: str, t: float) -> list[dict[str, Any]]:
        """A reveal is a new focus: dimming applied to the object while it was still hidden
        (an earlier ``dim_others``) is dropped, so it arrives at full strength."""
        path = f"{object_id}.dim"
        segs = self.c.doc.timeline.tracks.get(path, ())
        keep = [s for s in segs if self.c.time_map.resolve(s.t0) >= t - 1e-9]
        if len(keep) == len(segs):
            return []
        ops: list[dict[str, Any]] = [{"op": "clear_track", "path": path}]
        ops += [{"op": "add_segment", "path": path, "segment": s.model_dump(mode="json")}
                for s in keep]  # fmt: skip
        return ops

    def reveal(self, *ids: str, verb: str = "reveal") -> Motion:
        """Fade in. The object is hidden until then (its base opacity becomes 0), and any
        dimming from before is dropped: what is revealed is the new focus."""

        def make(object_id: str, delay: float):
            path = f"{object_id}.opacity"
            obj = self.state[object_id]
            # params.opacity already includes dim; the opacity track is the undimmed value.
            current = obj.params.opacity / obj.params.dim if obj.params.dim > 0 else 0.0
            target = current if current > 0 else 1.0
            ops = []
            if self._starts_hidden(path):
                target = self._base(path) if self._base(path) not in (None, 0, 0.0) else 1.0
                ops.append({"op": "set_param", "path": path, "value": 0.0})
            # Opacity never overshoots, so the fade always uses the reveal timing.
            seg, done = self._segment(path, float(target), verb, delay, timing="reveal")
            undim = self._undo_dimming_before(object_id, self.t + delay)
            return [*undim, *ops, seg], done

        return self._each(ids, verb, make)

    def exit(self, *ids: str) -> Motion:
        """Fade out."""

        def make(object_id: str, delay: float):
            seg, done = self._segment(f"{object_id}.opacity", 0.0, "exit", delay)
            return [seg], done

        return self._each(ids, "exit", make)

    def trace(self, *ids: str) -> Motion:
        """Draw a curve from its start to its end. Hidden (nothing drawn) until then."""

        def make(object_id: str, delay: float):
            if not self._has(object_id, "progress"):
                kind = self.state[object_id].type
                raise VerbError(f"trace needs a curve with 'progress'; {object_id!r} is a {kind}")
            path = f"{object_id}.progress"
            ops = [{"op": "set_param", "path": path, "value": 0.0}] if self._starts_hidden(
                path) else []  # fmt: skip
            seg, done = self._segment(path, 1.0, "trace", delay)
            return [*ops, seg], done

        return self._each(ids, "trace", make)

    def bloom_in(self, *ids: str) -> Motion:
        """Fade in while growing from a point with a small overshoot (objects with a
        radius ``r``); others simply fade in."""
        faded = self.reveal(*ids, verb="bloom_in")
        ops, end = list(faded.ops), faded.end
        for k, object_id in enumerate(ids):
            if not self._has(object_id, "r"):
                continue
            path = f"{object_id}.r"
            r = self.state[object_id].params.r
            if self._starts_hidden(path):
                ops.append({"op": "set_param", "path": path, "value": r * 1e-3})
            seg, done = self._segment(path, r, "bloom_in", k * self.motion["stagger"])
            ops.append(seg)
            end = max(end, done)
        return Motion(tuple(ops), end)

    def dim_others(self, *keep: str) -> Motion:
        """Fade everything except ``keep`` (and their children) to the theme's dim level.
        Backdrops and hidden objects are left alone."""
        for object_id in keep:
            self._object(object_id)
        level = self.motion["dim_level"]
        others = [
            o.id for o in self.state.objects
            if o.id not in keep and not any(o.id.startswith(k + ".") for k in keep)
            and o.params.dim > level + 1e-9
            and not o.impl.semantics(self.state).get("backdrop", False)
        ]  # fmt: skip
        # Dimming animates ``dim``, never ``opacity``: an object revealed later is still
        # hidden until its reveal, and arrives already dimmed.
        ops = [self._segment(f"{i}.dim", level, "dim_others", 0.0, "dim")[0] for i in others]
        return Motion(tuple(ops), self.motion["dim"]["duration"] if ops else 0.0)

    def undim(self) -> Motion:
        """Bring every dimmed object back to full opacity."""
        dimmed = [o.id for o in self.state.objects if o.params.dim < 1 - 1e-9]
        ops = [self._segment(f"{i}.dim", 1.0, "undim", 0.0, "dim")[0] for i in dimmed]
        return Motion(tuple(ops), self.motion["dim"]["duration"] if ops else 0.0)

    def focus_pull(self, object_id: str, margin: float = FOCUS_MARGIN) -> Motion:
        """Move the camera so ``object_id`` fills the frame, with ``margin`` around it."""
        obj = self._object(object_id)
        if obj.layer == "overlay":
            raise VerbError(f"{object_id!r} is on the overlay; the camera cannot move to it")
        b = obj.impl.world_bounds(self.state)
        cx, cy, _ = (float(v) for v in b.center)
        w, h = b.hi[0] - b.lo[0], b.hi[1] - b.lo[1]
        height = max(h, w / self.state.aspect) * (1 + 2 * margin)
        cam = self.c.doc.timeline.camera
        if self.c.doc.camera(cam).kind == "orbit":  # 3D: aim at it and come close enough
            radius = float(b.radius) * (1 + 2 * margin)
            fov = float(self.state.values[f"{cam}.fov"])
            dist = radius / max(1e-6, math.tan(math.radians(fov) / 2))
            s1, d1 = self._segment(f"{cam}.target", tuple(float(v) for v in b.center),
                                   "focus_pull", 0.0)  # fmt: skip
            s2, _ = self._segment(f"{cam}.distance", float(dist), "focus_pull", 0.0)
            return Motion((s1, s2), d1)
        z = float(self.state.values[f"{cam}.center"][2])  # keep the camera's depth
        s1, d1 = self._segment(f"{cam}.center", (cx, cy, z), "focus_pull", 0.0)
        s2, _ = self._segment(f"{cam}.height", float(height), "focus_pull", 0.0)
        return Motion((s1, s2), d1)

    def morph(self, path: str, value: Any, until: str | None = None) -> Motion:
        """Change any parameter (points of a curve, a constant, a color stop) smoothly, in
        the theme's ``morph`` time, or slowly until the marker ``until`` (a long, calm change:
        the constant of a function drifting while it is talked about)."""
        target = path.rpartition(".")[0]
        if self.c.doc.camera(target) is None:
            self._object(target)
        seg, done = self._segment(path, value, "morph", 0.0)
        if until is not None:
            done = self._until(until, "morph")
            seg["segment"]["t1"] = {"marker": until, "offset": 0.0}
        return Motion((seg,), done)

    def _until(self, until: str, verb: str) -> float:
        """Seconds from the anchor to the marker ``until``, which must come after it."""
        if until not in self.c.time_map.markers:
            raise VerbError(f"no marker {until!r}; markers are {sorted(self.c.time_map.markers)}")
        end = self.c.time_map.markers[until] - self.t
        if end <= 0:
            raise VerbError(f"marker {until!r} is not after the {verb} starts")
        return end

    def settle(self, path: str, value: Any) -> Motion:
        """Move to ``value`` like a released spring: overshoot, then come to rest."""
        if path.endswith(".opacity"):
            raise VerbError("settle overshoots, and opacity cannot go past 1; use reveal")
        seg, done = self._segment(path, value, "settle", 0.0)
        return Motion((seg,), done)

    def _orbit_camera(self, verb: str) -> str:
        cam = self.c.doc.timeline.camera
        if self.c.doc.camera(cam).kind != "orbit":
            raise VerbError(f"{verb} needs the 3D camera: s.use_camera('orbit')")
        return cam

    def orbit(self, degrees: float = 360.0, until: str | None = None) -> Motion:
        """Swing the camera ``degrees`` around its target (positive: counterclockwise seen
        from above). It takes the theme's ``orbit`` time per full turn, or runs until the
        marker ``until``. Camera goals keep working during it: the solver frames what you
        asked for at every angle."""
        cam = self._orbit_camera("orbit")
        path = f"{cam}.azimuth"
        start = self._written(path)
        spec = self.motion["orbit"]
        if until is None:
            seg, _ = self._segment(path, start + degrees, "orbit", 0.0)
            duration = spec["duration"] * abs(degrees) / 360.0
            seg["segment"]["t1"] = _ref(shift(self.at, duration))
            return Motion((seg,), duration)
        end = self._until(until, "orbit")
        seg = {
            "t0": _ref(self.at),
            "t1": {"marker": until, "offset": 0.0},
            "to": start + degrees,
            "ease": spec["ease"],
            "verb": "orbit",
        }
        return Motion(({"op": "add_segment", "path": path, "segment": seg},), end)

    def orbit_reveal(self, *ids: str) -> Motion:
        """Reveal objects one after another while the camera swings a little around them,
        so the eye gets depth from the motion (the theme's ``orbit_swing`` degrees)."""
        cam = self._orbit_camera("orbit_reveal")
        shown = self.reveal(*ids, verb="orbit_reveal")
        path = f"{cam}.azimuth"
        swing = self.motion["orbit_swing"]
        seg, done = self._segment(path, self._written(path) + swing, "orbit_reveal", 0.0)
        return Motion((*shown.ops, seg), max(shown.end, done))

    # ---------------------------------------------------------- the motion verbs

    def _instant(self, path: str, value: Any, verb: str, delay: float) -> dict[str, Any]:
        """A jump to ``value`` at ``delay`` seconds after the anchor."""
        t = _ref(shift(self.at, delay))
        seg = {"t0": t, "t1": t, "to": value, "ease": "linear", "verb": verb}
        return {"op": "add_segment", "path": path, "segment": seg}

    def _span(self, path: str, value: Any, verb: str, t0: float, t1: float, ease: str):
        seg = {"t0": _ref(shift(self.at, t0)), "t1": _ref(shift(self.at, t1)), "to": value,
               "ease": ease, "verb": verb}  # fmt: skip
        return {"op": "add_segment", "path": path, "segment": seg}

    def _show_at(self, object_id: str, verb: str, delay: float) -> list[dict[str, Any]]:
        """Appear (opacity 1) at ``delay``; hidden until then if nothing showed it before."""
        path = f"{object_id}.opacity"
        ops = []
        if self._starts_hidden(path):
            ops.append({"op": "set_param", "path": path, "value": 0.0})
        return [*ops, self._instant(path, 1.0, verb, delay)]

    def _of_type(self, object_id: str, verb: str, *names: str):
        obj = self._object(object_id)
        if obj.type not in names:
            raise VerbError(f"{verb} needs {' or '.join(names)}; {object_id!r} is a {obj.type}")
        return obj

    def transform(self, source: str, target: str) -> Motion:
        """An equation becoming the next: parts that appear in both glide from their old
        place (and color) to the new; parts with the same ``keys`` entry glide too, changing
        their text on the way, and arc round the parts they are carried past (the theme's
        ``transform_arc``); the old one's other parts fade out, the new one's fade in.
        ``target`` is hidden until now; ``source`` disappears as it starts."""
        a = self._of_type(source, "transform", "equation")
        self._of_type(target, "transform", "equation")
        ops: list[dict[str, Any]] = [
            {"op": "set_param", "path": f"{target}.from_parts", "value": list(a.params.parts)},
            {"op": "set_param", "path": f"{target}.from_colors", "value": list(a.params.colors)},
            # the place the script gave it; life's drift is added when drawn (``from_id``)
            {
                "op": "set_param",
                "path": f"{target}.from_pos",
                "value": [float(v) for v in self.state.values[f"{source}.pos"]],
            },
            {"op": "set_param", "path": f"{target}.from_keys", "value": list(a.params.keys)},
            {"op": "set_param", "path": f"{target}.from_id", "value": source},
            {"op": "set_param", "path": f"{target}.arc", "value": self.motion["transform_arc"]},
        ]
        if self._starts_hidden(f"{target}.morph"):
            ops.append({"op": "set_param", "path": f"{target}.morph", "value": 0.0})
        ops += self._show_at(target, "transform", 0.0)
        ops.append(self._instant(f"{source}.opacity", 0.0, "transform", 0.0))
        seg, done = self._segment(f"{target}.morph", 1.0, "transform", 0.0)
        return Motion((*ops, seg), done)

    def become(self, source: str, target: str) -> Motion:
        """One curve turning into another (a circle into a square): ``source``'s points move
        onto ``target``'s (both resampled to the same count, closed curves aligned so the
        motion is short), then ``target`` takes its place."""
        a = self._of_type(source, "become", "line")
        b = self._of_type(target, "become", "line")
        path = f"{source}.points"
        if not self._starts_hidden(path) or isinstance(self._base(path), ExprSpec):
            raise VerbError(f"become needs {source!r}'s points unanimated before it")
        pa, pb = _matched_shapes(np.asarray(a.params.points), np.asarray(b.params.points))
        spec = self.motion["become"]
        d = spec["duration"]
        # The same shape with more points (a track keeps one point count throughout).
        ops: list[dict[str, Any]] = [{"op": "set_param", "path": path, "value": pa.tolist()}]
        ops.append(self._span(path, pb.tolist(), "become", 0.0, d, spec["ease"]))
        ops += self._show_at(target, "become", d)
        ops.append(self._instant(f"{source}.opacity", 0.0, "become", d))
        return Motion(tuple(ops), d)

    def write(self, *ids: str) -> Motion:
        """Write text or an equation in, left to right (hidden until now)."""

        def make(object_id: str, delay: float):
            if not self._has(object_id, "write"):
                kind = self.state[object_id].type
                raise VerbError(f"write needs text or an equation; {object_id!r} is a {kind}")
            path = f"{object_id}.write"
            ops = [{"op": "set_param", "path": path, "value": 0.0}] if self._starts_hidden(
                path) else []  # fmt: skip
            seg, done = self._segment(path, 1.0, "write", delay)
            return [*ops, seg], done

        return self._each(ids, "write", make)

    def travel(self, *ids: str) -> Motion:
        """Send tracers along their paths, start to end, trailing light."""

        def make(object_id: str, delay: float):
            obj = self._of_type(object_id, "travel", "tracer")
            if obj.impl.semantics(self.state).get("path_missing"):
                raise VerbError(f"{object_id!r} travels along {obj.params.path!r}, which is not "
                                "a curve in the scene")  # fmt: skip
            path = f"{object_id}.progress"
            ops = [{"op": "set_param", "path": path, "value": 0.0}] if self._starts_hidden(
                path) else []  # fmt: skip
            seg, done = self._segment(path, 1.0, "travel", delay)
            return [*ops, seg], done

        return self._each(ids, "travel", make)

    def flash(self, *ids: str) -> Motion:
        """A quick brightening: light and glow swell, then fall back."""
        spec = self.motion["flash"]
        d = spec["duration"]

        def make(object_id: str, delay: float):
            path = f"{object_id}.flash"
            up = self._span(path, 1.0, "flash", delay, delay + 0.3 * d, spec["ease"])
            down = self._span(path, 0.0, "flash", delay + 0.3 * d, delay + d, "smooth")
            return [up, down], delay + d

        return self._each(ids, "flash", make)

    def circumscribe(self, object_id: str, shape: str = "circle") -> Motion:
        """Draw a ring (or a box) around an object, then fade it: "look here". The ring is
        the object's child ``<id>.ring``, made the first time and reused after."""
        self._object(object_id)
        if shape not in ("circle", "box"):
            raise VerbError(f"circumscribe draws a 'circle' or a 'box', not {shape!r}")
        ring = f"{object_id}.ring"
        spec = self.motion["circumscribe"]
        d = spec["duration"]
        ops: list[dict[str, Any]] = []
        if ring not in self.state:
            ops.append({"op": "add_object", "object": {"id": ring, "type": "callout", "params": {
                "target": object_id, "shape": shape, "progress": 0.0}}})  # fmt: skip
        else:
            ops += [self._instant(f"{ring}.progress", 0.0, "circumscribe", 0.0),
                    self._instant(f"{ring}.opacity", 1.0, "circumscribe", 0.0)]  # fmt: skip
        ops.append(self._span(f"{ring}.progress", 1.0, "circumscribe", 0.0, 0.6 * d, spec["ease"]))
        ops.append(self._span(f"{ring}.opacity", 0.0, "circumscribe", 0.6 * d, d, "smooth"))
        return Motion(tuple(ops), d)

    def wiggle(self, *ids: str) -> Motion:
        """A small shake from side to side: "this one"."""
        spec = self.motion["wiggle"]
        d = spec["duration"]

        def make(object_id: str, delay: float):
            if not self._has(object_id, "pos"):
                kind = self.state[object_id].type
                raise VerbError(f"wiggle moves an object's 'pos'; {object_id!r} is a {kind}")
            path = f"{object_id}.pos"
            self._base(path)  # refuses an expression
            obj = self.state[object_id]
            p0 = np.asarray(obj.params.pos, dtype=np.float64)
            size = float(obj.impl.world_bounds(self.state).radius)
            side = np.asarray(self.state.camera_for(obj).right) * max(0.06 * size, 1e-3)
            steps = (p0 + side, p0 - side, p0 + side / 2, p0)
            ops = [self._span(path, [float(v) for v in q], "wiggle", delay + k * d / 4,
                              delay + (k + 1) * d / 4, "smooth")
                   for k, q in enumerate(steps)]  # fmt: skip
            return ops, delay + d

        return self._each(ids, "wiggle", make)

    def sweep(self, path: str, value: float, show: bool = True) -> Motion:
        """Run a number from where it is to ``value`` at an even pace, showing it as it goes
        (``show``: a ``Counter`` beside the object, ``<id>.readout``, made the first time)."""
        target, _, name = path.rpartition(".")
        obj = self._object(target)
        current = getattr(obj.params, name, None)
        if not isinstance(current, int | float) or isinstance(current, bool):
            raise VerbError(f"sweep runs a number; {path} is {type(current).__name__}")
        self._base(path)  # refuses an expression
        seg, done = self._segment(path, float(value), "sweep", 0.0)
        ops: list[dict[str, Any]] = []
        readout = f"{target}.readout"
        if show and readout not in self.state:
            whole = type(obj.params).model_fields[name].annotation is int
            label = name.replace("_", "\\_")  # mathtext: an underscore is a subscript
            params = {
                "value": {"expr": path},
                "digits": 0 if whole else 2,
                "prefix": f"\\mathrm{{{label}}} = ",
                "opacity": 0.0,
            }
            ops.append(
                {"op": "add_object", "object": {"id": readout, "type": "counter", "params": params}}
            )
            ops.append({"op": "add_constraint", "rule": f"{readout} near {target}"})
            ops.append(self._instant(f"{readout}.opacity", 1.0, "sweep", 0.0))
        return Motion((*ops, seg), done)

    def unfold(self, room: str, ray: str, flips: int = 5, speed: float | None = None,
               index: int | None = None) -> Motion:  # fmt: skip
        """A ray bouncing in a room, shown as a straight line through mirror rooms: as the
        ray reaches each wall, a mirror copy of the room swings over that wall like a page
        and the ray carries straight on into it. The ray is a ``Line`` (its first point is
        where it starts, its second the way it goes) or ray ``index`` of an ``ac.SoundRays``
        (drawn as a new line, ``<ray>.unfolded_<index>``). ``room`` is a box (``ac.Room``, a
        box ``ac.Room3D``, anything whose bounds are the room: a flat one is 2D), a convex
        polygon (a closed ``Line``, an ``algebra.Polygon``: 2D) or a convex mesh
        ``ac.Room3D``. ``flips`` copies are made, each a ``HingedCopy``, the children
        ``<room>.mirror_<name>``: for a box, named by place (``mirror_e2_n1``: two rooms
        east, one north), otherwise by the walls crossed (``mirror_b_a_c``; walls ``a``,
        ``b``, ... in the room's order; ``core/unfold_math.py``). The ray runs straight to
        the far wall of the last copy at ``speed`` world units a second (default: one wall
        per flip on average); each flip takes the theme's ``unfold`` time and lands as the
        ray arrives. At a corner (several walls at once) the flips follow one another."""
        verb = "unfold"
        obj = self._object(room)
        if obj.type == "hinged_copy" or obj.layer == "overlay":
            raise VerbError(f"unfold needs a room in the world; {room!r} is a {obj.type}")
        start, direction, line = self._unfold_ray(room, ray, index)
        flat = self.c.doc.camera(self.c.doc.timeline.camera).kind != "orbit"
        if flat and abs(direction[2]) > 1e-12:
            raise VerbError("in 2D the ray must stay in the plane: give its points one z")
        if not 1 <= flips <= unfold_math.MAX_FLIPS:
            raise VerbError(f"flips must be 1 to {unfold_math.MAX_FLIPS}")
        shape = self._unfold_room(room, obj)
        try:
            if isinstance(shape, unfold_math.Polytope):
                u = unfold_math.room_unfold(shape, start, direction, flips)
            else:
                u = unfold_math.box_unfold(shape.lo, shape.hi, start, direction, flips)
        except ValueError as e:
            raise VerbError(f"unfold: {e}") from None
        spec = self.motion["unfold"]
        dur = spec["duration"]
        last = u.crossings[-1].distance
        if speed is None:
            speed = max(last, 1e-9) / len(u.crossings) / dur
        if speed <= 0:
            raise VerbError("unfold needs a positive speed")
        lead = max(0.0, dur - u.crossings[0].distance / speed)  # the first page has time
        straight = [list(u.start), list(u.end_point)]
        ops: list[dict[str, Any]] = []
        if line == ray:
            ops.append({"op": "set_param", "path": f"{ray}.points", "value": straight})
            if self._starts_hidden(f"{ray}.progress"):
                ops.append({"op": "set_param", "path": f"{ray}.progress", "value": 0.0})
        else:  # one of a SoundRays' rays: drawn as a line of its own
            ops.append({"op": "add_object", "object": {
                "id": line, "type": "line", "params": {"points": straight, "progress": 0.0},
                "style": {"glow": "strong", "stroke": "bold", "color": "highlight"}}})  # fmt: skip
        done = lead + u.end / speed
        ops.append(self._span(f"{line}.progress", 1.0, verb, lead, done, "linear"))
        arrive = [lead + c.distance / speed for c in u.crossings]
        windows = _flip_windows(arrive, dur, 1e-9 * max(1.0, done))
        for crossing, chain, (t0, t1) in zip(u.crossings, u.chains(), windows, strict=True):
            copy = f"{room}.mirror_{crossing.name}"
            if copy in self.state or self.c.doc.object(copy) is not None:
                raise VerbError(f"{copy!r} is already in the scene; unfold each room once")
            hinges = [[list(x) if isinstance(x, tuple) else x for x in h.as_params()]
                      for h in chain]  # fmt: skip
            ops.append({"op": "add_object", "object": {"id": copy, "type": "hinged_copy",
                        "params": {"source": room, "hinges": hinges, "turn": 0.0,
                                   "opacity": 0.0}}})  # fmt: skip
            ops.append(self._instant(f"{copy}.opacity", 1.0, verb, t0))
            ops.append(self._span(f"{copy}.turn", 1.0, verb, t0, t1, spec["ease"]))
        return Motion(tuple(ops), done)

    def _unfold_ray(self, room: str, ray: str, index: int | None):
        """Where the ray starts, the way it goes, and the id of the line that draws it."""
        obj = self._of_type(ray, "unfold", "line", "sound_rays")
        if obj.type == "line":
            if index is not None:
                raise VerbError(f"index picks one ray of a SoundRays; {ray!r} is a line")
            path = f"{ray}.points"
            if not self._starts_hidden(path) or isinstance(self._base(path), ExprSpec):
                raise VerbError(f"unfold needs {ray!r}'s points unanimated before it")
            pts = np.asarray(obj.params.points, dtype=np.float64)
            return pts[0], pts[1] - pts[0], ray
        if index is None:
            raise VerbError(f"unfold needs index= to pick one of {ray!r}'s rays")
        if obj.params.room != room:
            raise VerbError(f"{ray!r} bounces in {obj.params.room!r}; unfold that room")
        paths = obj.impl.paths(self.state)
        if paths is None:
            raise VerbError(f"{ray!r} has no room to bounce in")
        if not 0 <= index < len(paths["dirs"]):
            raise VerbError(f"{ray!r} has rays 0 to {len(paths['dirs']) - 1}, not {index}")
        line = f"{ray}.unfolded_{index}"
        if line in self.state or self.c.doc.object(line) is not None:
            raise VerbError(f"{line!r} is already in the scene; unfold each ray once")
        return np.asarray(obj.params.source, dtype=np.float64), paths["dirs"][index], line

    def _unfold_room(self, room: str, obj):
        """The room as a convex ``unfold_math.Polytope``, or as a box (its bounds)."""
        if obj.type == "room3d":  # a box or a mesh
            g = obj.impl.geometry(self.state)
            if g is None:
                raise VerbError(f"unfold: {room!r}'s mesh file is missing")
            if g.box is None:
                try:
                    return unfold_math.polyhedron(g.vertices, g.triangles)
                except ValueError as e:
                    raise VerbError(f"unfold needs a box room or a convex one; {room!r} is a "
                                    f"mesh and {e}") from None  # fmt: skip
        corners = obj.impl.features(self.state).get("vertices")
        if corners is None and obj.type == "line":
            pts = np.asarray(obj.params.points, dtype=np.float64)
            if len(pts) >= 4 and np.allclose(pts[0], pts[-1]):  # a closed outline
                corners = pts
        if corners is not None:
            try:
                return unfold_math.polygon(corners)
            except ValueError as e:
                raise VerbError(f"unfold needs a box room or a convex one; {room!r}: {e}") from None
        return obj.impl.world_bounds(self.state)


def _flip_windows(arrive: list[float], dur: float, tol: float) -> list[tuple[float, float]]:
    """When each flip turns (seconds after the anchor): the theme's time, landing as the ray
    arrives. Flips the ray reaches at once (a corner) follow one another, each starting as
    the one before lands: together they take the time since the last arrival or the
    theme's time, whichever is longer (each at most the theme's time)."""
    out: list[tuple[float, float]] = []
    prev, k = 0.0, 0
    while k < len(arrive):
        t, m = arrive[k], 1
        while k + m < len(arrive) and arrive[k + m] - t <= tol:
            m += 1
        w = dur if m == 1 else min(dur, max(t - prev, dur) / m)
        out += [(t - (m - j) * w, t - (m - 1 - j) * w) for j in range(m)]
        prev, k = t, k + m
    return out


def _resample(points: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    u = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(u, cum, pts[:, k]) for k in range(pts.shape[1])], axis=1)


def _matched_shapes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two polylines resampled to the same number of points; when both are closed, ``b`` is
    turned (and reversed if that is closer) so each point travels as little as possible."""
    n = max(len(a), len(b), 64)
    ra, rb = _resample(a, n), _resample(b, n)
    closed = np.allclose(a[0], a[-1]) and np.allclose(b[0], b[-1])
    if closed:
        best, best_cost = rb, np.inf
        ring = rb[:-1]
        for cand in (ring, ring[::-1]):
            for k in range(0, len(cand), max(1, len(cand) // 64)):
                c = np.roll(cand, -k, axis=0)
                c = np.vstack([c, c[:1]])
                cost = float(np.square(c - ra).sum())
                if cost < best_cost:
                    best, best_cost = c, cost
        rb = best
    return ra, rb
