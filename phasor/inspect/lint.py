"""The linter: visual problems found before anyone renders.

It looks at the baked scene at sampled times plus every marker, through the inspector, and
returns one issue per problem with the times it was seen. The rules:

| code      | severity | catches                                                        |
| overlap   | warning  | a label, or an object tagged ``key``, overlapping something    |
| offscreen | warning  | a label or key object cut by the frame                         |
| contrast  | warning  | text too close in brightness to what is behind it              |
| missing_glyph | warning | text with characters the font cannot draw (empty boxes)     |
| tiny      | warning  | objects or text too small to read                              |
| orphan    | warning  | an object that never appears                                   |
| clutter   | warning  | too many objects changing at once in one block                 |
| unsolved  | error    | a constraint that does not hold (from the bake, or a guard)    |
| occluded_label | warning | 3D: a label partly hidden by something in front of it      |
| missing_file | warning | a file an object reads (a mesh), a bed or a cue plays is not there |
| bad_sound_file | warning | a bed's or cue's file that cannot be decoded (it stays silent) |
| sound_cut | warning | a cue's file that runs past the end of the scene                    |
| bed_masks_voice | warning | a bed within ``[sound] masking`` dB of the voice in speech |
| cue_on_word | warning | a cue peaking within ``[sound] word_gap`` s of a stressed word |
| audio_clip | warning | the mix over full scale by ``[sound] clip`` dB before the limiter |
| cue_unanchored | warning | (strict) a cue at raw seconds instead of on a marker        |
| slow_layout | warning | the layout bake renders more occlusion buffers than ``bake_renders`` |

Under a perspective camera ``overlap`` only compares labels with labels and points with
points: a label in front of a surface is fine, and one behind it is ``occluded_label``.

Thresholds come from the theme's ``[lint]`` table; the sound rules' from ``[sound]``
(they sit beside the levels they judge).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np

from phasor.core import text as text_metrics
from phasor.core.color import contrast_ratio, over
from phasor.core.constraints import LABEL_SUFFIX
from phasor.core.ids import parent_of
from phasor.core.issues import Issue
from phasor.core.schema import MarkerRef
from phasor.core.state import SceneState
from phasor.inspect.passes import PassRunner, inspect_size
from phasor.layout.bake import Bake
from phasor.layout.geometry import footprint, view_of
from phasor.layout.plan import MOVE_PARAM, Plan
from phasor.time.compile import Compiled, CompiledExpr
from phasor.time.tracks import ResolvedTrack

DEFAULT_STEP = 0.5  # seconds between lint samples
FAINT = 0.05  # objects at or below this opacity are treated as not shown
CONTAINED = 0.98  # a label this much inside a shape is "on" it, not overlapping it
MAX_TIMES = 12


def sample_times(c: Compiled, step: float = DEFAULT_STEP) -> tuple[float, ...]:
    """Every marker, both ends, the ends of every transition, and a grid ``step`` apart."""
    end = c.duration
    ts = {0.0, end, *c.time_map.markers.values()}
    for node in c.nodes.values():
        if isinstance(node, ResolvedTrack):
            ts.update(s.t1 for s in node.segments)
    ts.update(np.arange(0.0, end + 1e-9, step).tolist())
    return tuple(sorted({round(t, 6) for t in ts if 0.0 <= t <= end}))


def _is_label(e: dict[str, Any]) -> bool:
    return e["semantic"].get("kind") == "label"


def _part_of(entries: dict[str, dict[str, Any]], a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether one of two entries is a part a composite built for the other (ADR 0005): a
    composite covers its own parts by design."""
    for x, y in ((a, b), (b, a)):
        up = x.get("part_of")
        while up is not None:
            if up == y["id"]:
                return True
            up = entries[up].get("part_of") if up in entries else None
    return False


def _shown(e: dict[str, Any]) -> bool:
    drawn = e["semantic"].get("drawn", 1.0) > 0  # a curve not yet traced shows nothing
    return e["opacity"] > FAINT and e["on_screen_fraction"] > 0 and drawn


def _area(b) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _box_overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def shown_opacity(c: Compiled, object_id: str) -> float:
    """The opacity an object is meant to have when it is fully shown: its constant value,
    or the highest its track ever reaches. Fades and dims in between are deliberate, so
    contrast is judged at this value, not at the moment's."""
    node = c.nodes.get(f"{object_id}.opacity")
    if isinstance(node, ResolvedTrack):
        return max([float(node.initial), *(float(s.to) for s in node.segments)])
    return float(getattr(node, "value", 1.0))


class _Frame:
    """One sampled time: the state, its inspector entries, and overlap measurements."""

    def __init__(
        self,
        state: SceneState,
        snap: dict[str, Any],
        runner: PassRunner | None,
        compiled: Compiled | None = None,
    ):
        self.state, self.runner, self.compiled = state, runner, compiled
        self.entries = {e["id"]: e for e in snap["objects"]}
        self.size = inspect_size(state)
        self._masks: dict[str, np.ndarray] = {}

    def box(self, object_id: str):
        return self.entries[object_id]["bbox"]["px"]

    def _mask(self, object_id: str) -> np.ndarray | None:
        """The object's silhouette at inspector resolution. None for a label: text is judged
        by its whole box, because anything crossing the box hurts reading."""
        if _is_label(self.entries[object_id]):
            return None
        if object_id not in self._masks:
            self._masks[object_id] = (
                self.runner.solo_mask(self.state, self.size, object_id)
                if self.runner is not None
                else footprint(self.state, self.state[object_id], self.size)
            )
        return self._masks[object_id]

    def _rect(self, box) -> tuple[slice, slice]:
        sx, sy = self.size[0] / self.state.width, self.size[1] / self.state.height
        x0, x1 = int(np.floor(box[0] * sx)), int(np.ceil(box[2] * sx))
        y0, y1 = int(np.floor(box[1] * sy)), int(np.ceil(box[3] * sy))
        w, h = self.size
        return (slice(min(max(y0, 0), h), min(max(y1, 0), h)),
                slice(min(max(x0, 0), w), min(max(x1, 0), w)))  # fmt: skip

    def covered(self, label_id: str, other_id: str) -> float:
        """Share of ``label_id``'s box covered by ``other_id``."""
        a, b = self.box(label_id), self.box(other_id)
        if _box_overlap(a, b) <= 0:
            return 0.0
        mask = self._mask(other_id)
        if mask is None:
            return _box_overlap(a, b) / max(_area(a), 1e-9)
        cut = mask[self._rect(a)]
        return float(cut.mean()) if cut.size else 0.0

    def shared(self, a_id: str, b_id: str) -> float:
        """Overlap of two objects as a share of the smaller one."""
        a, b = self.box(a_id), self.box(b_id)
        if _box_overlap(a, b) <= 0:
            return 0.0
        ma, mb = self._mask(a_id), self._mask(b_id)
        if ma is None and mb is None:
            return _box_overlap(a, b) / max(min(_area(a), _area(b)), 1e-9)
        if ma is None:
            return self.covered(a_id, b_id)
        if mb is None:
            return self.covered(b_id, a_id)
        smaller = min(int(ma.sum()), int(mb.sum()))
        return float((ma & mb).sum() / smaller) if smaller else 0.0

    def _backdrop(self) -> np.ndarray | None:
        """The color pass with every label left out, at inspector size (pixel mode only)."""
        if self.runner is None:
            return None
        if not hasattr(self, "_backdrop_px"):
            rest = tuple(o for o in self.state.objects
                         if o.impl.semantics(self.state).get("kind") != "label")  # fmt: skip
            others = dataclasses.replace(self.state, objects=rest)
            self._backdrop_px = self.runner.color(others, self.size)
        return self._backdrop_px

    def contrast(self, label_id: str) -> float:
        """Contrast of a label against what is behind it: measured from the color pass when
        there is one (so domain coloring and glow count), else the background and any shape
        the label mostly sits on."""
        me = self.state[label_id]
        c = me.style["color"]
        alpha = c[3] * (shown_opacity(self.compiled, label_id) if self.compiled else
                        me.params.opacity)  # fmt: skip
        backdrop = self._backdrop()
        if backdrop is not None:
            ys, xs = self._rect(self.box(label_id))
            behind = backdrop[ys, xs, :3].reshape(-1, 3)
            if len(behind):
                bg = (*np.clip(behind.mean(axis=0), 0, None), 1.0)
                return contrast_ratio(over((*c[:3], alpha), bg), bg)
        bg = self.state.background
        for obj in self.state.draw_order():
            if obj.id == label_id:
                break
            on_it = obj.params.opacity > FAINT and "color" in obj.style
            if on_it and self.covered(label_id, obj.id) > 0.5:
                under = obj.style["color"]
                bg = over((*under[:3], under[3] * obj.params.opacity), bg)
        return contrast_ratio(over((*c[:3], alpha), bg), bg)


class _Findings:
    def __init__(self) -> None:
        self.found: dict[tuple, dict[str, Any]] = {}

    def add(self, code: str, ids: tuple[str, ...], t: float, **info: Any) -> None:
        entry = self.found.setdefault((code, ids, info.get("path")), {"times": [], **info})
        entry["times"].append(t)

    def issues(self, make: Callable[..., Issue]) -> list[Issue]:
        return [make(code, ids, **info) for (code, ids, _), info in sorted(self.found.items(),
                key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or ""))]  # fmt: skip


def _when(times: list[float]) -> str:
    shown = ", ".join(f"{t:g}" for t in times[:MAX_TIMES])
    more = f" and {len(times) - MAX_TIMES} more" if len(times) > MAX_TIMES else ""
    return f"at t = {shown}{more} s"


def _movable(c: Compiled, object_id: str) -> bool:
    node = c.nodes.get(f"{object_id}.{MOVE_PARAM}")
    return node is not None and not isinstance(node, CompiledExpr)


def _near_fix(c: Compiled, plan: Plan, label_id: str) -> tuple[dict[str, Any], ...]:
    owner = parent_of(label_id)
    if (
        owner is None
        or not label_id.endswith(f".{LABEL_SUFFIX}")
        or label_id in plan.moving
        or not _movable(c, label_id)
    ):
        return ()
    return ({"op": "add_constraint", "rule": f"label({owner}) near {owner}"},)


def bake_rules(c: Compiled, plan: Plan, baked: Bake) -> list[Issue]:
    """``slow_layout``: the layout bake renders an occlusion buffer (a depth buffer in 3D)
    at every 10 Hz key time while something it places is shown, and again after every edit
    that changes what comes before that. More than the theme's ``bake_renders`` is slow on
    any machine. Counted the same with or without OpenGL; ``times`` are where each baked
    stretch starts. No fix: only the author knows whether the label must be there."""
    limit = c.theme.lint["bake_renders"]
    if baked.renders <= limit:
        return []
    ids = tuple(sorted(plan.moving))
    spans = ", ".join(f"{a:g}-{b:g} s" for a, b in baked.spans[:MAX_TIMES])
    more = f" and {len(baked.spans) - MAX_TIMES} more" if len(baked.spans) > MAX_TIMES else ""
    goals = " and the camera goals" if plan.goals else ""
    return [Issue(
        "slow_layout", f"the layout bake renders {baked.renders} occlusion buffers (limit "
        f"{limit:g}): one per key time (10 Hz) while {', '.join(map(repr, ids))} "
        f"{'is' if len(ids) == 1 else 'are'} shown ({spans}{more}){goals}",
        severity="warning", ids=ids, times=tuple(a for a, _ in baked.spans),
        fix_hint="show what layout places only while it is needed (reveal ... exit), or give "
        "it a fixed place (pos=...) instead of a rule",
    )]  # fmt: skip


def control_rules(c: Compiled) -> list[Issue]:
    """A slider whose parameter has gone, or is no longer a plain number, is a lint error."""
    from phasor.core.patch import param_value

    out = []
    for ctl in c.doc.controls:
        try:
            value = param_value(c.doc, ctl.path)
        except KeyError:
            out.append(
                Issue(
                    "orphan_control",
                    f"slider {ctl.key!r} targets a parameter that no longer exists",
                    path=ctl.path,
                    fix_hint=f"remove_control {ctl.key} or restore the parameter",
                )
            )
            continue
        animated = ctl.path in c.doc.timeline.tracks
        if animated or isinstance(value, dict | str):
            why = "animated by a track" if animated else "no longer a constant number"
            out.append(
                Issue(
                    "orphan_control",
                    f"slider {ctl.key!r}: {ctl.path} is {why}",
                    path=ctl.path,
                    fix_hint=f"remove_control {ctl.key}",
                )
            )
    return out


def file_rules(c: Compiled) -> list[Issue]:
    """``missing_file``: a parameter a type lists in ``Files`` (core/files.py) names a file
    that is not in the scene's folder. The object draws nothing until the file is there."""
    from phasor.core import files
    from phasor.core.objects import get_type
    from phasor.time.compile import Const

    out = []
    for o in c.objects:
        for name in getattr(get_type(o.type), "Files", ()):
            path = f"{o.id}.{name}"
            node = c.nodes.get(path)
            value = node.value if isinstance(node, Const) else None
            if isinstance(value, str) and value and not files.exists(c.base_dir, value):
                out.append(Issue(
                    "missing_file", f"{path}: no file {value!r} in the scene's folder",
                    severity="warning", ids=(o.id,), path=path, times=(0.0,),
                    fix_hint=f"put {value!r} next to the scene, or set {path} to a file that is",
                ))  # fmt: skip
    return out


SOUND_CODES = ("missing_file", "bad_sound_file", "sound_cut", "bed_masks_voice", "cue_on_word",
               "audio_clip", "cue_unanchored")  # fmt: skip


def sound_rules(c: Compiled, mix: Callable[[], Any]) -> list[Issue]:
    """The document's beds and cues (time/sound.py). Files: ``missing_file`` (not
    in the scene's folder), ``bad_sound_file`` (not audio ``soundfile`` can decode), and
    ``sound_cut`` (a file cue that plays past the end of the scene); each stays silent or is
    cut, never fails. The mix (``mix()``, made only when there are beds or cues), against
    thresholds in the theme's ``[sound]``: ``bed_masks_voice``, ``cue_on_word``,
    ``audio_clip``."""
    from phasor.core import files
    from phasor.time.sound import file_seconds, stressed_onsets

    out: list[Issue] = []
    named = [(p.id, p.bed.path, p.t0, None) for p in c.beds]
    named += [(p.id, p.cue.sound, p.t, p) for p in c.cues if not p.cue.builtin]
    for sid, path, t, cue in named:
        field = "sound" if cue is not None else "path"
        where = dict(severity="warning", ids=(sid,), path=f"{sid}.{field}", times=(t,))
        if not files.exists(c.base_dir, path):
            out.append(Issue("missing_file", f"{sid!r}: no file {path!r} in the scene's folder",
                             fix_hint=f"put {path!r} next to the scene, or name a file that is",
                             **where))  # fmt: skip
            continue
        length = file_seconds(c.base_dir, path)
        if length is None:
            out.append(Issue("bad_sound_file", f"{sid!r}: {path!r} is not audio that can be "
                             "decoded, so it is silent", fix_hint="convert it to WAV, FLAC or "
                             "OGG, e.g. ffmpeg -i in.m4a out.flac", **where))  # fmt: skip
        elif cue is not None and t + length > c.duration + 1e-6:
            end = t + length
            out.append(Issue("sound_cut", f"cue {sid!r} plays until {end:.3f} s but the scene "
                             f"ends at {c.duration:.3f} s", fix_hint="set meta.duration to "
                             f"{end:.3f} (or later), or start it earlier", **where))  # fmt: skip
    if not c.beds and not c.cues:
        return out
    try:
        m = mix()
    except (OSError, ValueError):  # the narration's own audio is broken: its rules say so
        return out
    if m is None or not m.mixed:
        return out
    level = c.theme.sound
    scripted = c.doc.meta.source == "script"
    for p in c.beds:
        lane = m.lane(p.id)
        if lane is None:
            continue
        gaps = [(a, m.voice_level(a, b) - lane.loudness(a, b)) for a, b in m.spans]
        close = [(a, g) for a, g in gaps if np.isfinite(g) and g < level["masking"]]
        if not close:
            continue
        worst = min(g for _, g in close)
        now = p.bed.gain if p.bed.gain is not None else level["bed"]
        lower = max(-60.0, round(now - (level["masking"] - worst) - 0.5, 1))
        fix: tuple[dict[str, Any], ...] = ()
        if not scripted:  # a script's bed is changed in the script, not by a tweak
            bed = {**p.bed.model_dump(mode="json"), "gain": lower}
            fix = ({"op": "set_bed", "bed": bed},)
        out.append(Issue(
            "bed_masks_voice", f"bed {p.id!r} is only {worst:.1f} dB under the voice while it "
            f"speaks {_when([a for a, _ in close])} (keep it {level['masking']:g} dB under)",
            severity="warning", ids=(p.id,), path=f"{p.id}.gain",
            times=tuple(round(a, 3) for a, _ in close[:MAX_TIMES]),
            fix_hint=f"lower the bed: gain={lower:g}", fix=fix,
        ))  # fmt: skip
    onsets = stressed_onsets(c.doc.narration)
    for p in c.cues:
        lane = m.lane(p.id)
        if lane is None or lane.peak_t is None:
            continue
        hit = [(t, w) for t, w in onsets if abs(t - lane.peak_t) <= level["word_gap"]]
        if hit:
            t, word = min(hit, key=lambda h: abs(h[0] - lane.peak_t))
            out.append(Issue(
                "cue_on_word", f"cue {p.id!r} peaks at {lane.peak_t:.3f} s, on the word "
                f"{word!r} ({t:.3f} s): it may cover it", severity="warning", ids=(p.id,),
                path=f"{p.id}.at", times=(round(lane.peak_t, 3),),
                fix_hint="move it into a pause, or just before the word with a negative offset",
            ))  # fmt: skip
    if m.over:
        out.append(Issue(
            "audio_clip", f"the mix reaches {m.pre_peak_db:+.1f} dBFS before the limiter "
            f"{_when(list(m.over))} (more than {level['clip']:g} dB over full scale): the "
            "limiter squashes it audibly", severity="warning", times=m.over,
            fix_hint="lower the loudest cues or beds there, or space out cues that pile up",
        ))  # fmt: skip
    return out


def type_rules(c: Compiled) -> list[Issue]:
    """``missing_type``: the document names code (``meta.modules``, ADR 0005) for a type or an
    updater that is not here: the module is missing, was refused (scripts may not run here)
    or failed to load. The object draws as a placeholder box, or without its updater, and
    keeps its parameters and overrides; nothing is lost."""
    return [
        Issue("missing_type", f"{object_id}: {why}", severity="error", ids=(object_id,),
              times=(0.0,), fix_hint="put the module next to the scene, or open it where "
              "scripts may run (allow_scripts=True, or the MCP server with --allow-scripts)")
        for object_id, why in c.missing
    ]  # fmt: skip


def _differ(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() != b.keys() or any(_differ(a[k], b[k], tol) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) != len(b) or any(_differ(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, int | float) and isinstance(b, int | float):
        return not abs(float(a) - float(b)) <= tol  # NaN differs from everything
    return a != b


def updater_rules(c: Compiled, step: float = DEFAULT_STEP) -> list[Issue]:
    """``impure_updater`` (ADR 0005): an updater must give the same result for the same t and
    the same reads. Each one is called again at every sample time, twice, the second time in
    the opposite order of times and of updaters, all with the same evaluated parameters; it
    fires when two answers differ by more than the theme's ``[lint] impure``. Hidden state (a
    counter, a global another updater writes, randomness) shows up this way."""
    from phasor.time.compile import _env
    from phasor.time.updaters import UpdaterOut

    ups = list({id(n.updater): n.updater for n in c.nodes.values()
                if isinstance(n, UpdaterOut)}.values())  # fmt: skip
    if not ups:
        return []
    tol = c.theme.lint["impure"]
    times = sample_times(c, step)
    envs = {t: _env(c.nodes, c.order, t) for t in times}

    def call(i: int, t: float) -> Any:
        try:
            return ups[i].call(envs[t], t)
        except ValueError as e:  # an updater that fails now and then is not pure either
            return ("failed", str(e))

    first = {(t, i): call(i, t) for t in times for i in range(len(ups))}
    again = {(t, i): call(i, t) for t in reversed(times) for i in reversed(range(len(ups)))}
    bad: dict[int, list[float]] = {}
    for key, value in first.items():
        if _differ(value, again[key], tol):
            bad.setdefault(key[1], []).append(key[0])
    return [
        Issue("impure_updater", f"{ups[i].object_id}: updater {ups[i].ref} gives different "
              f"answers for the same t {_when(sorted(ts))}: it keeps state, or reads something "
              "other than t and look(...)", severity="error", ids=(ups[i].object_id,),
              times=tuple(sorted(ts)), fix_hint="compute everything from t and look(...); "
              "something that needs the previous frame is a Simulation (fixed dt, a seed)")
        for i, ts in sorted(bad.items())
    ]  # fmt: skip


def strict_rules(c: Compiled) -> list[Issue]:
    """Rules for finished scenes only. ``raw_motion``: transitions written with raw ``to()``
    instead of a choreography verb (their timing is not the theme's). ``impure_updater``:
    see ``updater_rules``."""
    raw: dict[str, list[float]] = {}
    for path, segs in c.doc.timeline.tracks.items():
        for seg in segs:
            if seg.verb is None:
                raw.setdefault(path.rpartition(".")[0], []).append(c.time_map.resolve(seg.t0))
    return [
        Issue(
            "raw_motion",
            f"{object_id!r} moves with raw to() transitions {_when(sorted(ts))}",
            "warning",
            (object_id,),
            times=tuple(sorted(ts)),
            fix_hint="use a verb (reveal, trace, morph, settle, focus_pull, ...) so the "
            "theme sets timing and easing",
        )
        for object_id, ts in sorted(raw.items())
    ] + updater_rules(c) + [
        Issue("cue_unanchored", f"cue {p.id!r} is at {p.t:g} s, not on a marker: it will not "
              "follow the narration", "warning", (p.id,), path=f"{p.id}.at", times=(p.t,),
              fix_hint="anchor it: s.cue(\"<marker>\", ..., offset=...)")
        for p in c.cues if not isinstance(p.cue.at, MarkerRef)
    ]  # fmt: skip


def clutter(c: Compiled) -> list[Issue]:
    """Too many objects changing at once inside one block (the span between two markers)."""
    limit = c.theme.lint["clutter"]
    object_ids = {o.id for o in c.objects}
    spans = []  # (t0, t1, object id)
    for path, node in c.nodes.items():
        owner = path.rpartition(".")[0]
        if isinstance(node, ResolvedTrack) and owner in object_ids:
            spans += [(s.t0, s.t1, owner) for s in node.segments]
    edges = sorted({0.0, c.duration, *c.time_map.markers.values()})
    names = {t: n for n, t in sorted(c.time_map.markers.items(), reverse=True)}
    out = []
    for a, b in zip(edges, [*edges[1:], float("inf")], strict=True):
        worst: tuple[int, float, tuple[str, ...]] = (0, 0.0, ())
        for start in sorted({s[0] for s in spans if a <= s[0] < b}):
            busy = tuple(sorted({o for t0, t1, o in spans if t0 <= start < max(t1, t0 + 1e-9)}))
            if len(busy) > worst[0]:
                worst = (len(busy), start, busy)
        if worst[0] > limit:
            block = f"starting at {names[a]!r}" if a in names else f"from {a:g} s"
            block = f"the block {block}"
            out.append(Issue(
                "clutter", f"{worst[0]} objects change at once at t = {worst[1]:g} s in {block} "
                f"(limit {limit:g})", severity="warning", ids=worst[2], times=(worst[1],),
                fix_hint="stagger the transitions or split the block with another marker",
            ))  # fmt: skip
    return out


def run(
    c: Compiled,
    plan: Plan,
    state_at: Callable[[float], SceneState],
    snapshot_at: Callable[[float], dict[str, Any]],
    runner: PassRunner | None,
    *,
    step: float = DEFAULT_STEP,
) -> list[Issue]:
    """Every visual lint issue for the scene. ``state_at``/``snapshot_at`` give the baked
    scene, so the linter judges exactly what will render."""
    th = c.theme.lint
    f = _Findings()
    ever_shown: set[str] = set()
    solved_on_screen = {g.ids[0] for _, g in plan.guards("on_screen")} & plan.solved

    for t in sample_times(c, step):
        state = state_at(t)
        fr = _Frame(state, snapshot_at(t), runner, c)
        height = state.height
        shown = [e for e in fr.entries.values() if _shown(e)]
        ever_shown.update(e["id"] for e in shown)
        # A Value is a number with nothing to draw (ADR 0005): never "missing" from the frame.
        ever_shown.update(
            e["id"] for e in fr.entries.values() if e["semantic"].get("kind") == "value"
        )
        labels = [e for e in shown if _is_label(e)]
        persp = state.camera.perspective

        def both(a, b, kind) -> bool:
            return a["semantic"].get("kind") == kind and b["semantic"].get("kind") == kind

        seen_pairs = set()
        for a in shown:
            if not (_is_label(a) or "key" in a["tags"]):
                continue
            for b in shown:
                pair = tuple(sorted((a["id"], b["id"])))
                if a is b or pair in seen_pairs or _part_of(fr.entries, a, b):
                    continue
                if persp and not (both(a, b, "label") or both(a, b, "point")):
                    continue  # depth decides in 3D: see occluded_label
                if a["semantic"].get("backdrop") or b["semantic"].get("backdrop"):
                    continue  # labels are meant to sit on a backdrop (a plane, a fractal)
                if _is_label(a) and not _is_label(b):
                    share = fr.covered(a["id"], b["id"])
                elif _is_label(b) and not _is_label(a):
                    share = fr.covered(b["id"], a["id"])
                elif _is_label(a) or "key" in b["tags"]:
                    share = fr.shared(a["id"], b["id"])
                else:
                    continue
                if th["overlap"] < share:
                    seen_pairs.add(pair)
                    if share <= CONTAINED or (_is_label(a) and _is_label(b)):
                        f.add("overlap", pair, t)

        for e in shown:
            important = _is_label(e) or "key" in e["tags"]
            if important and e["on_screen_fraction"] < 0.999:
                f.add("offscreen", (e["id"],), t)
            x0, y0, x1, y1 = e["bbox"]["px"]
            if _is_label(e):
                if (y1 - y0) < th["text_min"] * height:
                    f.add("tiny", (e["id"],), t, px=y1 - y0, what="text")
            elif max(x1 - x0, y1 - y0) < th["tiny"] * height:
                f.add("tiny", (e["id"],), t, px=max(x1 - x0, y1 - y0), what="object")
        if persp:
            for e in labels:
                v = e["visible_fraction"]
                if e["layer"] != "overlay" and v is not None and v < th["occluded"]:
                    by = e["hidden_by"][0]["id"] if e["hidden_by"] else None
                    f.add("occluded_label", (e["id"],), t, visible=v, by=by)
        for e in labels:
            missing = text_metrics.missing_glyphs(e["semantic"].get("text", ""))
            if missing:
                f.add("missing_glyph", (e["id"],), t, chars="".join(missing))
            ratio = fr.contrast(e["id"])
            if ratio < th["contrast"]:
                f.add("contrast", (e["id"],), t, ratio=ratio)

        # Guards: explicit demands. Breaking one is an error, reported as "unsolved".
        for rule, g in plan.guards():
            es = [fr.entries[i] for i in g.ids]
            if any(e["opacity"] <= FAINT for e in es):
                continue
            if g.kind == "no_overlap":
                for i, a in enumerate(g.ids):
                    for b in g.ids[i + 1 :]:
                        if fr.shared(a, b) > th["overlap"]:
                            f.add("unsolved", (a, b), t, path=rule.path, rule=rule.spec.rule,
                                  why=f"{a!r} and {b!r} overlap")  # fmt: skip
            elif g.kind == "on_screen" and g.ids[0] not in solved_on_screen:
                unit = view_of(state, state[g.ids[0]]).units_per_height
                m = (c.theme.layout["margin"] if g.value is None else g.value / unit) * height
                x0, y0, x1, y1 = es[0]["bbox"]["px"]
                if min(x0, y0, state.width - x1, height - y1) < m - 0.5:
                    f.add("unsolved", g.ids, t, path=rule.path, rule=rule.spec.rule,
                          why=f"{g.ids[0]!r} crosses the frame margin")  # fmt: skip
            elif g.kind == "min_size_px":
                x0, y0, x1, y1 = es[0]["bbox"]["px"]
                if max(x1 - x0, y1 - y0) < g.value - 0.5:
                    side = max(x1 - x0, y1 - y0)
                    f.add("unsolved", g.ids, t, path=rule.path, rule=rule.spec.rule,
                          why=f"{g.ids[0]!r} is {side:.0f} px, under {g.value:g}")  # fmt: skip
            elif g.kind == "visible_in_3d":
                v = es[0]["visible_fraction"]
                if v is not None and es[0]["on_screen_fraction"] > 0 and v < th["occluded"]:
                    by = es[0]["hidden_by"][0]["id"] if es[0]["hidden_by"] else "something"
                    f.add("unsolved", g.ids, t, path=rule.path, rule=rule.spec.rule,
                          why=f"{g.ids[0]!r} is only {v:.0%} visible, behind {by!r}")  # fmt: skip
            elif g.kind == "readable" and "color" in state[g.ids[0]].style:
                ratio = fr.contrast(g.ids[0])
                if ratio < th["contrast"]:
                    f.add("unsolved", g.ids, t, path=rule.path, rule=rule.spec.rule,
                          why=f"{g.ids[0]!r} has contrast {ratio:.1f} (minimum "
                          f"{th['contrast']:g})")  # fmt: skip

    def make(code: str, ids: tuple[str, ...], times: list[float], **info: Any) -> Issue:
        when, ts = _when(times), tuple(times)
        if code == "overlap":
            fix = tuple(op for i in ids for op in _near_fix(c, plan, i))
            stuck = [i for i in ids if i in plan.placed]
            hint = (
                "apply the suggested fix (a 'near' constraint lets the placer find free space)"
                if fix else
                f"the placer found no free slot for {stuck[0]!r}; move or shrink a neighbor"
                if stuck else
                "move one of them, or name the label '<owner>.label' and add the constraint "
                "'label(<owner>) near <owner>'"
            )  # fmt: skip
            return Issue(code, f"{ids[0]!r} and {ids[1]!r} overlap {when}", "warning", ids,
                         fix_hint=hint, times=ts, fix=fix)  # fmt: skip
        if code == "offscreen":
            (i,) = ids
            can = _movable(c, i) and i not in plan.moving
            fix = ({"op": "add_constraint", "rule": f"on_screen({i})"},) if can else ()
            hint = "apply the suggested fix (on_screen)" if fix else "move it inward"
            return Issue(code, f"{i!r} is cut by the frame {when}", "warning", ids,
                         fix_hint=hint, times=ts, fix=fix)  # fmt: skip
        if code == "contrast":
            return Issue(code, f"{ids[0]!r} has contrast {info['ratio']:.1f} against what is "
                         f"behind it (minimum {th['contrast']:g}) {when}", "warning", ids,
                         fix_hint="use a brighter color role such as 'primary' or 'highlight', "
                         "raise its opacity, or move it", times=ts)  # fmt: skip
        if code == "missing_glyph":
            shown = ", ".join(repr(ch) for ch in info["chars"])
            return Issue(code, f"{ids[0]!r} uses characters the font cannot draw ({shown}); "
                         f"they show as empty boxes {when}", "warning", ids, times=ts,
                         fix_hint="write them in ASCII for now (z^2, pi), or use a math "
                         "label once LaTeX arrives")  # fmt: skip
        if code == "occluded_label":
            by = f" behind {info['by']!r}" if info.get("by") else ""
            return Issue(code, f"label {ids[0]!r} is hidden{by} (only {info['visible']:.0%} "
                         f"visible) {when}", "warning", ids, times=ts,
                         fix_hint="give it a 'near' constraint so the placer moves it, add the "
                         f"camera goal 'camera avoid_occlusion({ids[0]})', or make the "
                         "occluder glass")  # fmt: skip
        if code == "tiny":
            return Issue(code, f"{info['what']} {ids[0]!r} is only {info['px']:.0f} px "
                         f"{when}", "warning", ids,
                         fix_hint="use a larger size role, or zoom in", times=ts)  # fmt: skip
        return Issue("unsolved", f"constraint {info['rule']!r} does not hold: {info['why']} "
                     f"{when}", "error", ids, path=info["path"], times=ts,
                     fix_hint="make room, or change the rule")  # fmt: skip

    issues = f.issues(make)
    for o in c.objects:
        if o.id not in ever_shown:
            issues.append(Issue(
                "orphan", f"{o.id!r} never appears: it is invisible or off screen at every "
                "sampled time", "warning", (o.id,),
                fix_hint="animate its opacity or position into view, or remove it",
            ))  # fmt: skip
    return issues + clutter(c)
