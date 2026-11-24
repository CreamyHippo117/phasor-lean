"""Aesthetic lint rules: is it beautiful, not just correct.

Switched on with ``lint(aesthetic=True)`` (CLI ``--aesthetic``, MCP ``review``); the
ordinary linter stays about problems, these are about taste. Every threshold is the theme's
``[lint]`` entry, so your taste can move them.

| code            | fires when                                                              |
| focal_point     | a beat has no highlighted object, or more than one                      |
| color_restraint | more saturated hues on screen than the theme allows (the phase wheel    |
|                 | and the magnitude ramp are exempt: there, hue carries meaning)          |
| glow_clip       | too many pixels blow out to flat white (needs a renderer)               |
| density         | the content fills too little or too much of the frame                   |
| motion_rhythm   | unrelated objects start different motions within the same short window  |
| margin          | content crowds the outer safe margin of the frame                       |
| stroke_mix      | more stroke weights on screen than the theme allows                     |

A "beat" is a marker (or, with no markers, an evenly spaced sample). A fix is offered only
where it is certain to help: a margin rule, one stroke weight, one glow notch down.
"""

from __future__ import annotations

import colorsys
from collections.abc import Callable
from typing import Any

import numpy as np

from phasor.core.issues import Issue
from phasor.core.state import ObjectState, SceneState
from phasor.inspect.lint import _movable, _shown, _when, sample_times
from phasor.inspect.passes import PassRunner, inspect_size
from phasor.layout.geometry import footprint, is_backdrop, view_of
from phasor.layout.plan import Plan
from phasor.time.compile import Compiled
from phasor.time.tracks import ResolvedTrack

MAX_BEATS = 24
SATURATED = 0.25  # HSV saturation (sRGB) above which a color counts as a hue
HUE_BINS = 6  # 60-degree bins: "red-ish", "yellow-ish", ...
LUT_SHARE = 0.1  # a colormap contributes a hue when it spends this share of its length in it
RHYTHM_WINDOW_VERBS = "different motions"
# Light more than this many times plain paint is beyond what the tone-map shoulder can hold:
# it reaches the screen as flat white. (Softer glow only brightens.)
WHITE_OUT = 3.0


def beats(c: Compiled) -> tuple[float, ...]:
    """The moments judged: every marker, or evenly spaced samples when there are none."""
    marks = sorted({round(t, 6) for t in c.time_map.markers.values() if 0 <= t <= c.duration})
    if marks:
        return tuple(marks)
    ts = sample_times(c)
    if len(ts) <= MAX_BEATS:
        return ts
    return tuple(ts[round(i)] for i in np.linspace(0, len(ts) - 1, MAX_BEATS))


def _hue_bin(rgb_linear) -> int | None:
    # An analysis, not an encode (sRGB is encoded once, in render/output.py): a square root
    # is close enough to perceptual lightness to tell a saturated hue from a gray.
    r, g, b = (float(x) for x in np.sqrt(np.clip(np.asarray(rgb_linear[:3]), 0.0, 1.0)))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(h * HUE_BINS) % HUE_BINS if s >= SATURATED and v >= 0.2 else None


def _lut_hues(lut) -> set[int]:
    bins = [_hue_bin(c) for c in lut]
    return {b for b in set(bins) - {None} if bins.count(b) >= LUT_SHARE * len(bins)}  # type: ignore[arg-type]


# Colormaps whose hue is read as a quantity: a function's phase, a field's strength.
MEANINGFUL = ("phase", "magnitude")


def hues_on_screen(state: SceneState, shown: set[str]) -> set[int]:
    """Saturated hues of what is drawn: paint colors, and the colors of ramps and restrained
    wheels. The phase wheel and the magnitude ramp are left out: there, hue is how a
    function's phase or a field's strength is *read*."""
    hues: set[int] = set()
    for obj in state.objects:
        if obj.id not in shown or obj.params.opacity < 0.1:
            continue
        cm = obj.style.get("colormap")
        if getattr(obj.params, "hue_along", None) == 0:
            cm = None  # a line that does not run a hue along itself shows its paint color
        if getattr(obj.params, "color_by_size", None) is False:
            cm = None  # arrows not colored by size show their paint color
        if cm is not None:
            if obj.roles.get("colormap") not in MEANINGFUL:
                hues |= _lut_hues(cm)
            continue
        color = obj.style.get("color")
        if color is not None and obj.impl.semantics(state).get("kind") != "label":
            b = _hue_bin(color)
            if b is not None:
                hues.add(b)
    return hues


def _content(state: SceneState, shown: set[str]) -> list[ObjectState]:
    return [o for o in state.objects if o.id in shown and not is_backdrop(state, o)]


def _occupied(state: SceneState, runner: PassRunner | None, content: list[ObjectState]) -> float:
    size = inspect_size(state)
    if runner is not None:
        import dataclasses

        only = dataclasses.replace(state, objects=tuple(content))
        return float((runner.id_buffer(only, size) != 0).mean())
    mask = np.zeros(size[::-1], dtype=bool)
    for o in content:
        mask |= footprint(state, o, size)
    return float(mask.mean())


def rules(
    c: Compiled,
    plan: Plan,
    state_at: Callable[[float], SceneState],
    snapshot_at: Callable[[float], dict[str, Any]],
    runner: PassRunner | None,
) -> list[Issue]:
    th = c.theme.lint
    found: dict[tuple[str, tuple[str, ...], str], dict[str, Any]] = {}

    def add(code: str, ids: tuple[str, ...], t: float, variant: str = "", **info: Any) -> None:
        entry = found.setdefault((code, ids, variant), {"times": [], **info})
        entry["times"].append(t)

    for t in beats(c):
        state = state_at(t)
        snap = snapshot_at(t)
        entries = {e["id"]: e for e in snap["objects"]}
        shown = {i for i, e in entries.items() if _shown(e)}
        content = _content(state, shown)
        non_label = [
            o for o in content if o.impl.semantics(state).get("kind") != "label"
        ]  # labels annotate; the picture is the rest
        if not content:
            continue  # nothing yet: a blank beat is not "empty", it is not there yet

        # focal_point: exactly one thing the eye should land on
        highlighted = tuple(o.id for o in non_label if o.roles.get("color") == "highlight")
        if non_label and len(highlighted) != 1:
            if not highlighted:
                add("focal_point", (), t, "none")
            else:
                add("focal_point", highlighted, t, "many")

        # color_restraint
        n_hues = len(hues_on_screen(state, shown))
        if n_hues > th["hues"]:
            add("color_restraint", (), t, hues=n_hues)

        # density
        share = _occupied(state, runner, content)
        if share < th["density_min"]:
            add("density", (), t, "empty", share=share)
        elif share > th["density_max"]:
            add("density", (), t, "crowded", share=share)

        # glow_clip: light far beyond what the tone map can hold, before bloom (needs pixels)
        if runner is not None:
            pic = runner.color(state, inspect_size(state))
            clipped = float((pic[..., :3].max(axis=-1) > WHITE_OUT).mean())
            if clipped > th["clip"]:
                strong = tuple(o.id for o in content if o.roles.get("glow") == "strong")
                add("glow_clip", strong, t, share=clipped)

        # margin
        m, a = th["margin"], state.aspect  # the margin is in frame heights, like layout's
        for o in content:
            e = entries[o.id]
            if e["on_screen_fraction"] < 0.999:
                continue  # cut by the frame: that is `offscreen`, a bigger problem
            x0, y0, x1, y1 = e["bbox"]["norm"]
            m_ = m - 1e-3  # a solved margin lands exactly on it
            if x0 * a < m_ or y0 < m_ or (1 - x1) * a < m_ or 1 - y1 < m_:
                add("margin", (o.id,), t, unit=view_of(state, o).units_per_height)

        # stroke_mix
        weights = {o.id: o.roles["stroke"] for o in content if "stroke" in o.roles}
        if len(set(weights.values())) > th["stroke_mix"]:
            add("stroke_mix", tuple(sorted(weights)), t, weights=weights)

    issues = [
        _make(c, plan, th, code, ids, variant, info)
        for (code, ids, variant), info in sorted(
            found.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
        )
    ]
    return issues + motion_rhythm(c)


def _make(c: Compiled, plan: Plan, th, code: str, ids: tuple[str, ...], variant: str, info):
    ts = tuple(info["times"])
    when = _when(list(ts))
    if code == "focal_point":
        if variant == "none":
            return Issue(
                code,
                f"nothing is highlighted {when}: the eye has nowhere to land",
                "warning",
                ids,
                times=ts,
                fix_hint="give the one thing that matters color role 'highlight'",
            )
        return Issue(
            code,
            f"{len(ids)} objects are highlighted at once {when}: {', '.join(ids)}",
            "warning",
            ids,
            times=ts,
            fix_hint="keep 'highlight' on one; give the rest 'primary' or 'secondary'",
        )
    if code == "color_restraint":
        return Issue(
            code,
            f"{info['hues']} saturated hues on screen {when} (limit {th['hues']:g})",
            "warning",
            ids,
            times=ts,
            fix_hint="reduce to two or three hues: use theme roles 'primary', "
            "'secondary', 'muted' and one 'highlight', or a restrained colormap "
            "('soft_phase', 'ember', 'ocean')",
        )
    if code == "glow_clip":
        fix = tuple({"op": "set_style", "id": i, "key": "glow", "role": "soft"} for i in ids)
        return Issue(
            code,
            f"{info['share']:.1%} of the frame blows out to white {when} (limit {th['clip']:.1%})",
            "warning",
            ids,
            times=ts,
            fix_hint="lower a glow role from 'strong' to 'soft'",
            fix=fix,
        )
    if code == "density":
        what = "empty" if variant == "empty" else "crowded"
        bound = th["density_min"] if variant == "empty" else th["density_max"]
        return Issue(
            code,
            f"the frame is {what}: content covers {info['share']:.0%} of it "
            f"{when} (limit {bound:.0%})",
            "warning",
            ids,
            times=ts,
            fix_hint="add the missing element, or zoom the camera in"
            if variant == "empty"
            else "remove or dim something, or zoom out",
        )
    if code == "margin":
        (i,) = ids
        fix = ()
        if _movable(c, i) and i not in plan.moving:
            value = round(th["margin"] * info["unit"], 4)
            fix = ({"op": "add_constraint", "rule": f"on_screen({i}, {value:g})"},)
        return Issue(
            code,
            f"{i!r} crowds the frame edge {when} (keep {th['margin']:.0%} clear)",
            "warning",
            ids,
            times=ts,
            fix=fix,
            fix_hint="apply the fix (an on_screen rule with the theme's margin)"
            if fix
            else "move it inward",
        )
    # stroke_mix
    weights = info["weights"]
    keep = max(sorted(set(weights.values())), key=list(weights.values()).count)
    fix = tuple(
        {"op": "set_style", "id": i, "key": "stroke", "role": keep}
        for i, r in weights.items()
        if r != keep
    )
    return Issue(
        code,
        f"{len(set(weights.values()))} stroke weights on screen {when} "
        f"(limit {th['stroke_mix']:g})",
        "warning",
        ids,
        times=ts,
        fix=fix,
        fix_hint=f"use one weight ({keep!r}) for the plain strokes",
    )


def motion_rhythm(c: Compiled) -> list[Issue]:
    """Unrelated objects starting different motions inside one short window read as noise.
    Objects moved by the same verb at the same anchor are one gesture and are left alone;
    an object and its children (``pole`` and ``pole.label``) are related."""
    window = c.theme.lint["rhythm"]
    object_ids = {o.id for o in c.objects}
    starts: list[tuple[float, str, str]] = []  # (t, root id, verb or "raw")
    for path, node in c.nodes.items():
        owner = path.rpartition(".")[0]
        if not (isinstance(node, ResolvedTrack) and owner in object_ids):
            continue
        for seg, resolved in zip(c.doc.timeline.tracks.get(path, ()), node.segments, strict=False):
            starts.append((resolved.t0, owner.split(".")[0], seg.verb or "raw"))
    starts.sort()
    pairs: dict[tuple[str, str], list[float]] = {}
    for i, (t, root, verb) in enumerate(starts):
        for t2, root2, verb2 in starts[i + 1 :]:
            if t2 - t > window:
                break
            if root != root2 and verb != verb2:
                pairs.setdefault(tuple(sorted((root, root2))), []).append(t)
    return [
        Issue(
            "motion_rhythm",
            f"{a!r} and {b!r} start different motions within {window:g} s of "
            f"each other {_when(sorted(set(ts)))}",
            "warning",
            (a, b),
            times=tuple(sorted(set(ts))),
            fix_hint="start one of them later (another marker, or `.then()`), or give both "
            "the same verb",
        )
        for (a, b), ts in sorted(pairs.items())
    ]
