"""The art-director pass: a score for taste, and what to do next.

The deterministic half of the critic. It turns the aesthetic lint rules into one number
(0 is nothing to criticise) and a list of issues tied to object ids and the marker being
spoken, each with the operations that would fix it. Operations come in two kinds:

- ``fix``: certain (a margin rule, one stroke weight, a glow notch down);
- ``suggested``: judgement (which object deserves the highlight). A builder, human or AI,
  may take them or choose better; the picture and ``STYLE.md`` decide, this only proposes.

A vision model reads the storyboard against the rubric for what no rule can see; this is the
part that never gets tired and can be measured, so a "polish" loop can be *shown* to improve a
scene.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from phasor.core.issues import Issue
from phasor.inspect import aesthetic
from phasor.time.compile import Compiled

# How much each rule costs when it fires at every beat (scaled by the share of beats hit).
WEIGHTS = {
    "focal_point": 3.0,
    "color_restraint": 2.0,
    "glow_clip": 2.0,
    "density": 1.5,
    "motion_rhythm": 1.0,
    "margin": 1.0,
    "stroke_mix": 1.0,
}


def score(c: Compiled, issues: list[Issue]) -> float:
    """0 = nothing to criticise; each aesthetic issue costs its weight times the share of
    beats it appears at (motion_rhythm: a single occurrence counts as one beat)."""
    n = max(len(aesthetic.beats(c)), 1)
    total = 0.0
    for i in issues:
        w = WEIGHTS.get(i.code)
        if w is not None:
            total += w * min(len(i.times) / n, 1.0) if i.times else w
    return round(total, 3)


def _nearest_marker(c: Compiled, t: float) -> str | None:
    before = [(m, n) for n, m in c.time_map.markers.items() if m <= t + 1e-9]
    return max(before)[1] if before else None


def suggestions(
    c: Compiled,
    snapshot_at: Callable[[float], dict[str, Any]],
    issue: Issue,
) -> tuple[dict[str, Any], ...]:
    """Judgement fixes for an issue that has no certain one."""
    if issue.fix or issue.code != "focal_point" or not issue.times:
        return ()
    snap = snapshot_at(issue.times[0])
    entries = {e["id"]: e for e in snap["objects"]}
    shown = [
        e for e in entries.values()
        if e["opacity"] > 0.05 and e["semantic"].get("kind") not in ("label", None)
        and not e["semantic"].get("backdrop")
    ]  # fmt: skip

    def weight(e):  # tagged key things first, then the biggest on screen
        x0, y0, x1, y1 = e["bbox"]["norm"]
        return ("key" in e["tags"], (x1 - x0) * (y1 - y0))

    if issue.ids:  # several highlighted: keep the most important, calm the others
        keep = max((entries[i] for i in issue.ids if i in entries), key=weight)["id"]
        return tuple(
            {"op": "set_style", "id": i, "key": "color", "role": "primary"}
            for i in issue.ids if i != keep
        )  # fmt: skip
    # None highlighted at this beat. If something is highlighted elsewhere in the scene this
    # is probably "the story has not started yet", which a heuristic must not paper over;
    # only a scene with no highlight anywhere gets one proposed.
    if shown and not any(o.style.get("color") == "highlight" for o in c.doc.objects):
        pick = max(shown, key=weight)["id"]
        return ({"op": "set_style", "id": pick, "key": "color", "role": "highlight"},)
    return ()


def critique(
    c: Compiled,
    issues: list[Issue],
    snapshot_at: Callable[[float], dict[str, Any]],
) -> dict[str, Any]:
    """Aesthetic issues with markers and fixes, and the score."""
    found = [i for i in issues if i.code in WEIGHTS]
    out = []
    for i in found:
        d = i.to_dict()
        d["markers"] = sorted({m for t in i.times if (m := _nearest_marker(c, t))})
        d["suggested"] = list(suggestions(c, snapshot_at, i))
        out.append(d)
    return {"score": score(c, found), "beats": len(aesthetic.beats(c)), "issues": out}
