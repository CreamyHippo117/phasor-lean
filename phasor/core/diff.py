"""A structured difference between two documents (for ``phasor`` clients: AI, GUI, CLI).

Pure over ``Document``: it says *what* changed in terms of ids and parameter paths, not
which bytes moved. Nothing here writes; every change still goes through ``Engine.apply``.
"""

from __future__ import annotations

from typing import Any

from phasor.core.schema import Document


def _plain(v: Any) -> Any:
    if hasattr(v, "model_dump"):
        return v.model_dump(mode="json")
    if isinstance(v, tuple | list):
        return [_plain(x) for x in v]
    return v


def diff_documents(a: Document, b: Document) -> dict[str, Any]:
    """What changed going from ``a`` to ``b``: ``{"same", "summary", "changes": [...]}``.
    Each change has a ``kind`` and the ids or paths involved; order is stable."""
    out: list[dict[str, Any]] = []
    ao, bo = {o.id: o for o in a.objects}, {o.id: o for o in b.objects}
    for i in sorted(bo.keys() - ao.keys()):
        out.append({"kind": "object_added", "id": i, "type": bo[i].type})
    for i in sorted(ao.keys() - bo.keys()):
        out.append({"kind": "object_removed", "id": i, "type": ao[i].type})
    for i in sorted(ao.keys() & bo.keys()):
        x, y = ao[i], bo[i]
        if x.type != y.type:
            out.append({"kind": "type_changed", "id": i, "from": x.type, "to": y.type})
        for k in sorted(x.params.keys() | y.params.keys()):
            p, q = _plain(x.params.get(k)), _plain(y.params.get(k))
            if p != q:
                out.append(
                    {"kind": "param_changed", "id": i, "path": f"{i}.{k}", "from": p, "to": q}
                )
        if x.style != y.style:
            out.append(
                {"kind": "style_changed", "id": i, "from": dict(x.style), "to": dict(y.style)}
            )
        if x.tags != y.tags:
            out.append({"kind": "tags_changed", "id": i, "from": list(x.tags), "to": list(y.tags)})
        if x.layer != y.layer:
            out.append({"kind": "layer_changed", "id": i, "from": x.layer, "to": y.layer})
    am, bm = a.timeline.markers, b.timeline.markers
    for n in sorted(bm.keys() - am.keys()):
        out.append({"kind": "marker_added", "name": n, "t": bm[n]})
    for n in sorted(am.keys() - bm.keys()):
        out.append({"kind": "marker_removed", "name": n, "t": am[n]})
    for n in sorted(am.keys() & bm.keys()):
        if am[n] != bm[n]:
            out.append({"kind": "marker_moved", "name": n, "from": am[n], "to": bm[n]})
    at, bt = a.timeline.tracks, b.timeline.tracks
    for p in sorted(bt.keys() - at.keys()):
        out.append({"kind": "track_added", "path": p, "segments": len(bt[p])})
    for p in sorted(at.keys() - bt.keys()):
        out.append({"kind": "track_removed", "path": p, "segments": len(at[p])})
    for p in sorted(at.keys() & bt.keys()):
        if at[p] != bt[p]:
            out.append({"kind": "track_changed", "path": p, "from": len(at[p]), "to": len(bt[p])})

    def sounds(d: Document) -> dict[str, tuple[str, Any]]:
        beds = {x.id: ("bed", x) for x in d.sound.beds}
        return {**beds, **{x.id: ("cue", x) for x in d.sound.cues}}

    asnd, bsnd = sounds(a), sounds(b)
    for i in sorted(bsnd.keys() - asnd.keys()):
        out.append({"kind": "sound_added", "id": i, "sound": bsnd[i][0]})
    for i in sorted(asnd.keys() - bsnd.keys()):
        out.append({"kind": "sound_removed", "id": i, "sound": asnd[i][0]})
    for i in sorted(asnd.keys() & bsnd.keys()):
        if asnd[i] != bsnd[i]:
            out.append({"kind": "sound_changed", "id": i, "from": _plain(asnd[i][1]),
                        "to": _plain(bsnd[i][1])})  # fmt: skip
    ac = {c.rule: c.priority for c in a.constraints}
    bc = {c.rule: c.priority for c in b.constraints}
    for rule in sorted(bc.keys() - ac.keys()):
        out.append({"kind": "constraint_added", "rule": rule, "priority": bc[rule]})
    for rule in sorted(ac.keys() - bc.keys()):
        out.append({"kind": "constraint_removed", "rule": rule, "priority": ac[rule]})
    for rule in sorted(ac.keys() & bc.keys()):
        if ac[rule] != bc[rule]:
            out.append({"kind": "priority_changed", "rule": rule, "from": ac[rule], "to": bc[rule]})
    akeys, bkeys = {c.key: c for c in a.controls}, {c.key: c for c in b.controls}
    for k in sorted(bkeys.keys() - akeys.keys()):
        out.append(
            {"kind": "control_added", "control": k, "min": bkeys[k].min, "max": bkeys[k].max}
        )
    for k in sorted(akeys.keys() - bkeys.keys()):
        out.append({"kind": "control_removed", "control": k})
    for k in sorted(akeys.keys() & bkeys.keys()):
        if akeys[k] != bkeys[k]:
            out.append({"kind": "control_changed", "control": k})
    for i in sorted(set(b.retired_ids) - set(a.retired_ids)):
        out.append({"kind": "id_retired", "id": i})
    if a.timeline.camera != b.timeline.camera:
        out.append({"kind": "active_camera_changed", "from": a.timeline.camera,
                    "to": b.timeline.camera})  # fmt: skip
    if a.meta != b.meta:
        for k in sorted(type(a.meta).model_fields):
            if getattr(a.meta, k) != getattr(b.meta, k):
                out.append({"kind": "meta_changed", "field": k,
                            "from": getattr(a.meta, k), "to": getattr(b.meta, k)})  # fmt: skip
    if a.narration != b.narration:
        out.append({"kind": "narration_changed"})
    if a.cameras != b.cameras:
        out.append({"kind": "camera_changed"})
    return {"same": not out, "summary": _summary(out), "changes": out}


def _summary(changes: list[dict[str, Any]]) -> str:
    if not changes:
        return "no differences"
    counts: dict[str, int] = {}
    for c in changes:
        counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    return ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(counts.items()))
