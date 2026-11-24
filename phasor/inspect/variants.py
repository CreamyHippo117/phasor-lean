"""Variants of a beat: cheap alternatives to look at side by side.

A variant is a named list of patch operations over the current scene. They are tried on a
throwaway copy, so the scene itself never changes until somebody picks one.
"""

from __future__ import annotations

from typing import Any

from phasor.core.schema import Document

GLOW_UP = {"none": "soft", "soft": "strong"}
GLOW_DOWN = {"strong": "soft", "soft": "none"}
STROKE_UP = {"hairline": "regular", "regular": "bold"}


def _style_ops(doc: Document, key: str, mapping: dict[str, str]) -> list[dict[str, Any]]:
    ops = []
    for o in doc.objects:
        role = o.style.get(key)
        if role in mapping:
            ops.append({"op": "set_style", "id": o.id, "key": key, "role": mapping[role]})
    return ops


def make(doc: Document, n: int = 3) -> list[tuple[str, list[dict[str, Any]]]]:
    """Up to ``n`` named variants of ``doc``: as it is, calmer, bolder (in that order). Each
    changes only what a viewer reads as mood: glow, stroke weight, contour softness."""
    calmer = _style_ops(doc, "glow", GLOW_DOWN)
    bolder = _style_ops(doc, "glow", GLOW_UP) + _style_ops(doc, "stroke", STROKE_UP)
    for o in doc.objects:
        if o.type == "domain_coloring":
            calmer.append({"op": "set_param", "path": f"{o.id}.softness", "value": 0.85})
            bolder.append({"op": "set_param", "path": f"{o.id}.softness", "value": 0.2})
    return [("as_is", []), ("calmer", calmer), ("bolder", bolder)][: max(1, n)]
