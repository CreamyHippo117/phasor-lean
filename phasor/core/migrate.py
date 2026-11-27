"""Schema migrations. Each step upgrades a raw dict by exactly one version."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from phasor.core.issues import DocumentError, Issue
from phasor.core.schema import SCHEMA_VERSION


# from_version -> function returning the document at from_version + 1
def _v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 2 added ``constraints`` (layout rules)."""
    raw.setdefault("constraints", [])
    return raw


def _v2_to_v3(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 3 added ``retired_ids`` (ids of removed objects, never to be reused)."""
    raw.setdefault("retired_ids", [])
    return raw


def _v3_to_v4(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 4 added ``narration`` and ``Segment.verb``; both default to none."""
    raw.setdefault("narration", None)
    return raw


def _v4_to_v5(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 5 added ``meta.life`` (living motion), ``meta.seed`` and ``meta.look``. Older
    documents keep their stillness: life is off."""
    meta = dict(raw.get("meta", {}))
    meta.setdefault("life", "off")
    meta.setdefault("seed", 0)
    meta.setdefault("look", None)
    raw["meta"] = meta
    return raw


def _v5_to_v6(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 6 added ``controls`` (sliders). Older documents have none."""
    raw.setdefault("controls", [])
    return raw


def _v6_to_v7(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 7 added ``meta.palette`` (the colors run along strokes). None: the theme's."""
    raw.setdefault("meta", {}).setdefault("palette", None)
    return raw


def _v7_to_v8(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 8 added ``meta.modules`` (code the scene needs, ADR 0005) and per-object
    ``always`` (its updaters). Older documents need no code and have no updaters."""
    raw.setdefault("meta", {}).setdefault("modules", [])
    raw["objects"] = [{**o, "always": o.get("always", [])} for o in raw.get("objects", [])]
    return raw


def _v8_to_v9(raw: dict[str, Any]) -> dict[str, Any]:
    """Version 9 added the ``sound`` section (beds and cues beside the narration).
    Older documents have no sound apart from their narration."""
    raw.setdefault("sound", {"beds": [], "cues": []})
    return raw


MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _v1_to_v2,
    2: _v2_to_v3,
    3: _v3_to_v4,
    4: _v4_to_v5,
    5: _v5_to_v6,
    6: _v6_to_v7,
    7: _v7_to_v8,
    8: _v8_to_v9,
}


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    version = raw.get("schema_version")
    if not isinstance(version, int) or version < 1:
        raise DocumentError(Issue("schema_version", f"missing or bad schema_version: {version!r}"))
    if version > SCHEMA_VERSION:
        raise DocumentError(
            Issue(
                "schema_version",
                f"document is version {version}, this phasor reads up to {SCHEMA_VERSION}",
                fix_hint="upgrade phasor",
            )
        )
    while version < SCHEMA_VERSION:
        raw = MIGRATIONS[version](dict(raw))
        version += 1
        raw["schema_version"] = version
    return raw
