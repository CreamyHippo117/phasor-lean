"""Look presets: a theme plus a handful of choices, by name.

``Scene("x", look="calm")`` starts from a look: it picks the theme and the scene's ``life``
and supplies defaults for object parameters and style roles. A look is only named defaults:
anything a script or patch sets explicitly wins, and the document stores the resolved
values (``meta.theme``, ``meta.life`` and each object's own params), so it renders the same
later even if the look file changes. Files live in ``style/looks/*.toml``.
"""

from __future__ import annotations

import functools
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phasor.core.life import LEVELS
from phasor.style.theme import ThemeError, available_themes

LOOK_DIR = Path(__file__).parent / "looks"
DEFAULT_THEME = "midnight_glow"  # the theme when a scene has neither a look nor a theme
# The look a new scene gets when it names neither a look nor a theme (the
# prism look is the default style). ``look="none"`` opts out; PHASOR_DEFAULT_LOOK overrides
# it ("none" for plain; the test suite runs plain so its pictures stay what they check).
DEFAULT_LOOK = "prism"


def default_look(look: str | None, theme: str | None = None) -> str | None:
    """The look a new scene uses: ``look`` as given ("none": no look); with no look and no
    theme, the default look; with a theme but no look, none (the theme was a choice)."""
    if look is not None:
        return None if look in ("", "none") else look
    if theme is not None:
        return None
    env = os.environ.get("PHASOR_DEFAULT_LOOK", DEFAULT_LOOK)
    return None if env in ("", "none") else env


class LookError(ValueError):
    pass


@dataclass(frozen=True)
class Look:
    name: str
    theme: str
    life: str
    description: str = ""
    # object type -> {"params": {...}, "style": {...}}; filled in only where a scene says nothing
    defaults: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def fill(self, type_name: str, params: dict[str, Any], style: dict[str, str],
             palette: str | None = None):  # fmt: skip
        """``params`` and ``style`` with this look's defaults for ``type_name`` added where
        the object did not set them. With a ``palette``, what the look runs a hue along (its
        lines) takes the palette's colors."""
        d = self.defaults.get(type_name, {})
        d_style = dict(d.get("style", {}))
        if palette is not None and ("colormap" in d_style or "hue_along" in d.get("params", {})):
            d_style["colormap"] = palette
        return {**d.get("params", {}), **params}, {**d_style, **style}

    def palette_targets(self) -> set[str]:
        """The object types whose colors a palette sets (the ones the look runs a hue along)."""
        return {
            t
            for t, d in self.defaults.items()
            if "colormap" in d.get("style", {}) or "hue_along" in d.get("params", {})
        }


def available_looks() -> list[str]:
    return sorted(p.stem for p in LOOK_DIR.glob("*.toml"))


@functools.cache
def load_look(name: str) -> Look:
    path = LOOK_DIR / f"{name}.toml"
    if not path.exists():
        raise LookError(f"unknown look {name!r}; available: {available_looks()}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    # ``base = "prism"``: start from that look; this file changes only what it names.
    base = load_look(raw["base"]) if "base" in raw else None
    theme = raw.get("theme", base.theme if base else DEFAULT_THEME)
    life = raw.get("life", base.life if base else "off")
    if theme not in available_themes():
        raise ThemeError(f"look {name!r} names unknown theme {theme!r}")
    if life not in LEVELS:
        raise LookError(f"look {name!r}: life must be one of {LEVELS}")
    defaults = {
        t: {"params": dict(v["params"]), "style": dict(v["style"])}
        for t, v in (base.defaults if base else {}).items()
    }
    for t, v in raw.get("defaults", {}).items():
        d = defaults.setdefault(t, {"params": {}, "style": {}})
        d["params"].update(v.get("params", {}))
        d["style"].update(v.get("style", {}))
    return Look(name, theme, life, raw.get("description", ""), defaults)
