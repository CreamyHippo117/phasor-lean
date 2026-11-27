"""Themes and style roles.

Scenes name roles (``highlight``, ``bold``, ``label``); the theme decides what they look
like. Raw hex codes and pixel sizes never appear in a scene.
"""

from __future__ import annotations

import functools
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phasor.core.color import hex_to_linear, phase_lut, smooth_cycle

THEME_DIR = Path(__file__).parent / "themes"

# role group -> role names. "surface" values are the opacity of each material; how a
# material is lit (glass, wire, emissive) is decided by the renderer from the role name.
ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "emphasis": ("primary", "secondary", "muted", "highlight", "danger"),
    "stroke": ("hairline", "regular", "bold"),
    "glow": ("none", "soft", "strong"),
    "text": ("title", "label", "annotation", "math"),
    "surface": ("solid", "glass", "wire", "emissive"),
    # Cyclic wheels (phase) and ramps (the rest). `phase` is the full wheel: use it where hue
    # means something (domain coloring). `soft_phase` is a restrained three-hue wheel,
    # `ember` and `ocean` are two-tone ramps for fractals and plain fields.
    # The stroke palettes (bright from end to end, for hue run along glowing curves; a scene
    # picks one with ``palette``): `spectrum` is the prism rainbow, the rest are calmer.
    "colormap": (
        "phase",
        "soft_phase",
        "sequential",
        "ember",
        "ocean",
        "spectrum",
        "magnitude",
        "moonlight",
        "gold",
        "glacier",
        "borealis",
        "rose",
    ),
}
_SECTION = {
    "emphasis": "palette",
    "stroke": "stroke",
    "text": "text",
    "glow": "glow",
    "surface": "surface",
    "colormap": "colormap",
}
LUT_SIZE = 256


_TABLES = {
    "layout": ("gap", "margin", "leader"),
    "lint": (
        "overlap",
        "contrast",
        "tiny",
        "text_min",
        "clutter",
        "crowded",
        "occluded",
        "hues",
        "clip",
        "density_min",
        "density_max",
        "rhythm",
        "margin",
        "stroke_mix",
        "impure",
        "bake_renders",
    ),
    "life": (
        "glow",
        "glow_period",
        "point",
        "point_period",
        "field",
        "field_period",
        "shimmer",
        "shimmer_period",
        "subtle",
    ),
    "bloom": ("threshold", "strength", "levels", "exposure", "knee"),
    "narration": ("words_per_second", "lead", "gap", "pause_per_change", "pause_max"),
    "sound": (
        "target",
        "bed",
        "cue",
        "duck",
        "attack",
        "release",
        "ceiling",
        "knee",
        "masking",
        "word_gap",
        "clip",
    ),
    "camera": (
        "drift",
        "drift_period",
        "fill",
        "thirds",
        "smooth",
        "margin",
        "min_elevation",
        "max_elevation",
    ),
    "light": (
        "key_azimuth",
        "key_elevation",
        "key",
        "fill_azimuth",
        "fill_elevation",
        "fill",
        "rim_azimuth",
        "rim_elevation",
        "rim",
        "ambient",
        "specular",
        "shininess",
        "fog",
    ),
}


def _colormap(theme: str, role: str, spec: Any) -> tuple[tuple[float, float, float], ...]:
    """A cyclic color table as a tuple (hashable, frozen): ``"oklch L C"`` for the even
    phase wheel at OKLCH lightness L and chroma C (``"oklch"`` alone: 0.74 0.125), or a list
    of hex stops blended in linear light and wrapped back to the first."""
    if isinstance(spec, str) and spec.split()[:1] == ["oklch"]:
        try:
            lc = [float(v) for v in spec.split()[1:]]
        except ValueError:
            lc = []
        if len(lc) not in (0, 2):
            raise ThemeError(f"theme {theme!r}: colormap {role!r} must be 'oklch' or 'oklch L C'")
        lut = phase_lut(LUT_SIZE, *lc)
        return tuple(tuple(float(v) for v in row) for row in lut)
    if not isinstance(spec, list) or len(spec) < 2:
        raise ThemeError(f"theme {theme!r}: colormap {role!r} must be 'oklch' or 2+ hex stops")
    if role in PALETTES:  # palettes run along curves: smooth, no corners, no grey middles
        lut = smooth_cycle(np.array([hex_to_linear(h)[:3] for h in spec]), LUT_SIZE)
        return tuple(tuple(float(v) for v in row) for row in lut)
    stops = np.array([hex_to_linear(h)[:3] for h in spec] + [hex_to_linear(spec[0])[:3]])
    x = np.linspace(0, len(spec), LUT_SIZE, endpoint=False)
    k = np.minimum(x.astype(int), len(spec) - 1)
    u = (x - k)[:, None]
    lut = stops[k] * (1 - u) + stops[k + 1] * u
    return tuple(tuple(float(v) for v in row) for row in lut)


class ThemeError(ValueError):
    pass


@dataclass(frozen=True)
class Theme:
    name: str
    background: tuple[float, float, float, float]
    values: dict[str, dict[str, Any]]  # group -> role -> resolved value
    layout: dict[str, float] = field(default_factory=dict)  # fractions of the frame height
    lint: dict[str, float] = field(default_factory=dict)  # linter thresholds
    bloom: dict[str, float] = field(default_factory=dict)  # final-look glow settings
    narration: dict[str, float] = field(default_factory=dict)  # speaking rate, pacing
    sound: dict[str, float] = field(default_factory=dict)  # mix levels, ducking, sound lint
    camera: dict[str, float] = field(default_factory=dict)  # 3D drift and solver defaults
    light: dict[str, float] = field(default_factory=dict)  # three-point rig and depth fog
    life: dict[str, float] = field(default_factory=dict)  # living motion (see core/life.py)
    motion: dict[str, Any] = field(default_factory=dict)  # verb -> {duration, ease}, + scalars
    # [strokes]: a hue run along glowing strokes (hue_along, flow, lut); off when hue_along 0
    strokes: dict[str, Any] = field(default_factory=lambda: {"hue_along": 0.0, "flow": 0.0,
                                                              "lut": ()})  # fmt: skip

    def resolve(self, group: str, role: str) -> Any:
        if role not in ROLE_GROUPS.get(group, ()):
            raise ThemeError(f"{role!r} is not a {group} role; use one of {ROLE_GROUPS.get(group)}")
        try:
            return self.values[group][role]
        except KeyError:
            raise ThemeError(f"theme {self.name!r} defines no {group} role {role!r}") from None


PALETTES = ("spectrum", "moonlight", "gold", "glacier", "borealis", "rose")


def available_themes() -> list[str]:
    return sorted(p.stem for p in THEME_DIR.glob("*.toml"))


@functools.cache
def load_theme(name: str) -> Theme:
    raw = _raw(name)
    values: dict[str, dict[str, Any]] = {}
    for group, section in _SECTION.items():
        data = raw.get(section, {})
        missing = [r for r in ROLE_GROUPS[group] if r not in data]
        if missing:
            raise ThemeError(f"theme {name!r} is missing [{section}] roles {missing}")
        if group == "emphasis":
            values[group] = {r: hex_to_linear(data[r]) for r in ROLE_GROUPS[group]}
        elif group == "colormap":
            values[group] = {r: _colormap(name, r, data[r]) for r in ROLE_GROUPS[group]}
        else:
            values[group] = {r: float(data[r]) for r in ROLE_GROUPS[group]}
    extra = {}
    for section, required in _TABLES.items():
        data = raw.get(section, {})
        missing = [k for k in required if k not in data]
        if missing:
            raise ThemeError(f"theme {name!r} is missing [{section}] keys {missing}")
        extra[section] = {k: float(v) for k, v in data.items()}
    motion = _motion(name, raw.get("motion", {}))
    strokes = _strokes(name, raw.get("strokes", {}), values["colormap"])
    return Theme(name, hex_to_linear(raw["palette"]["background"]), values, **extra,
                 motion=motion, strokes=strokes)  # fmt: skip


def _raw(name: str, seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """A theme file's tables; ``base = "other"`` starts from that theme and overrides it
    table by table (key by key inside a table)."""
    path = THEME_DIR / f"{name}.toml"
    if not path.exists():
        raise ThemeError(f"unknown theme {name!r}; available: {available_themes()}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    base = raw.pop("base", None)
    if base is None:
        return raw
    if base in (*seen, name):
        raise ThemeError(f"theme {name!r}: base themes form a loop through {base!r}")
    merged = _raw(base, (*seen, name))
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _strokes(theme: str, data: dict[str, Any], colormaps: dict[str, Any]) -> dict[str, Any]:
    """``[strokes]`` (optional): ``hue_along`` > 0 runs a colormap along every glowing stroke
    that does not color itself (cycles per stroke), turning ``flow`` cycles a second."""
    along = float(data.get("hue_along", 0.0))
    cmap = str(data.get("colormap", "spectrum"))
    if cmap not in colormaps:
        raise ThemeError(f"theme {theme!r}: [strokes] colormap {cmap!r} is not a colormap role")
    if along < 0:
        raise ThemeError(f"theme {theme!r}: [strokes] hue_along must be >= 0")
    return {"hue_along": along, "flow": float(data.get("flow", 0.0)),
            "lut": tuple(tuple(c) for c in colormaps[cmap])}  # fmt: skip


VERBS = (
    "reveal",
    "exit",
    "trace",
    "bloom_in",
    "dim",
    "focus_pull",
    "morph",
    "settle",
    "orbit",
    "orbit_reveal",
    "transform",
    "become",
    "write",
    "travel",
    "flash",
    "circumscribe",
    "wiggle",
    "sweep",
    "unfold",
)


def _motion(theme: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("stagger", "dim_level", "orbit_swing", "camera_lead", "camera_speed"):
        if key not in data:
            raise ThemeError(f"theme {theme!r} is missing [motion] {key}")
        out[key] = float(data[key])
    # line heights a part arcs by when ``transform`` carries it past other parts
    out["transform_arc"] = float(data.get("transform_arc", 0.9))
    for verb in VERBS:
        spec = data.get(verb)
        if not isinstance(spec, dict) or "duration" not in spec or "ease" not in spec:
            raise ThemeError(f"theme {theme!r} needs [motion.{verb}] with duration and ease")
        out[verb] = {"duration": float(spec["duration"]), "ease": str(spec["ease"])}
    return out
