"""The scene document (contract 2): pydantic v2 models, frozen, extra keys forbidden.

This module checks *shape* only (ids, uniqueness, references that need no registry).
Meaning (object types, parameter types, expressions, tracks, cycles) is checked by
``phasor.time.compile.compile_document``, which every write path runs.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainValidator, model_validator

from phasor.core.constraints import PRIORITIES, RuleError, normalize_rule, parse_rule
from phasor.core.files import relative_path
from phasor.core.ids import is_valid_id, is_valid_name, parent_of, split_path
from phasor.core.values import normalize

SCHEMA_VERSION = 8  # 2: `constraints` (M2); 3: `retired_ids`; 4: `narration`, `Segment.verb` (M4);
# 5: `meta.life`, `meta.seed`, `meta.look` (M7); 6: `controls` (M12); 7: `meta.palette`;
# 8: `meta.modules`, `ObjectSpec.always` (M15, ADR 0005)
OVERRIDES_VERSION = 1


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExprSpec(Model):
    """A parameter computed from ``t`` and other parameters, e.g. ``dot.pos + vec(0, 0.4, 0)``."""

    expr: str


def _parse_param(v: Any) -> Any:
    if isinstance(v, ExprSpec):
        return v
    if isinstance(v, dict):
        return ExprSpec.model_validate(v)
    return normalize(v)


# A parameter is a constant value or {"expr": "..."}. Animation lives in timeline.tracks.
ParamSpec = Annotated[Any, PlainValidator(_parse_param)]
ConstValue = Annotated[Any, PlainValidator(normalize)]


class MarkerRef(Model):
    """A time anchored to a narration marker; re-recording moves the marker, not the track."""

    marker: str
    offset: float = 0.0


TimeRef = float | MarkerRef


class Segment(Model):
    """Move a parameter to ``to`` between ``t0`` and ``t1``. It holds its value in between."""

    t0: TimeRef
    t1: TimeRef
    to: ConstValue
    ease: str = "smooth"
    verb: str | None = None  # the choreography verb that made it; None is a raw transition


def _check_module(path: str) -> str:
    """A code module the document needs: a ``.py`` file inside the scene's folder."""
    path = relative_path(path)
    if not path.endswith(".py"):
        raise ValueError(f"module {path!r} must be a .py file (a scene script or *.types.py)")
    return path


def _check_updater(ref: str) -> str:
    module, sep, name = ref.rpartition(":") if isinstance(ref, str) else ("", "", "")
    if not sep or not module or not name:
        raise ValueError(f"updater {ref!r} must be written '<module>.py:<function>'")
    return f"{_check_module(module)}:{name}"


class ObjectSpec(Model):
    id: str
    type: str
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    style: dict[str, str] = Field(default_factory=dict)  # style key -> role name
    tags: tuple[str, ...] = ()
    layer: Literal["world", "overlay"] = "world"
    # Pure updaters (ADR 0005), each ``"<module>:<function>"``: a function in one of the
    # document's ``meta.modules`` that sets parameters of this object from (t, look).
    always: tuple[Annotated[str, PlainValidator(_check_updater)], ...] = ()


class CameraSpec(Model):
    id: str
    kind: Literal["ortho", "orbit"] = "ortho"  # orbit: perspective, around a target
    params: dict[str, ParamSpec] = Field(default_factory=dict)


def _check_rule(rule: str) -> str:
    rule = normalize_rule(rule)
    try:
        parse_rule(rule)
    except RuleError as e:
        raise ValueError(str(e)) from e
    return rule


class ConstraintSpec(Model):
    """One layout rule, e.g. ``label(pole) near pole``. The rule text is its identity."""

    rule: Annotated[str, PlainValidator(_check_rule)]
    priority: Literal["required", "strong", "medium", "weak"] = "strong"


assert ConstraintSpec.model_fields["priority"].annotation.__args__ == PRIORITIES


def _check_unique_rules(constraints: tuple[ConstraintSpec, ...]) -> None:
    rules = [c.rule for c in constraints]
    for r in rules:
        if rules.count(r) > 1:
            raise ValueError(f"duplicate constraint {r!r}")


class Word(Model):
    """One spoken word, in seconds from the start of its block."""

    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class NarrationBlock(Model):
    """One spoken chunk (a sentence), or a silent pause when ``text`` is empty."""

    text: str
    start: float = Field(ge=0)  # seconds on the scene timeline
    duration: float = Field(ge=0)
    audio: str | None = None  # WAV path relative to the scene's folder; None: silent/estimate
    words: tuple[Word, ...] = ()
    markers: dict[str, float] = Field(default_factory=dict)  # name -> seconds from block start


class Narration(Model):
    """The narration and where every marker in it falls.

    It is derived data: re-recording replaces it, and with it the markers it owns in
    ``timeline.markers``. Tracks refer to markers by name, so nothing else changes."""

    script: str
    voice: str
    blocks: tuple[NarrationBlock, ...] = ()

    def markers(self) -> dict[str, float]:
        return {name: round(b.start + dt, 4) for b in self.blocks for name, dt in b.markers.items()}

    @property
    def end(self) -> float:
        return max((b.start + b.duration for b in self.blocks), default=0.0)


class Timeline(Model):
    markers: dict[str, float] = Field(default_factory=dict)  # the TimeMap (contract 3)
    tracks: dict[str, tuple[Segment, ...]] = Field(default_factory=dict)  # "<id>.<param>" -> segs
    camera: str = "camera"


class Meta(Model):
    title: str
    width: int = Field(1920, gt=0)
    height: int = Field(1080, gt=0)
    fps: float = Field(60.0, gt=0)
    theme: str = "midnight_glow"
    color_space: Literal["srgb"] = "srgb"
    duration: float | None = Field(None, ge=0)  # None: derived from tracks and markers
    # Living motion (core/life.py): "off", "subtle" or "lively"; the seed varies the wiggle.
    life: Literal["off", "subtle", "lively"] = "off"
    seed: int = 0
    look: str | None = None  # the look preset the scene started from (the values are resolved)
    # The colors run along strokes (a colormap role: "spectrum", the prism rainbow, or
    # "moonlight", "gold", "glacier", "borealis", "rose"); None: the theme's own.
    palette: str | None = None
    # Code the scene needs (ADR 0005): the scene script or ``*.types.py`` files in its folder
    # that define its own object types and updaters. Loaded only where scripts may run.
    modules: tuple[Annotated[str, PlainValidator(_check_module)], ...] = ()
    # "script": built by a Python script; later tweaks go to name.overrides.json.
    source: Literal["document", "script"] = "document"


class Control(Model):
    """A slider: a number a viewer may change. ``path`` is a
    parameter (``julia.c``); for a vector parameter, ``component`` says which entry (0 is Re
    of a complex ``c``). The player, the editor, the notebook and the web export all build
    their sliders from the document's controls."""

    path: str
    component: int | None = Field(None, ge=0)
    label: str = ""
    min: float
    max: float
    step: float = Field(0.01, gt=0)

    @model_validator(mode="after")
    def _shape(self) -> Control:
        split_path(self.path)
        if self.max <= self.min:
            raise ValueError(f"control {self.path!r}: max must be above min")
        return self

    @property
    def key(self) -> str:
        return self.path if self.component is None else f"{self.path}[{self.component}]"


class Document(Model):
    schema_version: Literal[8] = SCHEMA_VERSION
    meta: Meta
    objects: tuple[ObjectSpec, ...] = ()
    constraints: tuple[ConstraintSpec, ...] = ()
    controls: tuple[Control, ...] = ()  # sliders a viewer may move (M12)
    # Ids of removed objects. They may never be used again (contract 2).
    retired_ids: tuple[str, ...] = ()
    narration: Narration | None = None
    cameras: tuple[CameraSpec, ...] = (CameraSpec(id="camera"),)
    timeline: Timeline = Field(default_factory=Timeline)

    @model_validator(mode="after")
    def _check_shape(self) -> Document:
        seen: set[str] = set()
        for thing in (*self.objects, *self.cameras):
            if not is_valid_id(thing.id):
                raise ValueError(f"invalid id {thing.id!r}: must match [a-z][a-z0-9_]* with dots")
            if thing.id in seen:
                raise ValueError(f"duplicate id {thing.id!r}")
            seen.add(thing.id)
        object_ids = {o.id for o in self.objects}
        modules = set(self.meta.modules)
        for o in self.objects:
            for ref in o.always:
                if ref.rpartition(":")[0] not in modules:
                    raise ValueError(f"{o.id}: updater {ref!r} names a module that is not in "
                                     f"meta.modules {sorted(modules)}")  # fmt: skip
            parent = parent_of(o.id)
            if parent is not None and parent not in object_ids:
                raise ValueError(f"object {o.id!r} needs its parent {parent!r} to exist")
        if self.timeline.camera not in {c.id for c in self.cameras}:
            raise ValueError(f"timeline.camera {self.timeline.camera!r} is not a camera id")
        for name in self.timeline.markers:
            if not is_valid_name(name):
                raise ValueError(f"invalid marker name {name!r}")
        for path in self.timeline.tracks:
            split_path(path)
        _check_unique_rules(self.constraints)
        keys = [c.key for c in self.controls]
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"duplicate controls: {sorted({k for k in keys if keys.count(k) > 1})}"
            )
        if self.narration is not None:
            for name, t in self.narration.markers().items():
                if self.timeline.markers.get(name) != t:
                    raise ValueError(f"narration marker {name!r} at {t:g} s is missing from, or "
                                     "different in, timeline.markers")  # fmt: skip
        back = sorted(seen & set(self.retired_ids))
        if back:
            raise ValueError(f"id {back[0]!r} was retired when its object was removed; ids are "
                             "never reused, so choose a new name")  # fmt: skip
        return self

    def object(self, object_id: str) -> ObjectSpec | None:
        return next((o for o in self.objects if o.id == object_id), None)

    def camera(self, camera_id: str) -> CameraSpec | None:
        return next((c for c in self.cameras if c.id == camera_id), None)


class Overrides(Model):
    """GUI/AI tweaks to a scripted scene, keyed by ``<id>.<param>`` / ``<id>.<style key>``,
    plus layout rules added on top of the script's own."""

    schema_version: Literal[1] = OVERRIDES_VERSION
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    styles: dict[str, str] = Field(default_factory=dict)
    constraints: tuple[ConstraintSpec, ...] = ()  # rules added on top of the script's own

    @model_validator(mode="after")
    def _check_shape(self) -> Overrides:
        _check_unique_rules(self.constraints)
        return self

    def is_empty(self) -> bool:
        return not self.params and not self.styles and not self.constraints


def json_schema() -> dict[str, Any]:
    """JSON Schema for the document, exported for MCP validation and docs."""
    return Document.model_json_schema()
