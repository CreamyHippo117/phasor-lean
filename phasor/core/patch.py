"""Patches: the vocabulary of ``Engine.apply`` (contract 1).

A patch is a list of small ops. Applying one is a pure function from document to document;
the Engine then validates the result as a whole, so a patch either applies fully or not at all.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, ValidationError

from phasor.core.constraints import normalize_rule, rule_ids
from phasor.core.ids import split_path
from phasor.core.issues import DocumentError, Issue
from phasor.core.objects import get_type
from phasor.core.schema import (
    CameraSpec,
    ConstraintSpec,
    ConstValue,
    Control,
    Document,
    ExprSpec,
    Meta,
    Model,
    Narration,
    ObjectSpec,
    Overrides,
    ParamSpec,
    Segment,
)


class AddObject(Model):
    op: Literal["add_object"] = "add_object"
    object: ObjectSpec


class RemoveObject(Model):
    """Remove an object and the tracks that animate it."""

    op: Literal["remove_object"] = "remove_object"
    id: str


class SetParam(Model):
    """Set an object or camera parameter to a constant or ``{"expr": ...}``."""

    op: Literal["set_param"] = "set_param"
    path: str
    value: ParamSpec


class SetStyle(Model):
    op: Literal["set_style"] = "set_style"
    id: str
    key: str
    role: str


class SetTags(Model):
    op: Literal["set_tags"] = "set_tags"
    id: str
    tags: tuple[str, ...]


class AddSegment(Model):
    """Append a transition to a parameter's track."""

    op: Literal["add_segment"] = "add_segment"
    path: str
    segment: Segment


class ClearTrack(Model):
    op: Literal["clear_track"] = "clear_track"
    path: str


class SetMarker(Model):
    op: Literal["set_marker"] = "set_marker"
    name: str
    t: float


class RemoveMarker(Model):
    op: Literal["remove_marker"] = "remove_marker"
    name: str


class SetMeta(Model):
    op: Literal["set_meta"] = "set_meta"
    field: str
    value: ConstValue | None


class SetNarration(Model):
    """Replace the narration (or remove it with ``None``). The markers the old narration
    owned leave ``timeline.markers`` and the new ones arrive, so every track anchored to a
    marker follows the new voice."""

    op: Literal["set_narration"] = "set_narration"
    narration: Narration | None


class AddConstraint(Model):
    """Add a layout rule, e.g. ``label(pole) near pole``."""

    op: Literal["add_constraint"] = "add_constraint"
    rule: str
    priority: Literal["required", "strong", "medium", "weak"] = "strong"


class SetCamera(Model):
    """Make camera ``id`` a camera of ``kind`` with ``params`` (replacing its parameters).
    ``orbit`` is the 3D perspective camera. Tracks on the camera stay and are re-checked."""

    op: Literal["set_camera"] = "set_camera"
    id: str = "camera"
    kind: Literal["ortho", "orbit"]
    params: dict[str, ParamSpec] = Field(default_factory=dict)


class RemoveConstraint(Model):
    op: Literal["remove_constraint"] = "remove_constraint"
    rule: str


class AddControl(Model):
    """Make a numeric parameter a slider. Replaces a control on the same parameter."""

    op: Literal["add_control"] = "add_control"
    control: Control


class RemoveControl(Model):
    op: Literal["remove_control"] = "remove_control"
    path: str
    component: int | None = None


Op = Annotated[
    AddObject
    | RemoveObject
    | SetParam
    | SetStyle
    | SetTags
    | AddSegment
    | ClearTrack
    | SetMarker
    | RemoveMarker
    | SetMeta
    | AddConstraint
    | SetNarration
    | SetCamera
    | RemoveConstraint
    | AddControl
    | RemoveControl,
    Field(discriminator="op"),
]


class Patch(Model):
    ops: tuple[Op, ...]


_PATCH = TypeAdapter(Patch)
TWEAK_OPS = (SetParam, SetStyle)  # the per-target ops an overrides file can hold


def parse_patch(patch: Any) -> Patch:
    """Accept a Patch, ``{"ops": [...]}``, a list of ops, or a single op."""
    if isinstance(patch, Patch):
        return patch
    if isinstance(patch, Model):
        patch = [patch]
    if isinstance(patch, dict) and "ops" not in patch:
        patch = [patch]
    if isinstance(patch, list | tuple):
        patch = {"ops": list(patch)}
    try:
        return _PATCH.validate_python(patch)
    except ValidationError as e:
        raise DocumentError(
            [
                Issue("patch_schema", err["msg"], path=".".join(str(p) for p in err["loc"]))
                for err in e.errors()
            ]
        ) from e


def _fail(code: str, message: str, **kw: Any) -> DocumentError:
    return DocumentError(Issue(code, message, **kw))


def _set_param(doc: Document, path: str, value: Any) -> Document:
    try:
        target, param = split_path(path)
    except ValueError as e:
        raise _fail("bad_path", str(e), path=path) from e
    for field in ("objects", "cameras"):
        items = getattr(doc, field)
        for i, item in enumerate(items):
            if item.id == target:
                new = item.model_copy(update={"params": {**item.params, param: value}})
                return doc.model_copy(update={field: (*items[:i], new, *items[i + 1 :])})
    raise _fail("unknown_id", f"no object or camera {target!r}", ids=(target,), path=path)


def param_value(doc: Document, path: str) -> Any:
    """The raw value of parameter ``path`` as the document holds it (``KeyError`` if none)."""
    target, param = split_path(path)
    for item in (*doc.objects, *doc.cameras):
        if item.id == target:
            if param in item.params:
                return item.params[param]
            cls = get_type(item.type) if hasattr(item, "type") else None
            field = cls.Params.model_fields.get(param) if cls is not None else None
            if field is not None:  # not set in the document: the type's own default
                return field.get_default(call_default_factory=True)
            raise KeyError(path)
    raise KeyError(path)


def _check_controllable(doc: Document, c: Control) -> None:
    """A control needs a constant numeric parameter (or entry of a vector): not an expression,
    not something animated by a track."""
    try:
        value = param_value(doc, c.path)
    except KeyError:
        raise _fail("bad_control", f"no parameter {c.path!r} to control", path=c.path) from None
    if c.path in doc.timeline.tracks:
        raise _fail("bad_control", f"{c.path} is animated by a track; a slider cannot drive it",
                    path=c.path)  # fmt: skip
    if isinstance(value, dict | str) or value is None:
        raise _fail("bad_control", f"{c.path} is not a number (or a vector of numbers)",
                    path=c.path)  # fmt: skip
    if isinstance(value, list | tuple):
        if c.component is None or c.component >= len(value):
            raise _fail("bad_control", f"{c.path} is a vector of {len(value)}: name a component "
                        f"(0 to {len(value) - 1})", path=c.path)  # fmt: skip
        if isinstance(value[c.component], dict | str | bool):
            raise _fail("bad_control", f"{c.path}[{c.component}] is not a number", path=c.path)
    elif c.component is not None or isinstance(value, bool):
        raise _fail("bad_control", f"{c.path} is a single number: leave component out",
                    path=c.path)  # fmt: skip


def _replace_object(doc: Document, object_id: str, **update: Any) -> Document:
    for i, o in enumerate(doc.objects):
        if o.id == object_id:
            new = o.model_copy(update=update)
            return doc.model_copy(
                update={"objects": (*doc.objects[:i], new, *doc.objects[i + 1 :])}
            )
    raise _fail("unknown_id", f"no object {object_id!r}", ids=(object_id,))


def _apply_op(doc: Document, op: Any) -> Document:
    tl = doc.timeline
    match op:
        case AddObject():
            if op.object.id in doc.retired_ids:
                raise _fail("id_retired", f"id {op.object.id!r} belonged to a removed object; "
                            "ids are never reused", ids=(op.object.id,),
                            fix_hint="give it a new id that says what it is")  # fmt: skip
            return doc.model_copy(update={"objects": (*doc.objects, op.object)})
        case RemoveObject():
            if doc.object(op.id) is None:
                raise _fail("unknown_id", f"no object {op.id!r}", ids=(op.id,))
            tracks = {p: s for p, s in tl.tracks.items() if split_path(p)[0] != op.id}
            return doc.model_copy(
                update={
                    "objects": tuple(o for o in doc.objects if o.id != op.id),
                    "timeline": tl.model_copy(update={"tracks": tracks}),
                    "retired_ids": tuple(sorted({*doc.retired_ids, op.id})),
                }
            )
        case SetParam():
            return _set_param(doc, op.path, op.value)
        case SetStyle():
            obj = doc.object(op.id)
            if obj is None:
                raise _fail("unknown_id", f"no object {op.id!r}", ids=(op.id,))
            return _replace_object(doc, op.id, style={**obj.style, op.key: op.role})
        case SetTags():
            return _replace_object(doc, op.id, tags=op.tags)
        case AddSegment():
            segs = (*tl.tracks.get(op.path, ()), op.segment)
            return doc.model_copy(
                update={"timeline": tl.model_copy(update={"tracks": {**tl.tracks, op.path: segs}})}
            )
        case ClearTrack():
            if op.path not in tl.tracks:
                raise _fail("unknown_track", f"no track on {op.path!r}", path=op.path)
            tracks = {p: s for p, s in tl.tracks.items() if p != op.path}
            return doc.model_copy(update={"timeline": tl.model_copy(update={"tracks": tracks})})
        case SetMarker():
            markers = {**tl.markers, op.name: op.t}
            return doc.model_copy(update={"timeline": tl.model_copy(update={"markers": markers})})
        case RemoveMarker():
            if op.name not in tl.markers:
                raise _fail("unknown_marker", f"no marker {op.name!r}")
            markers = {k: v for k, v in tl.markers.items() if k != op.name}
            return doc.model_copy(update={"timeline": tl.model_copy(update={"markers": markers})})
        case SetMeta():
            if op.field not in Meta.model_fields:
                raise _fail("unknown_meta", f"no meta field {op.field!r}")
            return doc.model_copy(update={"meta": doc.meta.model_copy(update={op.field: op.value})})
        case SetNarration():
            old = doc.narration.markers() if doc.narration else {}
            new = op.narration.markers() if op.narration else {}
            kept = {k: v for k, v in tl.markers.items() if k not in old}
            clash = sorted(set(kept) & set(new))
            if clash:
                raise _fail("marker_conflict", f"narration markers {clash} already exist as "
                            "hand-placed markers", fix_hint="rename one of them")  # fmt: skip
            timeline = tl.model_copy(update={"markers": {**kept, **new}})
            return doc.model_copy(update={"narration": op.narration, "timeline": timeline})
        case AddConstraint():
            try:
                spec = ConstraintSpec(rule=op.rule, priority=op.priority)
            except ValidationError as e:
                raise _fail("bad_constraint", e.errors()[0]["msg"]) from e
            if any(c.rule == spec.rule for c in doc.constraints):
                raise _fail("duplicate_constraint", f"constraint {spec.rule!r} already exists")
            return doc.model_copy(update={"constraints": (*doc.constraints, spec)})
        case SetCamera():
            cams = doc.cameras
            new = CameraSpec(id=op.id, kind=op.kind, params=op.params)
            if all(c.id != op.id for c in cams):
                raise _fail("unknown_id", f"no camera {op.id!r}; cameras are "
                            f"{[c.id for c in cams]}", ids=(op.id,))  # fmt: skip
            return doc.model_copy(
                update={"cameras": tuple(new if c.id == op.id else c for c in cams)}
            )
        case AddControl():
            _check_controllable(doc, op.control)
            kept = tuple(c for c in doc.controls if c.key != op.control.key)
            return doc.model_copy(update={"controls": (*kept, op.control)})
        case RemoveControl():
            key = op.path if op.component is None else f"{op.path}[{op.component}]"
            if all(c.key != key for c in doc.controls):
                raise _fail("unknown_control", f"no control {key!r}")
            return doc.model_copy(
                update={"controls": tuple(c for c in doc.controls if c.key != key)}
            )
        case RemoveConstraint():
            rule = normalize_rule(op.rule)
            if all(c.rule != rule for c in doc.constraints):
                raise _fail("unknown_constraint", f"no constraint {rule!r}")
            return doc.model_copy(
                update={"constraints": tuple(c for c in doc.constraints if c.rule != rule)}
            )
    raise _fail("unknown_op", f"unsupported op {op!r}")  # pragma: no cover


def apply_ops(doc: Document, patch: Patch) -> Document:
    """Pure: returns a new, shape-validated document. Raises DocumentError and changes nothing
    on failure. ``model_copy`` skips validation, so the result is re-validated as a whole."""
    for op in patch.ops:
        doc = _apply_op(doc, op)
    try:
        return Document.model_validate(doc.model_dump(warnings=False))
    except ValidationError as e:
        raise DocumentError(
            [
                Issue("schema", err["msg"], path=".".join(str(p) for p in err["loc"]) or None)
                for err in e.errors()
            ]
        ) from e


def apply_overrides(base: Document, overrides: Overrides) -> tuple[Document, list[Issue]]:
    """Lay tweaks over a base document. Tweaks whose target id is gone are reported as
    ``orphan_override`` lint errors and skipped; they are never dropped silently."""
    doc, orphans = base, []
    known = {o.id for o in base.objects} | {c.id for c in base.cameras}
    hint = "restore the object in the script, or delete this entry from the overrides file"
    for path, value in sorted(overrides.params.items()):
        target = path.rpartition(".")[0]
        if target not in known:
            orphans.append(
                Issue("orphan_override", f"override {path!r} targets a missing object",
                      ids=(target,), path=path, fix_hint=hint)
            )  # fmt: skip
            continue
        doc = _set_param(doc, path, value)
    for path, role in sorted(overrides.styles.items()):
        target, _, key = path.rpartition(".")
        if base.object(target) is None:
            orphans.append(
                Issue("orphan_override", f"style override {path!r} targets a missing object",
                      ids=(target,), path=path, fix_hint=hint)
            )  # fmt: skip
            continue
        doc = _apply_op(doc, SetStyle(id=target, key=key, role=role))
    have = {c.rule for c in base.constraints}
    for spec in overrides.constraints:
        missing = [i for i in rule_ids(spec.rule) if i not in known]
        if missing:
            orphans.append(
                Issue("orphan_override", f"constraint override {spec.rule!r} names missing "
                      f"object(s) {missing}", ids=tuple(missing),
                      path=f"constraints[{spec.rule}]", fix_hint=hint)
            )  # fmt: skip
        elif spec.rule not in have:  # the script may have adopted the same rule since
            doc = doc.model_copy(update={"constraints": (*doc.constraints, spec)})
    return doc, orphans


def record_tweaks(overrides: Overrides, patch: Patch) -> Overrides:
    """Fold tweak ops into an overrides file. Raises if the patch holds anything else."""
    params, styles = dict(overrides.params), dict(overrides.styles)
    constraints = list(overrides.constraints)
    for op in patch.ops:
        if isinstance(op, SetParam):
            params[op.path] = op.value
        elif isinstance(op, SetStyle):
            styles[f"{op.id}.{op.key}"] = op.role
        elif isinstance(op, AddConstraint):
            if all(c.rule != normalize_rule(op.rule) for c in constraints):
                constraints.append(ConstraintSpec(rule=op.rule, priority=op.priority))
        elif isinstance(op, RemoveConstraint) and any(
            c.rule == normalize_rule(op.rule) for c in constraints
        ):
            constraints = [c for c in constraints if c.rule != normalize_rule(op.rule)]
        else:
            raise _fail(
                "scripted_scene",
                f"'{op.op}' cannot be applied to a scene built by a script",
                fix_hint="change the script for structure and timing; "
                "tweaks may only use set_param, set_style and add_constraint",
            )
    return Overrides(params=params, styles=styles, constraints=tuple(constraints))


__all__ = [
    "AddConstraint",
    "AddControl",
    "AddObject",
    "AddSegment",
    "ClearTrack",
    "ExprSpec",
    "Op",
    "Patch",
    "RemoveConstraint",
    "RemoveControl",
    "RemoveMarker",
    "RemoveObject",
    "SetMarker",
    "SetMeta",
    "SetNarration",
    "SetParam",
    "SetStyle",
    "SetTags",
    "apply_ops",
    "apply_overrides",
    "parse_patch",
    "record_tweaks",
]  # fmt: skip
