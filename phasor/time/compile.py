"""Document -> parameter graph -> SceneState (contract 3).

``compile_document`` checks everything that has meaning: object types, parameter names and
types, style roles, expressions, tracks, markers and cycles. It reports every problem it
finds. ``evaluate`` is then a pure function of (compiled document, t).
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from phasor.core import life
from phasor.core.ids import is_valid_id
from phasor.core.issues import DocumentError, Issue
from phasor.core.objects import Placeholder, SceneObject, get_type, get_updater, registered_types
from phasor.core.schema import Document, ExprSpec
from phasor.core.spatial import (
    OrbitCameraParams,
    OrthoCameraParams,
    ortho_camera,
    overlay_camera,
    perspective_camera,
)
from phasor.core.state import ObjectState, SceneState
from phasor.core.values import normalize, same_shape
from phasor.style.theme import PALETTES, Theme, ThemeError, load_theme
from phasor.time.easing import EASINGS
from phasor.time.expr import CompiledExpr, ExprError, compile_expr
from phasor.time.tracks import ResolvedSegment, ResolvedTrack, TimeMap
from phasor.time.updaters import MEMO, Updater, UpdaterOut

CAMERA_PARAMS = {"ortho": OrthoCameraParams, "orbit": OrbitCameraParams}


@dataclass(frozen=True)
class Const:
    value: Any


Node = Const | ResolvedTrack | CompiledExpr


@dataclass(frozen=True)
class CompiledObject:
    id: str
    type: str
    index: int
    impl: SceneObject
    params_model: type[BaseModel]
    style: MappingProxyType
    roles: MappingProxyType
    tags: tuple[str, ...]
    layer: str


@dataclass(frozen=True)
class Compiled:
    doc: Document
    theme: Theme
    time_map: TimeMap
    nodes: MappingProxyType  # path -> Node
    order: tuple[str, ...]  # topological evaluation order
    objects: tuple[CompiledObject, ...]
    duration: float
    # The scene's folder, for objects that read files (``Files``); set by the Engine.
    base_dir: str = ""
    # (object id, why) for each object whose code is not available (lint missing_type).
    missing: tuple[tuple[str, str], ...] = ()


@functools.cache
def _adapter(model: type[BaseModel], name: str) -> TypeAdapter:
    f = model.model_fields[name]
    return TypeAdapter(Annotated[f.annotation, *f.metadata] if f.metadata else f.annotation)


def _coerce(model: type[BaseModel], name: str, value: Any) -> Any:
    """Validate one value against its field and return it in canonical form (tuples, floats)."""
    return normalize(_adapter(model, name).validate_python(value))


def _first_error(e: ValidationError) -> str:
    return e.errors()[0]["msg"]


def _toposort(nodes: dict[str, Node]) -> tuple[tuple[str, ...], list[str]]:
    """Kahn's algorithm over sorted keys, so the order is deterministic. Returns (order, cycle)."""
    deps = {p: (sorted(n.deps) if isinstance(n, CompiledExpr) else []) for p, n in nodes.items()}
    pending = {p: len(d) for p, d in deps.items()}
    users: dict[str, list[str]] = {p: [] for p in nodes}
    for p, ds in deps.items():
        for d in ds:
            users[d].append(p)
    ready = sorted(p for p, n in pending.items() if n == 0)
    order: list[str] = []
    while ready:
        p = ready.pop(0)
        order.append(p)
        for u in sorted(users[p]):
            pending[u] -= 1
            if pending[u] == 0:
                ready.append(u)
    return tuple(order), sorted(p for p, n in pending.items() if n > 0)


def _why(doc: Document, modules: Mapping[str, str] | None) -> str:
    """Why the code the document names is not here, from the Engine's module report."""
    bad = {m: s for m, s in (modules or {}).items() if s != "loaded"}
    if bad:
        return "; ".join(f"{m}: {s}" for m, s in sorted(bad.items()))
    return f"no module in meta.modules defines it: {', '.join(doc.meta.modules)}"


def compile_document(
    doc: Document, base_dir: str = "", modules: Mapping[str, str] | None = None
) -> Compiled:
    """Validate and resolve ``doc``. ``base_dir``: the scene's folder, which the files it
    names are relative to (``core/files.py``; a missing one is lint ``missing_file``).
    ``modules``: what became of each module in ``meta.modules`` (``"loaded"``, ``"missing"``,
    ``"refused"``, ``"failed: ..."``), from the Engine, which loads them (ADR 0005). An
    object whose type no loaded module defines draws as a placeholder (lint
    ``missing_type``); without ``meta.modules`` an unknown type is an error."""
    issues: list[Issue] = []
    add = issues.append
    missing: list[tuple[str, str]] = []  # (object id, why): lint missing_type
    placeholders: set[str] = set()

    try:
        theme = load_theme(doc.meta.theme)
    except ThemeError as e:
        raise DocumentError(Issue("unknown_theme", str(e), path="meta.theme")) from e
    if doc.meta.palette is not None and doc.meta.palette not in PALETTES:
        raise DocumentError(Issue("unknown_palette", f"palette must be one of {PALETTES}, got "
                                  f"{doc.meta.palette!r}", path="meta.palette"))  # fmt: skip
    time_map = TimeMap(doc.timeline.markers)
    nodes: dict[str, Node] = {}
    owner: dict[str, type[BaseModel]] = {}  # path -> params model, for coercing track values
    objects: list[CompiledObject] = []

    def add_params(target: str, model: type[BaseModel], given: dict[str, Any]) -> None:
        for name in given:
            if name not in model.model_fields:
                add(Issue("unknown_param", f"{target!r} has no parameter {name!r}; "
                          f"it has {sorted(model.model_fields)}", ids=(target,),
                          path=f"{target}.{name}"))  # fmt: skip
        for name, field in model.model_fields.items():
            path = f"{target}.{name}"
            owner[path] = model
            if name not in given:
                if field.is_required():
                    add(Issue("missing_param", f"{target!r} needs parameter {name!r}",
                              ids=(target,), path=path))  # fmt: skip
                    continue
                raw = field.get_default(call_default_factory=True)
            else:
                raw = given[name]
            if isinstance(raw, ExprSpec):
                try:
                    nodes[path] = compile_expr(raw.expr)
                except ExprError as e:
                    add(Issue("bad_expr", str(e), ids=(target,), path=path))
                continue
            try:
                nodes[path] = Const(_coerce(model, name, raw))
            except (ValidationError, ValueError) as e:
                msg = _first_error(e) if isinstance(e, ValidationError) else str(e)
                add(Issue("bad_param", f"{path}: {msg}", ids=(target,), path=path))

    for index, spec in enumerate(doc.objects, start=1):
        cls = get_type(spec.type)
        if cls is None and not doc.meta.modules:
            add(Issue("unknown_type", f"unknown object type {spec.type!r}; "
                      f"registered: {sorted(registered_types())}", ids=(spec.id,)))  # fmt: skip
            continue
        if cls is None:
            # The document names code that would define this type, but it is not here (lint
            # missing_type): a placeholder box keeps the object and all its parameters.
            missing.append((spec.id, f"type {spec.type!r} is not available ({_why(doc, modules)})"))
            cls = Placeholder
            placeholders.add(spec.id)
            known = {}
            for name, raw in spec.params.items():
                if name in cls.Params.model_fields and not isinstance(raw, ExprSpec):
                    with contextlib.suppress(ValidationError, ValueError):  # a bad value: default
                        known[name] = _coerce(cls.Params, name, raw)
                elif not isinstance(raw, ExprSpec):  # kept for expressions that read it
                    nodes[f"{spec.id}.{name}"] = Const(normalize(raw))
            add_params(spec.id, cls.Params, known)
        else:
            add_params(spec.id, cls.Params, spec.params)
        style, roles = {}, {}
        for key in spec.style:
            if key not in cls.Style and spec.id not in placeholders:
                add(Issue("unknown_style", f"{spec.type!r} has no style key {key!r}; "
                          f"it has {sorted(cls.Style)}", ids=(spec.id,)))  # fmt: skip
        for key, (group, default) in cls.Style.items():
            role = spec.style.get(key, default)
            try:
                style[key], roles[key] = theme.resolve(group, role), role
            except ThemeError as e:
                add(Issue("bad_role", f"{spec.id}.{key}: {e}", ids=(spec.id,)))
        objects.append(
            CompiledObject(spec.id, spec.type, index, cls(spec.id), cls.Params,
                           MappingProxyType(style), MappingProxyType(roles), spec.tags, spec.layer)
        )  # fmt: skip
    for cam in doc.cameras:
        add_params(cam.id, CAMERA_PARAMS[cam.kind], cam.params)

    duration = max(doc.timeline.markers.values(), default=0.0)
    if doc.narration is not None:
        duration = max(duration, doc.narration.end)  # the scene lasts as long as the voice
    for path, segs in sorted(doc.timeline.tracks.items()):
        node = nodes.get(path)
        if path not in owner and path.rpartition(".")[0] in placeholders:
            continue  # kept in the document; it animates again once the type is back
        if path not in owner:
            add(Issue("unknown_param", f"track on unknown parameter {path!r}", path=path))
            continue
        if not isinstance(node, Const):
            if isinstance(node, CompiledExpr):
                add(Issue("track_on_expr", f"{path!r} is an expression; it cannot have a track",
                          path=path, fix_hint="animate the parameters it reads"))  # fmt: skip
            continue
        target, name = path.rsplit(".", 1)
        resolved: list[ResolvedSegment] = []
        prev_end = float("-inf")
        for i, seg in enumerate(segs):
            where = f"{path}[{i}]"
            try:
                t0, t1 = time_map.resolve(seg.t0), time_map.resolve(seg.t1)
            except KeyError as e:
                add(Issue("unknown_marker", f"{where}: no marker {e.args[0]!r}", path=path))
                continue
            if seg.ease not in EASINGS:
                add(Issue("unknown_ease", f"{where}: no easing {seg.ease!r}; "
                          f"available: {sorted(EASINGS)}", path=path))  # fmt: skip
                continue
            if t1 < t0 or t0 < prev_end:
                add(Issue("bad_segment", f"{where}: spans {t0:g}-{t1:g} s but must not run "
                          f"backwards or start before the previous segment ends ({prev_end:g} s)",
                          path=path))  # fmt: skip
                continue
            try:
                to = _coerce(owner[path], name, seg.to)
            except (ValidationError, ValueError) as e:
                msg = _first_error(e) if isinstance(e, ValidationError) else str(e)
                add(Issue("bad_param", f"{where}: {msg}", ids=(target,), path=path))
                continue
            if not same_shape(node.value, to):
                add(Issue("bad_param", f"{where}: target {to!r} does not match the shape of "
                          f"{node.value!r}", ids=(target,), path=path))  # fmt: skip
                continue
            resolved.append(ResolvedSegment(t0, t1, to, EASINGS[seg.ease]))
            prev_end = t1
            duration = max(duration, t1)
        nodes[path] = ResolvedTrack(node.value, tuple(resolved))

    if any(o.always for o in doc.objects) and not issues:
        _attach_updaters(doc, nodes, objects, placeholders, modules, missing, add)

    for path, node in nodes.items():
        if isinstance(node, CompiledExpr):
            for dep in sorted(node.deps - nodes.keys()):
                if dep not in owner:  # a known but failed parameter was already reported
                    add(Issue("unknown_ref", f"{path}: expression reads unknown parameter "
                              f"{dep!r}", path=path))  # fmt: skip
    if issues:
        raise DocumentError(issues)

    order, cycle = _toposort(nodes)
    if cycle:
        raise DocumentError(
            Issue("cycle", f"parameters depend on each other in a cycle: {', '.join(cycle)}",
                  ids=tuple(sorted({p.rsplit('.', 1)[0] for p in cycle})))
        )  # fmt: skip
    if doc.meta.duration is not None:
        duration = doc.meta.duration
    compiled = Compiled(doc, theme, time_map, MappingProxyType(nodes), order, tuple(objects),
                        duration, str(base_dir), tuple(missing))  # fmt: skip
    evaluate(compiled, 0.0)  # surfaces type errors from expressions before anything is saved
    return compiled


def _env(nodes: Mapping[str, Node], order: tuple[str, ...], t: float,
         replace: Mapping[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """Every parameter's value at ``t``, in graph order (``replace`` stands in for some)."""
    env: dict[str, Any] = {}
    replace = replace or {}
    for path in order:
        node = nodes[path]
        try:
            if path in replace:
                env[path] = replace[path]
            elif isinstance(node, Const):
                env[path] = node.value
            elif isinstance(node, ResolvedTrack):
                env[path] = node.value_at(t)
            else:
                env[path] = normalize(node(env, t))
        except (ExprError, ValueError) as e:
            raise DocumentError(Issue("bad_expr", f"{path} at t={t:g}: {e}", path=path)) from e
    for key in [k for k in env if k.startswith(MEMO)]:
        del env[key]
    return env


def _attach_updaters(doc, nodes, objects, placeholders, modules, missing, add) -> None:
    """Make each object's updaters (``ObjectSpec.always``) nodes of the parameter graph. Each
    is run once at t = 0 to find what it reads (its edges) and what it sets."""
    models = {o.id: o.params_model for o in objects}
    order, cycle = _toposort(nodes)
    if cycle:
        return  # reported as a cycle below
    try:
        env0 = _env(nodes, order, 0.0)
    except DocumentError:
        return  # the same error surfaces when the document is evaluated
    for spec in doc.objects:
        if not spec.always or spec.id in placeholders:
            continue
        model = models[spec.id]
        for ref in spec.always:
            fn = get_updater(ref)
            if fn is None:
                why = _why(doc, modules)
                missing.append((spec.id, f"updater {ref!r} is not available ({why})"))
                continue
            probe = Updater(ref, spec.id, fn, frozenset(model.model_fields))
            reads: set[str] = set()
            try:
                sets = probe.call(env0, 0.0, reads)
                for name, value in sets.items():
                    _coerce(model, name, value)
            except (ExprError, ValueError) as e:
                msg = _first_error(e) if isinstance(e, ValidationError) else str(e)
                add(Issue("bad_updater", f"{spec.id}: {msg}", ids=(spec.id,)))
                continue
            up = Updater(ref, spec.id, fn, probe.params, frozenset(reads))
            for name in sets:
                path = f"{spec.id}.{name}"
                base = nodes[path]
                deps = up.reads | (base.deps if isinstance(base, CompiledExpr) else frozenset())
                nodes[path] = UpdaterOut(f"updater {ref}", frozenset(deps), True, None, up, name,
                                         base, model, _coerce)  # fmt: skip


def sway(theme: Theme, params: OrbitCameraParams, t: float) -> float:
    """The camera's slow drift in azimuth at ``t``, in degrees: a pure function of t, so
    time stays pure (contract 3). Amplitude from the camera's ``drift`` or the theme."""
    amp = theme.camera["drift"] if params.drift < 0 else params.drift
    period = theme.camera["drift_period"]
    return amp * math.sin(2 * math.pi * t / period) if amp and period > 0 else 0.0


@functools.cache
def _int_fields(model: type[BaseModel]) -> tuple[str, ...]:
    """Parameters declared as whole numbers (``terms: int``)."""
    return tuple(n for n, f in model.model_fields.items() if f.annotation is int)


def _strokes(c: Compiled) -> dict[str, Any]:
    """The theme's ``[strokes]`` with the scene's ``palette`` (when it picks one) as colors."""
    palette = c.doc.meta.palette
    if palette is None:
        return c.theme.strokes
    return {**c.theme.strokes, "lut": tuple(tuple(x) for x in c.theme.values["colormap"][palette])}


def evaluate(c: Compiled, t: float, replace: Mapping[str, Any] | None = None) -> SceneState:
    """Pure: depends only on (compiled document, t).

    ``replace`` maps parameter paths to values that stand in for the document's own; the
    layout bake uses it to put solved positions in, and expressions that read those
    parameters see the solved values.
    """
    t = float(t)
    env = _env(c.nodes, c.order, t, replace)

    meta = c.doc.meta
    life_amp = life.scale(meta.life, c.theme.life)

    def build(target: str, model: type[BaseModel], tags: tuple[str, ...] = (), style=None,
              channels: frozenset[str] = frozenset()):  # fmt: skip
        values = {n: env[f"{target}.{n}"] for n in model.model_fields}
        for n in _int_fields(model):  # an integer parameter animates in whole steps
            if isinstance(values[n], float) and math.isfinite(values[n]):
                values[n] = round(values[n])
        if "dim" in values and "opacity" in values:  # what is drawn is opacity x dim
            values["opacity"] = values["opacity"] * values["dim"]
        if style is not None:  # objects live (cameras do not): see core/life.py
            values, style = life.adjust(target, tags, set(model.model_fields), values, style,
                                        seed=meta.seed, amp=life_amp, theme_life=c.theme.life,
                                        t=t, channels=channels)  # fmt: skip
            built[target] = style
        try:
            return model.model_validate(values)
        except ValidationError as e:
            err = e.errors()[0]
            path = f"{target}.{'.'.join(str(p) for p in err['loc'])}"
            raise DocumentError(
                Issue("bad_param", f"{path} at t={t:g}: {err['msg']}", ids=(target,), path=path)
            ) from e

    built: dict[str, Any] = {}
    aspect = meta.width / meta.height
    cam_spec = c.doc.camera(c.doc.timeline.camera)
    cam_params = build(cam_spec.id, CAMERA_PARAMS[cam_spec.kind])
    if cam_spec.kind == "orbit":
        camera = perspective_camera(cam_params, aspect, sway(c.theme, cam_params, t))
    else:
        camera = ortho_camera(cam_params, aspect)
    objects = tuple(
        ObjectState(o.id, o.type, o.index, build(o.id, o.params_model, o.tags, o.style,
                                                 getattr(get_type(o.type), "Life", frozenset())),
                    MappingProxyType(built[o.id]), o.roles, o.tags, o.layer, o.impl)
        for o in c.objects
    )  # fmt: skip

    def scene(objects: tuple[ObjectState, ...]) -> SceneState:
        return SceneState(
            t=t,
            width=meta.width,
            height=meta.height,
            background=c.theme.background,
            objects=objects,
            camera=camera,
            overlay=overlay_camera(aspect),
            values=MappingProxyType(env),
            light=MappingProxyType(c.theme.light),
            strokes=MappingProxyType(_strokes(c)),
            base_dir=c.base_dir,
        )

    if any(callable(getattr(o.impl, "build", None)) for o in objects):
        objects = _expand(c, t, objects, scene)
    return scene(objects)


def _expand(c: Compiled, t: float, objects: tuple[ObjectState, ...], scene) -> tuple:
    """The parts composites build (ADR 0005), after the document's objects. A part's id is
    ``<composite>.<name>``; it may not be an id the document uses or has retired. A part's
    opacity is its own times its composite's, and it lies on the composite's layer, above
    it by its own ``z_index``. Parts are built in document order, a composite part's own
    parts right after it."""
    taken = {o.id for o in c.doc.objects} | {cam.id for cam in c.doc.cameras}
    retired = set(c.doc.retired_ids)
    out = list(objects)
    queue = [o for o in objects if callable(getattr(o.impl, "build", None))]

    def fail(parent: str, msg: str) -> DocumentError:
        return DocumentError(Issue("bad_part", f"{parent} at t={t:g}: {msg}", ids=(parent,)))

    while queue:
        parent = queue.pop(0)
        try:
            made = list(parent.impl.build(scene(tuple(out))))
        except DocumentError:
            raise
        except Exception as e:
            raise fail(parent.id, f"build failed: {type(e).__name__}: {e}") from e
        for part in made:
            cls = type(part)
            if get_type(getattr(cls, "type", "")) is not cls or not isinstance(part.id, str):
                raise fail(parent.id, f"build must return objects of registered types: {part!r}")
            full = f"{parent.id}.{part.id}"
            if not is_valid_id(full):
                raise fail(parent.id, f"part id {part.id!r} is not a valid name")
            if full in retired:
                raise fail(parent.id, f"part id {full!r} was retired; ids are never reused")
            if full in taken:
                raise fail(parent.id, f"part id {full!r} is already taken")
            taken.add(full)
            spec = part.to_spec()
            model = cls.Params
            values: dict[str, Any] = {}
            try:
                for name, raw in spec.params.items():
                    if name not in model.model_fields:
                        raise ValueError(f"{cls.type!r} has no parameter {name!r}")
                    if isinstance(raw, ExprSpec):
                        raise ValueError(f"{name}: a part takes values, not expressions")
                    values[name] = _coerce(model, name, raw)
                params = model.model_validate(values)
            except (ValidationError, ValueError) as e:
                msg = _first_error(e) if isinstance(e, ValidationError) else str(e)
                raise fail(parent.id, f"part {part.id!r}: {msg}") from e
            pp = parent.params  # a part fades with, and sits above, its composite
            seen = params.opacity * params.dim * pp.opacity
            params = model.model_validate(
                {**dict(params), "opacity": seen, "z_index": params.z_index + pp.z_index}
            )
            style, roles = {}, {}
            for key in spec.style:
                if key not in cls.Style:
                    raise fail(parent.id, f"part {part.id!r} has no style key {key!r}")
            for key, (group, default) in cls.Style.items():
                role = spec.style.get(key, default)
                try:
                    style[key], roles[key] = c.theme.resolve(group, role), role
                except ThemeError as e:
                    raise fail(parent.id, f"part {part.id!r}.{key}: {e}") from e
            part.id = full
            state = ObjectState(full, cls.type, len(out) + 1, params, MappingProxyType(style),
                                MappingProxyType(roles), spec.tags, parent.layer, part,
                                parent=parent.id)  # fmt: skip
            out.append(state)
            if callable(getattr(part, "build", None)):
                queue.insert(0, state)
    return tuple(out)
