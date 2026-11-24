"""A document as a Python builder script (the editor's code view).

``document_to_python(doc)`` writes a script that, run, builds the same scene: objects with
their parameters, style roles and tags, the camera, markers, animations (as ``at(...).to``),
constraints and controls. An object of a type defined in the scene's folder, or with
updaters, is written as the document's own ``add_object`` op (ADR 0005). Narration blocks
are not reproduced (their audio belongs to the voice module): a comment says so. Running the
result and diffing it with the document gives no differences except narration
(``tests/test_pycode.py``).
"""

from __future__ import annotations

from typing import Any

from phasor.core.objects import is_builtin, registered_types
from phasor.core.schema import Document, ExprSpec, MarkerRef

# class module -> how the generated script reaches it
_TOP = {"phasor.domains.basic", "phasor.domains.media"}  # from phasor import Dot, Image, ...
_ALIAS = {
    "phasor.domains.cx": "cx",
    "phasor.domains.acoustics": "ac",
    "phasor.domains.fourier": "fourier",
    "phasor.domains.linalg": "linalg",
    "phasor.domains.pde": "pde",
    "phasor.domains.algebra": "algebra",
    "phasor.domains.calc": "calc",
    "phasor.domains.geometry": "geo",
    "phasor.domains.cloud": "cloud",
    "phasor.domains.network": "nw",
    "phasor.domains.dynamics": "dyn",
    "phasor.domains.convolution": "conv",
    "phasor.domains.filters": "filt",
    "phasor.domains.waves": "ac",  # wave fields are named through ``ac``
}


def _lit(v: Any) -> str:
    if isinstance(v, ExprSpec):
        return f"expr({v.expr!r})"
    if isinstance(v, tuple | list):
        inner = ", ".join(_lit(x) for x in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_lit(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float):
        return repr(v)
    return repr(v)


def _class_ref(type_name: str, uses: set[str]) -> str:
    cls = registered_types().get(type_name)
    if cls is None:
        raise ValueError(f"unknown object type {type_name!r}")
    if cls.__module__ in _TOP:
        uses.add(cls.__name__)
        return cls.__name__
    alias = _ALIAS.get(cls.__module__)
    if alias is None:
        raise ValueError(f"no import for {cls.__module__}")
    uses.add(alias)
    return f"{alias}.{cls.__name__}"


def _op(name: str, **fields: Any) -> str:
    """A patch op as a Python literal (from its JSON form: lists, True, None)."""
    return repr({"op": name, **fields})


def _time(t: float | MarkerRef) -> tuple[str, float]:
    """(when, offset) arguments for ``s.at``."""
    if isinstance(t, MarkerRef):
        return repr(t.marker), float(t.offset)
    return repr(float(t)), 0.0


def _ref(t: float | MarkerRef) -> str:
    if isinstance(t, MarkerRef):
        return f"({t.marker!r}, {float(t.offset)!r})"
    return repr(float(t))


def document_to_python(doc: Document) -> str:
    uses: set[str] = {"Scene"}
    m = doc.meta
    lines: list[str] = []
    args = [repr(m.title), f"theme={m.theme!r}", f"life={m.life!r}", f"seed={m.seed}",
            f"width={m.width}", f"height={m.height}", f"fps={m.fps!r}"]  # fmt: skip
    if m.look:
        args.append(f"look={m.look!r}")
    if m.palette:
        args.append(f"palette={m.palette!r}")
    lines.append(f"s = Scene({', '.join(args)})")
    cam = doc.cameras[0] if doc.cameras else None
    if cam is not None and cam.kind == "orbit":
        kw = ", ".join(f"{k}={_lit(v)}" for k, v in cam.params.items())
        lines.append(f"s.use_camera('orbit'{', ' + kw if kw else ''})")
    elif cam is not None:
        for k, v in cam.params.items():
            lines.append(f"s.set('{cam.id}.{k}', {_lit(v)})")
    lines.append("")
    if m.modules:  # code in the scene's folder (ADR 0005): its types and updaters
        modules = _op("set_meta", field="modules", value=list(m.modules))
        lines.append(f"s.engine.apply({modules}, origin='script')")
    for o in doc.objects:
        cls = registered_types().get(o.type)
        if o.always or cls is None or not is_builtin(cls):
            # A scene-local type or an updater has no import: the document's own op says it.
            obj = o.model_dump(mode="json")
            lines.append(f"s.engine.apply({_op('add_object', object=obj)}, origin='script')")
            continue
        ref = _class_ref(o.type, uses)
        parts = [f"id={o.id!r}"] + [f"{k}={_lit(v)}" for k, v in o.params.items()]
        if o.style:
            parts.append(f"style={_lit(dict(o.style))}")
        if o.tags:
            parts.append(f"tags={_lit(list(o.tags))}")
        if o.layer != "world":
            parts.append(f"layer={o.layer!r}")
        lines.append(f"s.add({ref}({', '.join(parts)}))")
    if doc.objects:
        lines.append("")
    owned = set(doc.narration.markers()) if doc.narration else set()
    for name, t in doc.timeline.markers.items():
        if name not in owned:
            lines.append(f"s.marker({name!r}, {float(t)!r})")
    if doc.narration is not None:
        lines.append("# (narration is not reproduced here: record it with s.narrate(...))")
    for path, segs in doc.timeline.tracks.items():
        for seg in segs:
            same_anchor = (
                isinstance(seg.t0, MarkerRef) == isinstance(seg.t1, MarkerRef)
                and (not isinstance(seg.t0, MarkerRef) or seg.t0.marker == seg.t1.marker)
            )  # fmt: skip
            if same_anchor:
                when, off = _time(seg.t0)
                end = seg.t1.offset if isinstance(seg.t1, MarkerRef) else seg.t1
                start = seg.t0.offset if isinstance(seg.t0, MarkerRef) else seg.t0
                at = f"s.at({when}" + (f", {off!r}" if off else "") + ")"
                lines.append(f"{at}.to({path!r}, {_lit(seg.to)}, duration={float(end - start)!r}, "
                             f"ease={seg.ease!r})")  # fmt: skip
            else:
                lines.append(f"s.segment({path!r}, {_ref(seg.t0)}, {_ref(seg.t1)}, {_lit(seg.to)}, "
                             f"ease={seg.ease!r})")  # fmt: skip
    for c in doc.constraints:
        lines.append(f"s.constrain({c.rule!r}, priority={c.priority!r})")
    for c in doc.controls:
        extra = f", component={c.component}" if c.component is not None else ""
        lines.append(f"s.control({c.path!r}, min={c.min!r}, max={c.max!r}, step={c.step!r}, "
                     f"label={c.label!r}{extra})")  # fmt: skip
    top = sorted(u for u in uses if u[0].isupper())
    mods = sorted(u for u in uses if u[0].islower())
    head = [
        "from phasor import "
        + ", ".join([*top, *(["expr"] if "expr(" in "\n".join(lines) else []), *mods])
    ]
    return "\n".join([*head, "", *lines]) + "\n"
