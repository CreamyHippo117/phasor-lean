"""The object contract (contract 5, ADR 0005) and the type registry.

A type needs two members: ``Params`` and ``draw(ctx, state)`` (or, for a composite,
``build(state) -> [objects]`` instead of ``draw``). ``register`` refuses a type without
them. ``world_bounds``, ``features`` and ``semantics`` are the type's own when it defines
them and are derived from what ``draw`` records otherwise (``core/record.py``); a composite's
are the union of its parts'. ``draw`` may take a third argument, the pass (``color``, ``id``,
``solo``); one that does not is given a wrapper at registration that drops it, so every
renderer calls ``draw(ctx, state, pass_)``. After ``register`` every type has all five
``CONTRACT_MEMBERS``.

An instance is bound to one object id and holds no scene state: everything it needs comes
from the ``SceneState`` it is handed. The same instance doubles as the Python builder
(``Dot(id="dot", pos=(1, 2))``), so there is one class per type, not two.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from phasor.core.files import relative_path
from phasor.core.ids import is_valid_name
from phasor.core.schema import ObjectSpec
from phasor.core.state import ObjectState, SceneState
from phasor.core.values import Vec3

Pass = Literal["color", "id", "solo"]
PASSES: tuple[str, ...] = ("color", "id", "solo")
REQUIRED_MEMBERS = ("Params", "draw")  # ``build`` may stand in for ``draw`` (a composite)
DERIVED_MEMBERS = ("world_bounds", "features", "semantics")
CONTRACT_MEMBERS = ("Params", *DERIVED_MEMBERS, "draw")  # what every registered type has


class ParamsBase(BaseModel):
    """Parameters every object has."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    opacity: float = Field(1.0, ge=0, le=1)
    # A second opacity factor, owned by dim_others/undim so that dimming and revealing never
    # fight over one track. In a SceneState, ``opacity`` already includes it.
    dim: float = Field(1.0, ge=0, le=1)
    z_index: int = 0  # draw order among coplanar objects; higher draws on top
    # A brief brightening (the ``flash`` verb): the renderer adds light and glow by this much.
    flash: float = Field(0.0, ge=0, le=1)


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned box in world space, with its bounding sphere."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    @property
    def center(self) -> np.ndarray:
        return (np.asarray(self.lo) + np.asarray(self.hi)) / 2

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(np.asarray(self.hi) - np.asarray(self.lo)) / 2)

    def corners(self) -> np.ndarray:
        (x0, y0, z0), (x1, y1, z1) = self.lo, self.hi
        return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])

    @classmethod
    def of_points(cls, points, pad: float = 0.0) -> Bounds:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return cls(tuple(float(v) for v in p.min(0) - pad), tuple(float(v) for v in p.max(0) + pad))


class DrawContext(Protocol):
    """What an object may ask a renderer to draw. Renderer-agnostic and pass-agnostic: the
    renderer decides what a command means in the ``color``, ``id`` and ``solo`` passes."""

    def triangles(
        self,
        positions: np.ndarray,  # (N, 3) world space
        indices: np.ndarray | None = None,  # (M, 3) or None for a plain triangle list
        *,
        color: tuple[float, float, float, float],  # linear RGBA, not premultiplied
        glow: float = 1.0,  # emission: the color pass multiplies rgb by this (theme glow role)
        colors: np.ndarray | None = None,  # (N, 4) per vertex, replacing color's rgb (its alpha
        # still multiplies): a hue running along a curve. Only the color pass uses it.
    ) -> None: ...

    def field(
        self,
        corner: np.ndarray,  # world position of the quad's lower-left corner
        right: np.ndarray,  # world vector along the quad's bottom edge
        up: np.ndarray,  # world vector along its left edge
        *,
        pixels: Callable[[int, int, tuple[float, float], float], np.ndarray],
        opacity: float = 1.0,
        glow: float = 1.0,
    ) -> None:
        """A quad whose color is computed per pixel (domain coloring, fractals). It is one
        object in the id and solo passes. For the color pass the renderer calls
        ``pixels(width_px, height_px, center, height)`` with the visible part of the quad as
        a view in quad coordinates (``center`` and ``height`` measured along ``right`` and
        ``up`` from ``corner``, in world units), and expects linear RGBA (h, w, 4) float32,
        row 0 = top. Alpha is a per-pixel glow weight (0-1): with ``glow`` above 1 a pixel
        shines brighter and feeds the bloom by that weight; at ``glow`` 1 alpha is unused."""
        ...

    def surface(
        self,
        positions: np.ndarray,  # (N, 3) world space
        indices: np.ndarray,  # (M, 3)
        *,
        normals: np.ndarray,  # (N, 3), any sign: surfaces are lit from both sides
        colors: np.ndarray,  # (N, 4) linear RGBA per vertex, not premultiplied
        uv: np.ndarray | None = None,  # (N, 2) surface coordinates; the wire material grids them
        material: str = "solid",  # a surface role: solid | glass | wire | emissive
        opacity: float = 1.0,  # the material's opacity times the object's
        glow: float = 1.0,
    ) -> None:
        """A lit 3D surface (PBR-lite). The renderer lights it with the theme's
        three-point rig and depth fog; the id and solo passes see plain triangles."""
        ...

    def text(
        self,
        text: str,
        origin: np.ndarray,  # world position of the anchor
        height: float,  # world units, full line height
        *,
        color: tuple[float, float, float, float],
        anchor: str = "center",  # "center" | "left" | "right"
        glow: float = 1.0,  # as for triangles: above 1 the text emits light
        reveal: float = 1.0,  # the share shown, left to right (the ``write`` verb)
    ) -> None: ...


@runtime_checkable
class SceneObject(Protocol):
    type: ClassVar[str]
    Params: ClassVar[type[BaseModel]]
    id: str

    def world_bounds(self, state: SceneState) -> Bounds: ...
    def features(self, state: SceneState) -> dict[str, np.ndarray]: ...
    def semantics(self, state: SceneState) -> dict[str, Any]: ...
    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None: ...


class ObjectBase:
    """Convenience base. It deliberately implements none of the contract members."""

    type: ClassVar[str]
    # style key -> (role group, default role). Scenes may only use roles, never raw values.
    Style: ClassVar[dict[str, tuple[str, str]]] = {}
    # Named starting points: name -> {"params": {...}, "style": {...}}. ``preset="soft_phase"``
    # fills in what you did not set yourself; explicit values always win.
    Presets: ClassVar[dict[str, dict[str, dict[str, Any]]]] = {}
    # What living motion may do to this type (core/life.py): "glow" (light breathes),
    # "wander" (a point drifts by a share of its radius ``r``), "hue" (colors turn). Only
    # glow by default: geometry that means something (a polygon, a source) never drifts.
    Life: ClassVar[frozenset[str]] = frozenset({"glow"})
    # The parameters that name files the type reads (core/files.py): relative to the scene's
    # folder and inside it; a missing one is the lint issue ``missing_file``.
    Files: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        id: str,
        *,
        style: dict[str, str] | None = None,
        tags: tuple[str, ...] | list[str] = (),
        layer: str = "world",
        preset: str | None = None,
        **params: Any,
    ) -> None:
        self.id = id
        style = dict(style or {})
        if preset is not None:
            if preset not in self.Presets:
                raise ValueError(
                    f"{type(self).__name__} has no preset {preset!r}; presets are "
                    f"{sorted(self.Presets)}"
                )
            chosen = self.Presets[preset]
            params = {**chosen.get("params", {}), **params}
            style = {**chosen.get("style", {}), **style}
        self._spec_fields = {"style": style, "tags": tuple(tags), "layer": layer}
        self._params = params
        self._always: list[Callable] = []

    def always(self, fn: Callable) -> ObjectBase:
        """Keep some of this object's parameters set by ``fn(t, look) -> {param: value}``
        (ADR 0005). ``look("other")`` gives another object's parameters at ``t``
        (``look("dot").pos``); each one read is an edge in the parameter graph, so a cycle
        is an error. ``fn`` must be pure: the same answer for the same ``t`` and reads
        (lint ``impure_updater`` checks). It must be defined in the scene script or a
        ``*.types.py`` next to it. Call before ``Scene.add``; returns the object."""
        self._always.append(fn)
        return self

    def to_spec(self, base_dir: str | Path | None = None) -> ObjectSpec:
        """The object as the document stores it. ``base_dir``: the scene's folder, which
        names the modules its updaters live in."""
        always = tuple(register_updater(fn, base_dir or ".") for fn in self._always)
        return ObjectSpec(id=self.id, type=self.type, params=self._params, always=always,
                          **self._spec_fields)  # fmt: skip

    def me(self, state: SceneState) -> ObjectState:
        return state[self.id]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r})"


_REGISTRY: dict[str, type] = {}


class RegistrationError(TypeError):
    pass


def _takes_pass(draw: Callable) -> bool:
    """Whether ``draw`` accepts the pass as a third argument (after ``ctx`` and ``state``)."""
    if getattr(draw, "short_form", False):
        return True
    try:
        params = list(inspect.signature(draw, follow_wrapped=False).parameters.values())
    except (TypeError, ValueError):
        return True
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 4 or any(p.name == "pass_" for p in params)  # self, ctx, state, pass_


def _without_pass(draw: Callable) -> Callable:
    @functools.wraps(draw)
    def full(self, ctx: DrawContext, state: SceneState, pass_: Pass = "color") -> None:
        return draw(self, ctx, state)

    full.short_form = True  # type: ignore[attr-defined]
    return full


def _draw_parts(self, ctx: DrawContext, state: SceneState, pass_: Pass = "color") -> None:
    """A composite draws nothing itself: its parts are objects of their own in the state."""


def source_file(cls_or_fn: Any) -> str | None:
    """The file that defines a class or function, or None (built in C, typed at a prompt)."""
    code = getattr(cls_or_fn, "__code__", None)
    if code is not None:
        return str(Path(code.co_filename).resolve())
    try:
        src = inspect.getsourcefile(cls_or_fn)
    except (TypeError, OSError):
        return None
    return str(Path(src).resolve()) if src else None


def is_builtin(cls: type) -> bool:
    """Whether a type ships inside phasor (not defined in a scene or another package)."""
    return cls.__module__ == "phasor" or cls.__module__.startswith("phasor.")


def register(cls: type) -> type:
    """Class decorator: add an object type, or fail loudly if it breaks the contract.

    Required: ``Params`` (a ``ParamsBase``) and ``draw`` (or ``build``). The other members
    are filled in from ``core/record.py`` (or, for a composite, from its parts) when the
    type does not define them. A type may be registered again under the same name only by
    the file that defined it (a scene script run twice); built-in types never change."""
    from phasor.core import record

    name = getattr(cls, "type", None)
    if not isinstance(name, str) or not is_valid_name(name):
        raise RegistrationError(f"{cls.__name__}.type must be a name like 'dot', got {name!r}")
    composite = callable(getattr(cls, "build", None))
    missing = [m for m in REQUIRED_MEMBERS if not hasattr(cls, m)
               and not (m == "draw" and composite)]  # fmt: skip
    if missing:
        raise RegistrationError(f"{cls.__name__} breaks the object contract: missing {missing}")
    if not (isinstance(cls.Params, type) and issubclass(cls.Params, ParamsBase)):
        raise RegistrationError(f"{cls.__name__}.Params must subclass ParamsBase")
    if not cls.Params.__pydantic_complete__:
        # Annotations as text (``from __future__ import annotations``) in code that is not an
        # importable module (a notebook cell, exec'd text): resolve them where ``register`` ran.
        caller = inspect.currentframe().f_back  # type: ignore[union-attr]
        cls.Params.model_rebuild(_types_namespace=dict(caller.f_globals) if caller else None)
    for m in ("draw", *DERIVED_MEMBERS):
        if hasattr(cls, m) and not callable(getattr(cls, m)):
            raise RegistrationError(f"{cls.__name__}.{m} must be a method")
    bad = [f for f in getattr(cls, "Files", ()) if f not in cls.Params.model_fields]
    if bad:
        raise RegistrationError(f"{cls.__name__}.Files names {bad}, which are not parameters")
    source = source_file(cls)
    old = _REGISTRY.get(name)
    if old is not None and old is not cls:
        same_file = source is not None and getattr(old, "_phasor_source", None) == source
        if is_builtin(old) or not same_file:
            raise RegistrationError(f"object type {name!r} is already registered")
    if not hasattr(cls, "draw"):
        cls.draw = _draw_parts
    elif not _takes_pass(cls.draw):
        cls.draw = _without_pass(cls.draw)
    for m in DERIVED_MEMBERS:
        if not hasattr(cls, m):
            setattr(cls, m, (record.COMPOSITE if composite else record.DERIVED)[m])
    cls._phasor_source = source
    _REGISTRY[name] = cls
    return cls


def unregister(name: str) -> None:
    """Forget a type (a scene module that failed to load takes its types with it)."""
    _REGISTRY.pop(name, None)


def get_type(name: str) -> type | None:
    return _REGISTRY.get(name)


def registered_types() -> dict[str, type]:
    return dict(_REGISTRY)


def registry_snapshot() -> tuple[dict[str, type], dict[str, Callable]]:
    return dict(_REGISTRY), dict(_UPDATERS)


def restore_registry(snapshot: tuple[dict[str, type], dict[str, Callable]]) -> None:
    """Put the registries back as ``registry_snapshot`` found them (a module that failed to
    load leaves none of its types or updaters behind)."""
    _REGISTRY.clear()
    _REGISTRY.update(snapshot[0])
    _UPDATERS.clear()
    _UPDATERS.update(snapshot[1])


# ---------------------------------------------------------------- code in the scene's folder


def scene_module(thing: Any, base_dir: str | Path) -> str | None:
    """The module a scene-local type or updater lives in, relative to the scene's folder
    (``"scene.py"``, ``"shapes.types.py"``), or None when it is defined anywhere else (in
    phasor, an installed package, a test file outside the folder, a prompt)."""
    if isinstance(thing, type) and is_builtin(thing):
        return None
    src = getattr(thing, "_phasor_source", None) if isinstance(thing, type) else None
    src = src or source_file(thing)
    if not src or not src.endswith(".py"):
        return None
    root = Path(base_dir or ".").resolve()
    path = Path(src)
    if root not in path.parents:
        return None
    try:
        return relative_path(path.relative_to(root).as_posix())
    except ValueError:
        return None


_UPDATERS: dict[str, Callable] = {}  # "<module>:<name>" -> updater function


def updater_name(fn: Callable) -> str:
    """How a document names an updater inside its module: its qualified name, and for a
    lambda or a nested function also the line it starts on (``"<lambda>@12"``)."""
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "updater")
    if "<" in name:
        name += f"@{fn.__code__.co_firstlineno}"
    return name


def register_updater(fn: Callable, base_dir: str | Path) -> str:
    """Remember ``fn`` and return the reference a document stores for it,
    ``"<module>:<name>"``. ``fn`` must be defined in a file in the scene's folder (the
    document then lists the module), so the document can always find it again (contract 3:
    the state depends only on the document, t and the files it names)."""
    if not callable(fn):
        raise TypeError(
            f"an updater must be a function fn(t, look) -> {{param: value}}, got {fn!r}"
        )
    module = scene_module(fn, base_dir)
    if module is None:
        raise ValueError(
            f"updater {updater_name(fn)!r} is not defined in a file in the scene's folder "
            f"({base_dir}); define it in the scene script or a *.types.py next to it, so the "
            "document can name it"
        )
    ref = f"{module}:{updater_name(fn)}"
    _UPDATERS[ref] = fn
    return ref


def get_updater(ref: str) -> Callable | None:
    return _UPDATERS.get(ref)


class Placeholder(ObjectBase):
    """What an object of a type that is not available (its module is missing, refused or
    broken: lint ``missing_type``) draws: an outlined box at its ``pos``. It keeps the
    object's id, type name, parameters and overrides; nothing in the document changes.
    Not registered: a document cannot ask for one."""

    type: ClassVar[str] = "placeholder"
    Style: ClassVar = {"color": ("emphasis", "danger")}
    SIDE = 0.08  # share of the frame height

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)

    def world_bounds(self, state: SceneState) -> Bounds:
        c, right, up, h = self._frame(state)
        return Bounds.of_points([c + sx * h * right + sy * h * up
                                 for sx in (-1, 1) for sy in (-1, 1)])  # fmt: skip

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        return {"center": self._frame(state)[0]}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        return {"kind": "placeholder", "type": self.me(state).type}

    def _frame(self, state: SceneState):
        me = self.me(state)
        cam = state.camera_for(me)
        c = np.asarray(me.params.pos, dtype=np.float64)
        return c, cam.right, cam.up, self.SIDE / 2 * cam.units_per_height(c)

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass = "color") -> None:
        c, right, up, h = self._frame(state)
        w = h / 8
        color = (*self.me(state).style["color"][:3], self.me(state).params.opacity)
        for a, b in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)),
                     ((-1, 1), (-1, -1))):  # fmt: skip
            p0 = c + a[0] * h * right + a[1] * h * up
            p1 = c + b[0] * h * right + b[1] * h * up
            along = (p1 - p0) / np.linalg.norm(p1 - p0)
            side = np.cross(np.cross(right, up), along) * w
            quad = np.array([p0 - side, p1 - side, p1 + side, p0 + side])
            ctx.triangles(quad, np.array([[0, 1, 2], [0, 2, 3]]), color=color)
