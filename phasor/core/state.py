"""SceneState: everything resolved at one time t. Immutable (contract 3)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from phasor.core.spatial import Camera

if TYPE_CHECKING:
    from phasor.core.objects import SceneObject


@dataclass(frozen=True)
class ObjectState:
    id: str
    type: str
    index: int  # 1-based, document order; the value written to the id pass (0 = background)
    params: BaseModel  # the type's frozen Params model, fully resolved
    style: Mapping[str, Any]  # style key -> resolved theme value (linear RGBA, fractions, ...)
    roles: Mapping[str, str]  # style key -> role name
    tags: tuple[str, ...]
    layer: str
    impl: SceneObject = field(repr=False, compare=False)  # stateless behavior for this object
    # The composite that built this object (ADR 0005), or None for an object of the document.
    parent: str | None = None


@dataclass(frozen=True)
class SceneState:
    t: float
    width: int
    height: int
    background: tuple[float, float, float, float]  # linear RGBA
    objects: tuple[ObjectState, ...]
    camera: Camera = field(compare=False)  # the active world camera
    overlay: Camera = field(compare=False)
    values: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    # The theme's three-point light rig and fog (``[light]``), for lit surfaces.
    light: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}), compare=False)
    # The theme's ``[strokes]``: a hue run along glowing strokes (``hue_along`` 0: off).
    strokes: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), compare=False)
    # The scene's folder: the files objects read (``Files``, core/files.py) are relative to it.
    # Empty: the current folder.
    base_dir: str = field(default="", compare=False)
    _by_id: Mapping[str, ObjectState] = field(default=None, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_id", MappingProxyType({o.id: o for o in self.objects}))

    def __getitem__(self, object_id: str) -> ObjectState:
        return self._by_id[object_id]

    def __contains__(self, object_id: object) -> bool:
        return object_id in self._by_id

    def draw_order(self) -> tuple[ObjectState, ...]:
        """Back to front: world before overlay, then z_index, then document order. Every pass
        and every hit test uses this one ordering."""
        return tuple(
            sorted(self.objects, key=lambda o: (o.layer == "overlay", o.params.z_index, o.index))
        )

    def camera_for(self, obj: ObjectState) -> Camera:
        return self.overlay if obj.layer == "overlay" else self.camera

    @property
    def aspect(self) -> float:
        return self.width / self.height
