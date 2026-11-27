"""Mirror unfolding (the ``unfold`` verb): copies of an object carried through hinge turns.

    from phasor import HingedCopy

    s.add(HingedCopy(id="room.mirror_e1", source="room",
                     hinges=[((8, 2.5, 1.5), (0, 1, 0), 180.0)], turn=0.0))
    s.at("flip").morph("room.mirror_e1.turn", 1.0)      # the page turns

A ``HingedCopy`` draws another object (``source``) as it would look carried through a chain
of hinge turns (``hinges``: (point on the hinge line, its direction, degrees), first one
first), the last turned only ``turn`` of the way: at 0 the copy lies on top of where the
chain before it put it, at 1 it has swung right over. A hinge may carry a fourth entry, a
normal: the copy is then first mirrored in the plane through the hinge's point with that
normal (a half turn after it is the reflection in the wall: the true mirror image of a room
that is not symmetric). The ``unfold`` verb makes these copies for a ray bouncing in a room
(``core/unfold_math.py`` has the maths); a script can make others (a kaleidoscope, the net
of a cube) the same way.

The copy is the source drawn again, moved rigidly: the source draws itself for a camera
moved the opposite way, and every vertex it emits is moved back, so strokes still face the
real camera and keep their width mid-turn. It shows whatever the source shows (its own
opacity, dimming and colors) times the copy's opacity.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator

from phasor.core import unfold_math
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.spatial import Camera
from phasor.core.state import SceneState
from phasor.core.values import Vec3


def moved_camera(cam: Camera, m: np.ndarray) -> Camera:
    """The camera as seen from the copy's own frame: drawing there for this camera and then
    moving every vertex by ``m`` (a rigid 4x4) is the same as drawing the moved copy."""
    r = m[:3, :3]
    inv = np.linalg.inv(m)
    eye = inv[:3, :3] @ np.asarray(cam.eye, dtype=np.float64) + inv[:3, 3]
    return dataclasses.replace(
        cam, view_proj=np.asarray(cam.view_proj) @ m, eye=eye,
        right=r.T @ np.asarray(cam.right), up=r.T @ np.asarray(cam.up),
        forward=r.T @ np.asarray(cam.forward),
    )  # fmt: skip


class _Moved:
    """A ``DrawContext`` that moves everything drawn into it by ``m`` and fades it by
    ``opacity`` before passing it on."""

    def __init__(self, ctx: DrawContext, m: np.ndarray, opacity: float) -> None:
        self._ctx, self._m, self._r, self._op = ctx, m, m[:3, :3], opacity

    def _p(self, pts):
        return unfold_math.apply(self._m, pts)

    def _v(self, vecs):
        return np.asarray(vecs, dtype=np.float64) @ self._r.T

    def _c(self, color):
        return (*tuple(color)[:3], float(color[3]) * self._op)

    def triangles(self, positions, indices=None, *, color, glow=1.0, colors=None) -> None:
        self._ctx.triangles(self._p(positions), indices, color=self._c(color), glow=glow,
                            colors=colors)  # fmt: skip

    def field(self, corner, right, up, *, pixels, opacity=1.0, glow=1.0) -> None:
        self._ctx.field(self._p(corner), self._v(right), self._v(up), pixels=pixels,
                        opacity=opacity * self._op, glow=glow)  # fmt: skip

    def surface(self, positions, indices, *, normals, colors, uv=None, material="solid",
                opacity=1.0, glow=1.0) -> None:  # fmt: skip
        self._ctx.surface(self._p(positions), indices, normals=self._v(normals), colors=colors,
                          uv=uv, material=material, opacity=opacity * self._op,
                          glow=glow)  # fmt: skip

    def text(self, text, origin, height, *, color, anchor="center", glow=1.0,
             reveal=1.0) -> None:  # fmt: skip
        self._ctx.text(text, self._p(origin), height, color=self._c(color), anchor=anchor,
                       glow=glow, reveal=reveal)  # fmt: skip

    def note_path(self, points) -> None:
        note = getattr(self._ctx, "note_path", None)
        if note is not None:
            note(self._p(points))

    def __getattr__(self, name: str) -> Any:  # what else the renderer offers (stroke_hue)
        return getattr(self._ctx, name)


def _hinge(v):
    _point, axis, _degrees, *mirror = v
    if float(np.linalg.norm(axis)) < 1e-12:
        raise ValueError("a hinge needs a direction")
    if mirror and float(np.linalg.norm(mirror[0])) < 1e-12:
        raise ValueError("a hinge's mirror needs a normal")
    return v


@register
class HingedCopy(ObjectBase):
    """A copy of another object (``source``) carried through a chain of hinge turns, the
    last one ``turn`` of the way (0-1): a mirror room swinging over a wall like a page."""

    type: ClassVar[str] = "hinged_copy"
    Style: ClassVar = {}

    class Params(ParamsBase):
        source: str = Field(min_length=1)  # the id of the object copied (not a hinged copy)
        # (a point on the hinge line, its direction, degrees), first turn first; a fourth
        # entry, a normal, mirrors the copy first in the plane through the point
        hinges: tuple[tuple[Vec3, Vec3, float] | tuple[Vec3, Vec3, float, Vec3], ...] = Field(
            min_length=1
        )
        turn: float = 1.0  # share of the last hinge's turn done, 0-1

        @field_validator("hinges")
        @classmethod
        def _hinges(cls, v):
            return tuple(_hinge(h) for h in v)

        @field_validator("turn")
        @classmethod
        def _turn(cls, v):
            if not -1e-6 <= v <= 1 + 1e-6:  # an easing may land a hair past 1
                raise ValueError("turn runs from 0 to 1")
            return min(max(v, 0.0), 1.0)

    def _source(self, state: SceneState):
        """The source's state, or None when it is missing or cannot be copied."""
        p = self.me(state).params
        if p.source not in state:
            return None
        src = state[p.source]
        if src.type == self.type or src.layer == "overlay":
            return None
        return src

    def matrix(self, state: SceneState) -> np.ndarray:
        p = self.me(state).params
        return unfold_math.chain_matrix(p.hinges, p.turn)

    def world_bounds(self, state: SceneState) -> Bounds:
        src = self._source(state)
        if src is None:
            return Bounds.of_points([self.me(state).params.hinges[-1][0]])
        corners = src.impl.world_bounds(state).corners()
        return Bounds.of_points(unfold_math.apply(self.matrix(state), corners))

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        out = {"hinge": np.asarray(self.me(state).params.hinges[-1][0], dtype=np.float64)}
        src = self._source(state)
        if src is None:
            return out
        m = self.matrix(state)
        for name, pts in src.impl.features(state).items():
            arr = np.asarray(pts, dtype=np.float64)
            if arr.shape[-1:] == (3,):
                out.setdefault(name, unfold_math.apply(m, arr))
        return out

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        out: dict[str, Any] = {
            "kind": "hinged_copy", "of": p.source, "hinges": len(p.hinges),
            "turn": round(p.turn, 6), "landed": p.turn >= 1.0,
            "angle": round(p.hinges[-1][2] * p.turn, 4),
            "hinge": [round(float(v), 6) for v in p.hinges[-1][0]],
        }  # fmt: skip
        mirrors = sum(len(h) > 3 for h in p.hinges)
        if mirrors:  # each mirror flips the copy's handedness: it lands on the true image
            out["mirrors"] = mirrors
        src = self._source(state)
        if src is None:
            return {**out, "source_missing": p.source}
        center = unfold_math.apply(self.matrix(state), src.impl.world_bounds(state).center)
        out["center"] = [round(float(v), 6) for v in center]
        if src.impl.semantics(state).get("backdrop"):
            out["backdrop"] = True
        return out

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        src = self._source(state)
        if src is None or me.params.opacity <= 0:
            return
        m = self.matrix(state)
        seen = dataclasses.replace(state, camera=moved_camera(state.camera, m))
        src.impl.draw(_Moved(ctx, m, me.params.opacity), seen, pass_)
