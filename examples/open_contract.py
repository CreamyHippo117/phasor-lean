"""A scene that brings its own types: the open object contract (ADR 0005).

- ``Diamond`` is a new shape in about ten lines: ``Params`` and ``draw``, nothing else. Its
  bounds, features and semantics come from what it draws, so the inspector, layout and lint
  know it like any other object.
- ``Flag`` is a composite: ``build`` makes it out of a ``Line`` and a ``Diamond``, and its
  parts are inspected as themselves (``flag.pole``, ``flag.tip``).
- ``turn`` is a ``Value``: a number to animate, with no drawing. The planet goes round as it
  runs from 0 to 1 (an updater reads it), and another updater keeps a label on the planet.

The document names this file in ``meta.modules``, so a saved scene finds its types again
where scripts may run.

Run:  uv run python examples/open_contract.py        (prints the inspector's view and lint)
"""

from typing import ClassVar

import numpy as np

from phasor import Dot, Line, Scene, Text, Value
from phasor.core.objects import ObjectBase, ParamsBase, register
from phasor.core.values import Vec3


@register
class Diamond(ObjectBase):
    type: ClassVar[str] = "diamond"
    Style: ClassVar = {"color": ("emphasis", "highlight"), "glow": ("glow", "soft")}

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        r: float = 0.3

    def draw(self, ctx, state):
        me = self.me(state)
        (x, y, z), r = me.params.pos, me.params.r
        pts = np.array([[x, y, z], [x + r, y, z], [x, y + r, z], [x - r, y, z], [x, y - r, z]])
        rgba = (*me.style["color"][:3], me.style["color"][3] * me.params.opacity)
        ctx.triangles(pts, np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]), color=rgba,
                      glow=me.style["glow"])  # fmt: skip


@register
class Flag(ObjectBase):
    type: ClassVar[str] = "flag"

    class Params(ParamsBase):
        pos: Vec3 = (0.0, 0.0, 0.0)
        height: float = 1.6

    def build(self, state):
        (x, y, z), h = self.me(state).params.pos, self.me(state).params.height
        return [Line(id="pole", points=[(x, y, z), (x, y + h, z)], style={"color": "muted"}),
                Diamond(id="tip", pos=(x, y + h, z), r=0.28)]  # fmt: skip


def orbit(t, look):
    """The planet goes once round the sun as ``turn`` runs from 0 to 1."""
    a = 2 * np.pi * look("turn").value + 0.6
    return {"pos": (2.6 * np.cos(a), 1.7 * np.sin(a), 0.0)}


def follow(t, look):
    """The planet's name stays just above it."""
    x, y, z = look("planet").pos
    return {"pos": (x, y + 0.5, z)}


s = Scene("open_contract", look="prism")
s.set("camera.height", 7)
s.add(Value(id="turn", value=0.0))
s.add(Diamond(id="sun", r=0.45))
s.add(Dot(id="planet", r=0.14, style={"color": "secondary"}, tags=["exact"]).always(orbit))
s.add(Text(id="planet.label", text="planet").always(follow))
s.add(Flag(id="flag", pos=(-4.6, -1.6, 0)))
s.add(Text(id="flag.label", text="a composite", pos=(-4.6, -2.1, 0)))
s.add(Text(id="title", text="types of its own", pos=(8 / 9, 0.08), layer="overlay",
           style={"size": "title"}))  # fmt: skip
s.constrain("label(flag) near flag")

s.marker("start", 0.3)
s.marker("go", 1.2)
s.marker("end", 6.0)
s.at("start").reveal("title", "sun", "flag")
s.at("go").morph("turn.value", 1.0, until="end")

if __name__ == "__main__":
    e = s.engine
    for line in e.inspect(3.0, detail="brief")["objects"]:
        print(line)
    for issue in e.lint(strict=True):
        print("lint:", issue)
