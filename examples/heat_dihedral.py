"""The heat equation and the dihedral group.

Left: a rod whose temperature profile u(x, t) is computed by a finite-difference solver, drawn
beside the exact solution (they sit on top of each other); a hot plate cooling beside it. Right:
a square that the group D4 acts on (turn it a quarter, then flip it) next to the group's Cayley
table. Every number you see is something the inspector can read back.

Run:  uv run python examples/heat_dihedral.py      (prints the readings at each marker)
"""

from phasor import Scene, algebra, expr, pde

s = Scene("heat_dihedral", look="calm")
s.set("camera.height", 11)

s.add(
    pde.Profile(
        id="rod",
        kind="heat",
        length=2.0,
        diffusivity=0.5,
        modes=(1.0, 0.4),
        time=expr("t / 10"),
        pos=(-9, 1.0, 0),
        extent=(5.5, 2.6),
        preset="glowing",
    )
)
s.add(
    pde.Field(
        id="plate",
        kind="heat",
        mode=(0, 0),
        diffusivity=0.5,
        time=expr("t / 10"),
        size=(4.2, 4.2),
        pos=(-6.2, -3.2, 0),
    )
)

s.add(algebra.Polygon(id="square", n=4, pos=(0.2, 2.4, 0), r=1.5))
s.add(algebra.CayleyTable(id="table", group="D4", pos=(5.4, -1.2, 0), cell=0.7))

s.marker("turn", 2.0)
s.marker("flip", 4.0)
s.marker("turn_again", 6.0)
s.at("turn").morph("square.rotation", 90.0)
s.at("flip").morph("square.flip", 1.0)
s.at("turn_again").morph("square.rotation", 180.0)

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 3.5, 5.5, 8.0, 9.0):
        rod = e.inspect(t, id="rod")["objects"][0]["semantic"]
        plate = e.inspect(t, id="plate")["objects"][0]["semantic"]
        sq = e.inspect(t, id="square")["objects"][0]["semantic"]
        print(
            f"t={t:g}: rod max {rod['max']:.3f} (error {rod['max_error_vs_exact']:.4f}), "
            f"plate heat {plate['total_heat']:.3f}, square = {sq['element']}"
        )
