"""Echoes are mirror rooms (the ``unfold`` verb), in the prism look.

A ray of sound leaves the source and bounces round a room: first its real zig-zag path is
drawn. Then the same ray again, but instead of turning at a wall, the room makes a mirror
copy of itself across that wall, swinging over like a page turning on the wall's middle,
and the ray carries straight on into the copy. After five flips the zig-zag is one straight
line through a row of mirror rooms: each echo is the sound of a mirror room's source (an
image source).

Run:  uv run python examples/mirror_rooms.py      (prints the walls the ray reflects off)
"""

from phasor import Line, Scene, ac
from phasor.core import unfold_math

SIZE, SRC = (4.0, 2.5, 2.0), (0.8, 0.6, 1.0)
AIM = (1.0, 0.45, 0.0)  # the way the ray leaves: level, so it meets only the walls
FLIPS = 5

# The real path, from the same maths the verb uses: the straight ray folded back.
unfolded = unfold_math.box_unfold((0, 0, 0), SIZE, SRC, AIM, FLIPS)
far = unfolded.crossings[-1].cell_to  # the last mirror room: frame the whole row
middle = tuple((1 + c) * L / 2 for c, L in zip(far, SIZE, strict=True))

s = Scene("mirror_rooms", look="prism")
s.use_camera("orbit", azimuth=-96, elevation=52, distance=21.0, target=middle)
s.add(ac.Room3D(id="room", size=SIZE, material="wood", preset="outline"))
s.add(ac.Source(id="src", pos=SRC, tags=["key"]))
s.add(Line(id="bounces", points=[tuple(p) for p in unfold_math.zigzag(unfolded, (0, 0, 0), SIZE)]))
# The ray to unfold: where it starts and the way it goes (the verb makes it long enough).
s.add(Line(id="ray", points=[SRC, tuple(a + b for a, b in zip(SRC, AIM, strict=True))]))

s.marker("bounce", 0.5)
s.marker("unfold", 4.0)
s.at("bounce").trace("bounces")
s.at("unfold").dim_others("room", "ray", "src")
s.at("unfold").unfold("room", ray="ray", flips=FLIPS)

if __name__ == "__main__":
    e = s.engine
    print("walls:", [c.wall for c in unfolded.crossings])
    for t in (5.0, 8.0, 12.0):
        copies = [o for o in e.document.objects if o.type == "hinged_copy"]
        print(f"t={t:g}:", {o.id: e.inspect(t, id=o.id)["objects"][0]["semantic"]["turn"]
                            for o in copies})  # fmt: skip
    for issue in e.lint():
        print("lint:", issue)
