"""A kaleidoscope is mirror rooms (the ``unfold`` verb on a polygon), in the prism look.

A ray bounces round a triangle of mirrors: first its real path is drawn, a zig-zag that
never seems to settle. Then the same ray again, but at each mirror the triangle makes a
mirror copy of itself across it, swinging over the mirror's edge like a page, and the ray
carries straight on into the copy. The copies of an equilateral triangle tile the plane
(each corner is 60 degrees, a sixth of a turn), which is why a kaleidoscope's picture closes
up without a seam.

Run:  uv run python examples/kaleidoscope.py      (prints the mirrors the ray reflects off)
"""

import math

import numpy as np

from phasor import Line, Scene
from phasor.core import unfold_math

R = 1.5  # the triangle's circumradius, a corner at the top
CORNERS = [(R * math.cos(math.pi / 2 + 2 * math.pi * k / 3),
            R * math.sin(math.pi / 2 + 2 * math.pi * k / 3), 0.0) for k in range(3)]  # fmt: skip
START = (0.15, -0.35, 0.0)
AIM = (1.0, 0.3, 0.0)  # the way the ray leaves
FLIPS = 7

# The real path, from the same maths the verb uses: the straight ray folded back.
room = unfold_math.polygon(CORNERS)  # walls a, b, c: from each corner to the next
unfolded = unfold_math.room_unfold(room, START, AIM, FLIPS)
# Frame every copy: each is the triangle reflected in the walls crossed to reach it.
seen = np.vstack([unfold_math.apply(unfold_math.copy_matrix(room, c.word), room.vertices)
                  for c in unfolded.crossings] + [room.vertices])  # fmt: skip
lo, hi = seen.min(axis=0), seen.max(axis=0)

s = Scene("kaleidoscope", look="prism")
s.set("camera.center", tuple((lo + hi) / 2))
s.set("camera.height", 1.15 * max(hi[1] - lo[1], (hi[0] - lo[0]) * 9 / 16))
s.add(Line(id="mirrors", points=[*CORNERS, CORNERS[0]], preset="glowing"))
s.add(Line(id="bounces", points=[tuple(p) for p in unfold_math.zigzag(unfolded)]))
s.add(Line(id="ray", points=[START, tuple(a + b for a, b in zip(START, AIM, strict=True))],
           preset="beam"))  # fmt: skip

s.marker("bounce", 0.5)
s.marker("unfold", 4.0)
s.at("bounce").trace("bounces")
s.at("unfold").dim_others("mirrors", "ray")
s.at("unfold").unfold("mirrors", ray="ray", flips=FLIPS)

if __name__ == "__main__":
    e = s.engine
    print("mirrors:", [c.wall for c in unfolded.crossings])
    print("copies:", [o.id for o in e.document.objects if o.type == "hinged_copy"])
    for issue in e.lint():
        print("lint:", issue)
