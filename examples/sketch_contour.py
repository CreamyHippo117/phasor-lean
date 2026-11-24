"""Draw a contour; it integrates and animates.

A loop drawn by hand around the pole at i of 1/(z^2+1) (here a fixed wobbly stroke in
output pixels, as a pen would give it) becomes a glowing contour, traced from the marker
"draw", with its integral worked out along the drawn path and exactly from the residue:
pi. In the editor, the Sketch tab does the same with a real pen or mouse.

Run:  uv run python examples/sketch_contour.py      (prints the proposal)
"""

import math

from phasor import Scene, cx
from phasor.sketch.propose import Sketch

s = Scene("sketch_contour")
s.add(cx.DomainColoring(id="plane", function="1/(z**2+1)", size=(8.0, 4.5)))
s.marker("draw", 0.5)
s.marker("end", 3.5)

# A hand-drawn loop around i, counterclockwise on screen: wobbly, not quite closed.
sketch = Sketch(s.engine, t=0.5)
cx_, cy_ = sketch.to_pixels([[0.0, 1.0, 0.0]])[0]
r = sketch.to_pixels([[0.7, 1.0, 0.0]])[0][0] - cx_
stroke = [(cx_ + r * (1 + 0.06 * math.sin(3 * a)) * math.cos(a) + 2 * math.sin(11 * a),
           cy_ - r * (1 + 0.06 * math.sin(3 * a)) * math.sin(a))
          for a in (2 * math.pi * k / 90 * 0.97 for k in range(91))]  # fmt: skip

(proposal,) = sketch.propose([stroke])
result = s.engine.apply(proposal.ops, origin="script")  # the scene's own script accepts it
assert result.ok, result.errors
for step in proposal.animate:  # trace the contour, reveal its value, from the marker
    getattr(s.at("draw"), step["verb"])(*step["targets"])

if __name__ == "__main__":
    print(proposal.summary)
    print({k: proposal.data[k] for k in ("integral", "exact", "latex")})
