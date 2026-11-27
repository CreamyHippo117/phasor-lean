"""The s-plane rolls onto the z-plane (z = e^{sT}): a conformal roll, then an even dive.

The grid of the left half plane (Re s from -7 to 0) is pushed through the family
f_a(s) = (e^{as} - 1)/a (``cx.ComplexMap`` preset ``roll``), a holomorphic map at every
step: the jw axis, tangent to the unit circle at 1, rolls round it, and the lines
Re s = x curl into the circles |z| = e^x. Then the camera dives 600x into z = 0, where
those circles pile up. A camera's height moves by factors (``log:`` easing), so the
circles, evenly spaced in s, pass by at an even pace.

Run:  uv run python examples/s_plane_roll.py      (prints what the inspector reads)
"""

from phasor import Scene, cx

s = Scene("s_plane_roll")
s.set("camera.height", 8.0)
s.set("camera.center", (-2.0, 0.0, 0.0))
# Drawn at pos = center + 1, the roll ends on e^s itself (f_1 = e^s - 1).
s.add(cx.ComplexMap(id="splane", preset="roll", center=(-3.5, 0.0), pos=(-2.5, 0.0, 0.0),
                    extent=(7.0, 6.2832), lines=(15, 13), samples=240, amount=0.0))  # fmt: skip

for name, t in (("start", 0.0), ("roll", 0.5), ("rolled", 4.5), ("dive", 5.0), ("end", 11.0)):
    s.marker(name, t)
s.at("roll").morph("splane.amount", 1.0, until="rolled")
s.at("rolled").morph("camera.center", (0.0, 0.0, 0.0), until="dive")
s.at("dive").morph("camera.height", 8.0 / 600, until="end")  # by factors: an even dive

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 2.5, 4.5, 8.0, 11.0):
        print(t, e.state_at(t).values["camera.height"], e.inspect(t, id="splane")["objects"][0])
