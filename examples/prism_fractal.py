"""Fractals in the prism look: the edge of the set as light coming out of the dark.

First a Julia set of z^2 + c whose constant c drifts slowly along the circle |c| = 0.7885,
just outside the Mandelbrot set's main body, so the set morphs as it goes; then a short dive
(a million times closer) into a spiral of the Mandelbrot set. Inside each set is the
background; the edge glows, measured in pixels from the escape count itself (no derivative
carried through the iteration), so it stays as fine and as bright at every depth.

Run:  uv run python examples/prism_fractal.py      (prints where c is at each marker)
"""

from phasor import Scene, cx

SPIRAL = "-0.7756837680090537974694835039347410457282 0.1364673682946901247332744096178487651978"
# c = 0.7885 e^(i a) for a = 2.75, 2.95, 3.15, 3.35: four stops along the drift
PATH = [(-0.7288, 0.3010), (-0.7741, 0.1501), (-0.7885, -0.0066), (-0.7712, -0.1639)]

s = Scene("prism_fractal", look="prism")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 4.5)
s.add(cx.Fractal(id="julia", kind="julia", c=PATH[0], pos=(0, 0, 0), size=(8, 4.5),
                 center="0 0", zoom=0.25, max_iter=300))  # fmt: skip
s.add(cx.Fractal(id="dive", center=SPIRAL, pos=(0, 0, 0), size=(8, 4.5), max_iter=1500,
                 steady=True, opacity=0.0))  # fmt: skip

s.marker("appear", 0.3)
s.marker("drift", 1.5)
s.marker("dive", 8.0)
s.marker("end", 14.0)
s.at("appear").reveal("julia")
for k, c in enumerate(PATH[1:]):
    s.at("drift", 2.0 * k).to("julia.c", c, duration=2.0, ease="linear")
s.at("dive").exit("julia")
s.at("dive").reveal("dive")
s.at("dive", 0.5).to("dive.zoom", 6.0, duration=5.0)

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 1.5, 3.5, 5.5, 7.5, 13.0):
        st = e.state_at(t)
        print(f"t={t:g}: c = {st['julia'].params.c}, dive zoom {st['dive'].params.zoom:.1f}")
    for issue in e.lint(strict=True):
        print("lint:", issue)
