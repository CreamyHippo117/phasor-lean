"""The Buddhabrot: the Mandelbrot set's cloud mode, its escaping orbits drawn as light.

Pick c at random, iterate z -> z^2 + c from z = 0, and if the orbit escapes, add every point
it visited to the picture. Eight million such orbit points, each a tiny splat, build up a
glowing ghost of the Mandelbrot set: bright where many orbits pass (only points outside the
set have orbits that escape, yet those orbits light up the inside). Colour is how long the
orbit took to escape (on a log scale), so quick orbits and slow ones take different hues.
Then the camera drifts in toward the upper half while the palette turns a little.

Run:  uv run python examples/buddhabrot.py      (prints how many orbit points are in view)
"""

from phasor import Scene, cx

s = Scene("buddhabrot", look="prism")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 4.5)
s.add(cx.Fractal(id="cloud", kind="buddhabrot", pos=(0, 0, 0), size=(8, 4.5),
                 center="-0.45 0", zoom=0.2, max_iter=1000, count=8_000_000,
                 style={"colormap": "spectrum"}))  # fmt: skip

s.marker("appear", 0.3)
s.marker("drift", 2.5)
s.marker("end", 11.0)
s.at("appear").reveal("cloud")
s.at("drift").to("camera.center", (0.4, 0.5, 0.0), duration=8.0)
s.at("drift").to("camera.height", 3.0, duration=8.0)
s.at("drift").to("cloud.hue", 0.15, duration=8.0)

if __name__ == "__main__":
    sem = s.engine.inspect(0.0, id="cloud")["objects"][0]["semantic"]
    print({k: sem[k] for k in ("set", "max_iter", "points", "in_rectangle")})
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
