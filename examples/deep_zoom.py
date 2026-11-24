"""A dive into the Mandelbrot set down to 1e-30.

The view starts on the whole set and falls, at a steady ten-fold every third of a second,
into a spiral at a Misiurewicz point (where the orbit of 0 lands on a repelling fixed point
after 24 steps: it looks the same at every scale). Past 1e-5 the kernel switches to float64,
past 1e-10 per pixel to perturbation against a reference orbit computed exactly; at the
bottom the picture is 1e-30 tall and every pixel still agrees with exact iteration
(``tests/test_deep_zoom.py``).

Run:  uv run python examples/deep_zoom.py      (prints the precision at each marker)
"""

from phasor import Scene, Text, cx

SPIRAL = "-0.7756837680090537974694835039347410457282 0.1364673682946901247332744096178487651978"

s = Scene("deep_zoom", look="glow")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 4.5)
s.add(cx.Fractal(id="set", center=SPIRAL, pos=(0, 0, 0), size=(8, 4.5), max_iter=3000,
                 steady=True, preset="ocean"))  # fmt: skip
s.add(Text(id="depth", text="10^30 times closer", pos=(0, -1.8, 0), style={"size": "title"}))

s.marker("dive", 1.0)
s.marker("bottom", 11.0)
s.at("dive").to("set.zoom", 30.0, duration=10.0)  # an eased dive: 30 powers of ten
s.at("bottom").reveal("depth")

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 3.0, 4.0, 5.0, 11.0, 12.0):
        sem = e.inspect(t, id="set")["objects"][0]["semantic"]
        print(f"t={t:g}: zoom {sem['zoom']:.1f}, view {sem['view_height']:.2g} tall, "
              f"{sem['precision']}")  # fmt: skip
