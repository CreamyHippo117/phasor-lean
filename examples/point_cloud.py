"""A point cloud in the prism look: the roots of every polynomial with coefficients +-1, set
in motion.

All 2^16 = 65,536 polynomials of degree 16 whose coefficients are +1 or -1 (the leading one
+1), and their 1,048,576 roots, each a tiny splat of light. Where roots crowd (a ring round
the unit circle) the light builds up and runs white-hot; holes open at the roots of unity,
and the outer edge breaks into fractal lobes (the same shapes as the dragon curve). Colour
is each root's angle (its phase) taken round the palette, so the ring keeps its hues where
every polynomial crowds in.

Then the cloud moves: ``twist`` turns the coefficient +1 round the circle (to e^(2 pi i t)),
so every root glides, the ring tears open and fern-like tails grow from the tear; it turns
back as the camera falls toward z = i on the unit circle, where the hole is ringed with fine
structure, and the cloud settles into its first shape. The sparse points twinkle (life).
The splats are recomputed for the part on screen, so the detail stays sharp.

Run:  uv run python examples/point_cloud.py      (prints what the cloud is)
"""

from phasor import Equation, Scene, cloud

s = Scene("point_cloud", look="prism")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 3.6)
s.add(cloud.PointCloud(id="roots", source="roots", degree=16, coefficients=(-1.0, 1.0),
                       count=1_100_000, size=(6.4, 3.6), color_by="phase"))  # fmt: skip
s.add(Equation(id="title", pos=(0.0, -1.55, 0.0),
               parts=(r"$z^{16} \pm z^{15} \pm \cdots \pm z \pm 1 = 0$",)))  # fmt: skip

s.marker("appear", 0.3)
s.marker("turn", 2.0)
s.marker("fall", 4.0)
s.marker("end", 13.0)
s.at("appear").reveal("roots")
s.at("appear", 0.6).reveal("title")
s.at("turn").exit("title")
s.at("turn").to("roots.twist", 0.06, duration=3.5)  # the ring tears open
s.at("fall").to("camera.center", (0.0, 1.0, 0.0), duration=9.0)
s.at("fall").to("camera.height", 0.45, duration=9.0)
s.at("fall", 2.0).to("roots.twist", 0.0, duration=6.0)  # and closes as the camera lands

if __name__ == "__main__":
    sem = s.engine.inspect(0.0, id="roots")["objects"][0]["semantic"]
    print({k: sem[k] for k in ("source", "degree", "points", "in_rectangle")})
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
