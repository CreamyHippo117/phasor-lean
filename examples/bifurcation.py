"""The bifurcation diagram of the logistic map x -> r x (1 - x), as a glowing point cloud.

For every r across the picture the orbit of x = 1/2 runs 600 steps and its last 200 points
are drawn, each a tiny splat of light. Where an orbit settles on a cycle every point lands
on the same few pixels and the branch burns bright; past r = 3.5699 the orbit is chaotic and
its points smear into a glowing haze. Colour is the Lyapunov exponent: cycles take one hue,
chaos another.

The diagram sweeps in from the left; then the window zooms (r and x by different factors)
into the first period doublings, at r = 3 and r = 1 + sqrt(6), and the cascade that follows
them; it pulls back out and dives into the window of period three, which opens at
r = 1 + sqrt(8) and splits again into its own small copy of the whole picture. The columns
of r are recomputed for each view, about a pixel apart, so the zoom stays sharp.

Run:  uv run python examples/bifurcation.py      (prints what the diagram is)
"""

from phasor import Equation, Scene, cloud

s = Scene("bifurcation", look="prism")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 4.5)
s.add(cloud.PointCloud(id="diagram", preset="prism_bifurcation", map="logistic",
                       r_range=(2.8, 4.0), x_range=(0.0, 1.0), size=(7.6, 4.1), sweep=0.0,
                       r_focus=(2.95, 3.6), x_focus=(0.28, 0.94)))  # fmt: skip
s.add(Equation(id="rule", pos=(-2.35, 1.72, 0.0), drift=0.6,
               parts=(r"$x_{n+1} = r\,x_n\,(1 - x_n)$",)))  # fmt: skip

s.marker("appear", 0.3)
s.marker("doublings", 5.5)
s.marker("back", 11.0)
s.marker("three", 14.0)
s.marker("end", 22.0)
s.at("appear").reveal("diagram")
s.at("appear").to("diagram.sweep", 1.0, duration=4.5)  # r runs left to right
s.at("appear", 1.0).reveal("rule")
s.at("doublings").exit("rule")
s.at("doublings").to("diagram.focus", 1.0, duration=4.0)  # into the first doublings
s.at("back").to("diagram.focus", 0.0, duration=2.5)
# at focus 0 the window is r_range x x_range whatever the focus is, so it can move unseen
s.at("three").set("diagram.r_focus", (3.822, 3.862))
s.at("three").set("diagram.x_focus", (0.0, 1.0))
s.at("three", 0.2).to("diagram.focus", 1.0, duration=5.5)  # the window of period three

if __name__ == "__main__":
    sem = s.engine.inspect(20.0, id="diagram")["objects"][0]["semantic"]
    print({k: sem[k] for k in ("source", "map", "window", "points", "in_rectangle")})
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
