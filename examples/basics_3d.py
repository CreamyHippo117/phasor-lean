"""The 3D basics, in one scene you can turn around: axes with ticks, a helix (its formula
below) with the hue running along its length, a sphere defined by a formula that *maps* from
a flat input square to its output shape (the colors show height), and a vector field with its
flow lines.

The camera orbits once (the ``orbit`` verb); in the editor drag to turn it yourself.

Run:  uv run python examples/basics_3d.py      (prints what the inspector reads)
"""

import math

from phasor import Equation, Scene, expr, geo

s = Scene("basics_3d", look="prism_everywhere")
s.use_camera("orbit")
s.set("camera.target", (0.0, 0.0, 1.0))
s.set("camera.distance", 20.0)
s.set("camera.elevation", 24.0)
s.set("camera.azimuth", -55.0)

s.add(geo.Axes(id="axes", x_range=(-6, 6, 2), y_range=(-4, 4, 2), z_range=(-1, 5, 1),
               labels=("x", "y", "z"), pos=(0, 0, -1)))  # fmt: skip
s.add(geo.ParametricCurve(id="helix", fx="1.2*cos(t)", fy="1.2*sin(t)", fz="0.18*t - 1",
                          t_range=(0, 12 * math.pi), samples=900,
                          progress=expr("min(t / 4, 1)"), style={"stroke": "bold"}))  # fmt: skip
s.add(geo.ParametricSurface(id="ball", fx="2*sin(u)*cos(v)", fy="2*sin(u)*sin(v)",
                            fz="2*cos(u)", u_range=(0, math.pi / 2), v_range=(0, 2 * math.pi),
                            res=(36, 72), color_by="height", with_map=True,
                            domain_pos=(-5.5, 0, 0), morph=expr("clamp((t - 4) / 3, 0, 1)"),
                            pos=(3.0, 0, 0)))  # fmt: skip
s.add(geo.FlowLines(id="flow", fp="-y", fq="x", fr="0.15", pos=(3.0, 0.0, 0.5), extent=(5, 5, 0),
                    seeds=(4, 4, 1), length=3.0))  # fmt: skip
s.add(Equation(id="helix_eq", pos=(8 / 9, 0.9), layer="overlay",
               parts=("$\\gamma(t) = ($", "$1.2\\cos t,\\ 1.2\\sin t,\\ 0.18\\,t - 1$", "$)$"),
               colors=("primary", "secondary", "primary")))  # fmt: skip
s.marker("start", 0.0)
s.marker("end", 12.0)
s.at("start").orbit(360.0, until="end")

if __name__ == "__main__":
    e = s.engine
    for oid in ("helix", "ball", "flow"):
        print(oid, e.inspect(10.0, id=oid)["objects"][0]["semantic"])
