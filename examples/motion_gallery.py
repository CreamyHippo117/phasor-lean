"""The motion verbs, one beat each: an equation becomes the next, a circle becomes a
square, the plane bends under z^2, a comet travels a curve with its progress counting up,
particles stream along a field while a Fourier sum's terms are swept, and a point is
pointed at (flash, circumscribe, wiggle).

Run:  uv run python examples/motion_gallery.py      (prints what the inspector reads)
"""

import math

from phasor import Counter, Dot, Equation, Line, Scene, Text, Tracer, cx, expr, fourier, geo

s = Scene("motion_gallery", look="prism_everywhere")  # the hue runs along every curve
s.set("camera.height", 10.0)

# Top left: equations.
s.add(Equation(id="before", parts=("$x$", "$=2$"), colors=("primary", "secondary"),
               pos=(-4.5, 3.2, 0)))  # fmt: skip
s.add(Equation(id="after", parts=("$x$", "$+1$", "$=3$"), pos=(-4.5, 3.2, 0),
               colors=("primary", "highlight", "secondary")))  # fmt: skip
s.add(Text(id="caption", text="motion verbs", pos=(-4.5, 4.4, 0)))

# Top right: a circle that becomes a square.
ring = [(4.5 + math.cos(a), 3.0 + math.sin(a), 0.0)
        for a in (2 * math.pi * k / 48 for k in range(49))]  # fmt: skip
s.add(Line(id="circle", points=ring))
box = [(5.5, 4.0, 0.0), (3.5, 4.0, 0.0), (3.5, 2.0, 0.0), (5.5, 2.0, 0.0), (5.5, 4.0, 0.0)]
s.add(Line(id="square", points=box))

# Bottom left: the plane bending under z^2, with a comet on a curve.
s.add(cx.ComplexMap(id="plane_map", function="z**2", extent=(1.6, 1.6), pos=(-4.5, -2.0, 0),
                    lines=(9, 9), amount=0.0))  # fmt: skip
s.add(geo.ParametricCurve(id="loop", fx="-1.5 + 1.2*cos(t)", fy="-2.0 + 1.2*sin(2*t)",
                          t_range=(0.0, 6.2832), samples=300))  # fmt: skip
s.add(Tracer(id="comet", path="loop", preset="comet"))
s.add(Counter(id="comet.count", value=expr("comet.progress"), prefix="p = ", pos=(-1.5, -3.6, 0)))

# Bottom right: particles in a swirl, a Fourier sum and a point to point at.
s.add(geo.FlowParticles(id="swirl", fp="-y", fq="x", fr="0", pos=(4.5, -1.4, 0),
                        extent=(2.2, 2.2, 0.0), count=120, lifetime=1.5))  # fmt: skip
s.add(fourier.PartialSum(id="wave", kind="square", terms=2, pos=(4.5, -4.2, 0), extent=(3.5, 1.0)))
s.add(Dot(id="point", pos=(1.8, 0.2), r=0.15))

for name, t in (("start", 0.0), ("eq", 0.5), ("shape", 2.5), ("map", 4.5), ("flow", 7.0),
                ("look", 9.5), ("end", 12.0)):  # fmt: skip
    s.marker(name, t)
s.at("start").write("caption")
s.at("eq").transform("before", "after")
s.at("shape").become("circle", "square")
s.at("map").morph("plane_map.amount", 1.0)
s.at("map").travel("comet")
s.at("flow").sweep("wave.terms", 30)
s.at("look").flash("point").circumscribe("point")
s.at("look").then(0.2).wiggle("point")

if __name__ == "__main__":
    e = s.engine
    for oid in ("after", "plane_map", "comet", "swirl", "wave.readout"):
        print(oid, e.inspect(10.0, id=oid)["objects"][0]["semantic"])
