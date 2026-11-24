"""Follow one Lorenz trajectory as it spirals, crosses, and builds both lobes.

Three sweeps let the first crossing unfold at a slower rate. A small tracer
follows the visible endpoint, then fades as the orbit settles on the full form.
"""

from phasor import Scene, Tracer, dyn

s = Scene("strange_attractor", look="prism")
s.use_camera("orbit", azimuth=-75, elevation=20, distance=13.5)
s.add(dyn.StrangeAttractor(id="trajectory", seed=7, t_end=30, progress=0.009))
s.add(
    Tracer(
        id="moving_tip",
        path="trajectory",
        progress=1.0,
        trail=0.018,
        r=0.065,
        style={"color": "highlight", "glow": "strong", "stroke": "regular"},
    )
)

s.marker("start", 0.0)
s.marker("wind", 0.35)
s.marker("cross", 2.85)
s.marker("weave", 5.35)
s.marker("complete", 7.85)
s.marker("settle", 9.25)
s.marker("end", 11.25)

# This seed first crosses to the other lobe at about 14.34 simulation seconds.
s.at("wind").sweep("trajectory.progress", 13.5 / 30, show=False)
s.at("cross").sweep("trajectory.progress", 17 / 30, show=False)
s.at("weave").sweep("trajectory.progress", 1.0, show=False)
s.at("weave").orbit(55, until="settle")
s.at("complete").exit("moving_tip")
