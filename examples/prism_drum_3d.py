"""The prism look in 3D: a drum skin vibrating, seen as a landscape the camera circles.

The drum vibrates in its (2, 3) mode: six cells rise and fall in turn, the nodal lines
between them never move. The height of the surface is the skin's displacement. In the prism
look the surface is dark where the skin is at rest and takes the spectrum's hue of its height
where it swings, under a net of glowing lines in the same hues, so the shape reads as light
out of darkness. The camera swings a quarter turn round it.

Run:  uv run python examples/prism_drum_3d.py      (prints the drum's swing)
"""

from phasor import Scene, expr
from phasor.domains import pde

s = Scene("prism_drum_3d", look="prism")
s.use_camera("orbit")
s.set("camera.target", (0.0, 0.0, 0.0))
s.set("camera.distance", 16.0)
s.set("camera.elevation", 30.0)
s.set("camera.azimuth", -50.0)
s.add(pde.Surface(id="drum", kind="wave", mode=(2, 3), scale=1.2, height=1.6, n=64,
                  time=expr("t / 4"), size=(6.0, 6.0)))  # fmt: skip

s.marker("start", 0.0)
s.marker("end", 8.0)
s.at("start").orbit(90.0, until="end")

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 4.0, 8.0):
        drum = e.inspect(t, id="drum")["objects"][0]["semantic"]
        print(f"t={t:g}: drum swing {drum['min']:+.2f} to {drum['max']:+.2f}")
    for issue in e.lint(strict=True):
        print("lint:", issue)
