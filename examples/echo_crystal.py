"""The crystal of echoes: a lattice that flashes as a surface sweeps it.

Every echo of a clap is a mirror copy of the room (the image-source method), so the images
of the source form a 3D crystal round the room. The time since the clap is a sphere
growing from the listener at the speed of sound (slowed down); each point of the crystal
flashes the moment the sphere reaches it, which is the moment that echo is heard. The
flashes come faster and faster because the sphere's area grows: the t-squared law of echo
density, seen without a formula. Color is the number of reflections.

Then the sphere opens out into a plane tangent to it ahead of the listener (``flat`` 0 -> 1),
the picture of a feedback loop, whose equal-time surface is flat.

Run:  uv run python examples/echo_crystal.py      (prints the flash rate as the sphere grows)
"""

from phasor import Scene, ac, expr, sweep

SIZE, SRC, MIC = (8.0, 5.0, 3.0), (2.2, 2.0, 1.4), (6.0, 3.4, 1.4)
CLAP = 1.0  # scene seconds before the clap
SPEED = 3.0  # meters of sound per scene second (343 m/s slowed about 114 times)
EXTENT = 45.0  # meters round the listener drawn; order 19 fills that ball in every direction

s = Scene("echo_crystal", look="prism")
s.use_camera("orbit", azimuth=-50, elevation=24, distance=150, target=MIC)
s.add(ac.Room3D(id="room", size=SIZE, walls=0.0))
s.add(sweep.SweepLattice(id="crystal", size=SIZE, source=SRC, order=19, center=MIC,
                         normal=(1.0, 0.35, 0.2), time=expr(f"t - {CLAP}"), speed=SPEED,
                         extent=EXTENT, r=0.35, hue_step=0.045))  # fmt: skip
s.add(sweep.SweepFront(id="front", lattice="crystal"))
s.add(ac.Listener(id="mic", pos=MIC, tags=["key"]))
s.marker("start", 0.0)
s.marker("flatten", 4.5)
s.marker("end", 12.0)
s.at("start").orbit(40.0, until="end")
s.at("flatten").to("crystal.flat", 1.0, duration=3.0)

if __name__ == "__main__":
    e = s.engine
    for t in (2.0, 4.0, 6.0, 8.5, 11.5):
        st = e.state_at(t)
        lat = st["crystal"].impl.semantics(st)
        front = st["front"].impl.semantics(st)
        print(f"t={t:g}: front {front['shape']} at {lat['reach']:.0f} m, {lat['reached']} echoes "
              f"heard, {lat['rate']:.0f} per second, deepest {lat.get('deepest_reached', 0)} "
              "reflections")  # fmt: skip
    for issue in e.lint():
        print("lint:", issue)
