"""From a room's echoes to a feedback network's.

A clap in a room: its echoes are the image sources, a crystal of mirror rooms, and each one
flashes when the sphere of time since the clap (round the listener) reaches it. Then the
sphere opens out into the plane tangent to it ahead, the equal-time surface of feedback
loops. Where the plane has got to, three feedback loops take over: their echoes are the
grid N^3 (one axis per loop, each step one lap), and the same plane sweeps on through it,
lighting each grid point exactly when that echo arrives (the arrival sum n_i d_i is the
plane's height over the grid's corner). Then the loops grow from three to sixteen: an echo
of sixteen loops lives in 16 dimensions, its equal-time surface is a hyperplane there, not a
plane here, so the front fades and the star of arms fills the frame.

The two lattices are one look (``sweep.SweepLattice`` and ``nd.StarLattice`` share the flash
and the drawing) and one clock: the grid's corner sits on the plane's path and the grid's
arms are sized so its arrival times match the plane's speed.

Run:  uv run python examples/echo_room_to_loops.py      (prints each stage's echoes)
"""

import math

import numpy as np

from phasor import Scene, ac, expr, nd, sweep
from phasor.domains import ndlattice_math as nlm

SIZE, SRC, MIC = (8.0, 5.0, 3.0), (2.2, 2.0, 1.4), (6.0, 3.4, 1.4)
CLAP = 1.0  # scene seconds before the clap
SPEED = 3.0  # metres the front travels per scene second
LOOPS = 16
DELAYS = tuple(0.4 * d for d in nlm.default_delays(LOOPS))  # short loops: a dense grid soon
HANDOVER = 7.5  # scene time the loops take over: the plane is SPEED * (HANDOVER - CLAP) out
NORMAL = np.ones(3) / math.sqrt(3.0)  # the plane faces the grid's diagonal, sum n_i d_i
# the grid's corner, where the plane is at the handover, and its arm length: arm i is
# spacing * d_i / mean(d) along axis i, so a point's height over the corner along NORMAL is
# spacing / (mean(d) sqrt 3) * sum n_i d_i, which is SPEED * its arrival time
CORNER = tuple(np.asarray(MIC) + SPEED * (HANDOVER - CLAP) * NORMAL)
SPACING = SPEED * float(np.mean(DELAYS)) * math.sqrt(3.0)
STAR_CENTER = tuple(np.asarray(CORNER) + 3.0 * SPACING * NORMAL)

s = Scene("echo_room_to_loops", look="prism")
s.use_camera("orbit", azimuth=-80, elevation=20, distance=130, target=MIC)
s.add(ac.Room3D(id="room", size=SIZE, walls=0.0))
s.add(sweep.SweepLattice(id="crystal", size=SIZE, source=SRC, order=19, center=MIC,
                         normal=tuple(NORMAL), time=expr(f"t - {CLAP}"), speed=SPEED,
                         extent=45.0, r=0.35, hue_step=0.045))  # fmt: skip
s.add(sweep.SweepFront(id="front", lattice="crystal"))
s.add(nd.StarLattice(id="loops", delays=DELAYS, origin=CORNER, spacing=SPACING, arms=3.0,
                     max_time=3.2, max_points=100_000, time=expr(f"t - {HANDOVER}"), afterglow=0.5,
                     point_size=0.004, color_by="arm", crowd=120.0, opacity=0.0))  # fmt: skip

s.marker("start", 0.0)
s.marker("flatten", 4.5)  # the sphere opens into a plane
s.marker("handover", HANDOVER)  # the crystal fades, three loops' grid takes the plane
s.marker("more_loops", 11.0)  # three loops to sixteen: the front leaves 3D
s.marker("end", 16.0)
s.at("start").orbit(40.0, until="end")  # near edge-on to the plane at the handover
s.at("flatten").to("crystal.flat", 1.0, duration=3.0)
s.at("handover").to("crystal.opacity", 0.0, duration=2.0)
s.at("handover").to("room.opacity", 0.0, duration=2.0)
s.at("handover").to("loops.opacity", 1.0, duration=1.0)
s.at("handover").to("front.opacity", 0.45, duration=2.0)  # close up, a quieter veil
s.at("handover").to("camera.target", STAR_CENTER, duration=3.0)
s.at("handover").to("camera.distance", 50.0, duration=3.0)
s.at("more_loops").to("loops.arms", float(LOOPS), duration=4.0)
s.at("more_loops").to("front.opacity", 0.0, duration=1.5)

if __name__ == "__main__":
    e = s.engine
    for t in (3.0, 6.0, 9.0, 10.5, 13.0, 16.0):
        st = e.state_at(t)
        lat, front = st["crystal"].impl.semantics(st), st["front"].impl.semantics(st)
        star = st["loops"].impl.semantics(st)
        print(f"t={t:g}: front {front['shape']} at {lat['reach']:.1f} m, {lat['reached']} room "
              f"echoes; {star['arms']:g} loops, {star['heard']} loop echoes heard")  # fmt: skip
    for issue in e.lint():
        print("lint:", issue)
