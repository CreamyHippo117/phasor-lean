"""A 3D room with sound bouncing in it, in the prism look.

A hand clap at the source. Sound leaves it as rays in every direction and as an expanding
spherical front; both reflect off the walls (angle in = angle out), and with every
reflection they lose energy and shift color, so you watch the early echoes give way to the
dense, dimming reverberation. On the walls, glowing circles show where the fronts touch
them: the direct front first, then the reflected fronts from the image sources behind the
walls. Panel at the top: the impulse response at the listener filling in as the arrivals
reach it. The camera orbits the room. Time runs at 1/100 of real speed.

A room of another shape works the same way: put a Wavefront ``.obj`` next to the scene and
set ``path`` on the room (the repository has an L-shaped one, ``examples/assets/l_room.obj``;
the panel is for a shoebox only).

Run:  uv run python examples/room_3d.py      (prints the rays and fronts at a few times)
"""

from phasor import Scene, ac, expr

SIZE, SRC, MIC = (8.0, 5.0, 3.0), (2.2, 2.0, 1.4), (6.0, 3.4, 1.4)
CLAP = 1.0  # scene seconds before the clap
SLOW = 0.01  # seconds of sound per scene second
WINDOW = 0.06  # seconds of impulse response in the panel

s = Scene("room_3d", look="prism")
s.use_camera("orbit", azimuth=-58, elevation=26, distance=14.5, target=(4.0, 2.5, 1.2))
s.add(ac.Room3D(id="room", size=SIZE, material="wood"))
s.add(ac.Wavefront3D(id="front", room="room", source=SRC, time=expr(f"t - {CLAP}"), speed=SLOW,
                     order=4))  # fmt: skip
s.add(ac.SoundRays(id="rays", room="room", source=SRC, time=expr(f"t - {CLAP}"), speed=SLOW,
                   count=1500, bounces=12))  # fmt: skip
s.add(ac.Source(id="src", pos=SRC, tags=["key"]))
s.add(ac.Listener(id="mic", pos=MIC, tags=["key"]))
HEARD = f"min(max((t - {CLAP}) * {SLOW} / {WINDOW}, 0), 1)"  # share of the window heard so far
s.add(ac.ImpulseResponse(id="ir", size=SIZE, source=SRC, listener=MIC, material="wood", order=10,
                         duration=WINDOW, layer="overlay", pos=(0.05, 0.17, 0), extent=(0.45, 0.1),
                         reveal=expr(HEARD)))  # fmt: skip
s.marker("start", 0.0)
s.marker("end", 12.0)
s.at("start").orbit(70.0, until="end")

if __name__ == "__main__":
    e = s.engine
    print("room", e.inspect(0.0, id="room")["objects"][0]["semantic"])
    for t in (1.5, 3.0, 6.0, 11.0):
        rays = e.inspect(t, id="rays")["objects"][0]["semantic"]
        front = e.inspect(t, id="front")["objects"][0]["semantic"]
        print(f"t={t:g}: {rays['distance']:.1f} m, {rays['reflections_so_far']:.1f} reflections, "
              f"energy {rays['energy_left']:.2f}; {front['fronts_in_room']} fronts")  # fmt: skip
    for issue in e.lint():
        print("lint:", issue)
