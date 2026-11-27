"""The spider and the fly: a shortest path on a box is a straight line on its net, in the
prism look.

A room 30 x 12 x 12 (here a sixth of that). The spider sits on one end wall, in the middle,
1 below the ceiling; the fly on the far end wall, in the middle, 1 above the floor. The
shortest way the spider can walk to it is 40, not the 42 of going straight up, along and
down: it crosses five of the six faces, bending at every edge. Open the box into its net
and the bent path is one straight line, the hypotenuse of a 24-32-40 triangle. Then the box
folds up again, the line bending back over its edges.

Run:  uv run python examples/spider_and_fly.py      (prints the path and the faces it crosses)
"""

from phasor import Scene, net

K = 1 / 6  # the room, a sixth of its size
SIZE = (30 * K, 12 * K, 12 * K)
SPIDER, FLY = (0.0, 6 * K, 11 * K), (30 * K, 6 * K, 1 * K)

s = Scene("spider_and_fly", look="prism")
s.use_camera("orbit", azimuth=-75, elevation=58, distance=20.0, target=(2.5, 1.0, 1.5))
s.add(net.Polyhedron(id="room", size=SIZE, root="top", geodesic=[SPIDER, FLY], trace=0.0))

s.marker("walk", 0.5)
s.marker("open", 4.0)
s.marker("flat", 7.5)
s.marker("close", 10.0)
s.marker("folded", 13.0)
s.at("walk").morph("room.trace", 1.0, until="open")
s.at("open").morph("room.unfold", 1.0, until="flat")
s.at("close").morph("room.unfold", 0.0, until="folded")

if __name__ == "__main__":
    e = s.engine
    for t in (1.0, 9.0):
        sem = e.inspect(t, id="room")["objects"][0]["semantic"]
        print(f"t={t:g}:", sem["shape"], sem["geodesic"])
    print("length in the full-size room:", round(sem["geodesic"]["length"] / K, 4))
    for issue in e.lint(strict=True):
        print("lint:", issue)
