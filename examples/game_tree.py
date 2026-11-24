"""Every game of tic-tac-toe at once: its 765 different positions (boards that are rotations
or reflections of each other count once), each joined to the positions one move later.

The empty board is in the middle; each ring further out is one more move, and every
position fans out into the positions it leads to, so the graph opens like a flower. A
position reached by another order of moves is joined by a faint thread. Each position glows with
who wins it if both sides play perfectly (from the palette: one colour for X, one for a
draw, one for O). The graph grows one move at a time while the camera circles it. Load
your own graph the same way: ``nw.Network(id="g", path="my_graph.csv")`` (an edge list,
node-link JSON or GraphML next to the scene), or give the edges in the script.

Run:  uv run python examples/game_tree.py      (prints the size and how the game ends)
"""

import numpy as np

from phasor import Scene, Text
from phasor.domains import network as nw

s = Scene("game_tree", look="prism")
s.use_camera("orbit", azimuth=-90, elevation=60, distance=13.5, target=(0.0, -0.3, -0.8))
s.add(nw.Network(id="game", generator="tictactoe", layout="radial", dims=3, size=3.6,
                 color_by="outcome", grow=0.0, cross_edges=0.12))  # fmt: skip
s.add(Text(id="title", text="every game of tic-tac-toe", pos=(8 / 9, 0.07), layer="overlay",
           style={"size": "title"}))  # fmt: skip

# Grow one move per beat: the share of all edges that reach each move's ring.
g = nw.nm.tictactoe()  # the same graph the Network draws (symmetric=True)
d = g.depth()
reach = np.maximum(d[g.edges[:, 0]], d[g.edges[:, 1]])
share = np.cumsum(np.bincount(reach)[1:]) / len(g.edges)

s.marker("start", 0.3)
for move, frac in enumerate(share, start=1):
    s.marker(f"move_{move}", 0.3 + 1.0 * move)
    s.at(f"move_{move}").morph("game.grow", round(float(frac), 6))
s.marker("end", 11.5)
s.at("start").reveal("title")
s.at("start").orbit(60, until="end")

if __name__ == "__main__":
    e = s.engine
    sem = e.inspect(e.duration, id="game")["objects"][0]["semantic"]
    print(f"{sem['nodes']} positions, {sem['edges']} moves, {sem['depth']} moves deep")
    print("perfect play from the empty board:", {1: "X wins", 0: "a draw", -1: "O wins"}[
        int(g.node_attrs["outcome"][0])])  # fmt: skip
    print("positions by outcome:", {k: int(v) for k, v in zip(
        ("O wins", "draw", "X wins"), np.bincount(g.node_attrs["outcome"].astype(int) + 1),
        strict=True)})  # fmt: skip
    for issue in e.lint(strict=True):
        print("lint:", issue)
