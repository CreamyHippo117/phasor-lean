"""A linear system x' = A x and its modes.

A = [[1, 2], [2, 1]] has eigenvalues 3 and -1 with eigenvectors (1, 1) and (1, -1): a saddle.
The eigenvector lines are invariant (a point on one stays on it), cool where the motion comes
in (-1) and warm where it goes out (3). Trajectories are exact, e^(tA) x0, with colour running
along time. Then the grid (`linalg.TransformGrid(flow=A, flow_time=t)`) is carried along by
the flow e^(tA): it stretches along the warm line and squeezes along the cool one, and
every cell's area grows by e^(trace t) = e^(2t).

Run:  uv run python examples/linear_system.py      (prints the modes)
"""

from phasor import Scene, Text, dyn, linalg

A = ((1.0, 2.0), (2.0, 1.0))

s = Scene("linear_system", look="prism")
AT = (0, -0.3, 0)
s.add(linalg.TransformGrid(id="grid", flow=A, flow_time=0.0, extent=9, pos=AT, square=0,
                           basis=0, style={"glow": "soft"}))  # fmt: skip
s.add(linalg.EigenLines(id="modes", matrix=A, length=9, pos=AT, style={"glow": "strong"}))
s.add(dyn.LinearSystem(id="sys", matrix=A, extent=(12, 6.6), n_seeds=12, t_end=3.0,
                       progress=0.0, pos=AT))  # fmt: skip
s.add(Text(id="title", text="x' = A x: the modes", pos=(8 / 9, 0.07), layer="overlay",
           style={"size": "title"}))  # fmt: skip

s.marker("start", 0.3)
s.marker("run", 1.0)
s.marker("flow", 5.0)
s.marker("end", 9.5)
s.at("start").reveal("title")
s.at("run").sweep("sys.progress", 1.0, show=False)  # trace the trajectories
s.at("flow").sweep("grid.flow_time", 0.6, show=False)  # then carry the grid along the flow

if __name__ == "__main__":
    e = s.engine
    sem = e.inspect(e.duration, id="sys")["objects"][0]["semantic"]
    print(sem["type"], "eigenvalues", sem["eigenvalues"], "eigenvectors", sem["eigenvectors"])
    for issue in e.lint(strict=True):
        print("lint:", issue)
