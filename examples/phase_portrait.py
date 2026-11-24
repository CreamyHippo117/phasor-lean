"""A damped pendulum's phase portrait, then its rest points.

x is the angle and y the angular speed: x' = y, y' = -sin(x) - 0.3 y. Trajectories start from
a spread of seeds and are traced one after another with colour running along time; the small
arrows are the flow and the faint curves are the nullclines (where x or y stops changing).
Then the rest points appear: the hanging position is a stable spiral (the swing dies away) and
the upright ones are saddles (balanced, but any push falls off), each with its eigenvalues and
eigen-directions.

Run:  uv run python examples/phase_portrait.py      (prints the fixed points)
"""

from phasor import Scene, Text, dyn

F, G = "y", "-sin(x) - 0.3 * y"

s = Scene("phase_portrait", look="prism")
s.add(dyn.PhasePortrait(id="flow", f=F, g=G, extent=(12, 6.6), n_seeds=14, t_end=12.0,
                        progress=0.0, pos=(0, -0.3, 0)))  # fmt: skip
s.add(dyn.FixedPoints(id="rest", f=F, g=G, extent=(12, 6.6), reveal=0.0, pos=(0, -0.3, 0)))
s.add(Text(id="title", text="a damped pendulum", pos=(8 / 9, 0.07), layer="overlay",
           style={"size": "title"}))  # fmt: skip

s.marker("start", 0.3)
s.marker("run", 1.0)
s.marker("rest", 6.0)
s.marker("end", 9.5)
s.at("start").reveal("title")
s.at("run").sweep("flow.progress", 1.0, show=False)  # trace the trajectories, evenly
s.at("rest").sweep("rest.reveal", 1.0, show=False)  # then the rest points, one by one

if __name__ == "__main__":
    e = s.engine
    sem = e.inspect(e.duration, id="rest")["objects"][0]["semantic"]
    for pt in sem["points"]:
        print(pt["at"], pt["type"], "eigenvalues", pt["eigenvalues"])
    for issue in e.lint(strict=True):
        print("lint:", issue)
