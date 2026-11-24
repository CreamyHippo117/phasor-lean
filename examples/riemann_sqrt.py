"""The Riemann surface of sqrt(z), seen through a full orbit.

Two sheets, one for each value of the square root, meet at the branch point; a path that
walks once around it climbs from one sheet onto the other. The camera turns a full circle.
Nobody sets its elevation, distance or target: the camera solver chooses them from goals
(keep the surface in frame, keep the branch point unhidden). Every label is placed by
depth, so it stays readable from every side, and the event log has nothing to complain
about.

Run:  uv run python examples/riemann_sqrt.py      (lint, event log, storyboard)
      uv run phasor render examples/riemann_sqrt.py -o examples/out/riemann_sqrt.mp4
"""

from pathlib import Path

import numpy as np

from phasor import Dot, Line, Scene, Text, cx

SCRIPT = """
[[intro]] This is the Riemann surface of the square root.
[[sheets]] It has two sheets, one for each value of the root.
They meet at the branch point [[branch]].
Walk once around it [[walk]], and you climb from one sheet onto the other.
[[pause 0.5]]
Only a second turn [[home]] brings you back to where you started.
"""

R = 2.2  # radius of the surface
WALK_R = 1.0  # the walk circles the branch point at this distance

s = Scene("riemann_sqrt")
s.use_camera("orbit", azimuth=-60)  # elevation, distance, target: the camera solver's
# Glass: the two sheets fold over each other, so seen solid, one would always hide the other.
s.add(cx.RiemannSurface(id="surface", kind="sqrt", radius=R, style={"material": "glass"}))
s.add(Text(id="title", text="the Riemann surface of sqrt(z)", pos=(8 / 9, 0.08),
           layer="overlay", style={"size": "title"}))  # fmt: skip

s.add(Dot(id="branch", pos=(0, 0, 0), r=0.08, style={"color": "highlight"}, tags=["key"]))
s.add(Text(id="branch.label", text="branch point"))

# One point on each sheet, above 2 i, where the two sheets are far apart.
y = 2.0
h = np.sqrt(y) * np.cos(np.pi / 4)
for name, z, text in (("upper", h, "+sqrt(z)"), ("lower", -h, "-sqrt(z)")):
    s.add(Dot(id=name, pos=(0, y, z), r=0.07, tags=["key"]))
    s.add(Text(id=f"{name}.label", text=text, style={"size": "math"}))
    s.constrain(f"label({name}) near {name}")
s.constrain("label(branch) near branch")

# The walk: twice around the branch point, on the surface (lifted a hair so it is not
# buried in it). The first turn ends on the other sheet.
theta = np.linspace(0, 4 * np.pi, 361)
lift = 0.02
walk = [(WALK_R * np.cos(a), WALK_R * np.sin(a), np.sqrt(WALK_R) * np.cos(a / 2) + lift)
        for a in theta]  # fmt: skip
s.add(Line(id="walk", points=walk, style={"color": "highlight", "stroke": "bold",
                                         "glow": "none"}))  # fmt: skip

s.constrain("camera keep_visible(surface)")
s.constrain("camera avoid_occlusion(branch)", priority="medium")
s.constrain("camera avoid_occlusion(upper, lower)", priority="weak")

s.narrate(SCRIPT)
s.at("intro").reveal("surface", "title")
s.at("sheets").orbit_reveal("upper", "upper.label", "lower", "lower.label")
s.at("branch").bloom_in("branch").then().reveal("branch.label")
s.at("walk").trace("walk")
s.at("walk").orbit(360, until="home")
s.pace()

if __name__ == "__main__":
    from phasor.inspect.events import problems

    out = Path(__file__).parent / "out"
    e = s.engine
    n = e.document.narration
    print(f"voice {n.voice}: {len(n.blocks)} blocks, {e.duration:.1f} s")
    for issue in e.lint(strict=True):
        print("lint:", issue.severity, issue)
    log = e.events()
    print(f"event log: {len(log)} events, {len(problems(log))} problems")
    for ev in problems(log):
        print("  problem:", ev)
    try:
        from phasor.render.output import save_png

        save_png(out / "riemann_sqrt_storyboard.png", e.storyboard(marks=True)["image"])
        print("wrote", out / "riemann_sqrt_storyboard.png")
    except RuntimeError as err:
        print("no storyboard:", err)
