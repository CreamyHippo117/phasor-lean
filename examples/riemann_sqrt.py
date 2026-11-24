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
LIFT = 0.02  # lines on the surface are lifted a hair, so they are not buried in it

s = Scene("riemann_sqrt", look="prism")
s.use_camera("orbit", azimuth=-60)  # elevation, distance, target: the camera solver's
# Glass (the prism look's surface): the two sheets fold over each other, so seen solid, one
# would always hide the other. Its hue is the phase of sqrt(z), so each sheet has its own.
s.add(cx.RiemannSurface(id="surface", kind="sqrt", radius=R, preset="glass"))

# A net of faint glowing lines on the surface, so its shape reads as light out of the dark:
# three rings round the branch point, and spokes every 40 degrees over both turns (missing
# the two points above 2 i). One line draws it all: out along a spoke, round, back in
# along the next, with the rings hung on the first spoke. Below half opacity it hides
# nothing, like the glass: the camera solver and the label placer look straight through it.
R0 = 0.6  # where the spokes start


def on_surface(r, a):
    return (r * np.cos(a), r * np.sin(a), np.sqrt(r) * np.cos(a / 2) + LIFT)


def ring(r):
    return [on_surface(r, a) for a in np.linspace(0, 4 * np.pi, 241)]


net = []
for k, a in enumerate(np.radians(np.arange(0, 720, 40))):
    if k == 0:
        net += ring(R0) + [on_surface(r, a) for r in np.linspace(R0, 1.5, 10)]
        net += ring(1.5) + [on_surface(r, a) for r in np.linspace(1.5, R, 14)] + ring(R)
    rs = np.linspace(R, R0, 24) if k % 2 else np.linspace(R0, R, 24)
    if k:
        net += [on_surface(r, a) for r in rs]
    net += [on_surface(rs[-1], b) for b in np.linspace(a, a + np.radians(40), 12)[1:]]
s.add(Line(id="net", points=net, opacity=0.45, style={"stroke": "hairline", "glow": "soft"}))
s.add(Text(id="title", text="the Riemann surface of sqrt(z)", pos=(8 / 9, 0.08),
           layer="overlay", style={"size": "title"}))  # fmt: skip

s.add(Dot(id="branch", pos=(0, 0, 0), r=0.08, preset="beacon", tags=["key", "exact"]))
s.add(Text(id="branch.label", text="branch point"))

# One point on each sheet, above 2 i, where the two sheets are far apart.
y = 2.0
h = np.sqrt(y) * np.cos(np.pi / 4)
for name, z, text in (("upper", h, "+sqrt(z)"), ("lower", -h, "-sqrt(z)")):
    s.add(Dot(id=name, pos=(0, y, z), r=0.07, tags=["key", "exact"]))
    s.add(Text(id=f"{name}.label", text=text, style={"size": "math"}))
    s.constrain(f"label({name}) near {name}")
s.constrain("label(branch) near branch")

# The walk: twice around the branch point, on the surface (lifted a hair, like the net).
# The first turn ends on the other sheet.
theta = np.linspace(0, 4 * np.pi, 361)
walk = [on_surface(WALK_R, a) for a in theta]
s.add(Line(id="walk", points=walk, preset="beam"))  # the loudest stroke: it carries the story

s.constrain("camera keep_visible(surface)")
s.constrain("camera avoid_occlusion(branch)", priority="medium")
s.constrain("camera avoid_occlusion(upper, lower)", priority="weak")

s.narrate(SCRIPT)
s.at("intro").reveal("surface", "net", "title")
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
