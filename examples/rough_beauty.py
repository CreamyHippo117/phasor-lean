"""A deliberately rough scene that the critic pass improves.

Everything a first draft gets wrong, on purpose: three stroke weights, two things
highlighted at once, a point jammed against the edge of the frame, and two unrelated things
starting different motions in the same instant. The critic fixes what is certain, proposes
what is judgement, and leaves what only a person (or the AI looking at the picture) can
decide, such as the empty frame and the two motions that collide.

Run:  uv run python examples/rough_beauty.py
It prints the critic's score before and after three polish rounds, and writes before/after
storyboards next to this file.
"""

from pathlib import Path

import numpy as np

from phasor import Dot, Line, Scene, Text, cx

s = Scene("rough_beauty", look="calm")
s.set("camera.height", 5.2)
s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(9.2, 5.2)))
s.add(
    Dot(
        id="pole_i",
        pos=(0, 1),
        r=0.09,
        style={"color": "highlight", "glow": "strong"},
        tags=["key"],
    )
)
s.add(Dot(id="pole_minus_i", pos=(0, -1), r=0.09, style={"color": "highlight", "glow": "strong"},
          tags=["key"]))  # fmt: skip
s.add(
    Dot(id="probe", pos=(4.35, 0.0), r=0.1, style={"color": "secondary"}, tags=["key"])
)  # at the edge
ring = [(0.6 * np.cos(a), 1 + 0.6 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 120)]
s.add(
    Line(id="loop", points=ring, style={"color": "secondary", "stroke": "bold", "glow": "strong"})
)
s.add(Line(id="axis", points=[(-4, 0), (4, 0)], style={"color": "muted", "stroke": "hairline"}))
s.add(Line(id="guide", points=[(-4, -2), (4, -2)], style={"color": "muted", "stroke": "regular"}))
for name, text in (("pole_i", "pole at i"), ("pole_minus_i", "pole at -i")):
    s.add(Text(id=f"{name}.label", text=text))
    s.constrain(f"label({name}) near {name}")

s.marker("poles", 1.0)
s.marker("loop_draw", 3.0)
s.marker("end", 6.0)
s.at("poles").reveal("pole_i", "pole_minus_i", "pole_i.label", "pole_minus_i.label")
s.at("poles").morph("probe.pos", (4.35, 1.0))  # unrelated, a different motion, same instant
s.at("loop_draw").trace("loop")

if __name__ == "__main__":
    from phasor.render.output import save_png

    out = Path(__file__).parent / "out"
    e = s.engine
    before = e.critique()
    print(f"before: score {before['score']}")
    for i in before["issues"]:
        print("  ", i["code"], i["ids"], i["markers"])
    try:
        save_png(out / "rough_before.png", e.storyboard([1.0, 3.5, 6.0], marks=True)["image"])
    except RuntimeError as err:
        print("no storyboard:", err)
    report = e.polish(rounds=3)
    for r in report["rounds"]:
        print(f"round {r['round']}: score {r['score']} ({len(r['applied'])} changes)")
    print(f"after: score {report['after']}")
    try:
        save_png(out / "rough_after.png", e.storyboard([1.0, 3.5, 6.0], marks=True)["image"])
    except RuntimeError as err:
        print("no storyboard:", err)
