"""A residue theorem short, driven entirely by narration.

Every motion is a choreography verb anchored to a word in the script; nothing is timed in
seconds. Change the script or the voice and everything moves with it.

Run:  uv run python examples/residue_theorem.py        (lint report + storyboard)
      uv run phasor render examples/residue_theorem.py -o examples/out/residue_theorem.mp4
"""

from pathlib import Path

import numpy as np

from phasor import Dot, Line, Scene, Text, cx

SCRIPT = """
[[intro]] Here is the function one over z squared plus one.
It has two poles, one at i [[pole_top]] and one at minus i [[pole_bottom]].
We draw a small loop around [[loop_draw]] the pole at i.
[[pause 0.4]]
Only the pole inside the loop counts [[residue]].
Its residue is minus i over two [[res_value]], so the integral equals pi [[integral]].
"""

s = Scene("residue_theorem")
s.set("camera.height", 4.5)

s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(8.0, 4.5)))
s.add(Text(id="title", text="f(z) = 1 / (z^2 + 1)", pos=(8 / 9, 0.08), layer="overlay",
           style={"size": "title"}))  # fmt: skip
for name, y, text in (("pole_top", 1.0, "i"), ("pole_bottom", -1.0, "-i")):
    s.add(Dot(id=name, pos=(0, y), r=0.07, style={"color": "highlight"}, tags=["key"]))
    s.add(Text(id=f"{name}.label", text=text, style={"size": "math"}))
    s.constrain(f"label({name}) near {name}")
ring = [(0.55 * np.cos(a), 1 + 0.55 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 161)]
s.add(
    Line(id="loop", points=ring, style={"color": "highlight", "stroke": "bold", "glow": "strong"})
)
s.add(Text(id="loop.label", text="Res = -i/2"))
s.constrain("label(loop) near loop")
s.add(Text(id="result", text="integral around the loop = 2 pi i (-i/2) = pi", pos=(8 / 9, 0.92),
           layer="overlay"))  # fmt: skip

s.narrate(SCRIPT)
s.at("intro").reveal("plane", "title")
s.at("pole_top").bloom_in("pole_top").then().reveal("pole_top.label")
s.at("pole_bottom").bloom_in("pole_bottom").then().reveal("pole_bottom.label")
s.at("loop_draw").trace("loop")
s.at("residue").dim_others("loop", "pole_top")
s.at("res_value").reveal("loop.label")
s.at("integral").reveal("result")
s.pace()

if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    n = s.engine.document.narration
    print(f"voice {n.voice}: {len(n.blocks)} blocks, {s.engine.duration:.1f} s")
    for name, t in sorted(s.engine.document.timeline.markers.items(), key=lambda kv: kv[1]):
        print(f"  {t:6.2f} s  {name}")
    for issue in s.engine.lint(strict=True):
        print("lint:", issue.severity, issue)
    try:
        from phasor.render.output import save_png

        save_png(out / "residue_theorem_storyboard.png", s.engine.storyboard(marks=True)["image"])
        print("wrote", out / "residue_theorem_storyboard.png")
    except RuntimeError as e:
        print("no storyboard:", e)
