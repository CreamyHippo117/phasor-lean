"""A labeled dot crosses the frame, passes behind a disc, and leaves.

Run:  uv run python examples/moving_dot.py
It prints what the inspector sees at a few times and writes preview stills next to this file.
"""

from pathlib import Path

from phasor import Dot, Line, Scene, Text, expr

s = Scene("moving_dot")

s.add(Line(id="axis", points=[(-7, 0), (7, 0)], style={"color": "muted", "stroke": "hairline"}))
s.add(Dot(id="dot", pos=(-4, 0), r=0.25, style={"color": "highlight"}, tags=["key"]))
# The label follows the dot through an expression, so it can never be left behind.
s.add(Text(id="dot_label", text="z", pos=expr("dot.pos + vec(0, 0.7, 0)"), style={"size": "math"}))
# A larger disc drawn above the dot (z_index 1): the dot is hidden while it passes behind.
s.add(Dot(id="planet", pos=(0, 0), r=0.8, z_index=1, style={"color": "secondary"}, tags=["key"]))
s.set("camera.height", 8)

s.marker("go", 1.0)
s.marker("leave", 4.0)
s.at("go").to("dot.pos", (4, 0), duration=2.0)  # behind the planet at t = 2
s.at("leave").to("dot.pos", (9, 0), duration=2.0, ease="linear")  # exits the right edge

if __name__ == "__main__":
    here = Path(__file__).parent
    s.save(here / "moving_dot.phasor.json")
    for t in (0.0, 2.0, 3.0, 5.0, 6.0):
        print(f"t={t}:", *s.engine.inspect(t, detail="brief")["objects"], sep="\n  ")
    try:
        from phasor.render.output import save_png

        for t in (0.0, 2.0, 5.0):
            save_png(here / "out" / f"moving_dot_{t:.1f}.png", s.engine.still(t, (960, 540)))
        print("stills written to", here / "out")
    except RuntimeError as e:
        print("no stills:", e)
