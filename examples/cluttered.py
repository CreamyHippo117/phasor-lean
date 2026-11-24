"""A deliberately cluttered scene is flagged, then auto-fixed.

Three poles sit close together and move closer. Every label was dropped at the same naive
offset, so labels collide with each other, with a neighboring pole and with the contour
line; the title hangs off the top of the frame.

Run:  uv run python examples/cluttered.py
It prints the lint report, applies the suggested fixes, prints the report again, and writes
before/after storyboards (plain, and with object marks) next to this file.
"""

from pathlib import Path

from phasor import Dot, Line, Scene, Text

s = Scene("cluttered")
s.set("camera.height", 5)

s.add(Line(id="contour", points=[(-4, -2.7), (4, 1.3)], style={"color": "secondary"}))
POLES = {"pole_a": (-0.8, 0.0), "pole_b": (0.0, 0.1), "pole_c": (0.8, 0.0)}
for name, (x, y) in POLES.items():
    s.add(Dot(id=name, pos=(x, y), r=0.12, style={"color": "highlight"}, tags=["key"]))
    # Naive placement: every label just to the right of its pole, and never moved again.
    s.add(Text(id=f"{name}.label", text=f"residue {name[-1]}", pos=(x + 0.7, y + 0.05)))
s.add(Text(id="title", text="Three poles", pos=(0, 2.45), style={"size": "title"}))

s.marker("squeeze", 1.0)
s.at("squeeze").morph("pole_a.pos", (-0.45, 0.35))
s.at("squeeze").morph("pole_c.pos", (0.45, 0.4))


def report(title: str) -> None:
    issues = s.engine.lint()
    print(f"{title}: {len(issues)} issue(s)")
    for issue in issues:
        print("  ", issue.severity, issue)


if __name__ == "__main__":
    here = Path(__file__).parent

    def board(name: str) -> None:
        try:
            from phasor.render.output import save_png

            for marks in (False, True):
                image = s.engine.storyboard([0.0, 1.75, 2.5], marks=marks, size=(640, 360))
                suffix = "_marks" if marks else ""
                save_png(here / "out" / f"cluttered_{name}{suffix}.png", image["image"])
        except RuntimeError as e:
            print("no storyboard:", e)

    report("before")
    board("before")
    fixed = s.engine.autofix()
    print("applied:", *(op["rule"] for op in fixed["applied"]), sep="\n   ")
    report("after")
    board("after")
    print("label slots over time:", {k: sorted(set(v)) for k, v in s.engine.layout.slots.items()})
