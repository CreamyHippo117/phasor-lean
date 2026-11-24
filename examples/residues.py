"""Domain-colored 1/(z^2 + 1) with a glowing contour, at 4K.

The plane shows f(z) = 1/(z^2 + 1): hue is the phase, the bands step each time |f| doubles,
and the two poles at +i and -i are where every color meets. A loop around +i glows; it
grows, and a label names the residue it picks up.

Run:  uv run python examples/residues.py          (a 4K still at t = 2.5 s, and a storyboard)
      uv run phasor render examples/residues.py --preset 4k -o out/residues.mp4
"""

from pathlib import Path

import numpy as np

from phasor import Line, Scene, Text, cx

ANGLES = np.linspace(0, 2 * np.pi, 193)


def circle(center: complex, r: float) -> list[tuple[float, float]]:
    return [(center.real + r * np.cos(a), center.imag + r * np.sin(a)) for a in ANGLES]


s = Scene("residues")
s.set("camera.height", 4.5)
s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(8.0, 4.5)))
s.add(Line(id="loop", points=circle(1j, 0.35),
           style={"color": "highlight", "stroke": "bold", "glow": "strong"}))  # fmt: skip
s.add(Text(id="loop.label", text="Res = -i/2", style={"color": "primary"}))
s.add(Text(id="title", text="f(z) = 1 / (z^2 + 1)", pos=(8 / 9, 0.08), layer="overlay",
           style={"size": "title"}))  # fmt: skip
s.constrain("label(loop) near loop")

s.marker("grow", 1.0)
s.at("grow").to("loop.points", circle(1j, 0.7), duration=1.5)

if __name__ == "__main__":
    from phasor.render.output import encode_srgb8, save_png

    out = Path(__file__).parent / "out"
    for issue in s.engine.lint():
        print("lint:", issue.severity, issue)
    save_png(out / "residues_4k.png", encode_srgb8(s.engine.frame(2.5, (3840, 2160))))
    save_png(out / "residues_storyboard.png", s.engine.storyboard(marks=True)["image"])
    print("wrote", out / "residues_4k.png", "and", out / "residues_storyboard.png")
