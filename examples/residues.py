"""Domain-colored 1/(z^2 + 1) with a glowing contour, at 4K.

The plane shows f(z) = 1/(z^2 + 1) in the prism look (``prism_field``): the two poles at +i
and -i are points of light where every phase meets, the hue of f wrapped round each; rays of
constant phase and rings where |f| doubles are thin light on a near-dark field. A loop
around +i glows with a hue running along it; it grows, and a label names the residue it
picks up. The equation drifts.

Run:  uv run python examples/residues.py          (a 4K still at t = 2.5 s, and a storyboard)
      uv run phasor render examples/residues.py --preset 4k -o out/residues.mp4
"""

from pathlib import Path

import numpy as np

from phasor import Equation, Line, Scene, Text, cx

ANGLES = np.linspace(0, 2 * np.pi, 193)


def circle(center: complex, r: float) -> list[tuple[float, float]]:
    return [(center.real + r * np.cos(a), center.imag + r * np.sin(a)) for a in ANGLES]


s = Scene("residues", look="prism_field")
s.set("camera.height", 4.5)
s.add(cx.DomainColoring(id="plane", function="1/(z**2 + 1)", size=(8.0, 4.5)))
s.add(Line(id="loop", points=circle(1j, 0.35), style={"color": "highlight"}))
s.add(Text(id="loop.label", text="Res = -i/2"))
s.add(Equation(id="f_eq", pos=(2.6, 1.75, 0.0), parts=(r"$f(z) = \dfrac{1}{z^2 + 1}$",)))
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
