"""Heat maps in the prism look: a hot spot spreading, a drum vibrating, a chirp's spectrum.

Everything glows out of the dark in the hues of the spectrum, and every panel melts into the
background at its edges. On the left, heat put into the middle of a square plate spreads
out and fades: it glows where it is hot, with contour lines of light at even steps of
temperature, each in the hue of its temperature. On the right, a drum skin vibrating in its
(2, 3) mode: cells moving up and cells moving down glow in hues from opposite ends of the
spectrum, and the nodal lines between them, which never move, are drawn as light. Below,
the spectrogram of a tone sliding upward: the glowing trace is the tone's pitch over time,
running through the spectrum as it rises.

Run:  uv run python examples/prism_heat.py      (prints the plate's heat and the drum's state)
"""

from phasor import Scene, expr
from phasor.domains import pde
from phasor.domains.fourier import Spectrogram

s = Scene("prism_heat", look="prism")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 9.0)
s.add(pde.Field(id="plate", kind="heat", mode=(0, 0), diffusivity=0.25, scale=0.6,
                time=expr("0.02 + t / 10"), size=(4.6, 4.6), pos=(-3.4, 1.4, 0),
                n=64))  # fmt: skip
s.add(pde.Field(id="drum", kind="wave", mode=(2, 3), scale=1.2,
                time=expr("t / 4"), size=(4.6, 4.6), pos=(2.6, 1.4, 0), n=64))  # fmt: skip
s.add(Spectrogram(id="chirp", pos=(0.0, -3.0, 0.0), size=(13.0, 2.4)))

s.marker("start", 0.0)
s.marker("end", 8.0)

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 4.0, 8.0):
        plate = e.inspect(t, id="plate")["objects"][0]["semantic"]
        drum = e.inspect(t, id="drum")["objects"][0]["semantic"]
        print(f"t={t:g}: plate heat {plate.get('total_heat', 0):.3f}, "
              f"drum swing {drum['min']:+.2f} to {drum['max']:+.2f}")  # fmt: skip
    for issue in e.lint(strict=True):
        print("lint:", issue)
