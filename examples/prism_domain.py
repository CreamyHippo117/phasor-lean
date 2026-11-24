"""Domain coloring in the prism look: a rational function glowing out of the dark.

f(z) = (z^5 - 1) / (z^2 + 0.6 i) has five zeros on the unit circle and two poles inside it.
Each zero and pole is a point of light with the phase of f wrapped round it as colour (the
colours run one way round a zero and the other way round a pole); rays of constant phase and
rings where |f| doubles are drawn as thin light; the rest is near-dark, with a faint wash of
phase. The look is lively, so the colours drift a little round the wheel. It is
``prism_field``: prism with the equation drifting less and the field more see-through, its
lines carrying more light. ``examples/prism_transitions.py`` animates it.

Run:  uv run python examples/prism_domain.py      (prints the zeros and poles it shows)
"""

from phasor import Equation, Scene, cx

F = "(z**5 - 1)/(z**2 + 0.6j)"

s = Scene("prism_domain", look="prism_field")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 4.5)
s.add(cx.DomainColoring(id="plane", function=F, size=(8.0, 4.5), radius=0.2))
s.add(Equation(id="f_eq", pos=(0.0, 1.85, 0.0),
               parts=(r"$f(z) = \dfrac{z^5 - 1}{z^2 + 0.6\,i}$",)))  # fmt: skip

s.marker("appear", 0.3)
s.marker("end", 8.0)
s.at("appear").reveal("plane")
s.at("appear", 0.8).reveal("f_eq")

if __name__ == "__main__":
    sem = s.engine.inspect(0.0, id="plane")["objects"][0]["semantic"]
    print("zeros:", sem["zeros"])
    print("poles:", sem["poles"])
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
