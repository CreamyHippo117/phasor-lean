"""Transitions in the prism look: a function changing while the equations keep up with it.

f(z) = (z^5 - 1 - a) / (z^2 + a) glows out of the dark: five zeros near the unit circle (the
fifth roots of 1 + a) and two poles at z = +-sqrt(-a). Then, beat by beat:

1. the pole equation is solved beside it, z^2 + a = 0 -> z^2 = -a -> z = +-sqrt(-a): the
   "+a" travels over the "=" and becomes "-a", the "z^2" becomes "z";
2. the imaginary part of a goes from 0.6 to 0.7: the poles and zeros shift a little, and
   "0.6 i" turns into "0.7 i" in place;
3. a gains a real part x, which sweeps slowly from 0 to 0.5 while its value counts;
4. the function blends into (z^3 - 1 - a) / (z^2 + a): w = (1 - s) f + s g, so every frame
   is a true picture of a function, two zeros run off to infinity and the poles stay put; the
   equation opens up into the blend, then settles on the new function.

Every motion is a verb anchored to a marker; durations come from the theme, or run until
the next marker (``morph(..., until=)``). The look is ``prism_field``: equations drift less
than in ``prism`` and the field is more see-through, with more light in its lines.

Run:  uv run python examples/prism_transitions.py      (prints the poles at each beat)
"""

from phasor import Counter, Equation, Scene, cx, expr

F = "(z**5 - 1 - a)/(z**2 + a)"
G = "(z**3 - 1 - a)/(z**2 + a)"
F_TEX = r"$\dfrac{z^5 - 1 - a}{z^2 + a}$"
G_TEX = r"$\dfrac{z^3 - 1 - a}{z^2 + a}$"

s = Scene("prism_transitions", look="prism_field")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 4.5)
s.add(cx.DomainColoring(id="plane", function=F, to=G, a=(0.0, 0.6), size=(8.0, 4.5)))

# The function, bottom centre: f, then the blend of f and g, then g.
LHS = r"$f(z) =$"
s.add(Equation(id="f_eq", pos=(0.0, -1.75, 0.0), parts=(LHS, F_TEX), keys=("lhs", "f")))
s.add(Equation(id="mix_eq", pos=(0.0, -1.75, 0.0),
               parts=(LHS, r"$(1 - s)$", F_TEX, r"$+\;s$", G_TEX),
               colors=("primary", "secondary", "primary", "secondary", "highlight"),
               keys=("lhs", "", "f", "", "g")))  # fmt: skip
s.add(Equation(id="g_eq", pos=(0.0, -1.75, 0.0), parts=(LHS, G_TEX),
               colors=("primary", "highlight"), keys=("lhs", "g")))  # fmt: skip
s.add(Counter(id="s_value", value=expr("plane.blend"), prefix="s = ", pos=(2.9, -1.75, 0.0),
              opacity=0.0))  # fmt: skip

# The constant, top left: a = 0.6 i, then 0.7 i, then 0.5 + 0.7 i.
A_KEYS = ("a", "eq", "value")
s.add(Equation(id="a_06", pos=(-2.9, 1.8, 0.0), parts=("$a$", "$=$", r"$0.6\,i$"), keys=A_KEYS))
s.add(Equation(id="a_07", pos=(-2.9, 1.8, 0.0), parts=("$a$", "$=$", r"$0.7\,i$"), keys=A_KEYS))
s.add(Equation(id="a_x", pos=(-2.9, 1.8, 0.0), parts=("$a$", "$=$", r"$x + 0.7\,i$"),
               keys=A_KEYS))  # fmt: skip
s.add(Equation(id="a_end", pos=(-2.9, 1.8, 0.0), parts=("$a$", "$=$", r"$0.5 + 0.7\,i$"),
               keys=A_KEYS))  # fmt: skip
# while the real part x sweeps, its value counts below
s.add(Counter(id="x_value", value=expr("plane.a[0]"), prefix="x = ", pos=(-2.9, 1.25, 0.0),
              opacity=0.0))  # fmt: skip

# The poles, top right, solved step by step.
POLES = (2.9, 1.8, 0.0)
s.add(Equation(id="poles_1", pos=POLES, parts=("$z^2$", r"$+\,a$", "$=$", "$0$"),
               keys=("z", "a", "eq", "zero")))  # fmt: skip
s.add(Equation(id="poles_2", pos=POLES, parts=("$z^2$", "$=$", "$-a$"), keys=("z", "eq", "a")))
s.add(Equation(id="poles_3", pos=POLES, parts=("$z$", "$=$", r"$\pm\sqrt{-a}$"),
               colors=("highlight",), keys=("z", "eq", "a")))  # fmt: skip

for name, t in (
    ("appear", 0.4),
    ("solve", 3.5),
    ("rearrange", 5.3),
    ("root", 7.1),
    ("lift", 9.2),
    ("lift_done", 12.2),
    ("sweep", 13.2),
    ("sweep_go", 14.8),
    ("sweep_done", 20.0),
    ("blend", 21.6),
    ("blend_done", 29.0),
    ("settled", 29.7),
    ("end", 32.5),
):
    s.marker(name, t)  # fmt: skip

s.at("appear").reveal("plane")
s.at("appear", 0.8).write("f_eq")
s.at("appear", 1.6).reveal("a_06")

# 1. the pole equation, solved
s.at("solve").write("poles_1")
s.at("rearrange").transform("poles_1", "poles_2")
s.at("root").transform("poles_2", "poles_3")

# 2. the imaginary part of a: 0.6 -> 0.7 (the poles and zeros shift a little)
s.at("lift").morph("plane.a", (0.0, 0.7), until="lift_done")
s.at("lift").transform("a_06", "a_07")

# 3. a gains a real part, which sweeps slowly while its value counts
s.at("sweep").transform("a_07", "a_x").then().reveal("x_value")
s.at("sweep_go").morph("plane.a", (0.5, 0.7), until="sweep_done")
s.at("sweep_done").transform("a_x", "a_end")
s.at("sweep_done").exit("x_value")

# 4. one function becomes another
s.at("blend").transform("f_eq", "mix_eq")
s.at("blend").reveal("s_value")
s.at("blend").morph("plane.blend", 1.0, until="blend_done")
s.at("blend_done").transform("mix_eq", "g_eq")
s.at("blend_done").exit("s_value")
s.at("settled").flash("poles_3")

if __name__ == "__main__":
    e = s.engine
    markers = e.timeline()["markers"]
    for beat in ("appear", "lift_done", "sweep_done", "blend_done", "end"):
        sem = e.inspect(markers[beat], id="plane")["objects"][0]["semantic"]
        print(f"{beat:11s} a = {sem['a']}  poles = {sem['poles']}  zeros: {len(sem['zeros'])}")
    for issue in e.lint(strict=True):
        print("lint:", issue)
