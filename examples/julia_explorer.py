"""A Julia-set explorer with a slider for c.

The Julia set of z^2 + c changes shape completely as c moves: for c on the Mandelbrot set it
is one connected piece, outside it dust. Two sliders move the real and imaginary part of c; a
third sets the iteration depth. They are the scene's *controls*, so every viewer gets them:

    uv run phasor gui examples/julia_explorer.py                     # the editor: Controls tab
    uv run python examples/julia_explorer.py page.html               # one self-contained web page
    from phasor.interactive import notebook; notebook(scene.engine)  # inside Jupyter

Run:  uv run python examples/julia_explorer.py [page.html]
"""

import sys

from phasor import Scene, cx

s = Scene("julia_explorer", look="calm")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 2.6)
s.add(cx.Fractal(id="set", kind="julia", c=(-0.8, 0.156), pos=(0.0, 0.0, 0.0), size=(4.6, 2.6),
                 max_iter=120, preset="ocean"))  # fmt: skip
s.control("set.c", component=0, min=-1.2, max=1.2, step=0.05, label="Re c")
s.control("set.c", component=1, min=-1.2, max=1.2, step=0.05, label="Im c")
s.control("set.max_iter", min=30, max=300, step=30, label="iterations")

if __name__ == "__main__":
    from phasor.interactive import Player, export_web

    if len(sys.argv) > 1:
        info = export_web(s.engine, sys.argv[1], steps={"set.c[0]": 7, "set.c[1]": 7,
                                                        "set.max_iter": 2})  # fmt: skip
        print(info)
    else:
        p = Player(s.engine)
        print("controls:", [c.key for c in p.controls], p.values)
        for re, im in ((-0.8, 0.156), (0.285, 0.01), (0.0, 0.8)):
            p.set("set.c[0]", re)
            p.set("set.c[1]", im)
            frame = p.frame(0.0, (240, 135))
            inside = (frame.max(-1) < 40).mean()
            print(f"c = {re:+.3f}{im:+.3f}i: {inside:.0%} of the picture is inside")
