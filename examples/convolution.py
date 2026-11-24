"""Live convolution: a box and a decaying kernel.

The box f is the signal; the kernel g (a decaying exponential) is flipped and slid along it.
Where they overlap, their product is shaded under the curves, and the area of that shading is
the output (f * g)(tau): it rises while the kernel enters the box, then falls away as the kernel
leaves, tracing a curve that is the output point by point. Two hues for the two signals, the
overlap glowing, the output drawn as light as it is made.

Run:  uv run python examples/convolution.py      (prints the output at three moments)
"""

from phasor import Scene, Text, conv

s = Scene("convolution", look="prism")
s.add(conv.Convolution(id="c", f="box", g="decay", progress=0.0, pos=(0, -0.2, 0)))
s.add(Text(id="title", text="convolution: the area of the overlap", pos=(8 / 9, 0.07),
           layer="overlay", style={"size": "title"}))  # fmt: skip

s.marker("start", 0.3)
s.marker("slide", 1.0)
s.marker("end", 5.0)
s.at("start").reveal("title")
s.at("slide").sweep("c.progress", 1.0, show=False)  # slide the kernel across, evenly

if __name__ == "__main__":
    e = s.engine
    for t in (1.7, 2.3, 3.6):
        sem = e.inspect(t, id="c")["objects"][0]["semantic"]
        print(f"t={t}: shift {sem['shift']:.2f}, output {sem['output']:.3f}")
    for issue in e.lint(strict=True):
        print("lint:", issue)
