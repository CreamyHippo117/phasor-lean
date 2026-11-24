"""A picture in a scene: any PNG or JPEG, placed, sized and animated like any object.

The file sits next to the scene (examples/assets/sunset.png, made for this example); the
scene stores only its path. The picture fades in with an outline of light, a curve traces
the far ridge over it, and a caption appears. Put your own picture in instead:
``Image(id="art", path="my_painting.jpg", width=5)``.

Run:  uv run python examples/picture.py      (prints the picture's size)
"""

import numpy as np

from phasor import Image, Line, Scene, Text

s = Scene("picture", look="prism")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 4.5)
s.add(Image(id="sunset", path="assets/sunset.png", width=5.6, pos=(0.0, 0.25, 0.0),
            preset="luminous"))  # fmt: skip
# the far ridge of the picture, traced in light (its shape, roughly, in world units)
x = np.linspace(-2.8, 2.8, 200)
ridge = 0.25 + 1.75 * (0.5 - 0.58) + 0.12 * np.sin(2.1 * x + 0.9) + 0.05 * np.sin(5.3 * x)
s.add(Line(id="ridge", points=[(float(a), float(b), 0.0) for a, b in zip(x, ridge, strict=True)],
           style={"glow": "strong", "color": "highlight"}))  # fmt: skip
s.add(Text(id="caption", text="a picture is an object too", pos=(0.0, -1.95, 0.0)))

s.marker("appear", 0.3)
s.marker("trace", 1.8)
s.marker("caption", 3.6)
s.marker("end", 6.0)
s.at("appear").reveal("sunset")
s.at("trace").trace("ridge")
s.at("caption").reveal("caption")

if __name__ == "__main__":
    sem = s.engine.inspect(0.0, id="sunset")["objects"][0]["semantic"]
    print("picture:", sem["pixels"], "pixels, drawn", sem["size"], "world units")
    for issue in s.engine.lint(strict=True):
        print("lint:", issue)
