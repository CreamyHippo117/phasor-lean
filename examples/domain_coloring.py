"""Domain coloring of 1/(z^2 + 1) and a Mandelbrot field.

Run:  uv run python examples/domain_coloring.py
Uses CUDA when there is an NVIDIA GPU and the NumPy reference otherwise; the pictures match.
"""

from pathlib import Path

import numpy as np

from phasor.render import gpu
from phasor.render.output import encode_srgb8, save_png

if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    backend = "CUDA" if gpu.default_gpu() else "NumPy reference (no CUDA device)"
    print("backend:", backend)

    f = gpu.ComplexFunction.parse("1/(z**2 + 1)")
    print("f(z) =", f.source, "| LaTeX:", f.latex())
    picture = gpu.render("domain", 1280, 720, function=f, center=(0, 0), height=4.5, supersample=2)
    save_png(out / "domain_coloring.png", encode_srgb8(picture))

    field = gpu.render("escape", 1280, 720, center=(-0.6, 0), height=2.6, max_iter=300)
    # A plain gray ramp for now; themed fractal coloring arrives with the first theme.
    shade = np.where(field < 0, 0.0, 0.15 + 0.85 * (np.sqrt(np.maximum(field, 0)) % 4) / 4)
    rgba = np.dstack([shade, shade, shade, np.ones_like(shade)]).astype(np.float32)
    save_png(out / "mandelbrot.png", encode_srgb8(rgba))
    print("wrote", out / "domain_coloring.png", "and", out / "mandelbrot.png")
