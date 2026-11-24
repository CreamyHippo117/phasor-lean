"""A tour of the harmonic-analysis and linear-algebra modules.

A square wave's Fourier series with its Gibbs overshoot, the spectrum of a tone that does not
sit on a bin (leakage, then a Hann window), and a matrix that squashes space: the grid, the
unit square's area (the determinant) and the matrix entries, all moving together.

Run:  uv run python examples/fourier_linalg.py       (prints what the inspector sees)
"""

from phasor import Scene, expr, fourier, linalg

s = Scene("fourier_linalg", look="calm")
s.set("camera.height", 12)

s.add(fourier.PartialSum(id="wave", kind="square", terms=9, pos=(-4, 3.5, 0), extent=(7, 2.4),
                         preset="glowing"))  # fmt: skip
s.add(fourier.Spectrum(id="spec", frequencies=(100.5,), amplitudes=(1.0,), window="rect",
                       samples=256, fs=256.0, pos=(-7.5, -4.5, 0), extent=(5, 2.2)))  # fmt: skip

s.add(linalg.TransformGrid(id="grid", matrix=((1.0, 0.0), (0.0, 1.0)), pos=(5, -1, 0), extent=2))
s.add(linalg.Matrix(id="mat", matrix=expr("grid.matrix"), pos=(5, 4.2, 0)))
s.add(linalg.EigenLines(id="eig", matrix=((2.0, 1.0), (1.0, 2.0)), pos=(5, -1, 0), length=2.5))

s.marker("more_terms", 2.0)
s.marker("window", 4.0)
s.marker("squash", 6.0)
s.at("more_terms").morph("wave.terms", 40.0)  # more harmonics: the overshoot stays
s.at("window").morph("grid.matrix", ((2.0, 1.0), (1.0, 2.0)))
s.at("squash").morph("grid.matrix", ((1.0, 2.0), (0.5, 1.0)))

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 4.0, 8.0):
        print(f"t = {t:g}")
        for line in e.inspect(t, detail="brief")["objects"]:
            print("  ", line)
    sem = e.inspect(8.0, id="grid")["objects"][0]["semantic"]
    print("determinant at the end:", sem["determinant"], "(rank", sem["rank"], ")")
