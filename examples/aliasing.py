"""Sampling and aliasing: the wagon wheel.

A 5 Hz tone (cyan) is sampled at fs (white beacons on stems); the amber wave is the slowest one
through the same samples. While fs is above 10 Hz (twice the tone) the amber wave is the tone
itself. Sweep fs down past 10 Hz and the samples also fit a slower wave, running backwards
(its phase reflected); it slows as fs falls, stands still at fs = 5 Hz, then creeps forwards.
Below, the spectrum shows the true line, its alias mirrored about the Nyquist mark fs/2 and an
arc from one to the other; the wheel beside it is the tone's phasor at each sample: the inner
spiral is the turn it really makes between two samples, the outer arc the short turn the eye
sees.

Run:  uv run python examples/aliasing.py      (prints the alias at a few moments)
"""

from phasor import Counter, Scene, Text, expr, fourier

s = Scene("aliasing", look="prism")
s.add(fourier.Aliasing(id="a", frequencies=(5.0,), phases=(0.5,), fs=14.0, duration=1.6,
                       fmax=16.0, reveal=0.0, pos=(0, -0.55, 0), extent=(12.0, 5.8)))  # fmt: skip
s.add(Text(id="title", text="sampling: the slowest wave through the samples",
           pos=(8 / 9, 0.07), layer="overlay", style={"size": "title"}))  # fmt: skip

s.add(Counter(id="rate", value=expr("a.fs"), digits=1, prefix="f_s = ", suffix=r"\,\mathrm{Hz}",
              pos=(5.3, 2.8, 0), opacity=0.0))  # fmt: skip

s.marker("start", 0.3)
s.marker("sample", 0.8)
s.marker("sweep", 3.6)
s.marker("rate_low", 12.0)
s.marker("end", 13.0)
s.at("start").reveal("title")
s.at("sample").sweep("a.reveal", 1.0, show=False)  # take the samples, left to right
s.at("sweep", -0.6).reveal("rate")
# fs falls slowly, past 2f = 10 Hz (the alias appears, backwards), to f = 5 Hz (it stops)
s.at("sweep").morph("a.fs", 4.2, until="rate_low")

if __name__ == "__main__":
    e = s.engine
    for t in (3.0, 6.5, 9.0, 10.8, 12.5):
        sem = e.inspect(t, id="a")["objects"][0]["semantic"]
        print(f"t={t}: fs {sem['fs']:.2f} Hz, alias {sem['alias_frequencies'][0]:.2f} Hz, "
              f"{'backwards' if sem['reversed'][0] else 'forwards'}")  # fmt: skip
    for issue in e.lint(strict=True):
        print("lint:", issue)
