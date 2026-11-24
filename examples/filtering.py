"""A low-pass filter, seen four ways at once.

An order-6 Butterworth low-pass at 1200 Hz (sample rate 8 kHz). Left is its z plane: the
transfer function H(z) as colour, the six poles (crosses) inside the unit circle and the six
zeros (rings, all at z = -1) on it. A probe runs round the circle from DC to half the sample
rate while the gain and phase curves are traced beside it: the gain holds, bends at the cutoff
and falls away. Below, a sum of tones goes in and comes out: the high ones are gone from the
waveform, and in the spectrum the part the filter removes fades out.

Run:  uv run python examples/filtering.py      (prints what the filter is)
"""

from phasor import Scene, Text, filt

LOW = dict(kind="lowpass", order=6, cutoff=1200.0, fs=8000.0)
TONES = dict(frequencies=(250.0, 500.0, 750.0, 1750.0, 2000.0, 2500.0, 3000.0, 3500.0),
             amplitudes=(1.0, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6))  # fmt: skip
PLANE = dict(pos=(-4.3, 1.2, 0.0), scale=2.1)

s = Scene("filtering", look="prism")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 9.0)
s.add(filt.transfer_plane("plane", size=(6.8, 5.2), **PLANE, **LOW))
s.add(filt.PoleZero(id="poles", size=(6.8, 5.2), z_index=1, probe=0.0, **PLANE, **LOW))
s.add(filt.FrequencyResponse(id="gain", plot="gain", pos=(3.9, 2.85, 0.0), extent=(7.2, 1.9),
                             reveal=0.0, **LOW))  # fmt: skip
s.add(filt.FrequencyResponse(id="phase", plot="phase", pos=(3.9, 0.4, 0.0), extent=(7.2, 1.9),
                             reveal=0.0, **LOW))  # fmt: skip
s.add(filt.FilteredSignal(id="wave", pos=(-3.9, -3.0, 0.0), extent=(7.4, 2.4), progress=0.0,
                          **TONES, **LOW))  # fmt: skip
s.add(filt.FilteredSpectrum(id="spectrum", pos=(3.9, -3.0, 0.0), extent=(7.2, 2.4), **TONES,
                            **LOW))  # fmt: skip
s.add(Text(id="title", text="a low-pass filter", pos=(0.5, 0.045), layer="overlay",
           style={"size": "title"}))  # fmt: skip

s.marker("start", 0.3)
s.marker("response", 1.6)
s.marker("signal", 6.0)
s.marker("cut", 9.0)
s.marker("end", 12.0)
s.at("start").reveal("title")
s.at("start").reveal("plane")
s.at("start").reveal("poles")
s.at("start", 0.4).reveal("gain")
s.at("start", 0.6).reveal("phase")
s.at("response").sweep("gain.reveal", 1.0, show=False)  # the curves trace out with the probe
s.at("response").sweep("phase.reveal", 1.0, show=False)
s.at("response").sweep("poles.probe", 1.0, show=False)
s.at("signal").reveal("wave")
s.at("signal").sweep("wave.progress", 1.0, show=False)  # the signal goes in and comes out
s.at("signal", 1.0).reveal("spectrum")
s.at("cut").sweep("spectrum.fade", 1.0, show=False)  # what the filter removes fades out

if __name__ == "__main__":
    e = s.engine
    pz = e.inspect(e.duration, id="poles")["objects"][0]["semantic"]
    print("poles:", pz["poles"])
    print("zeros:", pz["zeros"], "stable:", pz["stable"])
    for issue in e.lint(strict=True):
        print("lint:", issue)
