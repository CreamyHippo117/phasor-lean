"""A spectrogram of any signal, with the playhead sweeping along it.

The signal is a formula in ``t`` seconds: a tone sliding upward, its octave above it (half as
loud), and a steady faint tone at 1500 Hz. Time runs across, frequency up, brightness is
strength; the playhead is a glowing line moving at the pace of the clock, and the scene
reports the strongest frequency under it.

To hear-and-see a sound file instead, put a ``.wav`` next to the scene and write
``Spectrogram(id="sg", wav="voice.wav", fmax=4000, playhead=expr("t / 5"))``: its length and
sample rate are the file's own. The ``__main__`` block below does exactly that with a short
file it writes itself, so no audio is stored in the repository.

Run:  uv run python examples/prism_spectrogram.py
"""

from phasor import Scene, expr
from phasor.domains.fourier import Spectrogram

SWEEP = "200 * t + 120 * t * t"  # phase of a tone sliding from 200 Hz up (in cycles)
SIGNAL = f"sin(tau * ({SWEEP})) + 0.5 * sin(tau * 2 * ({SWEEP})) + 0.15 * sin(tau * 1500 * t)"

s = Scene("prism_spectrogram", look="prism")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 4.4)
s.add(Spectrogram(id="sg", signal=SIGNAL, duration=6.0, fs=8000.0, nwin=512, size=(7.6, 3.6),
                  playhead=expr("clamp(t / 6, 0, 1)")))  # fmt: skip

s.marker("start", 0.0)
s.marker("end", 6.0)

if __name__ == "__main__":
    import tempfile
    import wave
    from pathlib import Path

    import numpy as np

    e = s.engine
    for t in (0.5, 3.0, 5.5):
        sem = e.inspect(t, id="sg")["objects"][0]["semantic"]
        print(f"t={t:g}: the strongest frequency under the playhead is "
              f"{sem['dominant_frequency']:.0f} Hz")  # fmt: skip
    for issue in e.lint(strict=True):
        print("lint:", issue)

    # The same, from a sound file next to the scene (here a short file written on the fly).
    with tempfile.TemporaryDirectory() as folder:
        fs = 16000
        t = np.arange(fs * 2) / fs
        pcm = (0.6 * np.sin(2 * np.pi * (300 * t + 300 * t * t)) * 32000).astype("<i2")
        with wave.open(str(Path(folder) / "sweep.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(fs)
            w.writeframes(pcm.tobytes())
        f = Scene("from_wav", base_dir=folder)
        f.add(Spectrogram(id="sg", wav="sweep.wav", fmax=2000.0, playhead=0.5))
        sem = f.engine.inspect(0.0, id="sg")["objects"][0]["semantic"]
        print(f"sweep.wav ({sem['duration']:.1f} s): {sem['dominant_frequency']:.0f} Hz halfway")
