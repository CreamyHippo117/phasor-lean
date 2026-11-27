"""Sound cues: the harmonic series, seen and heard.

A string's overtones sit at 1, 2, 3, 4, 5 times its pitch. Each one appears on its marker,
and on the same marker the soundtrack plays it: ``s.cue(marker, "tone", midi=...)`` puts a
dry synthesized note there, so moving the marker (or re-narrating: ``s.narrate`` with
``[[h1]]`` ... in the script makes these markers words) moves the sound with the picture.
The last beat plays the first five harmonics together, the chord a plucked string makes,
each partial quieter than the one below. (A recorded sound is ``s.cue("chord",
"audio/pluck.wav")``, a file beside this script; other built-ins: tick, chime, whoosh,
pop, swell.)

Run:  uv run python examples/sound_harmonics.py        (lint report and the sound report)
      uv run phasor render examples/sound_harmonics.py --preset draft -o examples/out/harmonics.mp4
"""

import math

from phasor import Dot, Line, Scene, Text

BASE = 45  # A2, 110 Hz

s = Scene("sound_harmonics")
s.set("camera.height", 6)
s.add(Line(id="string", points=[(-5, -1.5), (5, -1.5)], style={"color": "muted"}))
for k in range(1, 6):
    s.add(Dot(id=f"h{k}", pos=(-4 + 2 * (k - 1), -1.5 + 0.6 * k), r=0.12 + 0.03 * (6 - k),
              style={"color": "highlight" if k == 1 else "primary"}, tags=["key"]))  # fmt: skip
    s.add(Text(id=f"h{k}.label", text=f"{k}f", style={"size": "label"}))
    s.constrain(f"label(h{k}) near h{k} gap 0.15")

for k in range(1, 6):
    s.marker(f"h{k}", 0.5 + 1.5 * (k - 1))
s.marker("chord", 8.5)
s.marker("end", 11.0)
s.at("h1").reveal("string")
for k in range(1, 6):
    s.at(f"h{k}").bloom_in(f"h{k}").then().reveal(f"h{k}.label")
    # The k-th harmonic is 12 log2(k) semitones above the fundamental (not always a whole
    # number: the 3rd and 5th harmonics are a little off the piano's keys).
    s.cue(f"h{k}", "tone", midi=BASE + 12 * math.log2(k), dur=0.9, voice="soft")
s.at("chord").flash("string")  # the string itself rings
for k in range(1, 6):
    s.cue("chord", "tone", midi=BASE + 12 * math.log2(k), dur=2.0, voice="pluck",
          gain=-10 - 6 * math.log2(k), id=f"chord_{k}")  # fmt: skip

if __name__ == "__main__":
    for issue in s.engine.lint(strict=True):
        print(issue)
    report = s.engine.sound_report()
    print(f"mix: {report['loudness']} LUFS, true peak {report['true_peak_dbtp']} dBTP")
    for lane in report["lanes"]:
        print(f"  {lane['t']:6.2f} s  {lane['id']}  ({lane['loudness']} LUFS)")
