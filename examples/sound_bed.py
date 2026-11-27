"""A sound bed under the narration, with cues on the markers.

The fifth roots of unity appear around the unit circle while the narration says so. Under
it all runs ``assets/drift.ogg``, an 8 s ambient loop (made with NumPy for this example;
beds are your own files): ``s.bed`` loops it from the first word to the end, 18 dB under the
voice, and dips it 8 dB more while the voice speaks (theme ``[sound]``). Cues are anchored to
markers like motions: a whoosh lands just before the circle is named, each root pops on in
the pause, and a chime closes the last sentence. Re-narrate (another voice, an edited line)
and every cue and the bed's start move with the words.

Run:  uv run python examples/sound_bed.py        (lint, and the mix in numbers)
      uv run phasor render examples/sound_bed.py --preset draft -o examples/out/sound_bed.mp4
"""

import math

from phasor import Dot, Line, Scene, Text

SCRIPT = (
    "Take the [[circle]] unit circle. "
    "The fifth roots of unity sit evenly around it [[roots_said]]. "
    "[[pause 2.5]] "
    "Multiply all five [[product]] and you come back to one [[one]]."
)

s = Scene("sound_bed")
s.set("camera.height", 5)
ring = [(math.cos(a), math.sin(a)) for a in (2 * math.pi * k / 128 for k in range(129))]
s.add(Line(id="circle", points=[(1.6 * x, 1.6 * y) for x, y in ring], style={"color": "muted"}))
for k in range(5):
    a = 2 * math.pi * k / 5
    s.add(Dot(id=f"root{k}", pos=(1.6 * math.cos(a), 1.6 * math.sin(a)), r=0.12,
              style={"color": "highlight" if k == 0 else "primary"}, tags=["key"]))  # fmt: skip
s.add(Text(id="product", text="$\\omega^0 \\omega^1 \\omega^2 \\omega^3 \\omega^4 = 1$",
           pos=(0, -2.2), style={"size": "label"}))  # fmt: skip

s.narrate(SCRIPT)  # espeak-ng if installed, else timing estimated from the text
s.at("circle").trace("circle")
for k in range(5):  # one root at a time, in the pause after the sentence
    s.at("roots_said", 0.4 * k + 0.3).bloom_in(f"root{k}")
s.at("product").reveal("product")
s.at("one").flash("root0")

s.bed("assets/drift.ogg", start="circle", fade=1.5)
s.cue("circle", "whoosh", offset=-0.75, gain=-14)  # peaks just before "circle" is spoken
for k in range(5):
    s.cue("roots_said", "pop", offset=0.4 * k + 0.3, midi=72 + [0, 2, 4, 7, 9][k],
          pan=math.cos(2 * math.pi * k / 5) * 0.6)  # fmt: skip
s.cue("one", "chime", offset=0.2)

if __name__ == "__main__":
    for issue in s.engine.lint(strict=True):
        print(issue)
    r = s.engine.sound_report()
    print(f"mix: {r['loudness']} LUFS, true peak {r['true_peak_dbtp']} dBTP")
    for lane in r["lanes"]:
        print(f"  {lane['id']:<22} {lane['loudness']} LUFS  per block {lane['blocks']}")
