"""The modes of a reverb as a field of turning hands.

A reverb's impulse response is a sum of decaying modes, h[n] = sum_k r_k lambda_k^n: each pole
lambda_k = rho_k e^{i theta_k} of the filter is a hand that starts at its residue r_k, turns
theta_k radians every sample and shrinks by rho_k. Here 64 of them (32 conjugate pairs, so the
sum is real) sit on a sunflower spiral, the slow, low modes in the middle and the fast, high ones
outside; each takes the hue of the angle it points at. High modes die first (a darker room
absorbs the highs), so the field empties from the rim inwards.

The fast hands turn tens of degrees a frame, which a sharp frame shows as a stick jumping about
(strobing). With ``frame_ticks`` set, each is drawn as the exact smear of a camera shutter open
half a frame: the same light spread over the sector it swept, a faint fan that grows to a disc.

Run:  uv run python examples/reverb_modes.py      (prints the field as it decays)
"""

import math

from phasor import Scene, Text, expr, filt

PAIRS = 32
RATE = 30.0  # ticks (samples of the filter) per scene second
FPS = 60.0  # the scene's frame rate (the default); a preview at 30 fps smears half as far
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0

rho, theta, residues = [], [], []
for k in range(PAIRS):
    th = math.pi * (k + 0.2 + 0.6 * ((k * GOLDEN) % 1.0)) / PAIRS  # spread, not quite even
    life = 1800.0 * (1.0 - 0.85 * th / math.pi)  # ticks to fall 60 dB: the highs die first
    r = 10.0 ** (-3.0 / life)
    a = 2.0 * math.pi * ((k * GOLDEN * 7.0) % 1.0)  # each mode starts at its own phase
    for sign in (1.0, -1.0):  # conjugate pairs: a real filter
        rho.append(r)
        theta.append(sign * th)
        residues.append((math.cos(a), sign * math.sin(a)))

s = Scene("reverb_modes", look="prism")
s.set("camera.height", 10.4)
s.add(filt.PhasorField(id="modes", poles=filt.polar(rho, theta), residues=tuple(residues),
                       tick=expr(f"{RATE} * t"), frame_ticks=RATE / FPS, shutter=0.5,
                       spacing=1.0, pos=(0.0, -0.25, 0.0), tags=["key"]))  # fmt: skip
s.add(Text(id="title", text="the modes of a reverb", pos=(0.5, 0.045), layer="overlay",
           style={"size": "title"}))  # fmt: skip
s.marker("start", 0.0)
s.marker("end", 14.0)

if __name__ == "__main__":
    e = s.engine
    for t in (0.0, 3.0, 7.0, 13.0):
        sem = e.inspect(t, id="modes")["objects"][0]["semantic"]
        print(f"t={t:g}: tick {sem['tick']:g}, h[n] = {sem['sum'][0]:+.4f}, "
              f"fastest turns {sem['max_turn_per_frame_deg']:.1f} deg a frame, "
              f"{sem['blurred_count']} of {sem['count']} hands smeared")  # fmt: skip
    for issue in e.lint():
        print("lint:", issue)
