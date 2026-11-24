"""Interference and beamforming: two sources, then four in a line steering a beam.

Two point sources switch on one after the other, each sending out a ring. Where the rings
overlap they make the interference pattern: crests glow warm, troughs cool, and the lines
where the two waves cancel (the path difference is half a wavelength) stay dark. Those
lines stand still while the waves run outward along them; blending to the time-averaged
intensity leaves only the fringes. Then a phased array: four sources in a line, half a
wavelength apart, whose phase delays point the beam. Changing the delays swings the beam
from one side to the other; the elements glow as beacons and the beam direction is a ray.

Run:  uv run python examples/interference.py      (prints where the beam points)
"""

from phasor import Scene, ac, expr

s = Scene("interference", look="prism")
s.set("camera.height", 9.0)

s.add(ac.Interference(id="pair", sources=((-1.2, 0.0), (1.2, 0.0)), speed=1.2,
                      onsets=(0.0, 1.5), time=expr("t - 0.4"), size=(16.0, 9.0)))  # fmt: skip
s.add(ac.PhasedArray(id="array", center=(-6.0, 0.0), count=4, spacing=0.5, steer=-40.0,
                     speed=1.2, time=expr("t"), size=(16.0, 9.0)))  # fmt: skip

s.marker("pair", 0.0)
s.marker("average", 7.5)
s.marker("waves", 10.0)
s.marker("pair_exit", 12.0)
s.marker("array", 13.0)
s.marker("swing", 15.0)
s.marker("swing_back", 21.0)
s.marker("end", 27.0)

s.at("pair").reveal("pair")
s.at("average").morph("pair.blend", 1.0)
s.at("waves").morph("pair.blend", 0.0)
s.at("pair_exit").exit("pair")
s.at("array").reveal("array")
s.at("swing").morph("array.steer", 40.0, until="swing_back")
s.at("swing_back").morph("array.steer", 0.0, until="end")

if __name__ == "__main__":
    e = s.engine
    for t in (14.0, 18.0, 21.0, 26.0):
        beam = e.inspect(t, id="array")["objects"][0]["semantic"]
        print(f"t={t:g}: steer {beam['steer']:+.1f} deg, beamwidth {beam['beamwidth_deg']:.1f} deg")
    for issue in e.lint(strict=True):
        print("lint:", issue)
