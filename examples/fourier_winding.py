"""Wind a two-tone signal and let its centroid reveal the tones at 3 and 6 Hz."""

from phasor import Dot, Line, Scene, Text, fourier

CENTER = (-4.55, 0.55, 0)
RADIUS = 2.45
MAX_FREQ = 8.0
SCAN = 2.5
FIRST_RESONANCE = 5.0

s = Scene("fourier_winding", look="prism")
s.set("camera.height", 8.6)
signal = fourier.Winding(
    id="signal",
    pos=CENTER,
    size=RADIUS,
    max_freq=MAX_FREQ,
    fs=256,
    samples=1024,
    freq=0.75,
    progress=0,
    hue_along=0.52,
    spectrum_guide=False,
    style={"color": "secondary", "stroke": "hairline", "colormap": "borealis"},
)
s.add(signal)
x, y, _ = CENTER
baseline_y = y - RADIUS
scan_start, scan_end = x + 1.5 * RADIUS, x + 4.5 * RADIUS
s.add(
    Line(
        id="baseline",
        points=[(scan_start, baseline_y), (scan_end, baseline_y)],
        hue_along=0,
        style={"color": "muted", "glow": "none", "stroke": "hairline"},
    )
)
for frequency in (3, 6):
    peak_x = scan_start + (scan_end - scan_start) * frequency / MAX_FREQ
    s.add(
        Text(
            id=f"tone_{frequency}",
            text=f"{frequency} Hz",
            pos=(peak_x, baseline_y - 0.47, 0),
            style={"size": "annotation", "color": "highlight"},
        )
    )

s.marker("wind", 0.0)
s.marker("scan", SCAN)
s.marker("first_resonance", FIRST_RESONANCE)
s.marker("release_resonance", 6.7)
s.marker("resume", 7.25)
s.marker("second_resonance", 9.75)
s.marker("finish_scan", 10.7)
s.marker("end", 14.3)

s.at("wind").sweep("signal.progress", 1.0, show=False)
s.at("scan", 0.4).reveal("baseline")
s.at("scan").sweep("signal.freq", 3.0, show=False)
s.at("resume").sweep("signal.freq", 6.0, show=False)
s.at("second_resonance").reveal("tone_6")
s.at("finish_scan").sweep("signal.freq", MAX_FREQ, show=False)

# During the hold, both features are stationary. Put the light directly on each
# measured position so the highlight lands on the mark it is explaining.
resonance = signal.features(s.engine.state_at(FIRST_RESONANCE))
for feature, position in resonance.items():
    s.add(Dot(id=f"{feature}_light", pos=tuple(position), r=0.078, preset="beacon", z_index=2))
s.at("first_resonance").reveal("centroid_light", "spectrum_tip_light", "tone_3")
s.add(Text(id="spectrum_tip_light.label", text="Centroid magnitude", style={"size": "label"}))
s.constrain("label(spectrum_tip_light) near spectrum_tip_light")
s.at("first_resonance", 0.35).reveal("spectrum_tip_light.label")
for object_id in ("spectrum_tip_light.label", "centroid_light", "spectrum_tip_light"):
    s.at("release_resonance").exit(object_id)
