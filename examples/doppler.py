"""Three constant-speed views of the Doppler effect, in slow motion.

The wavefronts crowd ahead of a subsonic source. At Mach 1 their leading
edges meet at the source. Above Mach 1 the rings trail behind, tangent to
the two sides of the Mach cone. Warm hues show higher local pitch. The
views dissolve around the same source position and emission clock.
"""

from phasor import Line, Scene, Text, ac, expr

s = Scene("doppler", look="prism")
s.set("camera.height", 9.7)
size = 4.0
y = -0.34
clock = "0.76 + 0.14 * t"
into_sonic = "0.5 - 0.5 * cos(pi * clamp((t - 2.55) / 0.9, 0, 1))"
into_cone = "0.5 - 0.5 * cos(pi * clamp((t - 5.55) / 0.9, 0, 1))"
frame_for_cone = "0.5 - 0.5 * cos(pi * clamp((t - 3.45) / 2.15, 0, 1))"
source_x = f"2.65 + 0.20 * t + 1.85 * ({frame_for_cone})"

s.add(Line(id="travel_axis", points=((-7.7, y, 0), (7.7, y, 0)),
           style={"color": "muted", "stroke": "hairline", "glow": "none"}))  # fmt: skip

runs = (
    ("subsonic", 0.65, "Subsonic   Mach 0.65", f"1 - ({into_sonic})"),
    ("sonic", 1.0, "Sonic   Mach 1", f"({into_sonic}) - ({into_cone})"),
    ("supersonic", 2.2, "Supersonic   Mach 2.2", into_cone),
)
for name, mach, title, weight in runs:
    active = expr(weight)
    # Translate each constant-speed solution into the same source-centered view.
    origin = expr(f"vec({source_x} - {mach * size} * {name}.time, {y}, 0)")
    s.add(ac.Doppler(id=name, speed=mach, frequency=15, time=expr(clock),
                     pos=origin, size=size,
                     opacity=active))  # fmt: skip
    caption = f"{name}_caption"
    # Let the geometries overlap while each caption gets its own beat.
    caption_opacity = expr(f"max(0, 2 * ({weight}) - 1)")
    s.add(Text(id=caption, text=title, pos=(8 / 9, 0.055), layer="overlay",
               opacity=caption_opacity, style={"size": "label"}))  # fmt: skip

s.marker("subsonic", 1.7)
s.marker("sonic_match", 3.0)
s.marker("sonic", 4.7)
s.marker("cone_match", 6.0)
s.marker("supersonic", 8.4)
s.marker("end", 9.0)
