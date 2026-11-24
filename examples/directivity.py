"""Steer a linear array, narrow a piston beam, then compare normalized polar patterns."""

from phasor import Scene, Text, ac

s = Scene("directivity", look="prism")
s.set("camera.height", 7)
s.add(ac.Directivity(id="array", elements=5, steer=-30, size=2.3, opacity=0.82))
s.add(
    ac.Directivity(
        id="piston",
        kind="piston",
        frequency=480,
        size=2.3,
        opacity=0.82,
        style={"color": "secondary"},
    )
)

for name, title, detail in (
    ("array", "Linear array", "Electronic steering"),
    ("piston", "Baffled piston", "480 Hz to 1.7 kHz"),
):
    s.add(
        Text(
            id=f"{name}_title",
            text=title,
            layer="overlay",
            pos=(8 / 9, 0.115),
            style={"size": "title"},
        )
    )
    s.add(Text(id=f"{name}_detail", text=detail, layer="overlay", pos=(8 / 9, 0.89)))

for kind, title, x, color, params in (
    ("array", "Linear array", -3.85, "highlight", {"elements": 5, "steer": 35}),
    ("piston", "Baffled piston", 0.0, "secondary", {"frequency": 1700}),
    ("cardioid", "Cardioid", 3.85, "primary", {}),
):
    s.add(
        ac.Directivity(
            id=f"compare_{kind}",
            kind=kind,
            pos=(x, 0, 0),
            size=1.62,
            opacity=0.68,
            style={"color": color, "glow": "soft"},
            **params,
        )
    )
    s.add(
        Text(
            id=f"compare_{kind}_title",
            text=title,
            layer="overlay",
            pos=(8 / 9 + x / 7, 0.81),
            style={"size": "title"},
        )
    )

s.add(
    Text(
        id="shared_scale",
        text="Outer ring = 1 (normalized pressure)",
        layer="overlay",
        pos=(8 / 9, 0.16),
        style={"size": "label"},
    )
)

s.marker("array", 0.15)
s.marker("steer", 0.95)
s.marker("array_exit", 3.6)
s.marker("piston", 4.25)
s.marker("narrow", 5.05)
s.marker("piston_exit", 7.8)
s.marker("compare", 8.45)
s.marker("end", 11.5)

s.at("array").reveal("array", "array_title", "array_detail")
s.at("steer").sweep("array.steer", 35.0, show=False)
s.at("array_exit").exit("array", "array_title", "array_detail")
s.at("piston").reveal("piston", "piston_title", "piston_detail")
s.at("narrow").sweep("piston.frequency", 1700.0, show=False)
s.at("piston_exit").exit("piston", "piston_title", "piston_detail")
s.at("compare").reveal(
    "compare_array",
    "compare_piston",
    "compare_cardioid",
)
s.at("compare", 0.65).reveal(
    "compare_array_title",
    "compare_piston_title",
    "compare_cardioid_title",
    "shared_scale",
)
