"""An old-style Manim Community scene about room reverb: it runs in phasor unchanged.

This file is written exactly as you would write it for Manim (``from manim import *``,
``class ... (Scene)``, ``construct``, ``self.play``). Run it through phasor instead:

    uv run phasor manim examples/manim_reverb.py -o reverb.phasor.json
    uv run phasor gui reverb.phasor.json

A sound source in a room sends out a wavefront; every wall reflects it; the level at a
listener dies away exponentially, and the time to fall by 60 dB is the reverberation time
RT60 = 0.161 V / A (Sabine).
"""

import math

from manim import *


class RoomReverb(Scene):
    def construct(self):
        room = Rectangle(width=6.0, height=4.0, color=BLUE).shift(LEFT * 3.2)
        source = Dot(point=room.get_center() + LEFT * 1.6, color=YELLOW)
        listener = Dot(point=room.get_center() + RIGHT * 1.8 + DOWN * 0.8, color=GREEN)
        s_label = Text("source", font_size=30).next_to(source, UP, buff=0.2)
        l_label = Text("listener", font_size=30).next_to(listener, DOWN, buff=0.2)

        self.play(Create(room), run_time=1.5)
        self.play(FadeIn(source), FadeIn(listener), run_time=1)
        self.play(Write(s_label), Write(l_label), run_time=1)

        waves = VGroup(*[Circle(radius=0.01, color=YELLOW).move_to(source) for _ in range(3)])
        for k, ring in enumerate(waves):
            self.add(ring)
            self.play(ring.animate.scale(120 * (k + 1)), run_time=1.2, rate_func=linear)
            self.play(FadeOut(ring), run_time=0.3)

        axes = Axes(
            x_range=[0, 3, 1], y_range=[0, 1.2, 0.5], x_length=5.0, y_length=3.0, color=GREY
        ).shift(RIGHT * 3.6 + UP * 0.4)
        decay = axes.plot(lambda x: math.exp(-2.3 * x), x_range=[0, 3], color=RED)
        title = MathTex(r"RT_{60} = 0.161\,\frac{V}{A}", font_size=44).to_edge(UP)
        level = Text("level", font_size=26).next_to(axes, UP, buff=0.15)

        self.play(Create(axes), FadeIn(level), run_time=1.5)
        self.play(Create(decay), run_time=2.0)
        self.play(Write(title))
        self.play(Indicate(listener), run_time=1.0)
        self.wait(1.0)
