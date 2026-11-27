"""The critic pass measurably improves a rough scene in 3 rounds."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from phasor import Dot, Scene, load
from phasor.inspect.critic import WEIGHTS
from tests.conftest import gl_available

DEMO = Path(__file__).parent.parent / "examples" / "rough_beauty.py"
MODES = [None, pytest.param("auto", marks=[pytest.mark.gl, pytest.mark.skipif(not gl_available(), reason="gl")])]  # fmt: skip


@pytest.mark.parametrize("passes", MODES)
def test_exit_demo_the_critic_pass_improves_a_rough_scene_in_three_rounds(passes):
    e = load(DEMO, passes=passes)
    before = e.critique()
    codes = {i["code"] for i in before["issues"]}
    assert {"focal_point", "margin", "stroke_mix", "motion_rhythm", "density"} <= codes
    assert before["score"] > 5

    report = e.polish(rounds=3)
    assert 1 <= len(report["rounds"]) <= 3
    scores = [before["score"], *(r["score"] for r in report["rounds"])]
    assert all(b < a for a, b in itertools.pairwise(scores))  # each round helps
    assert report["after"] <= 0.55 * before["score"]  # measurably better: at least 45% less

    # what it did is what the report says: one highlight, strokes agree, nothing at the edge
    after = e.critique()
    left = {i["code"] for i in after["issues"]}
    assert not left & {"margin", "stroke_mix"}
    state = e.state_at(6.0)
    highlighted = [o.id for o in state.objects if o.roles.get("color") == "highlight"]
    assert len(highlighted) == 1 and highlighted[0] in ("pole_i", "pole_minus_i")
    assert len({o.roles["stroke"] for o in state.objects if "stroke" in o.roles}) == 1
    # judgement it must leave to the builder is still reported, not papered over
    assert {"density", "motion_rhythm"} <= left


def test_every_round_is_one_undo_step():
    e = load(DEMO, passes=None)
    start = e.critique()["score"]
    e.polish(rounds=3)
    assert e.critique()["score"] < start
    while e.undo():
        pass
    assert e.critique()["score"] == start


def test_a_clean_scene_scores_zero_and_polish_does_nothing():
    s = Scene("clean", passes=None)
    s.set("camera.height", 5)
    s.add(Dot(id="hero", pos=(0, 0), r=1.0, style={"color": "highlight"}, tags=["key"]))
    s.marker("beat", 1.0)
    e = s.engine
    assert e.critique() == {"score": 0.0, "beats": 1, "issues": []}
    r = e.polish()
    assert r["rounds"] == [] and r["before"] == r["after"] == 0.0


def test_the_critic_does_not_invent_a_hero_when_the_story_has_not_started():
    """ "nothing highlighted yet" while something highlighted appears later is reported, but
    no highlight is proposed: a heuristic cannot know the story."""
    s = Scene("late", passes=None)
    s.set("camera.height", 5)
    s.add(Dot(id="early", pos=(-1, 0), r=0.8, style={"color": "secondary"}))
    s.add(Dot(id="also", pos=(0, 1), r=0.8, style={"color": "secondary"}))  # a lone one would do
    s.add(Dot(id="hero", pos=(1, 0), r=0.8, style={"color": "highlight"}))
    s.marker("start", 1.0)
    s.marker("later", 4.0)
    s.at("later").reveal("hero")
    issue = next(i for i in s.engine.critique()["issues"] if i["code"] == "focal_point")
    assert "start" in issue["markers"] and issue["suggested"] == []


def test_weights_cover_every_aesthetic_rule():
    from phasor.inspect import aesthetic

    assert set(WEIGHTS) == {"focal_point", "color_restraint", "glow_clip", "density",
                            "motion_rhythm", "margin", "stroke_mix"}  # fmt: skip
    assert callable(aesthetic.rules)
