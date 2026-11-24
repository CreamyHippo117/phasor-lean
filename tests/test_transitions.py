"""Smooth transitions (examples/prism_transitions.py): a domain coloring whose constant ``a``
and whose function change smoothly, equations rearranging as they are solved, and the
``prism_field`` look. Checked through ``Engine.state_at`` and the inspector."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Equation, Scene, cx
from phasor.core import life
from phasor.domains.basic import crossing, pair_parts
from phasor.style.looks import load_look
from phasor.style.theme import load_theme
from phasor.time.verbs import VerbError
from tests.conftest import gl_available
from tests.test_beauty import _golden

ROOT = Path(__file__).resolve().parents[1]
F = "(z**5 - 1 - a)/(z**2 + a)"
G = "(z**3 - 1 - a)/(z**2 + a)"


def sem(e, i, t=0.0):
    return e.inspect(t, id=i)["objects"][0]["semantic"]


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(str(ROOT / "examples" / "prism_transitions.py"), run_name="x")["s"]


def _points(pts):
    return np.array([complex(x, y) for x, y in pts])


def _moved(a, b) -> float:
    """How far the nearest-matched points of ``a`` moved to ``b`` (the root order of a
    polynomial solver may change between frames)."""
    if len(a) == 0:
        return 0.0
    return max(float(np.min(np.abs(b - z))) for z in a)


# ---------------------------------------------------------------- function blends


def _plane(**kw):
    s = Scene("b", passes=None)
    s.add(cx.DomainColoring(id="plane", function=F, a=(0.0, 0.6), size=(8, 4.5), **kw))
    return sem(s.engine, "plane")


def test_a_blend_ends_on_each_function():
    alone_f, alone_g = _plane(), Scene("g", passes=None)
    alone_g.add(cx.DomainColoring(id="plane", function=G, a=(0.0, 0.6), size=(8, 4.5)))
    alone_g = sem(alone_g.engine, "plane")
    at0, at1 = _plane(to=G, blend=0.0), _plane(to=G, blend=1.0)
    assert at0["zeros"] == alone_f["zeros"] and at0["poles"] == alone_f["poles"]
    assert sorted(at1["zeros"]) == sorted(alone_g["zeros"]) and len(at1["zeros"]) == 3
    assert at0["to"] == G and at1["blend"] == 1.0
    half = _plane(to=G, blend=0.5)
    assert half["poles"] == alone_f["poles"]  # same denominator: the poles stay put
    assert len(half["zeros"]) == 5 and half["zeros"] != alone_f["zeros"]


def test_a_bad_second_function_is_refused_and_an_empty_one_is_none():
    s = Scene("bad", passes=None)
    with pytest.raises(Exception, match="cannot"):
        s.add(cx.DomainColoring(id="p", function="z", to="z +* 2"))
    s.add(cx.DomainColoring(id="q", function="z", to=""))
    assert "to" not in sem(s.engine, "q")


# ---------------------------------------------------------------- smooth tweens


def test_every_transition_is_smooth_frame_to_frame(example):
    """At 60 fps, ``a``, ``blend`` and the zeros and poles they move change a little each
    frame, never in a jump."""
    e = example.engine
    times = np.arange(8.0, 30.0, 1 / 60)
    prev = None
    worst_a = worst_b = worst_pole = 0.0
    for t in times:
        p = sem(e, "plane", float(t))
        cur = (np.array(p["a"]), p.get("blend", 0.0), _points(p["poles"]))
        if prev is not None:
            worst_a = max(worst_a, float(np.abs(cur[0] - prev[0]).max()))
            worst_b = max(worst_b, abs(cur[1] - prev[1]))
            worst_pole = max(worst_pole, _moved(prev[2], cur[2]))
        prev = cur
    assert 0 < worst_a < 0.004  # 0.5 over about six seconds, eased
    assert 0 < worst_b < 0.006
    assert worst_pole < 0.01  # world units per frame: a glide, not a jump


def test_zeros_glide_while_the_function_blends(example):
    """During the blend two zeros run off to infinity and three settle on the cube roots;
    the ones on screen move a little each frame."""
    e = example.engine
    m = e.timeline()["markers"]
    prev = None
    worst = 0.0
    for t in np.arange(m["blend"], m["blend_done"], 1 / 60):
        zeros = _points(sem(e, "plane", float(t))["zeros"])
        inner = zeros[np.abs(zeros) < 1.5]  # far ones leave the frame fast, as they should
        if prev is not None and len(inner) and len(prev):
            worst = max(worst, _moved(inner, prev))
        prev = zeros
    assert 0 < worst < 0.03


def test_the_example_is_deterministic(example):
    e = example.engine
    for t in (6.0, 15.0, 24.0):
        assert e.inspect(t) == e.inspect(t)
    again = runpy.run_path(str(ROOT / "examples" / "prism_transitions.py"), run_name="y")["s"]
    assert json.dumps(again.engine.document.model_dump(mode="json"), sort_keys=True) == \
        json.dumps(e.document.model_dump(mode="json"), sort_keys=True)  # fmt: skip


def test_verbs_give_the_same_document_in_any_order():
    def build(order):
        s = Scene("o", passes=None)
        s.add(cx.DomainColoring(id="plane", function=F, to=G, a=(0.0, 0.6)))
        s.add(Equation(id="one", parts=("$z^2$", r"$+\,a$", "$=$", "$0$"),
                       keys=("z", "a", "eq", "zero")))  # fmt: skip
        s.add(Equation(id="two", parts=("$z^2$", "$=$", "$-a$"), keys=("z", "eq", "a")))
        s.add(Equation(id="three", parts=("$z$", "$=$", r"$\pm\sqrt{-a}$"),
                       keys=("z", "eq", "a")))  # fmt: skip
        for name, t in (("go", 1.0), ("next", 3.0), ("blend", 5.0), ("done", 9.0)):
            s.marker(name, t)
        steps = {
            "t1": lambda: s.at("go").transform("one", "two"),
            "t2": lambda: s.at("next").transform("two", "three"),
            "a": lambda: s.at("go").morph("plane.a", (0.0, 0.7), until="next"),
            "b": lambda: s.at("blend").morph("plane.blend", 1.0, until="done"),
        }
        for k in order:
            steps[k]()
        return json.dumps(s.engine.document.model_dump(mode="json"), sort_keys=True)

    # (a chain of transforms on one equation is written in time order: two's opacity track)
    first = build(["t1", "t2", "a", "b"])
    assert first == build(["b", "a", "t1", "t2"]) == build(["a", "t1", "b", "t2"])


def test_morph_until_runs_to_the_marker_and_refuses_one_before():
    s = Scene("u", passes=None)
    s.add(cx.DomainColoring(id="plane", function=F, a=(0.0, 0.6)))
    s.marker("start", 1.0)
    s.marker("stop", 7.0)
    s.at("start").morph("plane.a", (0.5, 0.6), until="stop")
    e = s.engine
    assert sem(e, "plane", 1.0)["a"] == [0.0, 0.6]
    assert 0.0 < sem(e, "plane", 4.0)["a"][0] < 0.5
    assert sem(e, "plane", 7.0)["a"] == [0.5, 0.6]
    with pytest.raises(VerbError, match="not after"):
        s.at("stop").morph("plane.blend", 1.0, until="start")
    with pytest.raises(VerbError, match="no marker"):
        s.at("start").morph("plane.blend", 1.0, until="nope")


# ---------------------------------------------------------------- equations rearranging


def test_keyed_parts_pair_across_the_equals_sign_and_the_rest_by_text():
    one, two = ("$z^2$", r"$+\,a$", "$=$", "$0$"), ("$z^2$", "$=$", "$-a$")
    pairs = pair_parts(one, two, ("z", "a", "eq", "zero"), ("z", "eq", "a"))
    assert pairs == [(0, 0), (2, 1), (1, 2)]  # "+a" (1) becomes "-a" (2) over the "="
    assert crossing(pairs, one, two) == [False, False, True]  # only "+a" goes round
    # without keys: text only, in order (as before)
    assert pair_parts(one, two) == [(0, 0), (2, 1)]
    assert pair_parts(["x", "=2"], ["x", "+1", "=3"]) == [(0, 0)]
    # a value changing in place: paired, nothing crosses
    vals = pair_parts(["$a$", "$=$", "$0.6$"], ["$a$", "$=$", "$0.7$"], ("a", "eq", "v"),
                      ("a", "eq", "v"))  # fmt: skip
    assert vals == [(0, 0), (1, 1), (2, 2)]
    assert crossing(vals, ["$a$", "$=$", "$0.6$"], ["$a$", "$=$", "$0.7$"]) == [False] * 3


def test_the_solved_equation_reports_what_moves(example):
    e = example.engine
    m = e.timeline()["markers"]
    mid = sem(e, "poles_2", m["rearrange"] + 0.5)
    assert mid["matched"] == 3 and mid["changed"] == 1 and mid["carried_over"] == 1
    root = sem(e, "poles_3", m["root"] + 0.5)
    assert root["matched"] == 3 and root["changed"] == 2 and root["carried_over"] == 0
    value = sem(e, "a_07", m["lift"] + 0.5)
    assert value["matched"] == 3 and value["changed"] == 1  # "0.6 i" -> "0.7 i" in place
    mix = sem(e, "mix_eq", m["blend"] + 0.5)
    assert mix["matched"] == 2 and mix["changed"] == 0  # f(z) = and the fraction glide
    assert "morph" not in sem(e, "g_eq", m["end"])  # done: just the new equation
    assert sem(e, "g_eq", m["end"])["parts"][1].startswith(r"$\dfrac{z^3")


def test_a_carried_part_arcs_and_others_glide_straight():
    """Mid-way the carried part sits ``arc`` line heights off the line; at the ends it is on
    it. The draw is recorded: the carried part leaves the line only mid-way."""

    s = Scene("arc", passes=None)
    s.add(Equation(id="one", parts=("$z^2$", r"$+\,a$", "$=$", "$0$"),
                   keys=("z", "a", "eq", "zero")))  # fmt: skip
    s.add(Equation(id="two", parts=("$z^2$", "$=$", "$-a$"), keys=("z", "eq", "a")))
    s.marker("go", 1.0)
    s.at("go").transform("one", "two")
    e = s.engine
    params = e.state_at(1.5)["two"].params
    assert params.arc == pytest.approx(load_theme("midnight_glow").motion["transform_arc"])
    assert params.from_id == "one" and params.from_keys == ("z", "a", "eq", "zero")

    calls = []

    class Recorder:
        def text(self, text, origin, height, **kw):
            calls.append((text, np.asarray(origin, dtype=float), kw))

    for t, lifted in ((1.0, False), (1.65, True), (2.4, False)):
        calls.clear()
        state = e.state_at(t)
        state["two"].impl.draw(Recorder(), state, "color")
        ys = {text: o[1] for text, o, _ in calls}
        base = ys["$z^2$"]
        carried = [y for text, y in ys.items() if "a" in text]
        assert any(abs(y - base) > 0.1 for y in carried) == lifted, t


# ---------------------------------------------------------------- the prism_field look


def test_prism_field_is_prism_with_calmer_equations_and_a_clearer_field():
    prism, field = load_look("prism"), load_look("prism_field")
    assert field.theme == prism.theme and field.life == prism.life
    assert field.defaults["line"] == prism.defaults["line"]  # everything else is prism
    assert field.defaults["equation"]["params"]["drift"] < 1.0
    assert "drift" not in prism.defaults.get("equation", {}).get("params", {})
    dc_p, dc_f = prism.defaults["domain_coloring"], field.defaults["domain_coloring"]
    assert dc_f["params"]["radius"] < dc_p["params"]["radius"]  # less haze: more see-through
    assert dc_f["params"]["light"] > 1.0 and dc_f["params"]["line_glow"] > 0.6  # more light
    assert dc_f["style"] == dc_p["style"]


def test_equations_drift_by_their_share_of_the_theme_float():
    th = {"float": 0.08, "float_period": 9.0}

    def drift(share):
        v, _ = life.adjust("eq", (), {"pos"}, {"pos": (0.0, 0.0, 0.0), "drift": share}, {},
                           seed=0, amp=1.0, theme_life=th, t=3.0,
                           channels=frozenset({"float"}))  # fmt: skip
        return np.hypot(*v["pos"][:2])

    assert drift(1.0) > 0 and drift(0.0) == 0
    assert drift(0.6) == pytest.approx(0.6 * drift(1.0))


def test_a_transform_takes_life_drift_over_without_a_jump():
    """Mid-transform the drift blends from the source's to the target's: at the start the
    old place is drawn where the source was drifting, at the end the target's own drift."""
    th = {"float": 0.08, "float_period": 9.0}

    def pos(oid, values):
        v, _ = life.adjust(oid, (), set(values), values, {}, seed=0, amp=1.0, theme_life=th,
                           t=3.0, channels=frozenset({"float"}))  # fmt: skip
        return v

    src = pos("one", {"pos": (1.0, 0.0, 0.0)})["pos"]
    start = pos("two", {"pos": (2.0, 0.0, 0.0), "from_pos": (1.0, 0.0, 0.0), "from_id": "one",
                        "morph": 0.0})  # fmt: skip
    assert start["from_pos"] == pytest.approx(src)
    end = pos("two", {"pos": (2.0, 0.0, 0.0), "from_pos": (1.0, 0.0, 0.0), "from_id": "one",
                      "morph": 1.0})  # fmt: skip
    assert end["pos"] == pytest.approx(pos("two", {"pos": (2.0, 0.0, 0.0)})["pos"])


def test_the_example_is_lint_clean_strict(example):
    assert example.engine.lint(strict=True) == []


@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
@pytest.mark.slow
def test_prism_transitions_golden_storyboard(example):
    e = example.engine
    sheet = e.storyboard([2.5, 6.1, 16.0, 25.0], size=(320, 180))["image"]
    _golden("prism_transitions", sheet, e.pass_runner().renderer_class())
