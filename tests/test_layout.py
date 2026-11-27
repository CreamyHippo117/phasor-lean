"""Constraint language, stage 1 (kiwisolver), stage 2 (label placer), the bake."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from phasor import Dot, Engine, Line, Scene, Text, expr
from phasor.core import io
from phasor.core.constraints import Guard, Relation, RuleError, parse_rule, rule_ids
from phasor.core.issues import DocumentError
from phasor.core.schema import SCHEMA_VERSION, Document
from phasor.layout import labels, solver
from phasor.layout.bake import key_times
from phasor.layout.geometry import Box, Coverage, footprint, screen_box
from tests.conftest import gl_available

# ---------------------------------------------------------------- the constraint language


def test_parse_the_spec_example():
    (c,) = parse_rule("label(res.pole_i) above res.pole_i gap 0.2")
    assert c == Relation("above", "res.pole_i.label", "res.pole_i", 0.2)


def test_parse_guards_joined_by_and():
    a, b = parse_rule("on_screen(title, 0.2) and no_overlap(a, label(b), c)")
    assert a == Guard("on_screen", ("title",), 0.2)
    assert b == Guard("no_overlap", ("a", "b.label", "c"))
    assert rule_ids("label(p) near p and min_size_px(p, 12)") == ("p.label", "p")


@pytest.mark.parametrize(
    ("rule", "why"),
    [
        ("", "empty"),
        ("a over b", "unknown relation"),
        ("a above", "expected an object id"),
        ("a above a", "against itself"),
        ("a align_x b gap 1", "takes no gap"),
        ("a attach_to b", "needs a spot"),
        ("camera above b", "takes goals"),
        ("camera spin(a)", "unknown camera goal"),
        ("camera min_size(a)", "share of the frame"),
        ("no_overlap(a)", "at least 2"),
        ("min_size_px(a)", "needs a size"),
        ("a above b gap -1", "must not be negative"),
        ("a above b c", "expected and"),
        ("A above b", "cannot read"),
    ],
)
def test_bad_rules_say_why(rule, why):
    with pytest.raises(RuleError, match=why):
        parse_rule(rule)


def test_objects_may_be_named_like_keywords():
    (c,) = parse_rule("gap near label")
    assert c == Relation("near", "gap", "label")


def test_schema_validates_normalizes_and_rejects_duplicate_rules():
    meta = {"title": "t"}
    doc = Document(meta=meta, constraints=[{"rule": "  a   above  b "}])
    assert doc.constraints[0].rule == "a above b" and doc.constraints[0].priority == "strong"
    with pytest.raises(ValueError, match="unknown relation"):
        Document(meta=meta, constraints=[{"rule": "a over b"}])
    with pytest.raises(ValueError, match="duplicate constraint"):
        Document(meta=meta, constraints=[{"rule": "a above b"}, {"rule": "a  above b"}])


def test_version_1_documents_migrate():
    doc = io.parse_document({"schema_version": 1, "meta": {"title": "old"}})
    assert doc.schema_version == SCHEMA_VERSION and doc.constraints == () and doc.retired_ids == ()


# ---------------------------------------------------------------- stage 2: the label placer

FRAME = Box(0.0, 0.0, 16 / 9, 1.0)
ANCHOR = Box.around((0.8, 0.5), 0.1, 0.1)


def _req(id="l", anchor=ANCHOR, leader=0.09, w=0.2, h=0.04):
    return labels.Request(id, w, h, anchor, 0.01, leader)


def _n_box(req=None, share=1.0):
    """A box covering ``share`` of the label's 'n' slot, from its top edge down."""
    req = req or _req()
    (cx, cy) = next(c for s, c, _ in labels.candidates(req) if s == "n")
    return Box(cx - req.w / 2, cy + req.h / 2 - req.h * share, cx + req.w / 2, cy + req.h / 2)


def test_sixteen_candidates_eight_without_a_leader():
    cands = labels.candidates(_req())
    assert [s for s, _, _ in cands[:8]] == ["n", "e", "s", "w", "ne", "se", "sw", "nw"]
    assert len(cands) == 16 and all(lead for _, _, lead in cands[8:])
    assert len(labels.candidates(_req(leader=None))) == 8
    # Slots touch the anchor with exactly the gap between the two boxes.
    n = Box.around(cands[0][1], 0.2, 0.04)
    assert n.y0 == pytest.approx(ANCHOR.y1 + 0.01) and n.center[0] == pytest.approx(0.8)


def test_free_label_takes_the_nearest_slot_and_is_deterministic():
    a = labels.place([_req()], frame=FRAME)
    assert a["l"].slot == "n" and a["l"].blocked == 0 and not a["l"].leader
    assert a == labels.place([_req()], frame=FRAME)


def test_label_avoids_other_labels_objects_and_the_frame_edge():
    assert labels.place([_req()], frame=FRAME, obstacles=[_n_box()])["l"].slot == "s"
    mask = np.zeros((270, 480), dtype=bool)
    mask[:135] = True  # everything above the anchor's middle is covered by some object
    p = labels.place([_req()], frame=FRAME, coverage=Coverage(mask, 16 / 9))["l"]
    assert p.slot == "s" and p.blocked == 0
    top = Box.around((0.8, 0.96), 0.1, 0.06)
    assert labels.place([_req(anchor=top)], frame=FRAME)["l"].slot == "s"


def test_hysteresis_keeps_a_slot_until_another_is_20_percent_cheaper():
    prev = {"l": labels.Previous("n", (0.0, 0.08))}
    # Without history any obstacle on 'n' sends the label to 's'.
    assert labels.place([_req()], frame=FRAME, obstacles=[_n_box(share=0.02)])["l"].slot == "s"
    held = labels.place([_req()], frame=FRAME, obstacles=[_n_box(share=0.02)], previous=prev)
    assert held["l"].slot == "n"
    moved = labels.place([_req()], frame=FRAME, obstacles=[_n_box(share=0.2)], previous=prev)
    assert moved["l"].slot == "s"
    # The rule itself: it moved only because the winner cost under 80% of staying.
    stay = 0.5 * 2.0 + 25 * 0.2
    assert moved["l"].cost < 0.8 * stay


def test_a_wobbling_obstacle_never_makes_the_label_jitter():
    prev: dict[str, labels.Previous] = {}
    slots = []
    for k in range(40):
        share = 0.01 if k % 2 else 0.025
        p = labels.place([_req()], frame=FRAME, obstacles=[_n_box(share=share)], previous=prev)
        prev = {"l": labels.Previous(p["l"].slot, (p["l"].center[0] - 0.8, p["l"].center[1] - 0.5))}
        slots.append(p["l"].slot)
    assert len(set(slots)) == 1


def test_neighbors_make_room_for_each_other():
    a = _req("a", Box.around((0.70, 0.5), 0.04, 0.04))
    b = _req("b", Box.around((0.78, 0.5), 0.04, 0.04))
    c = _req("c", Box.around((0.86, 0.5), 0.04, 0.04))
    out = labels.place([a, b, c], frame=FRAME)
    boxes = [out[r.id].box(r) for r in (a, b, c)]
    assert all(boxes[i].overlap(boxes[j]) == 0 for i in range(3) for j in range(i + 1, 3))
    assert all(p.blocked == 0 for p in out.values())
    assert all(bx.overlap(r.anchor) == 0 for bx in boxes for r in (a, b, c))


def test_leader_slot_only_when_every_compass_slot_is_covered():
    ring = Box.around(ANCHOR.center, 0.1 + 2 * 0.25, 0.1 + 2 * 0.06)  # hides all 8 near slots
    p = labels.place([_req()], frame=FRAME, obstacles=[ring])["l"]
    assert p.leader and p.slot.endswith("+leader") and p.blocked < 0.5
    # An object that cannot draw a leader stays near and reports how blocked it is.
    q = labels.place([_req(leader=None)], frame=FRAME, obstacles=[ring])["l"]
    assert not q.leader and q.blocked > 0.5


def test_movement_cost_prefers_staying_close_to_the_previous_offset():
    square = _req(w=0.04)  # for a square label on a square anchor all four sides tie
    assert labels.place([square], frame=FRAME)["l"].slot == "n"
    prev = {"l": labels.Previous("gone", (0.08, 0.0))}  # it used to sit to the east
    assert labels.place([square], frame=FRAME, previous=prev)["l"].slot == "e"


def test_coverage_counts_pixels_inside_a_screen_box():
    mask = np.zeros((90, 160), dtype=bool)
    mask[:45, :80] = True  # top-left quarter
    cov = Coverage(mask, 16 / 9)
    assert cov.fraction(Box(0.0, 0.5, 8 / 9, 1.0)) == pytest.approx(1.0)
    assert cov.fraction(Box(8 / 9, 0.0, 16 / 9, 0.5)) == 0.0
    assert cov.fraction(Box(0.0, 0.0, 16 / 9, 1.0)) == pytest.approx(0.25)
    assert cov.fraction(Box(-1.0, 0.5, 0.0, 1.0)) == 0.0  # outside the frame: nothing there


# ---------------------------------------------------------------- stage 1: kiwisolver

A, B, C = (
    Box.around((0.2, 0.2), 0.2, 0.1),
    Box.around((1.0, 0.5), 0.1, 0.1),
    Box(1.4, 0.1, 1.5, 0.3),
)
BOXES = {"a": A, "b": B, "c": C}


def _solve(*rules, movable=("a",), boxes=BOXES):
    items = []
    for i, (text, priority, value) in enumerate(rules):
        (clause,) = parse_rule(text)
        items.append(solver.Item(i, priority, clause, value))
    return solver.solve(items, boxes, frozenset(movable), FRAME)


def test_above_puts_the_gap_between_the_boxes_and_centers():
    centers, unsolved = _solve(("a above b", "strong", 0.05))
    assert unsolved == []
    assert centers["a"] == pytest.approx((1.0, 0.5 + 0.05 + 0.05 + 0.05))


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("a below b", (1.0, 0.5 - 0.05 - 0.02 - 0.05)),
        ("a left_of b", (1.0 - 0.05 - 0.02 - 0.1, 0.5)),
        ("a right_of b", (1.0 + 0.05 + 0.02 + 0.1, 0.5)),
        ("a align_x b", (1.0, 0.2)),  # only x changes; y stays where the document put it
        ("a align_y b", (0.2, 0.5)),
    ],
)
def test_each_relation(rule, expected):
    centers, unsolved = _solve((rule, "strong", 0.02))
    assert unsolved == [] and centers["a"] == pytest.approx(expected)


def test_explicit_align_overrides_the_implied_centering():
    centers, unsolved = _solve(("a above b", "strong", 0.0), ("a align_x c", "strong", None))
    assert unsolved == [] and centers["a"] == pytest.approx((1.45, 0.6))


def test_chains_solve_together():
    centers, unsolved = _solve(
        ("b right_of c", "strong", 0.1), ("a right_of b", "strong", 0.1), movable=("a", "b")
    )
    assert unsolved == []
    assert centers["b"] == pytest.approx((1.5 + 0.1 + 0.05, 0.2))
    assert centers["a"] == pytest.approx((1.65 + 0.05 + 0.1 + 0.1, 0.2))


def test_inside_and_on_screen_move_only_as_far_as_needed():
    big = {"a": Box.around((0.2, 0.2), 0.1, 0.1), "b": Box(0.5, 0.4, 1.0, 0.9)}
    centers, unsolved = _solve(("a inside b", "strong", None), boxes=big)
    assert unsolved == [] and centers["a"] == pytest.approx((0.55, 0.45))
    out = {"a": Box.around((1.9, 1.2), 0.2, 0.1)}
    centers, unsolved = _solve(("on_screen(a)", "strong", 0.05), boxes=out)
    assert unsolved == []
    assert centers["a"] == pytest.approx((16 / 9 - 0.05 - 0.1, 1 - 0.05 - 0.05))


def test_conflicts_are_reported_never_dropped():
    centers, unsolved = _solve(("a above b", "strong", 0.0), ("a below b", "strong", 0.0))
    assert len(unsolved) >= 1 and all("off by" in u.message for u in unsolved)
    # The higher priority wins outright and the loser is the one reported.
    centers, unsolved = _solve(("a above b", "required", 0.0), ("a below b", "weak", 0.0))
    assert [u.index for u in unsolved] == [1]
    assert centers["a"][1] == pytest.approx(0.6)
    _, unsolved = _solve(("a above b", "required", 0.0), ("a below b", "required", 0.0))
    assert [u.index for u in unsolved] == [1] and "required" in unsolved[0].message


# ---------------------------------------------------------------- the bake, through Engine


def _scene(**kw):
    s = Scene("layout", **kw)
    s.set("camera.height", 8)
    s.add(Dot(id="dot", pos=(-3, 0), r=0.25, tags=["key"]))
    s.add(Text(id="dot.label", text="pole"))
    s.at(0.5).to("dot.pos", (3, 1), duration=2.0)
    return s


def _boxes(engine, t):
    state = engine.state_at(t)
    return screen_box(state, state["dot.label"]), screen_box(state, state["dot"])


def test_label_stays_glued_to_a_moving_anchor_between_key_times():
    s = _scene(passes=None)
    s.constrain("label(dot) above dot gap 0.2")
    e = s.engine
    for t in np.linspace(0, e.duration, 53):  # mostly not key times
        label, dot = _boxes(e, float(t))
        assert label.y0 - dot.y1 == pytest.approx(0.2 / 8, abs=1e-9)
        assert label.center[0] == pytest.approx(dot.center[0], abs=1e-9)
    assert e.lint() == []


def test_the_bake_is_pure_and_independent_of_query_order():
    def build():
        s = _scene(passes=None)
        s.constrain("label(dot) near dot")
        return s.engine

    a, b = build(), build()
    times = [2.5, 0.0, 1.2345, 0.7]
    forward = [a.state_at(t)["dot.label"].params for t in times]
    backward = [b.state_at(t)["dot.label"].params for t in reversed(times)][::-1]
    assert forward == backward
    assert a.layout.times == key_times(a._snap.compiled) and 0.5 in a.layout.times


def test_the_gap_holds_on_screen_while_the_camera_zooms():
    s = _scene(passes=None)
    s.constrain("label(dot) above dot")  # default gap comes from the theme, in screen heights
    s.at(0).to("camera.height", 3, duration=2.5)
    for t in (0.0, 0.33, 1.1, 2.5):
        label, dot = _boxes(s.engine, t)
        # Exact at key times; between them the dot's on-screen size is interpolated, which
        # is good to well under a tenth of a pixel at 1080p.
        exact = t in s.engine.layout.times
        assert label.y0 - dot.y1 == pytest.approx(0.012, abs=1e-9 if exact else 1e-4)


def test_expressions_read_solved_positions():
    s = _scene(passes=None)
    s.constrain("label(dot) above dot gap 0.2")
    s.add(Dot(id="echo", pos=expr("dot.label.pos + vec(0, 1, 0)"), r=0.1))
    state = s.engine.state_at(1.3)
    assert state["echo"].params.pos[1] == pytest.approx(state["dot.label"].params.pos[1] + 1)
    assert state["dot.label"].params.pos != (0.0, 0.0, 0.0)


def test_rules_that_cannot_work_are_rejected_with_a_reason():
    s = _scene(passes=None)
    s.add(Line(id="axis", points=[(-5, 0), (5, 0)]))
    s.add(Text(id="tied", text="x", pos=expr("dot.pos")))
    cases = {
        "axis above dot": "not_movable",
        "tied above dot": "not_movable",
        "label(ghost) near ghost": "unknown_id",
    }
    for rule, code in cases.items():
        r = s.engine.apply({"op": "add_constraint", "rule": rule}, origin="script")
        assert not r.ok and r.errors[0].code == code and r.errors[0].fix_hint, rule
    r = s.engine.apply({"op": "add_constraint", "rule": "dot upon axis"}, origin="script")
    assert not r.ok and r.errors[0].code == "bad_constraint"
    s.constrain("label(dot) near dot")
    with pytest.raises(DocumentError, match="conflicting_constraints"):
        s.constrain("label(dot) above dot")
    with pytest.raises(DocumentError, match="duplicate_constraint"):
        s.constrain("label(dot)  near dot")
    assert len(s.engine.document.constraints) == 1  # nothing half-applied


def test_unsatisfiable_rules_become_lint_errors_with_times():
    s = _scene(passes=None)
    s.constrain("label(dot) above dot")
    r = s.engine.apply({"op": "add_constraint", "rule": "label(dot) below dot"}, origin="script")
    assert r.ok
    unsolved = [i for i in s.engine.lint() if i.code == "unsolved"]
    assert unsolved and unsolved[0].severity == "error" and unsolved[0].times
    assert unsolved[0].path.startswith("constraints[") and "dot.label" in unsolved[0].ids
    assert [i for i in r.lint_added if i.code == "unsolved"] == unsolved
    back = s.engine.apply({"op": "remove_constraint", "rule": "label(dot) below dot"},
                          origin="script")  # fmt: skip
    assert back.ok and set(unsolved) <= set(back.lint_resolved)
    assert s.engine.lint() == []


def test_near_keeps_the_label_clear_and_does_not_jitter():
    s = _scene(passes=None)
    s.add(Line(id="wall", points=[(0, -4), (0, 4)], style={"stroke": "bold"}))
    s.add(Dot(id="rock", pos=(1.5, 1.2), r=0.3))
    s.constrain("label(dot) near dot")
    e = s.engine
    slots = e.layout.slots["dot.label"]
    changes = sum(a != b for a, b in itertools.pairwise(slots))
    assert 1 <= changes <= 6, slots  # it has to cross the wall, but it must not flicker
    assert [i for i in e.lint() if i.code == "unsolved"] == []
    size = (480, 270)
    for t in e.layout.times:
        state = e.state_at(t)
        label = footprint(state, state["dot.label"], size)
        for other in ("dot", "rock"):
            assert not (label & footprint(state, state[other], size)).any(), (t, other)


def test_leader_line_is_switched_on_when_the_label_has_to_stand_off():
    s = Scene("crowd", passes=None)
    s.set("camera.height", 8)
    s.add(Dot(id="hub", pos=(0, 0), r=0.1))
    s.add(Text(id="hub.label", text="hub"))
    for k in range(8):  # a ring of discs right where the eight near slots are
        a = k * np.pi / 4
        s.add(Dot(id=f"spoke_{'abcdefgh'[k]}", pos=(0.55 * np.cos(a), 0.4 * np.sin(a)), r=0.2))
    s.constrain("label(hub) near hub")
    p = s.engine.state_at(0)["hub.label"].params
    assert s.engine.layout.slots["hub.label"][0].endswith("+leader")
    assert p.leader_on == 1.0 and np.hypot(*p.leader[:2]) <= 0.1 * np.sqrt(2) + 1e-9


def test_constraints_on_a_scripted_scene_live_in_the_overrides_file(tmp_path):
    script = tmp_path / "demo.py"
    script.write_text(
        "from phasor import Dot, Scene, Text\n"
        "s = Scene('demo')\n"
        "s.add(Dot(id='dot', pos=(1, 1)))\n"
        "s.add(Text(id='dot.label', text='p'))\n",
        encoding="utf-8",
    )
    from phasor import load

    e = load(script, passes=None)
    assert e.apply({"op": "add_constraint", "rule": "label(dot) below dot gap 0.1"}).ok
    assert e.base.constraints == () and len(e.document.constraints) == 1
    e.save(tmp_path / "demo.phasor.json")
    again = load(script, passes=None)
    assert again.overrides.constraints[0].rule == "label(dot) below dot gap 0.1"
    assert again.state_at(0)["dot.label"].params.pos[1] < 1.0
    assert again.apply({"op": "remove_constraint", "rule": "label(dot) below dot gap 0.1"}).ok
    assert again.overrides.is_empty()
    # If the script drops the object, the rule is reported, not silently lost.
    script.write_text("from phasor import Scene\ns = Scene('demo')\n", encoding="utf-8")
    (issue,) = load(script, passes=None).lint()
    assert issue.code == "orphan_override" and "dot" in issue.ids


def test_undo_brings_the_previous_layout_back():
    s = _scene(passes=None)
    before = s.engine.state_at(1.0)["dot.label"].params.pos
    s.constrain("label(dot) above dot")
    assert s.engine.state_at(1.0)["dot.label"].params.pos != before
    s.engine.undo()
    assert s.engine.state_at(1.0)["dot.label"].params.pos == before


@pytest.mark.gl
def test_the_placer_sees_real_pixels_through_the_id_pass():
    """A long diagonal line covers a thin stripe, not its bounding box: with OpenGL the
    oracle is the id pass, and the label may sit inside the line's box."""
    s = Scene("stripe")
    s.set("camera.height", 8)
    s.add(Line(id="ramp", points=[(-6, -3.5), (6, 3.5)]))
    s.add(Dot(id="dot", pos=(2, -1.5), r=0.2))
    s.add(Text(id="dot.label", text="inside the box"))
    s.constrain("label(dot) near dot")
    assert s.engine.pass_runner() is not None and s.engine.layout.occlusion == "pixel"
    assert s.engine.layout.slots["dot.label"] == ("n",) and s.engine.lint() == []


# ---------------------------------------------------------------- outside, tangent_to, attach_to (M8)


def test_outside_leaves_by_the_nearest_side_and_moves_only_as_far_as_needed():
    overlapped = {
        "a": Box.around((1.12, 0.5), 0.2, 0.1),
        "b": B,
    }  # a sits on b, off to the right
    centers, unsolved = _solve(("a outside b", "strong", 0.02), boxes=overlapped)
    assert unsolved == []
    assert centers["a"][0] == pytest.approx(1.0 + 0.05 + 0.02 + 0.1)  # out to the right ...
    assert centers["a"][1] == pytest.approx(0.5)  # ... without moving in y
    away = {"a": Box.around((0.2, 0.2), 0.2, 0.1), "b": B}
    centers, _ = _solve(("a outside b", "strong", 0.02), boxes=away)
    assert centers["a"] == pytest.approx((0.2, 0.2))  # already outside: it stays put
    above = {"a": Box.around((1.0, 0.53), 0.2, 0.1), "b": B}
    centers, _ = _solve(("a outside b", "strong", 0.0), boxes=above)
    assert centers["a"][1] == pytest.approx(0.5 + 0.05 + 0.05)  # nearest exit is straight up


def test_tangent_to_touches_the_outside_edge_and_lines_up():
    near = {"a": Box.around((0.7, 0.55), 0.2, 0.1), "b": B}  # left of b, nearly touching
    centers, unsolved = _solve(("a tangent_to b", "strong", None), boxes=near)
    assert unsolved == []
    assert centers["a"] == pytest.approx((1.0 - 0.05 - 0.1, 0.5))  # edge to edge, centered on b


def test_attach_to_puts_the_object_at_a_named_spot_of_the_target():
    (clause,) = parse_rule("a attach_to b at top")
    assert clause == Relation("attach_to", "a", "b", None, "top")
    anchors = {("b", "top"): (1.0, 0.55), ("b", "right"): (1.05, 0.5), ("b", "tip"): (1.3, 0.8)}
    for rule, want in (
        ("a attach_to b at top", (1.0, 0.55 + 0.05 + 0.03)),  # sits on top of the spot
        ("a attach_to b at right", (1.05 + 0.1 + 0.03, 0.5)),
        ("a attach_to b at tip", (1.3 + 0.1 + 0.03, 0.8 + 0.05 + 0.03)),  # up and to the right
    ):
        (cl,) = parse_rule(rule)
        centers, unsolved = solver.solve([solver.Item(0, "strong", cl, 0.03)], BOXES,
                                         frozenset({"a"}), FRAME, anchors)  # fmt: skip
        assert unsolved == [] and centers["a"] == pytest.approx(want), rule


def test_the_new_relations_through_the_engine_in_both_modes():
    for passes in (None, "auto") if gl_available() else (None,):
        s = Scene("rel", passes=passes)
        s.set("camera.height", 6)
        s.add(Dot(id="hub", pos=(0, 0), r=0.5, tags=["key"]))
        s.add(Dot(id="moon", pos=(0.3, 0.1), r=0.2))
        s.add(Text(id="cap", text="north", pos=(3, 2)))
        s.add(Dot(id="rim", pos=(-3, -2), r=0.2))
        s.constrain("moon outside hub gap 0.1")
        s.constrain("rim tangent_to hub")
        s.constrain("cap attach_to hub at top")
        e = s.engine
        assert [i for i in e.lint() if i.code == "unsolved"] == []
        st = e.state_at(0.0)
        hub = screen_box(st, st["hub"])
        moon, rim, cap = (screen_box(st, st[i]) for i in ("moon", "rim", "cap"))
        assert moon.overlap(hub) == 0.0  # outside
        assert rim.x1 == pytest.approx(hub.x0, abs=1e-4) or rim.y1 == pytest.approx(
            hub.y0, abs=1e-4
        )
        assert cap.y0 == pytest.approx(hub.y1 + 0.012, abs=1e-3)  # on top of the hub, theme gap


def test_attach_to_an_unknown_spot_is_reported_with_the_choices():
    s = Scene("anch", passes=None)
    s.add(Dot(id="hub", pos=(0, 0)))
    s.add(Dot(id="x", pos=(2, 0)))
    s.constrain("x attach_to hub at roof")
    (issue,) = [i for i in s.engine.lint() if i.code == "unsolved"]
    assert "unknown spot 'roof'" in issue.message and "top" in issue.message


def test_attach_to_uses_an_objects_own_feature_points():
    from phasor import linalg

    s = Scene("tip", passes=None)
    s.set("camera.height", 8)
    s.add(linalg.Arrow(id="v", start=(0, 0, 0), end=(2, 1, 0)))
    s.add(Text(id="v.label", text="v"))
    s.constrain("v.label attach_to v at tip")
    e = s.engine
    assert [i for i in e.lint() if i.code == "unsolved"] == []
    st = e.state_at(0.0)
    lab, arrow = screen_box(st, st["v.label"]), screen_box(st, st["v"])
    assert lab.x0 >= arrow.x1 - 0.2 and lab.y0 >= arrow.y1 - 0.2  # beyond the tip, up and right


# ---------------------------------------------------------------- what the bake runs at (wishlist)
# A label near an object shown for a few seconds of a long 3D scene used to make the bake
# render an occlusion buffer at every key time of the whole scene, and every verb in the
# script baked it all again.

MODES = [None, pytest.param("auto", marks=pytest.mark.gl)]


def _brief(passes, duration=20.0, show=8.0, hide=11.0, more=()):
    from phasor import cx

    s = Scene("brief", passes=passes)
    s.use_camera("orbit", distance=9, elevation=25, azimuth=-90, drift=0)
    s.add(cx.RiemannSurface(id="surf", radius=2.2))
    s.add(Dot(id="x", pos=(0, 0, 0), r=0.07, tags=["key"]))
    s.add(Text(id="x.label", text="branch point"))
    for obj in more:
        s.add(obj)
    s.marker("show", show)
    s.marker("hide", hide)
    s.marker("end", duration)
    s.constrain("label(x) near x")
    s.at("show").reveal("x", "x.label")
    s.at(show + 1).to("x.pos", (0.6, 0.4, 0.2), duration=1.5)  # it moves while shown
    s.at("hide").exit("x", "x.label")
    return s


class _Counting:
    """Counts what the layout asks of the occlusion oracle."""

    def __init__(self, monkeypatch):
        from phasor.inspect.oracle import PassOracle
        from phasor.layout.bake import AnalyticOracle

        self.n = 0
        for cls in (AnalyticOracle, PassOracle):
            for name in ("depth", "coverage"):
                orig = getattr(cls, name)
                monkeypatch.setattr(cls, name, self._wrap(orig))

    def _wrap(self, orig):
        def counted(oracle, *a, **k):
            self.n += 1
            return orig(oracle, *a, **k)

        return counted


def _never_skip(monkeypatch):
    from phasor.layout import bake as bake_mod

    monkeypatch.setattr(bake_mod, "shown_keys", lambda c, plan, times: (True,) * len(times))


@pytest.mark.parametrize("passes", MODES)
def test_the_bake_runs_only_while_something_it_places_is_shown(passes, monkeypatch):
    built = _brief(passes).engine
    counting = _Counting(monkeypatch)
    e = Engine(built.document, passes=built._passes)  # baked from nothing
    lay = e.layout
    times = key_times(e._snap.compiled)
    assert len(times) > 200
    # The shown span (the label's fade starts at 8.12 s, after the dot's), to the end of
    # the exit fade, plus the key either side of it.
    assert lay.times[0] == pytest.approx(8.12) and lay.times[-1] < 12.0
    assert lay.spans == ((lay.times[0], lay.times[-1]),)
    assert lay.renders == len(lay.times) < 45
    assert counting.n == lay.renders  # one occlusion buffer per baked key, shared by labels
    # Hidden, the label simply holds its first or last place.
    assert e.state_at(2.0)["x.label"].params.pos == e.state_at(lay.times[0])["x.label"].params.pos
    assert e.state_at(15)["x.label"].params.pos == e.state_at(lay.times[-1])["x.label"].params.pos


@pytest.mark.parametrize("passes", MODES)
def test_placements_while_shown_are_those_a_bake_of_every_key_gives(passes, monkeypatch):
    e = _brief(passes).engine
    lay = e.layout
    shown = [e.state_at(t)["x.label"].params for t in lay.times]
    _never_skip(monkeypatch)
    ref = _brief(passes).engine
    assert len(ref.layout.times) == len(key_times(ref._snap.compiled)) > len(lay.times)
    assert [ref.state_at(t)["x.label"].params for t in lay.times] == shown
    for t in lay.times:
        k = ref.layout.times.index(t)
        assert ref.layout.slots["x.label"][k] == lay.slots["x.label"][lay.times.index(t)]


def test_a_label_read_by_an_expression_is_baked_everywhere():
    s = _brief(None, more=[Dot(id="echo", pos=expr("x.label.pos + vec(0, 0, 1)"), r=0.05)])
    lay = s.engine.layout
    assert lay.times == key_times(s.engine._snap.compiled)


def test_an_opacity_written_as_an_expression_is_baked_everywhere():
    s = _brief(None)
    s.set("x.label.dim", expr("1 - 0.5 * sin(t) ** 2"))
    lay = s.engine.layout
    assert lay.times == key_times(s.engine._snap.compiled)


def test_a_scene_whose_labels_never_show_bakes_nothing():
    s = Scene("never", passes=None)
    s.add(Dot(id="x", pos=(0, 0)))
    s.add(Text(id="x.label", text="x", opacity=0.0))
    s.marker("end", 30)
    s.constrain("label(x) near x")
    e = s.engine
    assert e.layout.times == () and e.layout.renders == 0
    assert e.state_at(3)["x.label"].params.pos == (0.0, 0.0, 0.0)  # the document's own


@pytest.mark.parametrize("passes", MODES)
def test_an_edit_reuses_the_bake_before_the_first_time_it_changes(passes, monkeypatch):
    s = _brief(passes, duration=30.0, show=8.0, hide=20.0)
    e = s.engine
    first = e.layout
    counting = _Counting(monkeypatch)
    s.at(15).morph("surf.radius", 2.0)  # the verb itself reads the scene at t = 15
    again = e.layout
    late = [t for t in again.times if t >= 15]
    assert counting.n == len(late) < len(again.times)
    assert [t for t in again.times if t < 15] == [t for t in first.times if t < 15]
    fresh = Engine(e.document, passes=e._passes)  # the same document, baked from nothing
    assert fresh.layout == again
    assert fresh.layout.renders == again.renders


def test_an_edit_before_the_shown_span_bakes_it_again_and_gets_the_same(monkeypatch):
    s = _brief(None, duration=30.0, show=8.0, hide=20.0)
    e = s.engine
    assert e.layout.times
    counting = _Counting(monkeypatch)
    s.at(3).morph("surf.radius", 2.0)
    lay = e.layout
    assert counting.n == len(lay.times) == lay.renders
    assert Engine(e.document, passes=None).layout == e.layout


def test_reuse_is_refused_when_anything_but_a_later_motion_changed():
    from phasor.layout.bake import unchanged_until

    s = _brief(None)
    c0 = s.engine._snap.compiled
    s.at(12).morph("surf.radius", 2.0)
    c1 = s.engine._snap.compiled
    assert unchanged_until(c0, c1) == pytest.approx(12.0)
    s.set("surf.radius", 2.5)  # a base value: every time changes
    assert unchanged_until(c1, s.engine._snap.compiled) == 0.0
    assert unchanged_until(c1, c1) == float("inf")


def test_a_verb_bakes_only_as_far_as_the_time_it_starts(monkeypatch):
    """A script that writes the rule first and its verbs after: each verb reads the scene
    where it starts, so it bakes up to there, and the next one carries on from it."""
    s = Scene("walk", passes=None)
    s.set("camera.height", 8)
    s.add(Dot(id="x", pos=(0, 0), r=0.2))
    s.add(Text(id="x.label", text="x"))
    s.add(Dot(id="y", pos=(2, 1), r=0.2))
    s.marker("end", 60)
    s.constrain("label(x) near x")  # shown from the start
    counting = _Counting(monkeypatch)
    s.at(5).morph("y.r", 0.4)
    assert counting.n < 60  # about 5 s of keys, not the scene's 600
    s.at(10).morph("y.pos", (1, 1, 0))
    assert counting.n < 120  # carried on from 5 s
    e = s.engine
    partial = [e._state_upto(t) for t in (0.0, 4.95, 9.0, 9.5)]
    assert [e.state_at(t) for t in (0.0, 4.95, 9.0, 9.5)] == partial
    assert e.layout == Engine(e.document, passes=None).layout


# ---------------------------------------------------------------- the camera solver's depth checks
# The solver checks avoid_occlusion on a depth buffer at every key; the refine rounds and
# the final check asked again about cameras already measured, so a long orbit rendered
# about two buffers per key. Each (time, camera, object) is now measured once, and no
# buffer is drawn when the answer needs none.


def _old_hidden_share(state, object_id, oracle):
    """``hidden_share`` as it was before (commit 70476b7): it always rendered first."""
    from phasor.layout.bake import DEPTH_SLACK

    obj = state[object_id]
    cam = state.camera_for(obj)
    b = obj.impl.world_bounds(state)
    labels_ = {o.id for o in state.objects if o.impl.semantics(state).get("kind") == "label"}
    depth = oracle.depth(state, frozenset({object_id} | labels_))
    h, w = depth.shape
    d = float(cam.depth(b.center)[0])
    if d <= 0:
        return 1.0
    ((px, py),) = cam.to_screen([b.center], w, h)
    r = max(0.5 * b.radius / (cam.units_per_height(b.center)) * h, 1.5)
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    disc = (xx - px) ** 2 + (yy - py) ** 2 <= r * r
    if not disc.any():
        return 0.0
    return float((depth[disc] < d * (1 - 2 * DEPTH_SLACK)).mean())


def _measure_every_time(monkeypatch):
    """The solver as it was: every check renders again."""
    from phasor.layout import bake as bake_mod
    from phasor.layout import camera
    from phasor.time.compile import evaluate

    def uncached(seen, t, p, object_id):
        state = evaluate(seen.c, t, {path: getattr(p, n) for path, n in seen.paths})
        return seen.hidden(state, object_id)

    monkeypatch.setattr(camera._Seen, "__call__", uncached)
    monkeypatch.setattr(bake_mod, "hidden_share", _old_hidden_share)


def _peek(passes, duration=6.0):
    s = Scene("peek_orbit", passes=passes)
    s.use_camera("orbit", drift=0, azimuth=0, target=(0, 0, 0))
    s.add(Dot(id="wall", pos=(2.5, 0, 0.3), r=0.5))  # a disc between the camera and the gem
    s.add(Dot(id="post", pos=(-1.5, 1.5, 0.2), r=0.5))
    s.add(Dot(id="gem", pos=(0, 0, 0), r=0.25, tags=["key"]))
    s.marker("spin", 0.5)
    s.marker("gone", 3.0)
    s.marker("back", 4.5)
    s.marker("end", duration)
    s.at("spin").orbit(200, until="end")
    s.at("gone").exit("wall", "post")  # for a while nothing can hide the gem
    s.at("back").reveal("wall", "post")
    s.constrain("camera keep_visible(gem) and camera avoid_occlusion(gem)")
    return s


@pytest.mark.parametrize("passes", MODES)
def test_camera_goals_measure_each_camera_once_and_solve_the_same(passes, monkeypatch):
    built = _peek(passes).engine
    counting = _Counting(monkeypatch)
    lay = Engine(built.document, passes=built._passes).layout
    assert counting.n == lay.renders
    _measure_every_time(monkeypatch)
    before = Engine(built.document, passes=built._passes).layout
    assert set(lay.camera) >= {"camera.elevation", "camera.distance"}
    assert lay.camera == before.camera and lay.camera_times == before.camera_times
    assert lay.issues == before.issues and lay.tracks == before.tracks
    # The same cameras, each measured once: about half the buffers (analytic: 274 -> 138).
    assert lay.renders < 0.6 * before.renders


def test_a_hidden_share_that_needs_no_buffer_draws_none(monkeypatch):
    from phasor.layout.bake import AnalyticOracle, hidden_share

    e = _peek(None).engine
    states = [e.state_at(t) for t in (1.0, 3.7, 5.5)]  # walls shown, gone, shown again
    oracle = AnalyticOracle()
    counting = _Counting(monkeypatch)
    old = [_old_hidden_share(st, "gem", oracle) for st in states]
    assert counting.n == 3
    assert [hidden_share(st, "gem", oracle) for st in states] == old
    assert counting.n == 3 + 2  # nothing occludes at 3.7 s: no buffer drawn


def test_pixel_mask_gives_what_testing_every_pixel_gives():
    from phasor.layout.geometry import pixel_mask

    rng = np.random.default_rng(3)
    h, w = 90, 160
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    cases = [(80.0, 45.0, 3.0), (-5.0, 20.0, 4.0), (300.0, 40.0, 500.0), (10.0, 10.0, 1e-9),
             (159.99, 89.99, 1.5), (1e9, -1e9, 1e9), (0.5, 0.5, 0.5)]  # fmt: skip
    cases += [(*rng.uniform(-50, 200, 2), float(rng.uniform(0.5, 80))) for _ in range(40)]
    for px, py, r in cases:

        def disc(xx, yy, px=px, py=py, r=r):
            return (xx - px) ** 2 + (yy - py) ** 2 <= r * r

        got = pixel_mask((h, w), (px - r, px + r, py - r, py + r), disc)
        assert np.array_equal(got, disc(xx, yy)), (px, py, r)
    unknown = pixel_mask((h, w), (np.nan, 1.0, 0.0, 1.0), lambda xx, yy: xx < 3)
    assert np.array_equal(unknown, xx < 3)  # no reach known: every pixel is tested


def test_a_point_footprint_is_the_disc_inside_its_box():
    s = Scene("fp", passes=None)
    s.add(Dot(id="d", pos=(0.7, -0.4), r=0.6))
    st = s.engine.state_at(0)
    obj = st["d"]
    w, h = 480, 270
    px = st.camera_for(obj).to_screen(obj.impl.world_bounds(st).corners(), w, h)
    (x0, y0), (x1, y1) = px.min(0), px.max(0)
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    rx, ry = max((x1 - x0) / 2, 1e-9), max((y1 - y0) / 2, 1e-9)
    disc = ((xx - (x0 + x1) / 2) / rx) ** 2 + ((yy - (y0 + y1) / 2) / ry) ** 2 <= 1.0
    assert disc.any() and np.array_equal(footprint(st, obj, (w, h)), disc)


# ---------------------------------------------------------------- edits the layout cannot see
# A parameter of an object that is hidden (opacity x dim = 0) at a key, that no rule names
# and nothing reads, cannot change the layout at that key: the bake keeps every key before
# the first one at which the object is shown.


def _late(passes, more=()):
    s = _brief(passes, duration=30.0, show=8.0, hide=25.0,
               more=[Dot(id="late", pos=(0.4, 0.3, 0.1), r=0.3), *more])  # fmt: skip
    s.at(20).reveal("late")
    return s


def _first_shown(e, object_id):
    return next(
        t for t in key_times(e._snap.compiled) if e.state_at(t)[object_id].params.opacity > 0
    )


@pytest.mark.parametrize("passes", MODES)
def test_an_edit_to_an_object_hidden_until_later_keeps_the_bake_before_it(passes, monkeypatch):
    s = _late(passes)
    e = s.engine
    first = e.layout
    shown = _first_shown(e, "late")
    assert 20.0 <= shown < 20.2
    counting = _Counting(monkeypatch)
    for name, value in (("r", 0.6), ("pos", (0.1, 0.2, 0.3))):
        counting.n = 0
        s.set(f"late.{name}", value)
        again = e.layout
        assert 0 < counting.n == len([t for t in again.times if t >= shown]) < len(again.times)
        assert [t for t in again.times if t < shown] == [t for t in first.times if t < shown]
        fresh = Engine(e.document, passes=e._passes)  # the same document, baked from nothing
        assert fresh.layout == again


def _until(s, edit):
    from phasor.layout.bake import layout_unchanged_until

    c0 = s.engine._snap.compiled
    edit(s)
    snap = s.engine._snap
    return layout_unchanged_until(c0, snap.compiled, snap.plan, key_times(snap.compiled))


def test_which_edits_the_layout_cannot_see():
    shown = _first_shown(_late(None).engine, "late")
    assert _until(_late(None), lambda s: s.set("late.r", 0.5)) == shown
    assert _until(_late(None), lambda s: s.at(22).morph("late.r", 0.5)) == shown
    assert _until(_late(None), lambda s: s.at(12).morph("x.r", 0.1)) == pytest.approx(12.0)
    # Shown from the start, so every key may see it.
    assert _until(_late(None), lambda s: s.set("surf.radius", 2.5)) == 0.0
    # What decides whether, and in what order, it draws.
    for name, value in (("opacity", 0.5), ("dim", 0.5), ("z_index", 3)):
        assert _until(_late(None), lambda s, n=name, v=value: s.set(f"late.{n}", v)) == 0.0
    # Read by another object's expression, named by another object, or named by a rule.
    echo = Dot(id="echo", pos=expr("late.pos + vec(0, 0, 1)"), r=0.05)
    assert _until(_late(None, more=[echo]), lambda s: s.set("late.r", 0.5)) == 0.0
    ringed = _late(None)
    ringed.at(26).circumscribe("late")
    assert _until(ringed, lambda s: s.set("late.r", 0.5)) == 0.0
    ruled = _late(None)
    ruled.constrain("on_screen(late)")
    assert _until(ruled, lambda s: s.set("late.r", 0.5)) == 0.0
    # A composite builds its parts from the whole scene, so it might read anything.
    from phasor import net

    boxed = _late(None, more=[net.Polyhedron(id="cube", size=(0.5, 0.5, 0.5), origin=(3, 3, 0))])
    assert _until(boxed, lambda s: s.set("late.r", 0.5)) == 0.0
    # Never shown: nothing it does reaches the layout.
    never = _late(None, more=[Dot(id="ghost", pos=(1, 1, 1), r=0.2, opacity=0.0)])
    assert _until(never, lambda s: s.set("ghost.r", 0.4)) == float("inf")


# ---------------------------------------------------------------- avoid_occlusion rests while hidden
# Owner, 2026-11-26: an avoid_occlusion goal protects nothing while its object is hidden
# (opacity x dim = 0), as long as the camera stays smooth. Before, the solver kept the view
# clear (and drew depth buffers for it) all the time. "Before" below is the solver with
# the goal never resting (no ``shown`` given to ``camera.solve``): the old code path exactly.


def _ringed(passes, kind):
    """A ring of posts around what the camera must see past them, while it orbits: "fade"
    (the gem appears at 12 s), "stagger" (gems at 11 and 15 s), "return" (the gem goes at
    3 s and comes back at 17 s), "blink" (it goes at 8 s and is back at 13 s: too soon to
    swing away and back), "always" (the gem is shown throughout)."""
    import math

    s = Scene(f"ringed_{kind}", passes=passes)
    s.use_camera("orbit", azimuth=-40, drift=0)
    gems = ["a", "b"] if kind == "stagger" else ["gem"]
    for k, g in enumerate(gems):
        s.add(Dot(id=g, pos=(0.7 * k, 0.3 * k, 0), r=0.2))
    posts = [f"post{k}" for k in range(8)]
    for k, p in enumerate(posts):
        a = 2 * math.pi * k / 8
        s.add(Dot(id=p, pos=(2.2 * math.cos(a), 2.2 * math.sin(a), 0.5), r=0.6))
    s.marker("spin", 0.5)
    s.marker("end", 22.0)
    s.at("spin").orbit(120, until="end")
    if kind == "fade":
        s.at(12.0).reveal("gem")
    elif kind == "stagger":
        s.at(11.0).reveal("b")
        s.at(15.0).reveal("a")
    elif kind in ("return", "blink"):
        s.at(3.0 if kind == "return" else 8.0).exit("gem")
        s.at(17.0 if kind == "return" else 13.0).reveal("gem")
    s.constrain(f"camera keep_visible({', '.join(posts)}) and "
                f"camera avoid_occlusion({', '.join(gems)})")  # fmt: skip
    return s


def _bake_both(passes, kind, monkeypatch):
    """(engine now, engine with the goal never resting, hidden-share times of each), both
    baked."""
    from phasor.layout import bake as bake_mod
    from phasor.layout import camera

    built = _ringed(passes, kind).engine
    times: list[float] = []
    orig = bake_mod.hidden_share

    def measured(state, object_id, oracle):
        times.append(state.t)
        return orig(state, object_id, oracle)

    monkeypatch.setattr(bake_mod, "hidden_share", measured)
    lay = Engine(built.document, passes=built._passes)
    assert lay.layout is not None  # baked now, while hidden_share is counted
    now = list(times)
    times.clear()
    solve = camera.solve
    monkeypatch.setattr(camera, "solve", lambda c, plan, times, hidden, shown: solve(
        c, plan, times, hidden))  # fmt: skip
    before = Engine(built.document, passes=built._passes)
    assert before.layout is not None
    return lay, before, now, list(times)


def _on_screen(e, fps=30.0):
    """What the viewer sees move: (sample times, speed, acceleration) of the fastest of
    every object's center on screen (NDC, the frame is 2 wide), as rendered."""
    ts = np.arange(0.0, e.layout.camera_times[-1] + 1e-9, 1.0 / fps)
    pts = []
    for t in ts:
        st = e.state_at(t)
        pts.append([st.camera.project([o.impl.world_bounds(st).center])[0][:2]
                    for o in st.objects])  # fmt: skip
    p = np.array(pts)
    vel = np.linalg.norm(np.diff(p, axis=0), axis=2).max(axis=1) * fps
    acc = np.linalg.norm(np.diff(p, 2, axis=0), axis=2).max(axis=1) * fps * fps
    return ts[1:-1], vel[1:], acc


def _motion(layout, fps=60.0):
    """Per solved camera parameter, sampled as rendered (``Bake.camera_at``) at ``fps``:
    (sample times, speed, acceleration). Elevation in degrees, distance in e-folds (a
    zoom), target in world units."""
    ts = np.arange(0.0, layout.camera_times[-1] + 1e-9, 1.0 / fps)
    out = {}
    for path in layout.camera:
        v = np.array([np.atleast_1d(layout.camera_at(t)[path]) for t in ts], dtype=float)
        if path.endswith("distance"):
            v = np.log(v)
        vel = np.linalg.norm(np.diff(v, axis=0), axis=1) * fps
        acc = np.linalg.norm(np.diff(v, 2, axis=0), axis=1) * fps * fps
        out[path.rpartition(".")[2]] = (ts[1:-1], vel[1:], acc)
    return out


@pytest.mark.parametrize("passes", MODES)
def test_avoid_occlusion_measures_nothing_while_its_object_is_hidden(passes, monkeypatch):
    e, before, now, then = _bake_both(passes, "fade", monkeypatch)
    assert not e.layout.issues and not before.layout.issues
    # Nothing until the lead-in before the gem appears at 12 s (the solver makes it at
    # least camera_lead long, longer for a bigger swing), then every key it is shown at.
    assert now and 4.0 < min(now) < 12.0 - 3.0
    assert {round(t, 6) for t in now} >= {round(t, 6) for t in e.layout.camera_times if t > 12}
    assert min(then) == 0.0
    assert e.layout.renders < 0.6 * before.layout.renders


def test_an_object_shown_throughout_is_protected_as_before(monkeypatch):
    e, before, _, _ = _bake_both(None, "always", monkeypatch)
    assert e.layout.camera == before.layout.camera and e.layout.issues == before.layout.issues
    assert e.layout.renders == before.layout.renders


@pytest.mark.parametrize("passes", MODES)
@pytest.mark.parametrize("kind", ["return", "blink"])
def test_no_swing_away_and_back_for_nothing(passes, kind, monkeypatch):
    """Hidden for less time than swinging away and back would take ("blink"), or than the
    camera would then spend free ("return": 14 s hidden, but the swing out and in needs
    more), the object stays protected: the camera does what it always did, and only the
    checks while it is hidden are left out."""
    e, before, _, _ = _bake_both(passes, kind, monkeypatch)
    assert e.layout.camera == before.layout.camera and not e.layout.issues
    assert e.layout.renders < before.layout.renders


@pytest.mark.parametrize("passes", MODES)
@pytest.mark.parametrize("kind", ["fade", "stagger"])
def test_the_camera_stays_smooth_when_avoid_occlusion_rests(passes, kind, monkeypatch):
    from phasor.style.theme import load_theme

    e, before, _, _ = _bake_both(passes, kind, monkeypatch)
    lay, old_lay = e.layout, before.layout
    assert not lay.issues and not old_lay.issues  # where it protects, the goal still holds
    assert lay.renders < 0.7 * old_lay.renders
    # Fade rests: the camera is free until its lead-in. Stagger hardly does: the swing for
    # the first gem would need longer than the 11 s before it, so that gem is protected
    # from the start, and the camera does about what it always did.
    gap = max(float(np.abs(np.array(lay.camera[k], float) - np.array(old_lay.camera[k], float)).max())
              for k in lay.camera)  # fmt: skip
    assert (gap > 1.0) == (kind == "fade")
    new, old = _motion(lay), _motion(old_lay)
    for name in new:
        # C1 as before: no velocity jump of a camera parameter much larger than the old
        # path's largest (fade, 2026-11-26: elevation 46 -> 17 deg/s^2, distance 0.69 ->
        # 0.66 e-folds/s^2, target 1.21 -> 1.45 units/s^2, at the reveal's end, on P).
        assert new[name][2].max() <= 1.25 * old[name][2].max(), name
    # What the viewer sees: no object on screen accelerates or moves faster than before.
    s_new, s_old = _on_screen(e), _on_screen(before)
    assert s_new[2].max() <= 1.05 * s_old[2].max()
    assert s_new[1].max() <= 1.1 * s_old[1].max()
    # The swing into a protecting view stays near camera_speed (the free and protected
    # paths add their own motion to it).
    cap = load_theme("midnight_glow").motion["camera_speed"]
    _, vel, _ = new["elevation"]
    assert vel.max() <= max(1.25 * cap, old["elevation"][1].max())
