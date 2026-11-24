"""Constraint language, stage 1 (kiwisolver), stage 2 (label placer), the bake."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from phasor import Dot, Line, Scene, Text, expr
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
