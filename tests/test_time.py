"""time: expressions, tracks, the parameter graph, and purity of evaluate()."""

from __future__ import annotations

import math
import random

import pytest

import phasor.domains  # noqa: F401
from phasor.core.issues import DocumentError
from phasor.core.schema import Document, MarkerRef, Meta, ObjectSpec, Segment, Timeline
from phasor.time.compile import compile_document, evaluate
from phasor.time.easing import EASINGS, OVERSHOOT
from phasor.time.expr import ExprError, compile_expr
from phasor.time.tracks import ResolvedSegment, ResolvedTrack, TimeMap

# ---------------------------------------------------------------- expressions


def test_expr_arithmetic_refs_and_time():
    e = compile_expr("2 * loop.r + sin(t) - res.pole_i.pos[1]")
    assert e.deps == {"loop.r", "res.pole_i.pos"} and e.uses_t
    env = {"loop.r": 1.5, "res.pole_i.pos": (0.0, 4.0, 0.0)}
    assert e(env, math.pi / 2) == pytest.approx(3.0 + 1.0 - 4.0)


def test_expr_vectors_broadcast():
    assert compile_expr("a.pos + vec(0, 0.5, 0)")({"a.pos": (1.0, 2.0, 3.0)}, 0) == (1, 2.5, 3)
    assert compile_expr("2 * a.pos")({"a.pos": (1.0, 2.0)}, 0) == (2.0, 4.0)
    assert compile_expr("-a.pos")({"a.pos": (1.0, -2.0)}, 0) == (-1.0, 2.0)
    assert compile_expr("midpoint(a.pos, b.pos)")({"a.pos": (0.0, 0.0), "b.pos": (2.0, 4.0)}, 0) \
        == (1.0, 2.0)  # fmt: skip
    assert compile_expr("length(vec(3, 4))")({}, 0) == 5.0
    assert compile_expr("(1, 2 + 3)")({}, 0) == (1.0, 5.0)
    assert not compile_expr("pi * 2").uses_t


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('x')",
        "open('f')",
        "a.b.__class__",
        "lambda: 1",
        "[x for x in y]",
        "a if b else c",
        "'text'",
        "True",
        "x",  # bare names other than t are not parameters
        "a.pos[i.j]",
        "sin(x=1)",
        "1 +",
        "a.b; c.d",
    ],
)
def test_expr_rejects_everything_outside_the_whitelist(source):
    with pytest.raises(ExprError):
        e = compile_expr(source)
        # attribute access is only a path lookup: dunder names can never reach an object
        e({}, 0)


def test_expr_runtime_errors_are_reported_with_source():
    with pytest.raises(ExprError, match="1 / a.r"):
        compile_expr("1 / a.r")({"a.r": 0.0}, 0)
    with pytest.raises(ExprError, match="lengths differ"):
        compile_expr("a.p + vec(1, 2)")({"a.p": (1.0, 2.0, 3.0)}, 0)


# ---------------------------------------------------------------- tracks


def _track(initial, *segs):
    return ResolvedTrack(
        initial, tuple(ResolvedSegment(a, b, to, EASINGS[e]) for a, b, to, e in segs)
    )


def test_track_holds_eases_and_holds():
    tr = _track(0.0, (1, 3, 10.0, "linear"), (5, 6, 0.0, "linear"))
    assert [tr.value_at(t) for t in (-1, 0, 1, 2, 3, 4, 5, 5.5, 6, 99)] == [
        0, 0, 0, 5, 10, 10, 10, 5, 0, 0,
    ]  # fmt: skip


def test_track_instant_segment_and_discrete_values():
    tr = _track("a", (2, 2, "b", "linear"), (3, 5, "c", "smooth"))
    assert [tr.value_at(t) for t in (1.99, 2, 4.99, 5)] == ["a", "b", "b", "c"]


@pytest.mark.parametrize("name", sorted(EASINGS))
def test_easings_hit_endpoints_and_stay_bounded(name):
    f = EASINGS[name]
    assert f(0.0) == pytest.approx(0.0) and f(1.0) == pytest.approx(1.0)
    values = [f(i / 50) for i in range(51)]
    if name in OVERSHOOT:  # these go past 1 on purpose, then come back
        assert 1.02 < max(values) < 1.25 and min(values) >= -1e-9
    else:
        assert all(-1e-9 <= v <= 1 + 1e-9 for v in values)


def test_timemap_resolves_markers_and_nearest():
    tm = TimeMap({"a": 1.0, "b": 4.0})
    assert tm.resolve(MarkerRef(marker="b", offset=-0.5)) == 3.5
    assert tm.resolve(2.0) == 2.0
    assert tm.nearest(2.0) == "a" and tm.nearest(3.0) == "b"
    with pytest.raises(KeyError):
        tm.resolve(MarkerRef(marker="nope"))


# ---------------------------------------------------------------- compile + evaluate


def _doc(objects, tracks=None, markers=None, **meta):
    return Document(
        meta=Meta(title="t", **meta),
        objects=objects,
        timeline=Timeline(tracks=tracks or {}, markers=markers or {}),
    )


def _dot(id="dot", **params):
    return ObjectSpec(id=id, type="dot", params=params)


def _codes(doc) -> set[str]:
    with pytest.raises(DocumentError) as e:
        compile_document(doc)
    return {i.code for i in e.value.issues}


def test_defaults_become_parameters_and_2d_points_are_padded():
    s = evaluate(compile_document(_doc([_dot(pos=(1, 2))])), 0)
    assert s["dot"].params.pos == (1.0, 2.0, 0.0)
    assert s.values["dot.opacity"] == 1.0 and s.values["camera.height"] == 8.0


def test_expression_follows_track_through_the_graph():
    doc = _doc(
        [_dot(pos=(0, 0)), ObjectSpec(id="label", type="text",
                                      params={"text": "z", "pos": {"expr": "dot.pos + vec(0,1,0)"}})],
        tracks={"dot.pos": (Segment(t0=MarkerRef(marker="go"), t1=MarkerRef(marker="go", offset=2),
                                    to=(4, 0), ease="linear"),)},
        markers={"go": 1.0},
    )  # fmt: skip
    c = compile_document(doc)
    assert c.duration == 3.0
    assert evaluate(c, 2.0)["label"].params.pos == (2.0, 1.0, 0.0)
    # Moving the marker (re-recording narration) retimes the track; nothing else changes.
    moved = doc.model_copy(
        update={"timeline": doc.timeline.model_copy(update={"markers": {"go": 5.0}})}
    )
    assert evaluate(compile_document(moved), 6.0)["label"].params.pos == (2.0, 1.0, 0.0)


def test_cycle_is_a_validation_error():
    doc = _doc([_dot("a", r={"expr": "b.r"}), _dot("b", r={"expr": "a.r + 1"})])
    with pytest.raises(DocumentError, match="cycle") as e:
        compile_document(doc)
    assert set(e.value.issues[0].ids) == {"a", "b"}
    assert "cycle" in _codes(_doc([_dot("a", r={"expr": "a.r"})]))


def test_all_problems_are_reported_together():
    doc = _doc(
        [
            ObjectSpec(id="a", type="nope"),
            _dot("b", r="big", wobble=1),
            _dot("c", r={"expr": "zzz.r"}),
            _dot("d", r={"expr": "1 +"}),
            ObjectSpec(id="e", type="text"),
            ObjectSpec(id="f", type="dot", style={"color": "#ff0000"}),
            ObjectSpec(id="g", type="dot", style={"sparkle": "primary"}),
        ],
        tracks={
            "b.opacity": (Segment(t0=0, t1=1, to=0.5, ease="bouncy"),),
            "c.r": (Segment(t0=0, t1=1, to=1.0),),
            "nope.r": (Segment(t0=0, t1=1, to=1.0),),
            "f.r": (Segment(t0=MarkerRef(marker="ghost"), t1=1, to=1.0),),
            "g.pos": (Segment(t0=0, t1=1, to=(1, 2, 3, 4)),),
            "f.opacity": (Segment(t0=2, t1=1, to=0.0),),
            "g.opacity": (Segment(t0=0, t1=2, to=0.0), Segment(t0=1, t1=3, to=1.0)),
        },
    )
    assert _codes(doc) == {
        "unknown_type", "bad_param", "unknown_param", "unknown_ref", "bad_expr", "missing_param",
        "bad_role", "unknown_style", "unknown_ease", "track_on_expr", "unknown_marker",
        "bad_segment",
    }  # fmt: skip


def test_expression_type_errors_surface_at_compile_time():
    assert "bad_param" in _codes(_doc([_dot(r={"expr": "vec(1, 2)"})]))
    assert "bad_expr" in _codes(_doc([_dot("a"), _dot("b", r={"expr": "1 / (a.r - a.r)"})]))


def test_constraint_violations_at_a_later_time_name_the_time():
    c = compile_document(_doc([_dot(r={"expr": "1 - t"})]))
    with pytest.raises(DocumentError, match="dot.r at t=2"):
        evaluate(c, 2.0)


def test_unknown_theme():
    assert _codes(_doc([], theme="nope")) == {"unknown_theme"}


def test_style_roles_resolve_to_linear_theme_values():
    s = evaluate(
        compile_document(_doc([ObjectSpec(id="a", type="dot", style={"color": "highlight"})])), 0
    )
    r, g, b, a = s["a"].style["color"]
    assert s["a"].roles == {"color": "highlight", "glow": "none"}
    assert a == 1.0 and r > g > b  # warm highlight, linear light
    assert r == pytest.approx(
        1.0
    )  # #ff.... stays 1.0; mid sRGB values drop below their sRGB number


def test_state_is_a_pure_function_of_document_and_t():
    doc = _doc(
        [_dot(pos=(0, 0)), _dot("b", pos={"expr": "dot.pos * sin(t)"}, r={"expr": "0.1 + dot.opacity"})],
        tracks={
            "dot.pos": (Segment(t0=0, t1=2, to=(4, 1)), Segment(t0=3, t1=5, to=(-2, 0), ease="in_out")),
            "dot.opacity": (Segment(t0=1, t1=4, to=0.2),),
        },
    )  # fmt: skip
    times = [i * 0.37 for i in range(20)]
    forward = [evaluate(compile_document(doc), t) for t in times]
    c = compile_document(doc)
    shuffled = times[:]
    random.Random(7).shuffle(shuffled)
    seeking = {t: evaluate(c, t) for t in shuffled + shuffled[::-1]}
    for t, state in zip(times, forward, strict=True):
        assert seeking[t] == state  # same (document, t) -> same state, whatever came before
        assert dict(seeking[t].values) == dict(state.values)


def test_state_is_immutable():
    s = evaluate(compile_document(_doc([_dot()])), 0)
    with pytest.raises(Exception):  # noqa: B017  (frozen dataclass / frozen pydantic / mappingproxy)
        s.t = 1
    with pytest.raises(Exception):  # noqa: B017
        s["dot"].params.r = 2
    with pytest.raises(TypeError):
        s.values["dot.r"] = 2
    with pytest.raises(ValueError):
        s.camera.view_proj[0, 0] = 5
