"""Variants and your taste as data (style/choices.jsonl)."""

from __future__ import annotations

import json

import pytest

from phasor import Dot, Scene, cx
from phasor.cli import main
from phasor.style import choices
from tests.conftest import gl_available


def test_record_and_read_back(tmp_path):
    path = tmp_path / "style" / "choices.jsonl"
    choices.record(path, scene="s", beat="intro", options=["as_is", "calmer", "bolder"],
                   winner="calmer", note="softer")  # fmt: skip
    choices.record(path, scene="s", beat="loop", options=["as_is", "calmer"], winner="as_is")
    log = choices.read(path)
    assert [c["winner"] for c in log] == ["calmer", "as_is"] and log[0]["note"] == "softer"
    assert all(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    with pytest.raises(ValueError, match="not one of the options"):
        choices.record(path, scene="s", beat="x", options=["a"], winner="b")
    assert choices.read(tmp_path / "missing.jsonl") == []
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"choices.jsonl:1"):
        choices.read(path)


def test_tune_needs_enough_evidence_and_a_better_than_fair_rate(tmp_path):
    path = tmp_path / "c.jsonl"
    opts = ["as_is", "calmer", "bolder"]
    for winner in ("calmer", "calmer", "bolder"):
        choices.record(path, scene="s", beat="b", options=opts, winner=winner)
    t = choices.tune(path)
    assert t["choices"] == 3 and t["favored"] == "calmer"
    assert t["options"]["calmer"] == {"offered": 3, "won": 2, "rate": 0.667}
    assert choices.tune(path, min_choices=4)["favored"] is None  # not enough yet
    even = tmp_path / "e.jsonl"
    for w in opts:
        choices.record(even, scene="s", beat="b", options=opts, winner=w)
    assert choices.tune(even)["favored"] is None  # everything wins a fair share: no favorite
    assert choices.tune(tmp_path / "none.jsonl") == {"choices": 0, "options": {}, "favored": None}


def test_cli_style_tune(tmp_path, capsys):
    path = tmp_path / "style" / "choices.jsonl"
    assert main(["style", "tune", "--choices", str(path)]) == 0
    assert "no choices yet" in capsys.readouterr().out
    for _ in range(3):
        choices.record(path, scene="s", beat="b", options=["as_is", "calmer"], winner="calmer")
    assert main(["style", "tune", "--choices", str(path)]) == 0
    out = capsys.readouterr().out
    assert "you favor: calmer" in out and (tmp_path / "style" / "preferences.json").exists()
    assert main(["style", "tune", "--choices", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["favored"] == "calmer"


def test_variant_operations_change_only_mood_and_never_the_scene():
    from phasor.inspect import variants

    s = Scene("v", passes=None, look="calm")
    s.add(Dot(id="d", pos=(0, 0), style={"glow": "soft"}))
    s.add(cx.DomainColoring(id="p", function="z"))
    names = [n for n, _ in variants.make(s.engine.document)]
    assert names == ["as_is", "calmer", "bolder"]
    ops = dict(variants.make(s.engine.document))
    assert ops["as_is"] == []
    assert {"op": "set_style", "id": "d", "key": "glow", "role": "none"} in ops["calmer"]
    assert {"op": "set_style", "id": "d", "key": "glow", "role": "strong"} in ops["bolder"]
    assert any(o.get("path") == "p.softness" for o in ops["calmer"])
    assert len(variants.make(s.engine.document, n=2)) == 2
    assert s.engine.document.object("d").style["glow"] == "soft"  # nothing applied


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_engine_variants_render_side_by_side_without_touching_the_scene(tmp_path):
    s = Scene("vr", passes="auto")
    s.add(Dot(id="d", pos=(0, 0), r=1.0, style={"glow": "soft", "color": "highlight"}))
    s.marker("beat", 1.0)
    e = s.engine
    version, doc = e.version, e.document
    r = e.variants("beat", size=(240, 135))
    assert [v["name"] for v in r["variants"]] == ["as_is", "calmer", "bolder"]
    assert r["image"].shape[1] > 3 * 240 and r["t"] == 1.0
    assert e.version == version and e.document == doc  # a throwaway copy was used
    assert all(isinstance(v["score"], float) for v in r["variants"])
