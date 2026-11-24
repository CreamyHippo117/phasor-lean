"""Narration sync, choreography verbs, pacing, audio, and the exit demo
("a residue theorem short driven entirely by narration")."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from phasor import Dot, Line, Scene, Text, load
from phasor.core.schema import MarkerRef
from phasor.style.theme import load_theme
from phasor.time.easing import EASINGS
from phasor.time.narration import (
    ScriptError,
    estimate_duration,
    mix_track,
    parse_script,
    write_wav,
)
from phasor.time.verbs import VerbError
from phasor.voice import TimingsFile, espeak_available, from_recordings, narrate

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "examples" / "residue_theorem.py"
THEME = load_theme("midnight_glow")
needs_espeak = pytest.mark.skipif(not espeak_available(), reason="espeak-ng not installed")

SCRIPT = """We start with a function that has two poles [[poles_appear]].
Draw a small loop around [[loop_draw]] the one at i.
[[pause 1.5]]
Only that pole contributes [[residue_glow]]."""


# ---------------------------------------------------------------- the script format


def test_script_splits_into_blocks_pieces_and_markers():
    a, b, pause, c = parse_script(SCRIPT)
    assert a.pieces == ("We start with a function that has two poles.",)
    assert a.markers == {"poles_appear": 1}  # at the end of the sentence
    assert b.pieces == ("Draw a small loop around", "the one at i.")
    assert b.markers == {"loop_draw": 1}  # where "the one at i" starts
    assert pause.pause == 1.5 and pause.pieces == ()
    assert c.markers == {"residue_glow": 1}
    (d,) = parse_script("[[first]] Before any word.")
    assert d.markers == {"first": 0} and d.pieces == ("Before any word.",)
    assert len(parse_script("One. Two! Three?\nFour")) == 4


@pytest.mark.parametrize(
    ("script", "why"),
    [("", "empty"), ("Say [[Bad Name]] this.", "bad marker"), ("A [[m]]. B [[m]].", "twice"),
     ("[[pause soon]]", "bad pause"), ("[[only_a_marker]]", "no words")],
)  # fmt: skip
def test_bad_scripts_say_why(script, why):
    with pytest.raises(ScriptError, match=why):
        parse_script(script)


def test_estimated_timing_is_deterministic_and_lays_blocks_end_to_end(tmp_path):
    n = narrate(SCRIPT, "estimate", tmp_path / "clips", THEME)
    assert n.voice == "estimate" and all(b.audio is None for b in n.blocks)
    assert not (tmp_path / "clips").exists()  # nothing to write
    first, second, pause, last = n.blocks
    assert first.start == THEME.narration["lead"]
    wps = THEME.narration["words_per_second"]
    assert first.duration == pytest.approx(estimate_duration(first.text, wps), abs=1e-3)
    gap = THEME.narration["gap"]
    assert second.start == pytest.approx(first.start + first.duration + gap, abs=1e-3)
    assert pause.start == pytest.approx(second.start + second.duration + gap, abs=1e-3)
    assert last.start == pytest.approx(pause.start + 1.5, abs=1e-3)
    assert n.markers()["loop_draw"] == pytest.approx(
        second.start + estimate_duration("Draw a small loop around", wps), abs=1e-3)  # fmt: skip
    assert [w.text for w in second.words][:2] == ["Draw", "a"]
    assert narrate(SCRIPT, "estimate", tmp_path, THEME) == n


@needs_espeak
def test_espeak_makes_markers_exact_and_caches_clips(tmp_path):
    folder = tmp_path / "scene.narration"
    n = narrate(SCRIPT, "espeak", folder, THEME)
    assert n.voice.startswith("espeak:")
    b = n.blocks[1]
    samples, rate = _read(tmp_path / b.audio)
    assert len(samples) / rate == pytest.approx(b.duration, abs=2e-3)
    # The marker sits exactly at the seam between the two separately spoken pieces: the
    # first piece alone has exactly that duration.
    alone = narrate("Draw a small loop around.", "espeak", tmp_path / "solo", THEME).blocks[0]
    assert b.markers["loop_draw"] == pytest.approx(alone.duration, abs=0.05)
    clips = sorted(p.name for p in folder.iterdir())
    again = narrate(SCRIPT, "espeak", folder, THEME)  # cached: no new files, same result
    assert again == n and sorted(p.name for p in folder.iterdir()) == clips


def _read(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16), w.getframerate()


def test_recordings_with_word_timings_place_markers_on_words(tmp_path):
    rate = 8000
    rec = tmp_path / "rec"
    sentences = ["Draw a small loop around the one at i.", "Only that pole contributes."]
    for k, text in enumerate(sentences):
        n_words = len(text.split())
        write_wav(rec / f"{k}.wav", np.full(int(0.4 * n_words * rate), 1000, np.int16), rate)
        words = [{"text": w.strip("."), "start": 0.4 * j, "end": 0.4 * j + 0.3}
                 for j, w in enumerate(text.split())]  # fmt: skip
        (rec / f"{k}.json").write_text(json.dumps(words), encoding="utf-8")
    script = "Draw a small loop around [[loop_draw]] the one at i.\nOnly that pole contributes."
    n = from_recordings(script, [rec / "0.wav", rec / "1.wav"], TimingsFile(), tmp_path, THEME)
    assert n.blocks[0].markers["loop_draw"] == pytest.approx(5 * 0.4)  # "the" is word 6
    assert n.blocks[0].audio == "rec/0.wav" and n.voice == "recorded"
    with pytest.raises(ScriptError, match="2 sentences but 1 recordings"):
        from_recordings(script, [rec / "0.wav"], TimingsFile(), tmp_path, THEME)


# ---------------------------------------------------------------- narration drives time


def _scene(voice="estimate", **kw):
    s = Scene("narrated", passes=None, **kw)
    s.add(Dot(id="pole", pos=(0, 1), r=0.2))
    s.add(Line(id="loop", points=[(0.5, 1), (0, 1.5), (-0.5, 1), (0, 0.5), (0.5, 1)]))
    s.add(Text(id="loop.label", text="Res", pos=(1, 1)))
    s.narrate(SCRIPT, voice=voice)
    return s


def test_markers_come_from_the_narration_and_tracks_follow_a_new_voice(tmp_path):
    s = _scene(base_dir=tmp_path)
    s.at("loop_draw").trace("loop")
    e = s.engine
    t_draw = e.document.timeline.markers["loop_draw"]
    assert e.state_at(t_draw)["loop"].params.progress == 0.0
    assert e.state_at(t_draw + 5)["loop"].params.progress == 1.0
    assert e.duration == pytest.approx(e.document.narration.end)
    # Re-voice: markers move, the trace moves with them, nothing else needs touching.
    s.narrate(SCRIPT.replace("Draw a small loop", "Now we carefully draw a small loop"),
              voice="estimate")  # fmt: skip
    later = e.document.timeline.markers["loop_draw"]
    assert later > t_draw + 0.5
    assert e.state_at(later)["loop"].params.progress == 0.0


def test_narration_markers_and_hand_markers_coexist_but_never_clash(tmp_path):
    s = _scene(base_dir=tmp_path)
    s.marker("handmade", 0.2)
    s.narrate("Short [[blip]] line.", voice="estimate")
    assert set(s.engine.document.timeline.markers) == {"handmade", "blip"}
    r = s.engine.apply({"op": "set_marker", "name": "blip", "t": 9.0}, origin="script")
    assert not r.ok  # the narration owns that marker
    s.marker("taken", 1.0)
    with pytest.raises(Exception, match="marker_conflict"):
        s.narrate("Now [[taken]] again.", voice="estimate")


def test_pace_adds_room_after_busy_sentences_and_is_idempotent(tmp_path):
    s = _scene(base_dir=tmp_path)
    s.at("poles_appear").reveal("pole", "loop.label")  # two objects change in block 0
    before = s.engine.document.narration
    s.pace()
    after = s.engine.document.narration
    extra = 2 * THEME.narration["pause_per_change"]
    assert after.blocks[1].start == pytest.approx(before.blocks[1].start + extra)
    assert after.blocks[0].start == before.blocks[0].start
    s.pace()
    assert s.engine.document.narration == after
    with pytest.raises(ValueError, match="narrate"):
        Scene("x", passes=None).pace()


def test_the_audio_track_places_each_clip_at_its_block(tmp_path):
    rate = 1000
    write_wav(tmp_path / "a.wav", np.full(500, 7, np.int16), rate)
    from phasor.core.schema import Narration, NarrationBlock

    n = Narration(script="x", voice="test", blocks=(
        NarrationBlock(text="a", start=0.25, duration=0.5, audio="a.wav"),
        NarrationBlock(text="b", start=2.0, duration=0.5, audio="a.wav")))  # fmt: skip
    track, r = mix_track(n, tmp_path, duration=3.0)
    assert r == rate and len(track) == 3000
    assert track[249] == 0 and track[250] == 7 and track[749] == 7 and track[750] == 0
    assert track[2000] == 7
    est = Narration(script="x", voice="estimate", blocks=(NarrationBlock(text="a", start=0,
                                                                        duration=1),))  # fmt: skip
    assert mix_track(est, tmp_path, 1.0) is None


# ---------------------------------------------------------------- verbs


def _verbs_scene():
    s = Scene("verbs", passes=None)
    s.set("camera.height", 8)
    s.add(Dot(id="a", pos=(-2, 0), r=0.3))
    s.add(Dot(id="b", pos=(2, 0), r=0.3))
    s.add(Line(id="curve", points=[(-3, -2), (3, -2)]))
    s.add(Text(id="note", text="note", pos=(0, 2)))
    s.marker("go", 1.0)
    return s


def test_reveal_hides_until_then_and_staggers():
    s = _verbs_scene()
    s.at("go").reveal("a", "b")
    e = s.engine
    m = THEME.motion
    assert e.state_at(0.5)["a"].params.opacity == 0 and e.state_at(0.5)["b"].params.opacity == 0
    t_a = 1.0 + m["reveal"]["duration"]
    assert e.state_at(t_a)["a"].params.opacity == 1.0 and e.state_at(t_a)["b"].params.opacity < 1
    assert e.state_at(t_a + m["stagger"])["b"].params.opacity == 1.0
    segs = e.document.timeline.tracks["a.opacity"]
    assert segs[0].verb == "reveal" and segs[0].ease == m["reveal"]["ease"]
    assert segs[0].t0 == MarkerRef(marker="go")


def test_trace_draws_a_curve_progressively():
    s = _verbs_scene()
    s.at("go").trace("curve")
    e = s.engine
    d = THEME.motion["trace"]["duration"]
    assert e.state_at(0.5)["curve"].params.progress == 0
    mid = e.state_at(1.0 + d / 2)["curve"]
    assert mid.params.progress == pytest.approx(0.5)
    x0, _, x1 = mid.impl.world_bounds(e.state_at(1.0 + d / 2)).lo[0], 0, \
        mid.impl.world_bounds(e.state_at(1.0 + d / 2)).hi[0]  # fmt: skip
    assert x0 == pytest.approx(-3, abs=0.1) and x1 == pytest.approx(0, abs=0.1)
    with pytest.raises(VerbError, match="needs a curve"):
        s.at("go").trace("a")


def test_dim_others_and_reveal_do_not_fight_whatever_the_authoring_order():
    s = _verbs_scene()
    s.marker("later", 3.0)
    s.at("go").dim_others("a")  # 'note' and 'b' dim here...
    s.at("later").reveal("note")  # ...but 'note' is revealed afterwards: hidden, then full
    e = s.engine
    level = THEME.motion["dim_level"]
    assert e.state_at(2.0)["b"].params.opacity == pytest.approx(level)
    assert e.state_at(2.0)["note"].params.opacity == 0  # hidden until its reveal
    assert e.state_at(5.0)["note"].params.opacity == 1.0  # arrives undimmed: it is the focus
    assert e.state_at(2.0)["a"].params.opacity == 1.0  # the kept object never dims
    s.at(6.0).undim()
    assert e.state_at(8.0)["b"].params.opacity == 1.0


def test_focus_pull_frames_the_object():
    s = _verbs_scene()
    s.at("go").focus_pull("a")
    e = s.engine
    end = e.state_at(5.0)
    assert end.values["camera.center"] == pytest.approx((-2.0, 0.0, 0.0))
    assert end.values["camera.height"] == pytest.approx(0.6 * 1.3)
    assert e.where("a", 5.0)["region"] == "center"


def test_settle_overshoots_then_rests_and_bloom_in_grows():
    s = _verbs_scene()
    s.at("go").settle("a.pos", (-2, 2)).then().bloom_in("b")
    e = s.engine
    d = THEME.motion["settle"]["duration"]
    ys = [e.state_at(1.0 + d * k / 40)["a"].params.pos[1] for k in range(41)]
    assert max(ys) > 2.05 and ys[-1] == pytest.approx(2.0)
    b_start = 1.0 + d
    assert e.state_at(b_start - 0.01)["b"].params.r < 0.01
    rs = [e.state_at(b_start + THEME.motion["bloom_in"]["duration"] * k / 40)["b"].params.r
          for k in range(41)]  # fmt: skip
    assert max(rs) > 0.3 and rs[-1] == pytest.approx(0.3)
    assert max(e.state_at(b_start + k / 20)["b"].params.opacity for k in range(30)) <= 1.0


def test_verbs_fail_loudly():
    s = _verbs_scene()
    with pytest.raises(VerbError, match="no object 'ghost'"):
        s.at("go").reveal("ghost")
    with pytest.raises(VerbError, match="no marker 'nope'"):
        s.at("nope").reveal("a")
    with pytest.raises(VerbError, match="3D camera"):
        s.at("go").orbit_reveal("a")
    with pytest.raises(VerbError, match="opacity"):
        s.at("go").settle("a.opacity", 0.5)


def test_strict_lint_flags_raw_motion_only():
    s = _verbs_scene()
    s.at("go").to("a.pos", (0, 0))
    s.at("go").morph("b.pos", (0, 1))
    assert [i.code for i in s.engine.lint()] == []
    (raw,) = [i for i in s.engine.lint(strict=True) if i.code == "raw_motion"]
    assert raw.ids == ("a",) and raw.times == (1.0,) and "verb" in raw.fix_hint


@pytest.mark.parametrize("name", ["back", "spring"])
def test_overshooting_easings_land_exactly(name):
    assert EASINGS[name](1.0) == pytest.approx(1.0) and EASINGS[name](0.0) == pytest.approx(0)


# ---------------------------------------------------------------- the exit demo


@pytest.fixture(scope="module")
def short():
    return load(DEMO, passes=None)


def test_exit_demo_is_driven_entirely_by_narration(short):
    doc = short.document
    assert doc.narration is not None and len(doc.narration.blocks) == 6
    assert set(doc.timeline.markers) == set(doc.narration.markers())  # no hand-timed markers
    segs = [seg for track in doc.timeline.tracks.values() for seg in track]
    assert segs and all(isinstance(x.t0, MarkerRef) and isinstance(x.t1, MarkerRef) for x in segs)
    assert all(x.verb for x in segs)  # every motion is a verb
    assert short.lint(strict=True) == []
    # The scene lasts until the last word, or the last motion if that ends later.
    ends = [short._snap.compiled.time_map.resolve(x.t1) for x in segs]
    assert short.duration == pytest.approx(max(doc.narration.end, *ends))


def test_exit_demo_tells_the_story_in_order(short):
    m = short.document.timeline.markers
    at = short.state_at
    assert at(0.0)["plane"].params.opacity == 0  # nothing before the first word
    assert at(m["pole_top"] - 0.01)["pole_top"].params.opacity == 0
    assert at(m["pole_bottom"] + 2)["pole_bottom"].params.opacity == 1
    assert at(m["loop_draw"])["loop"].params.progress == 0
    assert at(m["residue"])["loop"].params.progress == 1  # drawn by the time it is discussed
    level = THEME.motion["dim_level"]
    assert at(m["res_value"])["pole_bottom"].params.opacity == pytest.approx(level)
    assert at(m["integral"] - 0.01)["result"].params.opacity == 0
    assert at(short.duration)["result"].params.opacity == 1.0
    sem = short.inspect(short.duration, id="loop.label")["objects"][0]
    assert sem["semantic"]["text"] == "Res = -i/2" and sem["opacity"] == 1.0


@needs_espeak
def test_exit_demo_has_a_voice_and_renders_with_it(tmp_path):
    from tests.conftest import gl_available

    e = load(DEMO, passes=None)
    track = e.audio_track()
    assert track is not None and e.document.narration.voice.startswith("espeak")
    samples, rate = track
    assert len(samples) / rate == pytest.approx(e.duration, abs=0.05)
    first = e.document.narration.blocks[0]
    i0 = int(first.start * rate)
    assert not samples[: i0 - 10].any() and np.abs(samples[i0 : i0 + rate]).max() > 1000
    if not gl_available():
        return
    av = pytest.importorskip("av")
    from phasor.cli import main

    out = tmp_path / "short.mp4"
    assert main(["render", str(DEMO), "-o", str(out), "--fps", "10", "--start", "6",
                 "--end", "7", "--quiet"]) == 0  # fmt: skip
    with av.open(str(out)) as c:
        assert [s.type for s in c.streams] == ["video", "audio"]
        a = c.streams.audio[0]
        assert float(a.duration * a.time_base) == pytest.approx(1.0, abs=0.1)
