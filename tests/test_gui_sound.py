"""Sound in the editor: add music and effects, move them and the narration
on the timeline, master, listen. The logic only (``gui/controller.py``, no Qt); the window
that draws it is tested in ``test_gui_window.py``."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from phasor.core.schema import Narration, NarrationBlock, Word
from phasor.gui.audio import Player, backend
from phasor.gui.controller import Editor
from phasor.time.narration import write_wav
from phasor.workspace import Workspace

RATE = 22_050


def _tone(path, freq, seconds, amp=8000, rate=RATE):
    t = np.arange(round(seconds * rate)) / rate
    write_wav(path, (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16), rate)


def _narration(starts, length=1.5, audio="v.wav"):
    """One sentence per start, one stressed word each, a marker on the word."""
    return Narration(script="x", voice="test", blocks=tuple(
        NarrationBlock(text=f"word{k}", start=s, duration=length, audio=audio,
                       words=(Word(text=f"word{k}", start=0.1, end=length - 0.1),),
                       markers={f"m{k}": 0.1})
        for k, s in enumerate(starts)))  # fmt: skip


@pytest.fixture
def ed(tmp_path):
    ws = Workspace(tmp_path / "scenes")
    ws.scene_new("with sound")
    folder = ws.engine.base_dir
    _tone(folder / "v.wav", 220, 1.5, amp=12000)
    e = Editor(ws)
    nar = _narration((1.0, 5.0)).model_dump(mode="json")
    assert e.ws.scene_edit([{"op": "set_narration", "narration": nar},
                            {"op": "set_marker", "name": "end", "t": 8.0}])["ok"]  # fmt: skip
    return e


@pytest.fixture
def ambient(tmp_path):
    """A music file somewhere outside the scene's folder, as a person would pick it."""
    path = tmp_path / "elsewhere" / "Night Drift.wav"
    path.parent.mkdir()
    _tone(path, 110, 2.0, amp=6000)
    return path


def test_adding_music_copies_it_into_the_scene_folder(ed, ambient):
    r = ed.add_bed(ambient)
    assert r["ok"]
    (bed,) = ed.engine.document.sound.beds
    assert bed.id == "bed_night_drift" and bed.path == "audio/Night Drift.wav"
    assert (ed.engine.base_dir / "audio" / "Night Drift.wav").read_bytes() == ambient.read_bytes()
    assert ed.add_bed(ambient)["ok"]  # the same file again: a second bed, no second copy
    assert [b.id for b in ed.engine.document.sound.beds] == ["bed_night_drift",
                                                             "bed_night_drift_2"]  # fmt: skip
    assert len(list((ed.engine.base_dir / "audio").iterdir())) == 1
    other = ambient.with_name("x") / ambient.name
    other.parent.mkdir()
    _tone(other, 330, 1.0)
    ed.add_bed(other)  # same name, other sound: kept apart
    assert ed.engine.document.sound.beds[-1].path == "audio/Night Drift_1.wav"


def test_an_effect_at_the_playhead_ties_itself_to_the_nearest_marker(ed):
    ed.set_time(5.3)  # just after the second sentence's marker (5.1 s)
    assert ed.add_cue("chime")["ok"]
    (cue,) = ed.engine.document.sound.cues
    assert cue.id == "cue_chime_m1" and cue.at.marker == "m1"
    assert cue.at.offset == pytest.approx(0.2)
    assert ed.add_cue("pop", marker="m0", offset=-0.1)["ok"]
    assert ed.engine.document.sound.cues[-1].at.marker == "m0"


def test_dragging_a_cue_snaps_it_to_a_marker_and_it_follows_the_voice(ed):
    ed.add_cue("tick", marker="m0")
    assert ed.move_cue("cue_tick_m0", 4.8)["ok"]
    (cue,) = ed.engine.document.sound.cues
    assert cue.at.marker == "m1" and cue.at.offset == pytest.approx(-0.3)
    assert ed.move_sentence(1, 6.0)["ok"]  # move the second sentence: its marker and the cue
    (t,) = [c["t"] for c in ed.timeline()["cues"]]
    assert t == pytest.approx(5.8)
    ed.undo()
    assert ed.timeline()["cues"][0]["t"] == pytest.approx(4.8)


def test_a_sentence_moves_only_between_its_neighbours(ed):
    assert ed.move_sentence(1, 0.0)["ok"]  # cannot go before the first one ends
    blocks = ed.engine.document.narration.blocks
    assert blocks[1].start == pytest.approx(blocks[0].start + blocks[0].duration)
    assert ed.timeline()["markers"]["m1"] == pytest.approx(blocks[1].start + 0.1)
    assert not ed.move_sentence(7, 1.0)["ok"]


def test_sliding_a_bed_keeps_its_length(ed, ambient):
    ed.add_bed(ambient, start="m0", end="m1")
    before = ed.timeline()["beds"][0]
    assert ed.move_bed("bed_night_drift", before["start"] + 0.5)["ok"]
    after = ed.timeline()["beds"][0]
    assert after["start"] == pytest.approx(before["start"] + 0.5)
    assert after["end"] - after["start"] == pytest.approx(before["end"] - before["start"])


def test_gain_and_removal_are_one_undo_step_each(ed, ambient):
    ed.add_bed(ambient)
    v = ed.version
    assert ed.update_sound("bed_night_drift", gain=-24.0)["ok"] and ed.version == v + 1
    assert ed.engine.document.sound.beds[0].gain == -24.0
    assert not ed.update_sound("nothing", gain=1.0)["ok"]
    assert ed.remove_sound("bed_night_drift")["ok"] and not ed.engine.document.sound.beds
    ed.undo()
    assert ed.engine.document.sound.beds[0].gain == -24.0
    assert [a["who"] for a in ed.activity()][-2:] == ["you", "you"]


def test_master_turns_down_a_bed_covering_the_voice_and_moves_a_cue_off_a_word(ed, ambient):
    ed.add_bed(ambient, gain=6.0)
    ed.add_cue("pop", marker="m0")  # on the stressed word itself
    codes = {i["code"] for i in ed.sound()["issues"]}
    assert {"bed_masks_voice", "cue_on_word"} <= codes
    r = ed.master()
    assert r["ok"] and r["rounds"]
    assert not {"bed_masks_voice", "cue_on_word"} & set(r["after"]["issues"])
    assert ed.engine.document.sound.beds[0].gain < 6.0
    assert ed.engine.document.sound.cues[0].at.offset < 0  # now just before the word
    for _ in r["rounds"]:
        ed.undo()
    assert ed.engine.document.sound.beds[0].gain == 6.0


def test_master_with_nothing_to_fix_changes_nothing(ed, ambient):
    ed.add_bed(ambient)
    v = ed.version
    r = ed.master()
    assert r["ok"] and r["rounds"] == [] and ed.version == v


def test_the_sound_tab_reads_lanes_levels_and_switches(ed, ambient):
    ed.add_bed(ambient)
    ed.add_cue("chime", marker="m0")
    s = ed.sound()
    assert [x["id"] for x in s["lanes"]] == ["voice", "bed_night_drift", "cue_chime_m0"]
    assert s["beds"][0]["path"] == "audio/Night Drift.wav" and s["cues"][0]["sound"] == "chime"
    assert not s["scripted"] and s["muted"] == [] and s["defaults"]["bed"] == -18.0
    ed.toggle_mute("bed_night_drift")
    ed.toggle_solo("voice")
    s = ed.sound()
    assert s["muted"] == ["bed_night_drift"] and s["solo"] == ["voice"]
    picture = ed.sound_picture()
    assert picture and picture.endswith(".png")


def _read(path):
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 2 and w.getsampwidth() == 2
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)
        return data, w.getframerate()


def test_listening_plays_the_mix_from_the_playhead_with_mutes(ed, ambient):
    ed.add_bed(ambient, fade=0.0)
    full, rate = _read(ed.listen_file(0.0))
    track, rate2 = ed.engine.audio_track()
    assert rate == rate2 and (full == track).all()  # what you hear is what gets encoded
    later, _ = _read(ed.listen_file(2.0))
    assert len(later) == len(full) - 2 * rate
    ed.toggle_solo("voice")  # only the voice: silent before the first sentence
    only_voice, r1 = _read(ed.listen_file(0.0))
    assert not only_voice[: r1 // 2].any() and full[: rate // 2].any()
    ed.toggle_solo("voice")
    ed.toggle_mute("voice")
    ed.toggle_mute("bed_night_drift")
    silent, _ = _read(ed.listen_file(0.0))
    assert not silent.any()


def test_a_silent_scene_has_nothing_to_listen_to(tmp_path):
    ws = Workspace(tmp_path)
    ws.scene_new("quiet")
    e = Editor(ws)
    assert e.listen_file() is None and not e.sound()["lanes"]
    assert e.master() == {"ok": True, "silent": True}


def test_narrating_from_the_editor(ed):
    r = ed.narrate("One sentence [[first]]. Then another [[second]].", voice="estimate")
    assert r["ok"] and set(r["markers"]) >= {"first", "second"}
    assert ed.activity()[-1]["what"] == "narration (estimate)"


def test_a_script_scene_becomes_an_editable_copy_and_the_script_stays(tmp_path):
    script = tmp_path / "s.py"
    code = "from phasor import Scene\ns = Scene('scripted')\ns.marker('go', 1.0)\n"
    script.write_text(code, encoding="utf-8")
    ws = Workspace(tmp_path)
    ws.allow_scripts = True
    ws.scene_open(str(script))
    e = Editor(ws)
    assert e.sound()["scripted"]
    r = e.add_cue("pop", marker="go")  # the first sound edit: a copy, then the edit
    assert r["ok"] and e.copied == str(tmp_path / "s_edit.phasor.json")
    assert not e.engine.scripted and e.engine.document.sound.cues[0].id == "cue_pop_go"
    assert script.read_text(encoding="utf-8") == code  # the script is untouched
    assert not (tmp_path / "s.overrides.json").exists()  # and so are its tweaks
    assert any("editing a copy" in a["what"] for a in e.activity())
    e.save()
    from phasor.scene import load

    again = load(tmp_path / "s_edit.phasor.json", passes=None)
    assert again.document.sound.cues[0].id == "cue_pop_go" and not again.scripted


def test_a_copy_never_overwrites_an_earlier_copy(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("from phasor import Scene\ns = Scene('x')\n", encoding="utf-8")
    ws = Workspace(tmp_path)
    ws.allow_scripts = True
    for name in ("s_edit.phasor.json", "s_edit_2.phasor.json"):
        ws.scene_open(str(script))
        assert ws.scene_copy()["saved"].endswith(name)


# ---------------------------------------------------------------- your own voice


def _take(path, seconds, amp, freq=200):
    _tone(path, freq, seconds, amp=amp)
    return path


def test_your_recordings_become_the_narration_where_you_drop_them(tmp_path):
    ws = Workspace(tmp_path / "scenes")
    ws.scene_new("my voice")
    e = Editor(ws)
    one = _take(tmp_path / "take one.wav", 1.5, 20000)
    two = _take(tmp_path / "take two.wav", 1.0, 2000, freq=250)
    assert e.add_voice(one, t=0.5, text="Here is a pole.")["ok"]
    assert e.add_voice(two, t=3.0)["ok"]
    n = e.engine.document.narration
    assert [b.text for b in n.blocks] == ["Here is a pole.", "take two"]
    assert [b.audio for b in n.blocks] == ["audio/take one.wav", "audio/take two.wav"]
    assert [b.start for b in n.blocks] == [0.5, 3.0] and n.blocks[1].duration == 1.0
    assert n.voice == "recorded" and n.script == "Here is a pole.\ntake two"
    over = e.add_voice(one, t=3.5)  # on top of the second take
    assert not over["ok"] and "overlap" in over["errors"][0]["message"]
    track, rate = e.engine.audio_track()
    assert np.abs(track[round(0.6 * rate) : round(1.9 * rate)]).max() > 1000  # heard


def test_music_ducks_under_your_voice(tmp_path, ambient):
    ws = Workspace(tmp_path / "scenes")
    ws.scene_new("duck")
    e = Editor(ws)
    e.add_voice(_take(tmp_path / "v.wav", 1.5, 12000), t=2.0)
    e.add_bed(ambient, fade=0.0)
    m = e.engine.sound_mix()
    assert m.spans == ((2.0, 3.5),) and m.duck_db(2.3, 3.2) < -6
    assert e.sound()["issues"] == []


def test_a_marker_in_your_recording_moves_with_it(tmp_path):
    ws = Workspace(tmp_path / "scenes")
    ws.scene_new("marks")
    e = Editor(ws)
    e.add_voice(_take(tmp_path / "v.wav", 2.0, 12000), t=1.0)
    assert e.add_marker("pole", t=1.6)["ok"]  # inside the take: its own marker
    assert e.add_marker("later", t=2.9)["ok"]  # inside too (the take ends at 3 s)
    assert not e.add_marker("pole", t=2.0)["ok"] and not e.add_marker("Bad name!")["ok"]
    e.add_cue("chime", marker="pole")
    assert e.move_sentence(0, 2.0)["ok"]
    tl = e.timeline()
    assert tl["markers"]["pole"] == pytest.approx(2.6) and tl["markers"]["later"] == 3.9
    assert tl["cues"][0]["t"] == pytest.approx(2.6)


def test_taking_a_recording_out(tmp_path):
    ws = Workspace(tmp_path / "scenes")
    ws.scene_new("out")
    e = Editor(ws)
    e.add_voice(_take(tmp_path / "a.wav", 1.0, 9000), t=0.0)
    e.add_voice(_take(tmp_path / "b.wav", 1.0, 9000), t=2.0)
    assert e.remove_sentence(0)["ok"]
    assert [b.audio for b in e.engine.document.narration.blocks] == ["audio/b.wav"]
    assert e.remove_sentence(0)["ok"] and e.engine.document.narration is None
    assert not e.remove_sentence(0)["ok"]


def test_the_ai_adds_a_recording_from_the_scene_folder(tmp_path):
    from phasor.workspace import ToolError

    ws = Workspace(tmp_path)
    ws.scene_new("ai")
    (ws.engine.base_dir / "audio").mkdir(parents=True)
    _take(ws.engine.base_dir / "audio" / "intro.wav", 1.0, 9000)
    ws.scene_edit({"op": "set_marker", "name": "start", "t": 0.5})
    r = ws.voice_add("audio/intro.wav", at="start", text="Hello.")
    assert r["ok"] and ws.engine.document.narration.blocks[0].start == 0.5
    with pytest.raises(ToolError):
        ws.voice_add("../outside.wav")


def test_a_player_without_a_backend_stays_silent(tmp_path):
    p = Player(kind=None)
    assert not p.available and not p.play(tmp_path / "x.wav")
    p.stop()
    assert backend() in (None, "winsound", "afplay", "paplay", "aplay", "ffplay")


@pytest.mark.skipif(backend() not in ("paplay", "aplay", "ffplay"), reason="no player here")
def test_a_player_plays_and_stops(tmp_path):
    path = tmp_path / "x.wav"
    write_wav(path, (3000 * np.sin(np.arange(8000) * 0.1)).astype(np.int16), 8000)
    p = Player()
    assert p.play(path)
    p.stop()
    assert not p.playing
