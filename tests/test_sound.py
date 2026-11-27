"""Sound: beds and cues in the document, mixed with the narration on
a stereo bus, by loudness, with ducking from the word times, a final trim and a limiter."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from phasor import Scene
from phasor.core import io
from phasor.core.issues import DocumentError
from phasor.core.migrate import migrate
from phasor.core.schema import SCHEMA_VERSION, Cue, Document, Meta, Narration, NarrationBlock, Word
from phasor.style.theme import load_theme
from phasor.time import sound as snd
from phasor.time.narration import mix_track, write_wav
from tests.test_lint import MODES

LEVELS = load_theme("midnight_glow").sound
RATE = snd.RATE


def scene(tmp_path, **kw):
    return Scene("sounds", passes=None, base_dir=tmp_path, **kw)


def sine(freq, seconds, dbfs, rate=RATE, channels=2):
    """A sine whose peak is ``dbfs`` (EBU Tech 3341 states its signals this way)."""
    t = np.arange(round(seconds * rate)) / rate
    x = 10 ** (dbfs / 20) * np.sin(2 * np.pi * freq * t)
    return np.stack([x] * channels, axis=1)


def write_audio(path, x, rate):
    import soundfile

    soundfile.write(str(path), x, rate)


def voice_wav(tmp_path, seconds=2.0, rate=16000, name="v.wav", level=0.3):
    """A stand-in for a recorded voice: a tone with a syllable-like wobble."""
    t = np.arange(round(seconds * rate)) / rate
    x = level * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    write_wav(tmp_path / name, (x * 32767).astype(np.int16), rate)
    return name


def narrated(s, audio, start=0.5, duration=2.0, words=None, markers=None):
    words = words or (Word(text="listen", start=0.0, end=0.5), Word(text="closely", start=0.6,
                                                                      end=duration))  # fmt: skip
    block = NarrationBlock(text="listen closely", start=start, duration=duration, audio=audio,
                           words=words, markers=markers or {"go": 0.0})  # fmt: skip
    s._apply({"op": "set_narration", "narration": Narration(script="x", voice="test",
                                                            blocks=(block,))})  # fmt: skip


# ---------------------------------------------------------------- the loudness meter (BS.1770)


def test_k_weighting_matches_the_bs1770_table_at_48k():
    sos = snd.k_weighting(48000)
    assert sos[0, :3] == pytest.approx([1.53512485958697, -2.69169618940638, 1.19839281085285],
                                       abs=1e-9)  # fmt: skip
    assert sos[0, 4:] == pytest.approx([-1.69065929318241, 0.73248077421585], abs=1e-9)
    assert sos[1, :3] == pytest.approx([1.0, -2.0, 1.0])
    assert sos[1, 4:] == pytest.approx([-1.99004745483398, 0.99007225036621], abs=1e-9)


@pytest.mark.parametrize(
    ("segments", "expected"),
    [
        ([(20.0, -23.0)], -23.0),  # EBU Tech 3341 case 1
        ([(20.0, -33.0)], -33.0),  # case 2
        ([(10.0, -36.0), (60.0, -23.0), (10.0, -36.0)], -23.0),  # case 3: relative gate
        ([(10.0, -72.0), (10.0, -36.0), (60.0, -23.0), (10.0, -36.0), (10.0, -72.0)], -23.0),
        ([(20.0, -26.0), (20.1, -20.0), (20.0, -26.0)], -23.0),  # case 5
    ],
)
def test_the_meter_reads_the_ebu_reference_signals(segments, expected):
    x = np.concatenate([sine(1000, sec, db) for sec, db in segments])
    assert snd.integrated(x, RATE) == pytest.approx(expected, abs=0.1)


def test_mono_is_heard_on_both_channels_and_silence_is_minus_infinity():
    mono = sine(1000, 5, -23.0, channels=1)[:, 0]
    assert snd.integrated(mono, RATE) == pytest.approx(-23.0, abs=0.1)
    assert snd.integrated(np.zeros(RATE * 2), RATE) == -math.inf
    assert snd.momentary_max(sine(1000, 0.1, -23.0), RATE) == pytest.approx(
        -23.0 + 10 * math.log10(0.1 / 0.4), abs=0.1
    )  # a short sound: itself, then silence


def test_the_meter_works_at_other_rates():
    assert snd.integrated(sine(1000, 10, -23.0, rate=44100), 44100) == pytest.approx(-23, abs=0.1)


# ---------------------------------------------------------------- the limiter


def test_true_peak_finds_the_peak_between_samples():
    n = np.arange(4800)
    x = np.sin(np.pi / 2 * n + np.pi / 4)  # fs/4, sampled at 45 degrees: samples read 0.707
    assert np.abs(x).max() == pytest.approx(math.sqrt(0.5))
    assert snd.true_peak(x).max() == pytest.approx(1.0, abs=0.02)


def test_the_limiter_holds_the_true_peak_at_the_ceiling_and_leaves_quiet_sound_alone():
    gen = np.random.default_rng(0)
    loud = np.cumsum(gen.standard_normal((RATE, 2)), axis=0)
    loud *= 3.0 / np.abs(loud).max()  # ~ +9.5 dBFS
    out = snd.limit(loud.astype(np.float32), RATE, -1.0, 2.0)
    assert 20 * math.log10(snd.true_peak(out).max()) <= -1.0 + 0.05
    assert np.abs(np.diff(out, axis=0)).max() < 0.2  # no clicks: the gain moves smoothly
    quiet = (0.5 * sine(440, 1, 0.0)).astype(np.float32)  # -6 dBFS, below the knee
    assert snd.limit(quiet, RATE, -1.0, 2.0) is quiet


# ---------------------------------------------------------------- ducking (tests first)


def test_the_duck_envelope_dips_during_speech_with_attack_and_release():
    rate = 1000
    g = snd.duck_envelope([(2.0, 3.0)], 6000, rate, depth=8.0, attack=0.25, release=0.5)
    db = 20 * np.log10(g)
    assert db[:1749].max() == 0.0 and db[3500:].min() == pytest.approx(0.0)  # untouched outside
    assert db[2000:3001] == pytest.approx(-8.0)  # the whole span is down by `duck`
    ramp_in, ramp_out = db[1750:2001], db[3000:3501]
    assert np.all(np.diff(ramp_in) <= 1e-12) and np.all(np.diff(ramp_out) >= -1e-12)
    assert db[1875] == pytest.approx(-4.0)  # halfway down at the middle of the attack


def test_spans_closer_than_attack_plus_release_stay_ducked_between_them():
    g = snd.duck_envelope([(1.0, 2.0), (2.5, 3.0)], 4000, 1000, 8.0, 0.25, 0.6)
    assert (20 * np.log10(g[2000:2500])) == pytest.approx(-8.0)


def test_speech_spans_come_from_the_word_times():
    n = Narration(script="x", voice="test", blocks=(
        NarrationBlock(text="a b", start=1.0, duration=2.0, words=(
            Word(text="a", start=0.2, end=0.5), Word(text="b", start=0.6, end=1.5))),
        NarrationBlock(text="", start=3.5, duration=1.0),  # a pause does not speak
        NarrationBlock(text="c", start=5.0, duration=1.0),  # no words: the whole block
    ))  # fmt: skip
    assert snd.speech_spans(n) == ((1.2, 2.5), (5.0, 6.0))


# ---------------------------------------------------------------- the document


def test_beds_and_cues_round_trip_through_the_saved_document(tmp_path):
    s = scene(tmp_path)
    s.marker("intro", 0.5)
    s.marker("outro", 4.0)
    s.bed("audio/drift.ogg", start="intro", end="outro", gain=-20, fade=1.5)
    s.cue("intro", "chime", gain=-12, midi=88)
    s.cue(("outro", -0.2), "audio/whoosh.wav", pan=-0.5)
    doc = s.engine.document
    assert [b.id for b in doc.sound.beds] == ["bed_drift"]
    assert [c.id for c in doc.sound.cues] == ["cue_chime_intro", "cue_whoosh_outro"]
    assert doc.cue("cue_whoosh_outro").at.offset == -0.2
    path = tmp_path / "s.phasor.json"
    io.save(path, doc)
    again = io.load_document(path)
    assert again == doc and again.schema_version == SCHEMA_VERSION == 9
    assert Document.model_validate(doc.model_dump(mode="json")) == doc


def test_a_version_8_document_migrates_to_an_empty_sound_section():
    raw = migrate({"schema_version": 8, "meta": {"title": "old"}, "objects": []})
    assert raw["schema_version"] == SCHEMA_VERSION and raw["sound"] == {"beds": [], "cues": []}
    doc = io.parse_document({"schema_version": 8, "meta": {"title": "old"}})
    assert doc.sound.beds == () and doc.sound.cues == ()


@pytest.mark.parametrize(
    ("cue", "why"),
    [
        ({"sound": "/etc/x.wav"}, "absolute"),
        ({"sound": "../x.wav"}, "leaves"),
        ({"sound": "x.txt"}, "must be one of"),
        ({"sound": "chimes"}, "not a built-in"),
        ({"sound": "tick", "midi": 60}, "takes no midi"),
        ({"sound": "a.wav", "dur": 1.0}, "a sound file takes no dur"),
        ({"sound": "chime", "voice": "bell"}, "takes no voice"),
    ],
)
def test_a_cue_names_a_built_in_or_an_audio_file_in_the_folder(cue, why):
    with pytest.raises(ValueError, match=why):
        Cue(id="c", at=0.0, **cue)


def test_ids_share_the_object_space_are_retired_and_edits_undo(tmp_path):
    from phasor.engine import Engine

    e = Engine(Document(meta=Meta(title="doc")), passes=None, base_dir=tmp_path)  # not a script
    assert e.apply({"op": "add_object", "object": {"id": "dot", "type": "dot"}}).ok
    clash = e.apply({"op": "add_cue", "cue": {"id": "dot", "at": 0, "sound": "tick"}})
    assert not clash.ok and "duplicate id" in clash.errors[0].message
    assert e.apply({"op": "add_cue", "cue": {"id": "ping", "at": 0.5, "sound": "pop"}}).ok
    assert e.apply({"op": "add_bed", "bed": {"id": "pad", "path": "pad.flac"}}).ok
    moved = {"id": "ping", "at": 0.75, "sound": "pop", "midi": 80}
    assert e.apply({"op": "set_cue", "cue": moved}).ok
    assert e.document.cue("ping").at == 0.75 and e.document.cue("ping").midi == 80
    assert e.apply({"op": "set_bed", "bed": {"id": "pad", "path": "pad.flac", "gain": -24}}).ok
    assert not e.apply({"op": "set_bed", "bed": {"id": "nope", "path": "pad.flac"}}).ok
    assert e.apply({"op": "remove_sound", "id": "ping"}).ok
    assert "ping" in e.document.retired_ids and e.document.sound.cues == ()
    back = e.apply({"op": "add_cue", "cue": moved})
    assert not back.ok and back.errors[0].code == "id_retired"
    e.undo()  # undo the removal: the cue is back, its id no longer retired
    assert e.document.cue("ping") is not None and "ping" not in e.document.retired_ids
    assert Scene.of(e).cue(0.75, "pop") == "cue_pop_750ms"  # a fresh id, never a retired one


def test_a_cue_at_an_unknown_marker_and_a_backwards_bed_are_refused(tmp_path):
    s = scene(tmp_path)
    with pytest.raises(DocumentError, match="no marker 'nowhere'"):
        s.cue("nowhere", "tick")
    s.marker("a", 1.0)
    s.marker("b", 2.0)
    with pytest.raises(DocumentError, match="end after it starts"):
        s.bed("x.wav", start="b", end="a")
    assert s.engine.document.sound.cues == () and s.engine.document.sound.beds == ()


def test_diff_and_generated_script_carry_beds_and_cues(tmp_path):
    import runpy

    from phasor.core.diff import diff_documents
    from phasor.core.pycode import document_to_python

    s = scene(tmp_path)
    before = s.engine.document
    s.marker("go", 0.5)
    s.marker("stop", 3.0)
    s.cue(("go", 0.25), "tone", midi=64, voice="bell", dur=0.8, gain=-14)
    s.cue("go", "drum.wav", pan=0.3)
    s.bed("pad.ogg", start="go", end="stop", fade_in=0.5, fade_out=1.0, loop=False)
    s.bed("air.wav")
    kinds = [c["kind"] for c in diff_documents(before, s.engine.document)["changes"]]
    assert kinds.count("sound_added") == 4
    path = tmp_path / "again.py"
    path.write_text(document_to_python(s.engine.document), encoding="utf-8")
    again = runpy.run_path(str(path), run_name="x")["s"].engine.document
    assert again.sound == s.engine.document.sound


# ---------------------------------------------------------------- timing


def test_cues_and_bed_ends_follow_their_markers_when_the_narration_is_retimed(tmp_path):
    s = scene(tmp_path)
    s.narrate("A first sentence. Then the bell [[ring]] sounds here [[after]] softly.",
              voice="estimate")  # fmt: skip
    cid = s.cue("ring", "chime", offset=0.1)
    bid = s.bed("pad.wav", start="ring", end="after")
    e = s.engine

    def placed():
        tl = e.timeline()
        cue = next(c for c in tl["cues"] if c["id"] == cid)
        bed = next(b for b in tl["beds"] if b["id"] == bid)
        return cue["t"], bed["start"], bed["end"]

    t, b0, b1 = placed()
    m = e.document.timeline.markers
    assert (t, b0, b1) == pytest.approx((m["ring"] + 0.1, m["ring"], m["after"]))
    s.narrate("A much, much longer first sentence than before, to push everything later. "
              "Then the bell [[ring]] sounds here [[after]] softly.", voice="estimate")  # fmt: skip
    t2, c0, c1 = placed()
    m = e.document.timeline.markers
    assert (t2, c0, c1) == pytest.approx((m["ring"] + 0.1, m["ring"], m["after"]))
    assert t2 > t + 1


def test_a_built_in_cue_holds_the_scene_open_until_it_ends(tmp_path):
    s = scene(tmp_path)
    s.marker("end", 2.0)
    s.cue(("end", 0.5), "tone", dur=1.25)
    assert s.engine.duration == pytest.approx(3.75)
    s.bed("pad.wav")  # a bed does not
    assert s.engine.duration == pytest.approx(3.75)


# ---------------------------------------------------------------- the mix


def test_a_narration_only_scene_is_its_narration_untouched_in_stereo(tmp_path):
    """Step 1: bit-identical to the mono track ``mix_track`` makes, on both channels."""
    s = scene(tmp_path)
    narrated(s, voice_wav(tmp_path, rate=22050))
    samples, rate = s.engine.audio_track()
    mono, r = mix_track(s.engine.document.narration, tmp_path, s.engine.duration)
    assert rate == r == 22050 and samples.shape == (len(mono), 2) and samples.dtype == np.int16
    assert np.array_equal(samples[:, 0], mono) and np.array_equal(samples[:, 1], mono)
    assert not s.engine.sound_mix().mixed


def test_no_sound_and_no_narration_is_silence(tmp_path):
    assert scene(tmp_path).engine.audio_track() is None


def test_levels_are_set_by_loudness_relative_to_the_voice(tmp_path):
    s = scene(tmp_path)
    narrated(s, voice_wav(tmp_path, seconds=4.0), duration=4.0)
    s._apply({"op": "set_meta", "field": "duration", "value": 7.0})  # bed alone at the end
    pad = sine(110, 3.0, -3.0, rate=44100) + sine(165, 3.0, -9.0, rate=44100)
    write_audio(tmp_path / "pad.flac", pad, 44100)  # another rate and format: resampled
    s.bed("pad.flac", fade=0.0)
    m = s.engine.sound_mix()
    assert m.mixed and m.rate == RATE and m.samples.shape[1] == 2
    voice, bed = m.lane("voice"), m.lane("bed_pad")
    assert voice.loudness() == pytest.approx(LEVELS["target"], abs=0.1)
    a, b = m.spans[0]
    under = LEVELS["bed"] - LEVELS["duck"]  # 18 dB under, ducked 8 more while speaking
    assert bed.loudness(a + 0.5, b) - voice.loudness(a + 0.5, b) == pytest.approx(
        under, abs=1.0)  # fmt: skip
    out = m.samples.astype(np.float64) / 32768
    assert snd.integrated(out, RATE) == pytest.approx(LEVELS["target"], abs=0.3)  # the trim
    assert 20 * math.log10(snd.true_peak(out).max()) <= LEVELS["ceiling"] + 0.1
    # After the speech (and its release) the bed comes back up by the ducking depth.
    after = bed.loudness(b + LEVELS["release"] + 0.1, m.samples.shape[0] / RATE)
    assert after - bed.loudness(a + 0.5, b) == pytest.approx(LEVELS["duck"], abs=0.3)


def test_beds_loop_fade_and_crossfade_with_equal_power(tmp_path):
    s = scene(tmp_path)
    s.marker("a", 0.0)
    s.marker("b", 2.0)
    s.marker("c", 4.0)
    tone = sine(500, 0.3, -6.0)  # 0.3 s, looped to fill
    write_audio(tmp_path / "one.wav", tone, RATE)
    write_audio(tmp_path / "two.wav", tone, RATE)
    s.bed("one.wav", start="a", end=("b", 1.0), fade_in=0.0, fade_out=2.0)
    s.bed("two.wav", start=("b", -1.0), end="c", fade_in=2.0, fade_out=0.0)
    m = s.engine.sound_mix()
    one, two = m.lane("bed_one"), m.lane("bed_two")
    assert one.t1 == pytest.approx(3.0) and two.t0 == pytest.approx(1.0)  # looped to fill
    out = m.samples[:, 0].astype(np.float64)
    # Equal power through the crossfade (sin in, cos out: sin^2 + cos^2 = 1), so two
    # uncorrelated beds keep the loudness steady across it.
    env_in = snd._fades(2 * RATE, 2.0, 0.0, RATE)
    env_out = snd._fades(2 * RATE, 0.0, 2.0, RATE)
    assert env_in[0] == 0.0 and env_out[-1] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(env_in[1:] ** 2 + env_out[:-1] ** 2, 1.0, atol=1e-3)
    mid = one.loudness(1.5, 2.5)
    assert one.loudness(0.0, 1.0) - mid == pytest.approx(two.loudness(3.0, 4.0) - two.loudness(
        1.5, 2.5), abs=0.5)  # the fades mirror each other  # fmt: skip
    assert np.abs(out[round(0.5 * RATE) : round(0.8 * RATE)]).max() > 0  # one plays alone
    assert not out[round(4.0 * RATE) + 10 :].any()  # nothing after the last bed ends


def test_a_bed_that_does_not_loop_stops_with_its_file(tmp_path):
    s = scene(tmp_path)
    s.marker("end", 3.0)
    write_audio(tmp_path / "short.wav", sine(300, 1.0, -6.0), RATE)
    s.bed("short.wav", loop=False, fade=0.1)
    m = s.engine.sound_mix()
    assert m.lane("bed_short").t1 == pytest.approx(1.0)
    assert not m.samples[RATE + 10 :].any()


def test_cues_land_on_their_sample_with_their_pan(tmp_path):
    s = scene(tmp_path)
    s.marker("hit", 1.0)
    s.marker("end", 2.0)
    s.cue(("hit", 0.01), "pop", pan=-1.0)  # hard left
    m = s.engine.sound_mix()
    i0 = round(1.01 * RATE)
    assert not m.samples[:i0].any() and m.samples[i0:, 0].any()
    assert not m.samples[:, 1].any()  # nothing on the right
    assert m.lane("cue_pop_hit").peak_t == pytest.approx(1.01, abs=0.01)


@pytest.mark.parametrize("name", sorted(snd.DEFAULTS))
def test_each_built_in_sits_at_its_nominal_loudness(tmp_path, name):
    """Step 3: within 0.5 LU of the level it is set to (voice target + cue gain).
    Estimated narration: no trim, so the cue sits where it will under a real voice."""
    s = scene(tmp_path)
    s.narrate("Short line [[go]] here.", voice="estimate")
    s.marker("cue", s.engine.duration + 1.0)
    s.cue("cue", name)
    m = s.engine.sound_mix()
    i0 = round(m.lane(f"cue_{name}_cue").t0 * RATE)
    heard = snd.momentary_max(m.samples[i0:].astype(np.float64) / 32768, RATE)
    assert heard == pytest.approx(LEVELS["target"] + LEVELS["cue"], abs=0.5)


@pytest.mark.parametrize("name", sorted(snd.DEFAULTS))
def test_a_built_in_is_a_pure_function_of_its_id_and_the_seed(name):
    def digest(cue_id, seed):
        x = snd.builtin(Cue(id=cue_id, at=0.0, sound=name), seed)
        return hashlib.sha256(x.tobytes()).hexdigest()

    snd._builtin.cache_clear()
    first = digest("a", 0)
    snd._builtin.cache_clear()
    assert digest("a", 0) == first  # made again from nothing: the same samples
    if name != "tone":  # tone has no noise in it; the others vary with id and seed
        assert digest("b", 0) != first and digest("a", 1) != first
    x = snd.builtin(Cue(id="a", at=0.0, sound=name), 0)
    assert x.dtype == np.float32 and np.abs(x).max() == pytest.approx(1.0)
    assert len(x) == round(snd.builtin_seconds(Cue(id="a", at=0.0, sound=name)) * RATE)


def test_the_tone_built_in_plays_its_pitch(tmp_path):
    x = snd.builtin(Cue(id="t", at=0.0, sound="tone", midi=69, dur=0.5), 0).astype(np.float64)
    assert x[0] == 0 and x[-1] == 0
    hz = np.fft.rfftfreq(len(x), 1 / RATE)[np.argmax(np.abs(np.fft.rfft(x)))]
    assert hz == pytest.approx(440, abs=3)


def test_the_mix_is_deterministic(tmp_path):
    write_audio(tmp_path / "pad.ogg", sine(220, 2.0, -6.0, rate=24000), 24000)

    def build():
        s = scene(tmp_path)
        s.narrate("One line [[go]] here, then [[next]] another.", voice="estimate")
        s.bed("pad.ogg")
        s.cue("go", "whoosh", offset=-0.2)
        s.cue("next", "chime")
        return s.engine

    a = build()
    digest = hashlib.sha256(a.audio_track()[0].tobytes()).hexdigest()
    snd._builtin.cache_clear()
    snd._at_rate.cache_clear()
    snd._decode.cache_clear()
    assert hashlib.sha256(build().audio_track()[0].tobytes()).hexdigest() == digest


def test_an_edited_file_is_heard_at_once(tmp_path):
    import os

    s = scene(tmp_path)
    s.marker("end", 1.0)
    write_audio(tmp_path / "a.wav", sine(300, 1.0, -6.0), RATE)
    s.cue(0.0, "a.wav")
    first = s.engine.audio_track()[0].copy()
    write_audio(tmp_path / "a.wav", sine(600, 1.0, -6.0), RATE)
    st = os.stat(tmp_path / "a.wav")
    os.utime(tmp_path / "a.wav", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert not np.array_equal(s.engine.audio_track()[0], first)


# ---------------------------------------------------------------- lint (no OpenGL needed)


def test_missing_undecodable_and_cut_files_are_lint_warnings(tmp_path):
    s = scene(tmp_path)
    s.marker("end", 1.0)
    s.cue(0.5, "nope.wav")
    (tmp_path / "bad.wav").write_bytes(b"not audio")
    s.cue(0.25, "bad.wav")
    write_audio(tmp_path / "long.wav", sine(200, 1.0, -20.0), RATE)
    s.cue(0.5, "long.wav")
    s.bed("gone.ogg")
    by = {}
    for i in s.engine.lint():
        by.setdefault(i.code, set()).update(i.ids)
    assert by["missing_file"] == {"cue_nope_500ms", "bed_gone"}
    assert by["bad_sound_file"] == {"cue_bad_250ms"}
    assert by["sound_cut"] == {"cue_long_500ms"}
    samples, _ = s.engine.audio_track()  # the broken ones are silent; the good one plays
    assert samples[round(0.6 * RATE) : round(0.61 * RATE)].any() and len(samples) == RATE


def test_bed_masks_voice_fires_with_a_certain_fix(tmp_path):
    from phasor.engine import Engine

    write_audio(tmp_path / "pad.wav", sine(150, 2.0, -6.0), RATE)
    name = voice_wav(tmp_path, seconds=3.0)
    e = Engine(Document(meta=Meta(title="doc")), passes=None, base_dir=tmp_path)
    s = Scene.of(e)
    narrated(s, name, duration=3.0)
    s.bed("pad.wav", gain=-6.0)  # 6 under, ducked 8 more: 14 under, but masking is 12 ...
    assert "bed_masks_voice" not in {i.code for i in e.lint()}
    s.engine.apply({"op": "set_bed", "bed": {"id": "bed_pad", "path": "pad.wav", "gain": 0.0}})
    issue = next(i for i in e.lint() if i.code == "bed_masks_voice")
    assert issue.ids == ("bed_pad",) and issue.times and issue.fix
    assert e.autofix()["applied"]
    assert "bed_masks_voice" not in {i.code for i in e.lint()}
    assert e.document.bed("bed_pad").gain < 0.0


def test_a_scripted_bed_gets_a_hint_not_a_tweak(tmp_path):
    write_audio(tmp_path / "pad.wav", sine(150, 2.0, -6.0), RATE)
    s = scene(tmp_path)
    narrated(s, voice_wav(tmp_path, seconds=3.0), duration=3.0)
    s.bed("pad.wav", gain=3.0)
    issue = next(i for i in s.engine.lint() if i.code == "bed_masks_voice")
    assert not issue.fix and "gain=" in issue.fix_hint


def test_cue_on_word_and_cue_unanchored(tmp_path):
    s = scene(tmp_path)
    narrated(s, voice_wav(tmp_path), words=(Word(text="a", start=0.0, end=0.1),
                                            Word(text="wonderful", start=0.5, end=1.0)))  # fmt: skip
    s.cue(("go", 0.5), "tick")  # on "wonderful"
    s.cue(("go", 0.05), "tick", id="on_short_word")  # "a" carries no stress
    s.cue(3.0, "tick", id="raw")
    codes = {(i.code, i.ids) for i in s.engine.lint()}
    assert ("cue_on_word", ("cue_tick_go",)) in codes
    assert not any(ids == ("on_short_word",) for _, ids in codes)
    assert not any(c == "cue_unanchored" for c, _ in codes)  # strict only
    strict = {(i.code, i.ids) for i in s.engine.lint(strict=True)}
    assert ("cue_unanchored", ("raw",)) in strict
    assert ("cue_unanchored", ("cue_tick_go",)) not in strict


def test_audio_clip_reports_a_pile_up_before_the_limiter(tmp_path):
    s = scene(tmp_path)
    s.narrate("Then [[hit]] it all at once.", voice="estimate")  # no trim: where a voice would be
    for k in range(6):
        s.cue("hit", "pop", gain=6.0, id=f"pop_{k}")
    issue = next(i for i in s.engine.lint() if i.code == "audio_clip")
    hit = s.engine.document.timeline.markers["hit"]
    assert issue.times and issue.times[0] == pytest.approx(hit, abs=0.05)
    m = s.engine.sound_mix()
    assert m.pre_peak_db > LEVELS["clip"] and m.peak_dbtp <= LEVELS["ceiling"] + 0.1


@pytest.mark.parametrize("passes", MODES)
def test_the_sound_rules_fire_alike_with_and_without_opengl(tmp_path, passes):
    """CI on Windows has no OpenGL: the sound rules need none, and give the same issues."""
    from tests.conftest import gl_available

    if passes == "auto" and not gl_available():
        pytest.skip("needs OpenGL")
    write_audio(tmp_path / "pad.wav", sine(150, 2.0, -6.0), RATE)
    s = Scene("modes", passes=passes, base_dir=tmp_path)
    narrated(s, voice_wav(tmp_path, seconds=3.0), duration=3.0, words=(
        Word(text="listen", start=0.0, end=0.5), Word(text="closely", start=0.6, end=3.0)))  # fmt: skip
    s.bed("pad.wav", gain=3.0)
    s.cue(("go", 0.6), "tick")
    s.cue(4.0, "missing.wav")
    for k in range(6):
        s.cue(("go", 3.5), "pop", gain=6.0, id=f"pile_{k}")
    codes = {i.code for i in s.engine.lint(strict=True)}
    assert {"bed_masks_voice", "cue_on_word", "missing_file", "cue_unanchored"} <= codes


# ---------------------------------------------------------------- the AI's tools


def test_the_cue_and_bed_tools_add_and_undo_and_the_report_has_numbers(tmp_path):
    from phasor.workspace.tools import ToolError, Workspace

    ws = Workspace(tmp_path)
    ws.scene_new("tool")
    ws.narrate("Listen [[ping]] now, and [[later]] again.", voice="estimate", pace=False)
    r = ws.cue("ping", "chime", offset=0.1, midi=86)
    assert r["id"] == "cue_chime_ping" and r["t"] == pytest.approx(
        ws.engine.document.timeline.markers["ping"] + 0.1)  # fmt: skip
    write_audio(tmp_path / "pad.ogg", sine(220, 1.0, -6.0), RATE)
    assert ws.bed("pad.ogg", start="ping")["id"] == "bed_pad"
    with pytest.raises(ToolError):
        ws.cue("ping", "../outside.wav")
    with pytest.raises(ToolError, match="takes no voice"):
        ws.cue("ping", "chime", voice="bell")
    summary = ws.summary()
    assert summary["cues"] == ["cue_chime_ping"] and summary["beds"] == ["bed_pad"]
    report = ws.sound_report()
    assert report["mixed"] and {x["id"] for x in report["lanes"]} == {"bed_pad", "cue_chime_ping"}
    bed = next(x for x in report["lanes"] if x["id"] == "bed_pad")
    assert len(bed["blocks"]) == len(report["blocks"]) and bed["duck_db"][0] == pytest.approx(
        -LEVELS["duck"], abs=0.5)  # fmt: skip
    assert isinstance(report["issues"], list)
    ws.undo()
    assert ws.summary()["beds"] == []
    assert ws.scene_edit({"op": "remove_sound", "id": "cue_chime_ping"})["ok"]
    assert ws.engine.document.sound.cues == ()


# ---------------------------------------------------------------- loudness lanes (storyboard)


def lanes_scene(tmp_path, passes=None):
    from phasor import Dot

    write_audio(tmp_path / "pad.ogg", sine(220, 2.0, -6.0, rate=24000), 24000)
    s = Scene("lanes", passes=passes, base_dir=tmp_path)
    s.add(Dot(id="dot", pos=(0, 0)))
    s.narrate("A dot appears [[dot]] here. [[pause 2]] Then it rings [[ring]] out.",
              voice="estimate")  # fmt: skip
    s.at("dot").reveal("dot")
    s.bed("pad.ogg", fade=0.5)
    s.cue("ring", "chime")
    return s


def test_the_loudness_lanes_show_the_bed_dipping_under_speech(tmp_path):
    s = lanes_scene(tmp_path)
    img = s.engine.sound_lanes(800, [0.0, 1.0])
    assert img.dtype == np.uint8 and img.shape[1] == 800 and img.shape[2] == 3
    from phasor.render.storyboard import LANE_COLORS, LANE_H, PAD

    rows = img.shape[0] - PAD - 16
    assert rows == 2 * (LANE_H + 4)  # the bed and the cues (no recorded voice yet)
    bed = img[PAD : PAD + LANE_H]
    drawn = np.all(bed == LANE_COLORS["bed"], axis=-1)
    heights = [np.flatnonzero(drawn[:, x]).min() if drawn[:, x].any() else None
               for x in range(130, 790)]  # fmt: skip
    ys = [h for h in heights if h is not None]
    assert ys and max(ys) - min(ys) >= LANE_H * 8 / 50 * 0.8  # the 8 dB dip shows
    m = s.engine.sound_mix()
    assert m.lane("bed_pad") is not None and not m.lane("voice")


@pytest.mark.gl
def test_a_storyboard_has_the_loudness_lanes_under_its_stills(tmp_path):
    from tests.conftest import gl_available

    if not gl_available():
        pytest.skip("needs OpenGL")
    s = lanes_scene(tmp_path, passes="auto")
    with_sound = s.engine.storyboard(times=[0.0, 1.0, 2.0], size=(160, 90))
    assert with_sound.get("sound_lanes")
    plain = Scene("plain", base_dir=tmp_path)
    from phasor import Dot

    plain.add(Dot(id="dot", pos=(0, 0)))
    without = plain.engine.storyboard(times=[0.0, 1.0, 2.0], size=(160, 90))
    assert "sound_lanes" not in without
    assert with_sound["image"].shape[0] > without["image"].shape[0]
    assert with_sound["image"].shape[1] == without["image"].shape[1]


# ---------------------------------------------------------------- the rendered video


@pytest.mark.gl
def test_a_rendered_video_carries_the_mix_in_stereo(tmp_path):
    from tests.conftest import gl_available

    if not gl_available():
        pytest.skip("needs OpenGL")
    av = pytest.importorskip("av")
    from phasor import Dot
    from phasor.render.job import render_file

    s = Scene("beep", base_dir=tmp_path)
    s.add(Dot(id="dot", pos=(0, 0)))
    s.marker("go", 0.25)
    s.cue("go", "tone", pan=1.0, dur=0.25)
    out = tmp_path / "beep.mp4"
    r = render_file(s.engine, out, preset="draft", end=1.0)
    assert r["audio"] and r["sounds"] == 1 and not r["narration"]
    with av.open(str(out)) as c:
        assert [st.type for st in c.streams] == ["video", "audio"]
        assert c.streams.audio[0].layout.name == "stereo"
        pcm = np.concatenate([f.to_ndarray().reshape(-1, 2) if f.format.is_packed
                              else f.to_ndarray().T for f in c.decode(audio=0)])  # fmt: skip
    left, right = np.abs(pcm[:, 0]).max(), np.abs(pcm[:, 1]).max()
    assert right > 0.05 and left < 0.1 * right  # panned right
    loud = np.flatnonzero(np.abs(pcm[:, 1]) > 0.5 * right)
    assert abs(loud[0] / 48000 - 0.25) < 0.05 and loud[-1] / 48000 < 0.6
