"""The spectrogram of any signal, a formula in t or a .wav file next to
the scene, with a glowing playhead."""

from __future__ import annotations

import math
import runpy
import wave
from pathlib import Path

import numpy as np
import pytest

from phasor import Engine, Scene
from phasor.core.issues import DocumentError
from phasor.core.schema import Document, Meta
from phasor.domains import fourier_math as fm
from phasor.domains.fourier import Spectrogram
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _write_wav(path: Path, x: np.ndarray, fs: int, channels: int = 1) -> Path:
    pcm = np.round(np.clip(x, -1, 1) * 32000).astype("<i2")
    if channels == 2:
        pcm = np.stack([pcm, pcm], axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())
    return path


def _scene(tmp_path: Path, **params) -> Scene:
    s = Scene("sg", passes=None)
    s.engine = Engine(Document(meta=Meta(title="sg")), passes=None, base_dir=tmp_path)
    s.add(Spectrogram(id="sg", **params))
    return s


def _sem(e) -> dict:
    return e.inspect(0.0, id="sg")["objects"][0]["semantic"]


# ---------------------------------------------------------------- the maths


def test_a_pure_tone_peaks_in_its_bin_and_nowhere_else():
    fs, nwin = 4000.0, 256
    x = fm.sinusoids([750.0], [1.0], 4000, fs)  # 750 Hz = bin 48 exactly (df = 15.625)
    sg = fm.spectrogram(x, fs, nwin)
    assert np.all(np.argmax(sg["mag"], axis=0) == 48)
    assert sg["mag"][48].mean() == pytest.approx(1.0, rel=0.02)  # amplitude-corrected
    assert fm.dominant_frequency(sg, 0.5) == pytest.approx(750.0, abs=0.5)


def test_an_off_bin_tone_is_found_between_bins():
    fs = 4000.0
    sg = fm.spectrogram(fm.sinusoids([803.0], [1.0], 4000, fs), fs, 256)
    assert fm.dominant_frequency(sg, 0.5) == pytest.approx(803.0, abs=2.0)  # a bin is 15.6 wide


def test_a_chirps_ridge_rises_along_the_sweep():
    fs = 8000.0
    sg = fm.spectrogram(fm.chirp(300.0, 2300.0, 2.0, fs), fs, 512)
    ridge = sg["freq"][np.argmax(sg["mag"], axis=0)]
    assert np.all(np.diff(ridge) >= 0)
    assert fm.dominant_frequency(sg, 1.0) == pytest.approx(1300.0, abs=40.0)


def test_window_and_hop_change_the_frames_not_the_pitch():
    fs = 4000.0
    x = fm.sinusoids([500.0], [1.0], 4000, fs)
    a = fm.spectrogram(x, fs, 256, hop=64)
    b = fm.spectrogram(x, fs, 256, hop=128, win="blackman")
    assert b["mag"].shape[1] < a["mag"].shape[1]
    for sg in (a, b):
        assert fm.dominant_frequency(sg, 0.5) == pytest.approx(500.0, abs=1.0)
    with pytest.raises(ValueError, match="unknown window"):
        fm.spectrogram(x, fs, 256, win="nope")


def test_a_signal_shorter_than_the_window_still_gives_one_frame():
    sg = fm.spectrogram(np.ones(40), 1000.0, 256)
    assert sg["mag"].shape == (129, 1)


def test_the_old_default_is_unchanged():
    fs = 4000.0
    x = fm.chirp(200.0, 1200.0, 1.0, fs)
    w = np.hanning(256)
    old = np.array(
        [np.abs(np.fft.rfft(x[i : i + 256] * w)) for i in range(0, len(x) - 256 + 1, 64)]
    ).T / (w.sum() / 2)
    np.testing.assert_allclose(fm.spectrogram(x, fs, 256)["mag"], old)


def test_sampling_a_formula_is_the_function_at_k_over_fs():
    x = fm.sampled(lambda t: 2 * t, 0.5, 100.0)
    np.testing.assert_allclose(x, 2 * np.arange(50) / 100.0)


# ---------------------------------------------------------------- the object


def test_a_formula_in_t_is_the_signal(tmp_path):
    s = _scene(tmp_path, signal="sin(tau * 600 * t)", duration=1.0, fs=4000.0, playhead=0.5)
    sem = _sem(s.engine)
    assert sem["source"] == "formula" and sem["max_frequency"] == 2000.0
    assert sem["playhead_time"] == pytest.approx(0.5)
    assert sem["dominant_frequency"] == pytest.approx(600.0, abs=2.0)


def test_the_playhead_moves_and_the_dominant_frequency_follows_it(tmp_path):
    s = _scene(
        tmp_path, signal="sin(tau * (200 * t + 500 * t * t))", duration=1.0, fs=4000.0, playhead=0.1
    )  # 200 Hz rising to 1200 Hz
    seen = []
    for u in (0.1, 0.5, 0.9):
        e = s.engine.fork()
        assert e.apply({"op": "set_param", "path": "sg.playhead", "value": u}, origin="script").ok
        seen.append(_sem(e)["dominant_frequency"])
    assert seen[0] < seen[1] < seen[2]
    assert seen[1] == pytest.approx(700.0, abs=60.0)


def test_the_playhead_can_follow_the_clock(tmp_path):
    from phasor import expr

    s = _scene(
        tmp_path,
        signal="sin(tau * (200 * t + 500 * t * t))",
        duration=1.0,
        fs=4000.0,
        playhead=expr("t / 4"),
    )
    s.marker("end", 4.0)
    early = s.engine.inspect(0.4, id="sg")["objects"][0]["semantic"]["dominant_frequency"]
    late = s.engine.inspect(3.6, id="sg")["objects"][0]["semantic"]["dominant_frequency"]
    assert early < late


def test_no_playhead_means_no_dominant_frequency(tmp_path):
    s = _scene(tmp_path, signal="sin(tau * 300 * t)")
    assert _sem(s.engine)["dominant_frequency"] is None


def test_the_default_is_still_the_chirp(tmp_path):
    s = _scene(tmp_path, f0=300, f1=900, playhead=0.5)
    sem = _sem(s.engine)
    assert sem["source"] == "chirp" and sem["dominant_frequency"] == pytest.approx(600, abs=40)


def test_a_wav_next_to_the_scene_roundtrips(tmp_path):
    fs = 8000
    t = np.arange(fs) / fs
    _write_wav(tmp_path / "tone.wav", 0.8 * np.sin(2 * math.pi * 1000 * t), fs)
    s = _scene(tmp_path, wav="tone.wav", playhead=0.5)
    sem = _sem(s.engine)
    assert sem["source"] == "wav" and sem["duration"] == pytest.approx(1.0)
    assert sem["max_frequency"] == 4000.0
    assert sem["dominant_frequency"] == pytest.approx(1000.0, abs=2.0)


def test_a_stereo_wav_is_mixed_down_and_the_axis_can_be_cropped(tmp_path):
    fs = 16000
    t = np.arange(fs // 2) / fs
    _write_wav(tmp_path / "st.wav", np.sin(2 * math.pi * 440 * t), fs, channels=2)
    s = _scene(tmp_path, wav="st.wav", fmax=2000.0, playhead=0.5)
    sem = _sem(s.engine)
    assert sem["max_frequency"] == 2000.0
    assert sem["dominant_frequency"] == pytest.approx(440.0, abs=4.0)


def test_a_missing_wav_is_missing_file_and_draws_nothing(tmp_path):
    s = Scene("m", passes=None)
    s.engine = Engine(Document(meta=Meta(title="m")), passes=None, base_dir=tmp_path)
    s.add(Spectrogram(id="a", wav="gone.wav"))
    (issue,) = [i for i in s.engine.lint() if i.code == "missing_file"]
    assert "gone.wav" in issue.message
    st = s.engine.state_at(0.0)
    assert st["a"].impl.semantics(st)["missing_file"] == "gone.wav"
    with pytest.raises(DocumentError, match="notes.txt"):
        s.add(Spectrogram(id="b", wav="notes.txt"))


def test_a_bad_formula_is_refused(tmp_path):
    s = _scene(tmp_path)
    for bad, why in (
        ("sin(tau * x)", "unknown name"),
        ("sg.f0 * t", "formula in t"),
        ("sin(", "syntax"),
    ):
        with pytest.raises(DocumentError, match=why):
            s.add(Spectrogram(id="z", signal=bad))
    with pytest.raises(DocumentError, match="too long"):
        s.add(Spectrogram(id="y", signal="t", duration=500.0, fs=44100.0))


def test_a_formula_and_a_wav_together_are_refused(tmp_path):
    _write_wav(tmp_path / "a.wav", np.zeros(800), 8000)
    with pytest.raises(DocumentError, match="not both"):
        _scene(tmp_path, wav="a.wav", signal="sin(t)")


def test_the_same_scene_gives_the_same_numbers(tmp_path):
    sig = "sin(tau * 440 * t) + 0.5 * sin(tau * 880 * t)"
    a = _scene(tmp_path, signal=sig, playhead=0.3)
    b = _scene(tmp_path, signal=sig, playhead=0.3)
    assert a.engine.inspect(0.0, id="sg") == b.engine.inspect(0.0, id="sg")


def test_the_wav_travels_with_the_scene_when_it_is_saved(tmp_path):
    fs = 8000
    _write_wav(tmp_path / "tone.wav", np.sin(np.arange(fs) * 0.3), fs)
    s = _scene(tmp_path, wav="tone.wav")
    out = tmp_path / "elsewhere"
    out.mkdir()
    s.engine.save(out / "x.phasor.json")
    assert (out / "tone.wav").is_file()


def test_the_json_path_refuses_what_the_python_path_refuses(tmp_path):
    """A patch from the AI or a file is validated the same way as a script's object."""
    _write_wav(tmp_path / "a.wav", np.zeros(800), 8000)
    (tmp_path / "a.txt").write_text("not sound", encoding="utf-8")
    s = _scene(tmp_path)
    cases = [({"signal": "sg.f0 * t"}, "formula in t"), ({"signal": "foo(t)"}, "unknown function"),
             ({"signal": "sin(t)", "wav": "a.wav"}, "not both"),
             ({"signal": "t", "duration": 500, "fs": 44100}, "too long"),
             ({"wav": "a.txt"}, "not a .wav")]  # fmt: skip
    for i, (params, why) in enumerate(cases):
        op = {"op": "add_object", "object": {"id": f"z{i}", "type": "spectrogram",
                                             "params": params}}  # fmt: skip
        r = s.engine.apply(op, origin="script")
        assert not r.ok and why in " ".join(e.message for e in r.errors), (params, r.errors)


def test_a_later_edit_cannot_make_the_formula_too_long(tmp_path):
    s = _scene(tmp_path, signal="sin(t)", duration=1.0, fs=4000.0)
    r = s.engine.apply({"op": "set_param", "path": "sg.duration", "value": 500.0}, origin="script")
    assert not r.ok and "too long" in r.errors[0].message


def test_the_ai_cannot_use_a_wav_from_outside_its_folder(tmp_path):
    from phasor.workspace import ToolError, Workspace

    _write_wav(tmp_path / "outside.wav", np.zeros(800), 8000)
    root = tmp_path / "project"
    root.mkdir()
    _write_wav(root / "inside.wav", np.zeros(800), 8000)
    ws = Workspace(root)
    ws.scene_new("sounds")
    ok = ws.scene_edit({"op": "add_object", "object": {"id": "a", "type": "spectrogram",
                        "params": {"wav": "inside.wav"}}})  # fmt: skip
    assert ok["ok"]
    with pytest.raises(ToolError, match="leaves the scene's folder"):
        ws.scene_edit({"op": "add_object", "object": {"id": "b", "type": "spectrogram",
                       "params": {"wav": "../outside.wav"}}})  # fmt: skip
    assert [o.id for o in ws.engine.document.objects] == ["a"]


# ---------------------------------------------------------------- the picture


@needs_gl
def test_the_playhead_is_a_glowing_line_where_it_says():
    s = Scene("ph", look="prism")
    s.set("camera.height", 4.0)
    s.add(Spectrogram(id="sg", signal="sin(tau * 600 * t)", size=(6.0, 3.0), playhead=0.75))
    _pic, emit = s.engine.pass_runner().color_and_emission(s.engine.state_at(0.0), (240, 135))
    # world x = -3 + 0.75 * 6 = 1.5 -> pixel column 120 + 1.5 * (135 / 4) = 170
    col = emit[..., :3].sum(axis=(0, 2))
    assert abs(int(np.argmax(col)) - 170) <= 2
    assert col[170] > 3 * np.median(col)


@needs_gl
@pytest.mark.slow
def test_spectrogram_golden_storyboard():
    from tests.test_beauty import _golden

    e = runpy.run_path(str(ROOT / "examples" / "prism_spectrogram.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([0.5, 3.0, 5.5], size=(320, 180))["image"]
    _golden("prism_spectrogram", sheet, e.pass_runner().renderer_class())
