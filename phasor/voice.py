"""Voices: where narration audio and timing come from.

- ``estimate``: no audio; timing from the text at the theme's speaking rate.
- ``espeak[:lang[:wpm]]``: WAV via espeak-ng; exact markers (each piece is its own clip).
- ``piper:<model.onnx>``: WAV via Piper; exact markers the same way.
- recordings: your own WAVs; words placed by WhisperX, or read from a timings file.

Synthesizing piece by piece makes every marker exact without any alignment model. The
price is a tiny seam in the voice at each marker. Clips are cached by (text, voice), so
re-running a script re-synthesizes only sentences that changed.

This module does I/O (subprocesses, files), so it sits beside ``phasor.scene`` at the top of
the layer order and nothing below imports it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from phasor.core.proc import NO_WINDOW
from phasor.core.schema import Narration, Word
from phasor.style.theme import Theme
from phasor.time.narration import (
    Block,
    ScriptError,
    Spoken,
    build,
    estimate_duration,
    parse_script,
    read_wav,
    write_wav,
)


class VoiceError(RuntimeError):
    pass


class Synth(Protocol):
    name: str

    def synthesize(self, text: str, path: Path) -> None:
        """Write ``text`` spoken as a mono 16-bit WAV at ``path``."""
        ...


@dataclass(frozen=True)
class Espeak:
    lang: str = "en-us"
    wpm: int = 160

    @property
    def name(self) -> str:
        return f"espeak:{self.lang}:{self.wpm}"

    def synthesize(self, text: str, path: Path) -> None:
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if exe is None:
            raise VoiceError("espeak-ng is not installed (apt install espeak-ng, or the "
                             "Windows installer from github.com/espeak-ng)")  # fmt: skip
        r = subprocess.run([exe, "-v", self.lang, "-s", str(self.wpm), "-w", str(path), text],
                           capture_output=True, text=True, creationflags=NO_WINDOW)  # fmt: skip
        if r.returncode != 0:
            raise VoiceError(f"espeak-ng failed: {r.stderr.strip()}")


@dataclass(frozen=True)
class Piper:
    """Untested here: the model files could not be downloaded in the build environment."""

    model: str

    @property
    def name(self) -> str:
        return f"piper:{self.model}"

    def synthesize(self, text: str, path: Path) -> None:
        exe = shutil.which("piper")
        if exe is None:
            raise VoiceError("piper is not installed (pip install piper-tts)")
        r = subprocess.run([exe, "--model", self.model, "--output_file", str(path)],
                           input=text, capture_output=True, text=True,
                           creationflags=NO_WINDOW)  # fmt: skip
        if r.returncode != 0:
            raise VoiceError(f"piper failed: {r.stderr.strip()[-400:]}")


def espeak_available() -> bool:
    return bool(shutil.which("espeak-ng") or shutil.which("espeak"))


def parse_voice(voice: str) -> Synth | None:
    """``auto`` (espeak when installed, else estimate), ``estimate``, ``espeak[:lang[:wpm]]``
    or ``piper:<model>``. None means estimated timing with no audio."""
    if voice == "auto":
        voice = "espeak" if espeak_available() else "estimate"
    if voice == "estimate":
        return None
    kind, _, rest = voice.partition(":")
    if kind == "espeak":
        parts = rest.split(":") if rest else []
        return Espeak(parts[0] if parts else "en-us", int(parts[1]) if len(parts) > 1 else 160)
    if kind == "piper" and rest:
        return Piper(rest)
    raise VoiceError(f"unknown voice {voice!r}; use auto, estimate, espeak[:lang[:wpm]] "
                     "or piper:<model.onnx>")  # fmt: skip


def _trim(samples: np.ndarray, rate: int, keep: float = 0.03) -> np.ndarray:
    """Cut silence at both ends (synthesizers pad clips), keeping ``keep`` seconds."""
    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > 300)
    if loud.size == 0:
        return samples[:0]
    pad = int(keep * rate)
    return samples[max(loud[0] - pad, 0) : loud[-1] + pad + 1]


def _speak_block(synth: Synth, block: Block, folder: Path) -> Spoken:
    """Synthesize each piece separately and join them into one clip for the block."""
    key = hashlib.sha1(json.dumps([synth.name, block.pieces]).encode()).hexdigest()[:16]
    clip = folder / f"{key}.wav"
    meta = folder / f"{key}.json"
    if clip.exists() and meta.exists():
        return Spoken(tuple(json.loads(meta.read_text(encoding="utf-8"))), clip.name)
    parts, rate = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for i, piece in enumerate(block.pieces):
            path = Path(tmp) / f"{i}.wav"
            synth.synthesize(piece, path)
            samples, r = read_wav(path)
            if rate not in (None, r):
                raise VoiceError("the voice changed sample rate between pieces")
            rate = r
            parts.append(_trim(samples, r))
    assert rate is not None
    durations = [len(p) / rate for p in parts]
    write_wav(clip, np.concatenate(parts), rate)
    meta.write_text(json.dumps(durations), encoding="utf-8")
    return Spoken(tuple(durations), clip.name)


def narrate(script: str, voice: str, folder: Path, theme: Theme) -> Narration:
    """Speak (or time) a script. Audio clips go to ``folder``; the ``Narration`` refers to
    them by path relative to ``folder.parent`` (the scene's folder)."""
    blocks = parse_script(script)
    synth = parse_voice(voice)
    n = theme.narration
    spoken: list[Spoken | None] = []
    for block in blocks:
        if block.pause is not None:
            spoken.append(None)
        elif synth is None:
            spoken.append(Spoken(tuple(estimate_duration(p, n["words_per_second"])
                                       for p in block.pieces)))  # fmt: skip
        else:
            folder.mkdir(parents=True, exist_ok=True)
            said = _speak_block(synth, block, folder)
            rel = f"{folder.name}/{said.audio}"
            spoken.append(Spoken(said.durations, rel))
    name = synth.name if synth else "estimate"
    return build(script, name, blocks, spoken, lead=n["lead"], gap=n["gap"])


# ------------------------------------------------------------------ recordings


class Aligner(Protocol):
    def align(self, wav: Path, text: str) -> list[Word]:
        """Word timings (seconds from the start of the clip) for a recording of ``text``."""
        ...


@dataclass(frozen=True)
class TimingsFile:
    """Word timings from a JSON file per clip (``clip.json``: ``[{"text", "start", "end"}]``),
    e.g. exported from any aligner or editor."""

    def align(self, wav: Path, text: str) -> list[Word]:
        path = wav.with_suffix(".json")
        if not path.exists():
            raise VoiceError(f"no word timings at {path}")
        return [Word.model_validate(w) for w in json.loads(path.read_text(encoding="utf-8"))]


@dataclass(frozen=True)
class WhisperX:
    """Forced alignment with WhisperX. Untested here: it needs model downloads."""

    language: str = "en"
    device: str = "cuda"

    def align(self, wav: Path, text: str) -> list[Word]:
        try:
            import whisperx
        except ImportError as e:
            raise VoiceError("whisperx is not installed (pip install whisperx)") from e
        audio = whisperx.load_audio(str(wav))
        model, meta = whisperx.load_align_model(language_code=self.language, device=self.device)
        segments = [{"text": text, "start": 0.0, "end": len(audio) / 16000}]
        out = whisperx.align(segments, model, meta, audio, self.device)
        return [Word(text=w["word"], start=w["start"], end=w["end"])
                for w in out["word_segments"] if "start" in w]  # fmt: skip


def _norm(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalnum())


def from_recordings(
    script: str, clips: Sequence[str | Path], aligner: Aligner, base_dir: Path, theme: Theme
) -> Narration:
    """Use your own recordings, one WAV per spoken sentence in order. Markers fall where
    the aligner says the next word starts."""
    blocks = parse_script(script)
    spoken_blocks = [b for b in blocks if b.pause is None]
    if len(clips) != len(spoken_blocks):
        raise ScriptError(f"the script has {len(spoken_blocks)} sentences but "
                          f"{len(clips)} recordings were given")  # fmt: skip
    spoken: list[Spoken | None] = []
    it = iter(clips)
    for block in blocks:
        if block.pause is not None:
            spoken.append(None)
            continue
        clip = Path(next(it))
        samples, rate = read_wav(clip)
        words = aligner.align(clip, block.text)
        counts = [len(p.split()) for p in block.pieces]
        if len(words) != sum(counts) or any(
            _norm(w.text) != _norm(s) for w, s in zip(words, block.text.split(), strict=False)
        ):
            raise VoiceError(f"{clip.name}: the aligned words do not match the sentence "
                             f"{block.text!r}")  # fmt: skip
        total = len(samples) / rate
        bounds, k = [0.0], 0
        for c in counts[:-1]:
            k += c
            bounds.append(words[k].start)
        bounds.append(total)
        durations = tuple(b - a for a, b in itertools.pairwise(bounds))
        rel = clip.resolve().relative_to(base_dir.resolve()).as_posix()
        spoken.append(Spoken(durations, rel, tuple(words)))
    n = theme.narration
    return build(script, "recorded", blocks, spoken, lead=n["lead"], gap=n["gap"])
