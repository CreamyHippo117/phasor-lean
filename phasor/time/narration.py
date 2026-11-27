"""Narration as time. Pure: no audio libraries.

A script is plain text with markers:

    We start with a function that has two poles [[poles_appear]].
    Draw a small loop around [[loop_draw]] the one at i.
    [[pause 1.5]]
    Only that pole contributes [[residue_glow]].

Each sentence is a *block* (a macroblock). Markers cut a block into *pieces* (microblocks);
a marker's time is where the piece after it starts. ``[[pause s]]`` is a silent block.

How long each piece lasts comes from a voice (``phasor.voice``): synthesized piece by
piece (exact), aligned from a recording, or estimated from the text. This module turns
those durations into a ``Narration`` (core.schema) whose markers become the TimeMap.
"""

from __future__ import annotations

import re
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from phasor.core.ids import is_valid_name
from phasor.core.schema import Narration, NarrationBlock, Word

_TAG = re.compile(r"\[\[\s*([^\]]*?)\s*\]\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PAUSE = re.compile(r"pause\s+(\d+(?:\.\d+)?)")
_WORD = re.compile(r"\w")


class ScriptError(ValueError):
    pass


@dataclass(frozen=True)
class Block:
    """A sentence split at its markers, or a pause (``pause`` set, no pieces)."""

    pieces: tuple[str, ...] = ()
    markers: Mapping[str, int] = field(default_factory=dict)  # name -> piece index it starts
    pause: float | None = None

    @property
    def text(self) -> str:
        return " ".join(self.pieces)


def _split_block(sentence: str, seen: set[str]) -> Block:
    pieces: list[str] = []
    markers: dict[str, int] = {}
    pos = 0
    for m in _TAG.finditer(sentence):
        pieces.append(sentence[pos : m.start()])
        name = m.group(1)
        if not is_valid_name(name):
            raise ScriptError(f"bad marker [[{name}]]: use a name like loop_draw, or [[pause 1.5]]")
        if name in seen:
            raise ScriptError(f"marker [[{name}]] is used twice")
        seen.add(name)
        markers[name] = len(pieces)
        pos = m.end()
    pieces.append(sentence[pos:])
    # A piece with no words (". " after a final marker) joins the piece before it; markers
    # that pointed at it now mean "the end of the piece before".
    out: list[str] = []
    for piece in pieces:
        piece = " ".join(piece.split())
        if _WORD.search(piece) or not out:
            out.append(piece)
            continue
        out[-1] = (out[-1] + piece).strip()
        for name, k in markers.items():
            if k > len(out):
                markers[name] = k - 1
    if not _WORD.search(out[0]):  # nothing spoken before the first marker
        if len(out) > 1:
            out.pop(0)
            markers = {n: max(k - 1, 0) for n, k in markers.items()}
        else:
            raise ScriptError(f"block {sentence.strip()!r} has no words")
    return Block(tuple(out), markers)


def parse_script(script: str) -> list[Block]:
    """Blocks in order. Sentences end at . ! ? or a line break; pauses stand alone."""
    blocks: list[Block] = []
    seen: set[str] = set()
    for line in script.splitlines():
        # A pause tag splits the line; text on either side is spoken normally.
        for part in re.split(r"(\[\[\s*pause\b[^\]]*\]\])", line):
            part = part.strip()
            if not part:
                continue
            tag = _TAG.fullmatch(part)
            if tag and tag.group(1).startswith("pause"):
                m = _PAUSE.fullmatch(tag.group(1))
                if m is None:
                    raise ScriptError(f"bad pause {part}: write [[pause 1.5]]")
                blocks.append(Block(pause=float(m.group(1))))
                continue
            for sentence in _SENTENCE_END.split(part):
                if sentence.strip():
                    blocks.append(_split_block(sentence, seen))
    if not blocks:
        raise ScriptError("the script is empty")
    return blocks


# ------------------------------------------------------------------ timing


def estimate_words(text: str, duration: float, offset: float = 0.0) -> tuple[Word, ...]:
    """Spread the words of ``text`` over ``duration`` in proportion to their length."""
    words = text.split()
    if not words:
        return ()
    weights = np.array([len(w) + 1 for w in words], dtype=np.float64)
    edges = offset + duration * np.concatenate([[0], np.cumsum(weights)]) / weights.sum()
    return tuple(Word(text=w, start=round(float(a), 4), end=round(float(b), 4))
                 for w, a, b in zip(words, edges[:-1], edges[1:], strict=True))  # fmt: skip


def estimate_duration(text: str, words_per_second: float) -> float:
    """Speaking time for ``text``: words at a steady rate plus short stops at punctuation."""
    n = len(text.split())
    stops = 0.15 * len(re.findall(r"[,;:]", text)) + 0.3 * len(re.findall(r"[.!?]", text))
    return n / words_per_second + stops


@dataclass(frozen=True)
class Spoken:
    """What a voice produced for one block: piece durations, optional audio and words."""

    durations: tuple[float, ...]
    audio: str | None = None
    words: tuple[Word, ...] = ()


def build(
    script: str,
    voice: str,
    blocks: Sequence[Block],
    spoken: Sequence[Spoken | None],
    *,
    lead: float,
    gap: float,
    extra: Sequence[float] = (),
) -> Narration:
    """Lay blocks end to end: ``lead`` seconds of silence first, ``gap`` between blocks,
    plus ``extra[k]`` after block k (pacing). ``spoken[k]`` is None for pauses."""
    out: list[NarrationBlock] = []
    t = lead
    for k, (block, said) in enumerate(zip(blocks, spoken, strict=True)):
        if block.pause is not None:
            out.append(NarrationBlock(text="", start=round(t, 4), duration=block.pause))
            t += block.pause + (extra[k] if k < len(extra) else 0.0)
            continue
        assert said is not None and len(said.durations) == len(block.pieces)
        starts = np.concatenate([[0.0], np.cumsum(said.durations)])
        markers = {name: round(float(starts[i]), 4) for name, i in block.markers.items()}
        duration = float(starts[-1])
        words = said.words
        if not words:
            words = tuple(w for piece, a, d in zip(block.pieces, starts[:-1], said.durations,
                                                   strict=True)
                          for w in estimate_words(piece, d, a))  # fmt: skip
        out.append(NarrationBlock(text=block.text, start=round(t, 4), duration=round(duration, 4),
                                  audio=said.audio, words=words, markers=markers))  # fmt: skip
        t += duration + gap + (extra[k] if k < len(extra) else 0.0)
    return Narration(script=script, voice=voice, blocks=tuple(out))


def repace(narration: Narration, extra: Sequence[float], gap: float, lead: float) -> Narration:
    """Move blocks so that block k is followed by ``gap + extra[k]`` (pauses: ``extra[k]``).
    Durations and audio stay; only start times, and so the markers, change."""
    out, t = [], lead
    for k, b in enumerate(narration.blocks):
        out.append(NarrationBlock.model_validate({**b.model_dump(), "start": round(t, 4)}))
        t += b.duration + (0.0 if not b.text else gap) + (extra[k] if k < len(extra) else 0.0)
    return Narration(script=narration.script, voice=narration.voice, blocks=tuple(out))


def move_block(narration: Narration, k: int, start: float) -> Narration:
    """Block ``k`` moved to start at ``start`` seconds, kept between its neighbours (it may
    not overlap the sentence before or after). Its words and markers move with it, and so
    does everything anchored to those markers. ``IndexError`` for a block that is not there."""
    blocks = narration.blocks
    b = blocks[k]
    lo = blocks[k - 1].start + blocks[k - 1].duration if k > 0 else 0.0
    hi = blocks[k + 1].start - b.duration if k + 1 < len(blocks) else float("inf")
    t = round(min(max(float(start), lo), max(hi, lo)), 4)
    moved = NarrationBlock.model_validate({**b.model_dump(), "start": t})
    out = (*blocks[:k], moved, *blocks[k + 1 :])
    return Narration(script=narration.script, voice=narration.voice, blocks=out)


def add_block(narration: Narration | None, block: NarrationBlock) -> Narration:
    """``narration`` with one more spoken block (your own recording, say) in time order.
    ``ValueError`` when it would overlap a block already there: two voices never talk at
    once. Its text joins the script, so the script still reads in order."""
    blocks = narration.blocks if narration else ()
    end = block.start + block.duration
    for k, b in enumerate(blocks):
        if b.text and block.start < b.start + b.duration and b.start < end:
            raise ValueError(f"it would overlap sentence {k} ({b.start:g}-{b.start + b.duration:g}"
                             " s); drop it before or after, or move that one")  # fmt: skip
    out = sorted((*blocks, block), key=lambda b: b.start)
    script = "\n".join(b.text for b in out if b.text)
    voice = "recorded" if narration is None else narration.voice
    return Narration(script=script, voice=voice, blocks=tuple(out))


def remove_block(narration: Narration, k: int) -> Narration | None:
    """``narration`` without block ``k`` (None when nothing is left). Its markers go too,
    so whatever was anchored to them must move first (the document refuses otherwise)."""
    out = tuple(b for i, b in enumerate(narration.blocks) if i != k)
    if not any(b.text for b in out):
        return None
    script = "\n".join(b.text for b in out if b.text)
    return Narration(script=script, voice=narration.voice, blocks=out)


def mark_block(narration: Narration, k: int, name: str, at: float) -> Narration:
    """``narration`` with a marker ``name`` at ``at`` seconds into block ``k``: it belongs to
    the block, so it moves when the block moves (a word in your recording, say)."""
    b = narration.blocks[k]
    markers = {**b.markers, name: round(min(max(at, 0.0), b.duration), 4)}
    moved = NarrationBlock.model_validate({**b.model_dump(), "markers": markers})
    blocks = (*narration.blocks[:k], moved, *narration.blocks[k + 1 :])
    return Narration(script=narration.script, voice=narration.voice, blocks=blocks)


def block_of(narration: Narration, t: float) -> int | None:
    """Index of the block that is playing, or last played, at time ``t``."""
    k = None
    for i, b in enumerate(narration.blocks):
        if b.start <= t + 1e-9:
            k = i
    return k


# ------------------------------------------------------------------ audio track


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono int16 samples and the sample rate. Stereo is mixed down."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit WAV is supported")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        return data, w.getframerate()


def write_wav(path: str | Path, samples: np.ndarray, rate: int) -> None:
    """16-bit WAV: ``samples`` shape (n,) mono or (n, channels)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def mix_track(
    narration: Narration, base_dir: str | Path, duration: float
) -> tuple[np.ndarray, int] | None:
    """The whole narration as one mono int16 track ``duration`` seconds long, each block's
    audio placed at its start. None when no block has audio (estimated timing)."""
    root = Path(base_dir).resolve()
    clips = []
    for b in narration.blocks:
        if not b.audio:
            continue
        path = (root / b.audio).resolve()
        if root not in path.parents:  # a document must not read files outside its folder
            raise ValueError(f"narration audio {b.audio!r} is outside the scene's folder")
        clips.append((b.start, read_wav(path)))
    if not clips:
        return None
    rates = {rate for _, (_, rate) in clips}
    if len(rates) != 1:
        raise ValueError(f"narration clips have different sample rates: {sorted(rates)}")
    rate = rates.pop()
    end = max(duration, max(t + len(s) / rate for t, (s, _) in clips))
    track = np.zeros(int(np.ceil(end * rate)), dtype=np.int32)
    for t, (samples, _) in clips:
        i = round(t * rate)
        track[i : i + len(samples)] += samples
    return np.clip(track, -32768, 32767).astype(np.int16), rate
