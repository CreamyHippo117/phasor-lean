"""Video output.

``open_writer(path, width, height, fps, fast=False)`` returns a writer with ``write(frame)`` and
``close()``; frames are (h, w, 3) uint8 sRGB from ``render.output.encode_srgb8``.

- ``*.png`` with a ``%d`` pattern, or a directory: a PNG sequence (for editing).
- ``*.mov``: ProRes 4444 through a system ffmpeg (for editing).
- anything else (``*.mp4``): the best working encoder, probed in this order: PyAV NVENC
  (h264, then hevc), a system ffmpeg fed raw frames over a pipe (NVENC first), then PyAV
  libx264. A probe counts only when it really encoded frames (``render.doctor``).
"""

from __future__ import annotations

import contextlib
import functools
import queue
import shutil
import subprocess
import threading
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import numpy as np

from phasor.core.proc import NO_WINDOW
from phasor.render.doctor import check_ffmpeg, check_pyav, pick_encoder


class EncodeError(RuntimeError):
    pass


class Writer(Protocol):
    path: Path
    encoder: str

    def write(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


@functools.cache
def best_encoder() -> str | None:
    """``"pyav:<codec>"``, ``"ffmpeg:<codec>"`` or None. Probed once per process."""
    return pick_encoder(check_pyav(), check_ffmpeg())


def _rate(fps: float) -> Fraction:
    return Fraction(fps).limit_denominator(1001)


def _check(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
        raise EncodeError(
            f"frames must be ({height}, {width}, 3) uint8, got {frame.dtype} {frame.shape}"
        )
    return np.ascontiguousarray(frame)


class PngSequence:
    def __init__(self, path: Path, width: int, height: int) -> None:
        self.pattern = path / "frame_%05d.png" if "%" not in path.name else path
        self.pattern.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.encoder = path, "png"
        self.size, self.count = (width, height), 0

    def write(self, frame: np.ndarray) -> None:
        from PIL import Image

        frame = _check(frame, *self.size)
        Image.fromarray(frame, "RGB").save(str(self.pattern) % self.count)
        self.count += 1

    def close(self) -> None:
        pass


# Drafts (``render.job`` preset ``draft``): the quickest encodes, for frames nobody keeps.
FAST_X264 = {"crf": "28", "preset": "veryfast"}
FAST_NVENC = {"preset": "p1", "rc": "vbr", "cq": "30"}


class PyAVWriter:
    def __init__(
        self, path: Path, width: int, height: int, fps: float, codec: str, fast: bool = False
    ) -> None:
        import av

        self.path, self.encoder, self.size = path, f"pyav:{codec}", (width, height)
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream(codec, rate=_rate(fps))
        self._stream.width, self._stream.height = width, height
        self._stream.pix_fmt = "yuv420p"
        if codec == "libx264":
            self._stream.options = {"crf": "16", "preset": "medium"} if not fast else FAST_X264
        elif codec == "h264_nvenc":
            # Constant quality, not NVENC's fixed default bitrate: on prism_primes cq 19 matched
            # the default's 43 dB PSNR at a fifth of the size, and hard frames get more bits.
            self._stream.bit_rate = 0
            self._stream.options = (
                {"preset": "p5", "rc": "vbr", "cq": "19"} if not fast else FAST_NVENC
            )
        self._frame_type = av.VideoFrame

    def write(self, frame: np.ndarray) -> None:
        vf = self._frame_type.from_ndarray(_check(frame, *self.size), format="rgb24")
        for packet in self._stream.encode(vf):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def _fast_flags(codec: str, fast: bool) -> list[str]:
    """ffmpeg arguments for a draft encode of ``codec`` (none when not ``fast``)."""
    if not fast:
        return []
    opts = FAST_X264 if codec == "libx264" else FAST_NVENC if codec.endswith("_nvenc") else {}
    return [a for k, v in opts.items() for a in (f"-{k}", v)]


class FFmpegWriter:
    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        codec: str,
        pix_fmt: str,
        fast: bool = False,
    ) -> None:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise EncodeError("no ffmpeg on PATH")
        self.path, self.encoder, self.size = path, f"ffmpeg:{codec}", (width, height)
        rate = _rate(fps)
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-",
               "-c:v", codec, "-pix_fmt", pix_fmt, *_fast_flags(codec, fast),
               str(path)]  # fmt: skip
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                      creationflags=NO_WINDOW)  # fmt: skip

    def write(self, frame: np.ndarray) -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(_check(frame, *self.size).tobytes())
        except BrokenPipeError as e:
            raise EncodeError(self._proc.stderr.read().decode(errors="replace")) from e

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise EncodeError(self._proc.stderr.read().decode(errors="replace"))


class BackgroundWriter:
    """Any ``Writer`` on its own thread: ``write`` hands the frame over and returns, so a
    frame is encoded while the next one renders (PyAV, the ffmpeg pipe and PNG compression
    all let go of Python's lock while they work). Frames reach ``inner`` in order, every one
    of them; ``write`` waits only when the encoder is ``depth`` frames behind. An error in
    the encoder is raised by the next ``write`` or by ``close``. ``abort`` stops without
    writing what is still queued (a failed or cancelled render). The caller must not change
    a frame after handing it over."""

    def __init__(self, inner: Writer, depth: int = 2) -> None:
        self.inner, self.path, self.encoder = inner, inner.path, inner.encoder
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._error: BaseException | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="phasor-encode", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while (frame := self._queue.get()) is not None:
            if self._error is None and not self._stop:
                try:
                    self.inner.write(frame)
                except BaseException as e:  # handed to the caller's thread
                    self._error = e

    def write(self, frame: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put(frame)

    def _finish(self) -> BaseException | None:
        self._queue.put(None)
        self._thread.join()
        return self._error

    def close(self) -> None:
        error = self._finish()
        try:
            self.inner.close()
        except BaseException:
            if error is None:
                raise
        if error is not None:
            raise error

    def abort(self) -> None:
        self._stop = True
        self._finish()
        # The render already failed; its own error is the one to report.
        with contextlib.suppress(Exception):
            self.inner.close()


def open_writer(path: str | Path, width: int, height: int, fps: float,
                fast: bool = False) -> Writer:  # fmt: skip
    path = Path(path)
    if path.suffix == "" or (path.suffix == ".png" and "%" in path.name):
        return PngSequence(path, width, height)
    if width % 2 or height % 2:
        raise EncodeError(f"video needs an even width and height, got {width}x{height}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".mov":
        return FFmpegWriter(path, width, height, fps, "prores_ks", "yuva444p10le")
    choice = best_encoder()
    if choice is None:
        raise EncodeError("no working video encoder (run `phasor doctor`); write PNGs instead")
    backend, codec = choice.split(":", 1)
    if backend == "pyav" and codec.endswith("_nvenc") and not _nvenc_takes(codec, width, height):
        codec = "libx264"  # NVENC has a minimum frame size (found 2026-11-21: 64x48 refused)
    if backend == "pyav":
        return PyAVWriter(path, width, height, fps, codec, fast)
    return FFmpegWriter(path, width, height, fps, codec, "yuv420p", fast)


@functools.cache
def _nvenc_takes(codec: str, width: int, height: int) -> bool:
    """Whether NVIDIA's encoder accepts this frame size: it refuses small frames (the limit
    depends on the card), and the refusal only comes when the first frame is encoded."""
    import io

    import av

    try:
        with av.open(io.BytesIO(), "w", format="mp4") as c:
            st = c.add_stream(codec, rate=30)
            st.width, st.height, st.pix_fmt = width, height, "yuv420p"
            frame = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), np.uint8), "rgb24")
            for packet in [*st.encode(frame), *st.encode()]:
                c.mux(packet)
        return True
    except Exception:
        return False


def mux_audio(video: str | Path, samples: np.ndarray, rate: int) -> None:
    """Add an int16 soundtrack (``(n, 2)`` stereo, or ``(n,)`` mono, which is heard on both
    channels) to a finished video file, in place. The video stream is copied, not
    re-encoded; the audio becomes stereo AAC (MP4) or PCM (MOV)."""
    import av

    video = Path(video)
    tmp = video.with_name(video.stem + ".mux" + video.suffix)
    with av.open(str(video)) as src, av.open(str(tmp), mode="w") as dst:
        vin = src.streams.video[0]
        vout = dst.add_stream_from_template(vin)
        codec = "pcm_s16le" if video.suffix == ".mov" else "aac"
        aout = dst.add_stream(codec, rate=rate, layout="stereo")
        for packet in src.demux(vin):
            if packet.dts is None:
                continue
            packet.stream = vout
            dst.mux(packet)
        pcm = np.asarray(samples, dtype=np.int16)
        if pcm.ndim == 1:
            pcm = np.stack([pcm, pcm], axis=1)
        pcm = np.ascontiguousarray(pcm).reshape(1, -1)  # packed: L R L R ...
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="stereo")
        frame.sample_rate = rate
        for packet in aout.encode(frame):
            dst.mux(packet)
        for packet in aout.encode(None):
            dst.mux(packet)
    tmp.replace(video)
