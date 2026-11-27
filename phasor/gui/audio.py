"""Hearing the scene in the editor: play a WAV file with what the system already has.

No audio library is needed: Windows plays WAV through Python's own ``winsound``; macOS
through ``afplay``; Linux through ``paplay`` (PulseAudio and PipeWire), ``aplay`` (ALSA) or
``ffplay``. The editor writes the mix from the playhead to a file and plays that, so what
you hear is exactly what the encoder writes (``Engine.audio_track``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from phasor.core.proc import NO_WINDOW

# Linux players, best first: (program, arguments before the file)
_LINUX = (("paplay", ()), ("aplay", ("-q",)), ("ffplay", ("-nodisp", "-autoexit", "-loglevel",
                                                        "quiet")))  # fmt: skip


def backend() -> str | None:
    """How this system plays sound: ``winsound``, a program's name, or None (silent)."""
    if sys.platform == "win32":
        return "winsound"
    if sys.platform == "darwin":
        return "afplay" if shutil.which("afplay") else None
    return next((name for name, _ in _LINUX if shutil.which(name)), None)


class Player:
    """Plays one file at a time in the background; ``play`` again replaces it."""

    def __init__(self, kind: str | None = "auto") -> None:
        self.kind = backend() if kind == "auto" else kind
        self._proc: subprocess.Popen | None = None

    @property
    def available(self) -> bool:
        return self.kind is not None

    def play(self, path: str | Path) -> bool:
        """Start playing ``path`` (a WAV file); False when this system cannot play sound."""
        self.stop()
        if self.kind is None:
            return False
        if self.kind == "winsound":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        args = dict(_LINUX).get(self.kind, ())
        quiet = subprocess.DEVNULL
        command = [shutil.which(self.kind) or self.kind, *args, str(path)]
        self._proc = subprocess.Popen(command, stdin=quiet, stdout=quiet, stderr=quiet,
                                      creationflags=NO_WINDOW)  # fmt: skip
        return True

    def stop(self) -> None:
        if self.kind == "winsound":
            import winsound

            winsound.PlaySound(None, 0)
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None

    @property
    def playing(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
