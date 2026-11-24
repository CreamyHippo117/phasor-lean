"""Where things are on this machine: the person's scenes folder, the example scenes, how to
start the ``phasor`` command line. Shared by the desktop app and the AI kit (``kit.py``),
neither of which may import the other.
"""

from __future__ import annotations

import sys
from pathlib import Path


def windows_documents() -> Path | None:
    """Windows' own Documents folder (``FOLDERID_Documents``), which OneDrive may have
    moved; None elsewhere or if Windows will not say."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        guid = (ctypes.c_byte * 16)(*uuid.UUID("FDD39AD0-238F-46AF-ADB4-6C85480369C7").bytes_le)
        out = ctypes.c_wchar_p()
        shell32, ole32 = ctypes.windll.shell32, ctypes.windll.ole32  # type: ignore[attr-defined]
        if shell32.SHGetKnownFolderPath(guid, wintypes.DWORD(0), None, ctypes.byref(out)):
            return None
        try:
            return Path(out.value) if out.value else None
        finally:
            ole32.CoTaskMemFree(out)
    except Exception:
        return None


def frozen() -> bool:
    """Whether this is the packaged app (PyInstaller) rather than Python with phasor."""
    return bool(getattr(sys, "frozen", False))


def examples_dir() -> Path | None:
    """The example scenes: beside the source tree, inside the installed package, or inside
    the packaged app."""
    here = Path(__file__).resolve()
    candidates = [here.parents[2] / "examples", here.parents[1] / "_docs" / "examples"]
    if frozen():
        candidates.insert(0, Path(getattr(sys, "_MEIPASS", "")) / "examples")
    return next((p for p in candidates if p.is_dir()), None)


def cli_command(*args: str) -> list[str]:
    """The ``phasor`` command line with ``args``, as a process to start."""
    if frozen():
        return [sys.executable, "--cli", *args]
    return [sys.executable, "-m", "phasor", *args]


def scenes_folder() -> Path:
    """Where new scenes and copied examples go by default: Documents/phasor. On Windows,
    the Documents folder Explorer shows (often moved into OneDrive), when it is inside
    the home folder."""
    home = Path.home()
    docs = windows_documents() or home / "Documents"
    if not docs.is_relative_to(home) or not docs.is_dir():
        docs = home / "Documents"
    return (docs if docs.is_dir() else home) / "phasor"
