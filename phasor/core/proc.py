"""Starting other programs (TeX, ffmpeg, nvidia-smi, speech engines, PowerShell).

Pass ``creationflags=NO_WINDOW`` to every ``subprocess`` call. On Windows the desktop app has
no console, and a console program started from it without this flag opens a visible console
window of its own: a black flash on every typeset formula (checked in the packaged app,
2026-11-20). Elsewhere it is 0 and changes nothing.
"""

from __future__ import annotations

import subprocess
import sys

NO_WINDOW: int = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
