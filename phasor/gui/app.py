"""The desktop app around the editor: what is not one scene's business. No Qt in here (the
window draws it), so all of it can be tested directly.

- ``Settings``: recent scenes, the last folder, the math engine; a small JSON file in the
  platform's settings folder (``config_dir``).
- ``TEMPLATES`` and ``examples()``: what "New" and the start screen offer.
- ``ExportJob``: a picture, a video or a web page made by a separate ``phasor`` process, so
  the window stays responsive, the GL context is never shared between threads, and the job
  can be cancelled. Progress is read from what the process prints.
- ``install_shortcut``: a launcher in the desktop's application menu.
- ``cli_command``: how to run the ``phasor`` command line from here, also from the
  packaged app (which has no Python of its own: it runs itself with ``--cli``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phasor.core.proc import NO_WINDOW
from phasor.workspace.paths import (  # noqa: F401  (part of this module's interface)
    cli_command,
    examples_dir,
    frozen,
    windows_documents,
)

APP_NAME = "phasor"
MAX_RECENT = 12
ASSETS = Path(__file__).parent / "assets"
ICON = ASSETS / "icon.png"


# ---------------------------------------------------------------- where things live


def config_dir() -> Path:
    """The per-user settings folder: %APPDATA%\\phasor on Windows, ~/Library/Application
    Support/phasor on macOS, $XDG_CONFIG_HOME/phasor (~/.config/phasor) elsewhere.
    ``PHASOR_CONFIG_DIR`` overrides it (tests, portable installs)."""
    if os.environ.get("PHASOR_CONFIG_DIR"):
        return Path(os.environ["PHASOR_CONFIG_DIR"])
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME


# ---------------------------------------------------------------- settings


@dataclass
class Settings:
    recent: list[str] = field(default_factory=list)
    last_dir: str = ""
    math_engine: str = "mathtext"  # typeset by default; "latex" or "auto" is the option

    @classmethod
    def path(cls) -> Path:
        return config_dir() / "settings.json"

    @classmethod
    def load(cls) -> Settings:
        """The saved settings; defaults when there are none or the file is damaged."""
        try:
            raw = json.loads(cls.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        s = cls()
        s.recent = [str(p) for p in raw.get("recent", []) if isinstance(p, str)][:MAX_RECENT]
        s.last_dir = str(raw.get("last_dir", ""))
        if raw.get("math_engine") in ("mathtext", "latex", "auto"):
            s.math_engine = raw["math_engine"]
        return s

    def save(self) -> None:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"recent": self.recent, "last_dir": self.last_dir, "math_engine": self.math_engine}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(p)

    def opened(self, path: str | Path) -> None:
        """Put ``path`` first in the recent list (and remember its folder)."""
        p = str(Path(path).resolve())
        self.recent = [p, *[r for r in self.recent if r != p]][:MAX_RECENT]
        self.last_dir = str(Path(p).parent)

    def existing_recent(self) -> list[str]:
        """Recent scenes that are still on disk."""
        return [r for r in self.recent if Path(r).exists()]


# ---------------------------------------------------------------- new scenes


# What "New" offers: a title and the patch ops that set the scene up (beyond scene_new).
TEMPLATES: dict[str, dict[str, Any]] = {
    "2D scene": {"about": "the plane, seen from above", "ops": []},
    "3D scene": {
        "about": "an orbit camera: drag to turn it",
        "ops": [{"op": "set_camera", "id": "camera", "kind": "orbit", "params": {}}],
    },
    "Complex function": {
        "about": "a domain-colored f(z), the phase as hue",
        "ops": [
            {
                "op": "add_object",
                "object": {
                    "id": "plane",
                    "type": "domain_coloring",
                    "params": {"function": "1/(z**2 + 1)", "size": [8.0, 4.5]},
                },
            }
        ],
    },
}


_NOTE = re.compile(r"^Milestone \w+(?: exit demo)?(?: \([^)]*\))?(?:[:,]\s*|\s+(?=tour))")


def examples() -> list[dict[str, str]]:
    """The example scenes: name, file and the first line of what each shows."""
    d = examples_dir()
    if d is None:
        return []
    out = []
    for p in sorted(d.glob("*.py")):
        if p.stem in ("domain_coloring", "__init__"):
            continue  # a kernel script, not a scene
        about = ""
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        m = re.match(r'\s*"""(.*?)(?:\n\n|""")', text, re.S)
        if m:
            about = " ".join(m.group(1).split())
            # The docstrings start with project notes ("Milestone 9 exit demo: ..."): the
            # start screen shows only what the scene is.
            about = _NOTE.sub("", about)
            about = about[:1].upper() + about[1:160]
        out.append({"name": p.stem.replace("_", " "), "path": str(p), "about": about})
    return out


# ---------------------------------------------------------------- exports


EXPORTS = {
    "picture": {"suffix": ".png", "label": "Picture (PNG)"},
    "video": {"suffix": ".mp4", "label": "Video (MP4, or .mov for ProRes)"},
    "web": {"suffix": ".html", "label": "Web page with sliders (HTML)"},
}
_FRAME = re.compile(r"frame (\d+)/(\d+)")


class ExportJob:
    """One export in a ``phasor`` process of its own, from a snapshot of the scene.

    ``start`` saves the scene as it is now (overrides included) to a hidden file beside
    the original, so relative paths (narration audio) still resolve, and runs the command
    line on it. Call ``poll`` from a timer: it returns the progress and, at the end, the
    outcome. ``cancel`` stops it. The snapshot is removed when the job ends.
    """

    def __init__(self, engine: Any, kind: str, out: str | Path, *, scene_dir: str | Path,
                 t: float = 0.0, preset: str = "preview", steps: int = 5) -> None:  # fmt: skip
        if kind not in EXPORTS:
            raise ValueError(f"export kind must be one of {sorted(EXPORTS)}, got {kind!r}")
        self.engine, self.kind, self.out = engine, kind, Path(out)
        self.scene_dir, self.t, self.preset, self.steps = Path(scene_dir), t, preset, steps
        self.proc: subprocess.Popen | None = None
        self.done, self.ok, self.message = False, False, ""
        self.frame, self.frames = 0, 0
        self._buf = ""
        self._snapshot: Path | None = None
        self._log: Any = None

    def command(self, scene: str) -> list[str]:
        if self.kind == "picture":
            return cli_command("render", scene, "-o", str(self.out), "--frame", f"{self.t:g}",
                               "--preset", self.preset)  # fmt: skip
        if self.kind == "video":
            return cli_command("render", scene, "-o", str(self.out), "--preset", self.preset)
        return cli_command("web", scene, "-o", str(self.out), "--steps", str(self.steps))

    def start(self) -> None:
        """Snapshot the scene and start the process. Raises ``OSError`` (leaving nothing
        behind) when the output folder cannot be made or the process cannot start."""
        self.out.parent.mkdir(parents=True, exist_ok=True)  # before anything is written
        fd, name = tempfile.mkstemp(prefix=".phasor-export-", suffix=".phasor.json",
                                    dir=self.scene_dir)  # fmt: skip
        os.close(fd)
        self._snapshot = Path(name)
        try:
            self.engine.save(self._snapshot)
            # Kept open for the life of the job (the process writes to it); closed in _cleanup.
            self._log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")  # noqa: SIM115
            self.proc = subprocess.Popen(
                self.command(str(self._snapshot)), stdout=self._log, stderr=self._log,
                cwd=self.scene_dir, creationflags=NO_WINDOW,
            )  # fmt: skip
        except BaseException:
            self._cleanup()
            raise

    def _read(self) -> str:
        self._log.seek(0)
        return self._log.read()

    def poll(self) -> dict[str, Any]:
        """{"frame", "frames", "done", "ok", "message"}: where the job is."""
        if self.proc is None or self.done:
            return self.status()
        text = self._read()
        found = _FRAME.findall(text)
        if found:
            self.frame, self.frames = (int(x) for x in found[-1])
        code = self.proc.poll()
        if code is not None:
            self.done, self.ok = True, code == 0 and self.out.exists()
            lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
            msg = [ln for ln in lines if not _FRAME.fullmatch(ln)]
            self.message = msg[-1] if msg else ("done" if self.ok else f"exit code {code}")
            self._cleanup()
        return self.status()

    def status(self) -> dict[str, Any]:
        return {"frame": self.frame, "frames": self.frames, "done": self.done, "ok": self.ok,
                "message": self.message, "out": str(self.out)}  # fmt: skip

    def cancel(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.done, self.ok, self.message = True, False, "cancelled"
        self._cleanup()

    def _cleanup(self) -> None:
        if self._snapshot is not None:
            from phasor.core import io

            for p in (self._snapshot, io.overrides_path(self._snapshot)):
                p.unlink(missing_ok=True)
            self._snapshot = None
        if self._log is not None:
            self._log.close()
            self._log = None


# ---------------------------------------------------------------- the launcher


def launch_command() -> list[str]:
    """How a desktop shortcut starts the editor."""
    if frozen():
        return [sys.executable]
    return [sys.executable, "-m", "phasor.gui"]


def install_shortcut() -> dict[str, Any]:
    """Add phasor to the desktop's application menu. Linux: a ``.desktop`` file in
    ~/.local/share/applications (opens .phasor.json files too). Windows: a Start-menu
    shortcut (untested on Windows). macOS: use the packaged ``.app`` instead."""
    cmd = launch_command()
    if sys.platform.startswith("linux"):
        apps = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        target = apps / "applications" / "phasor.desktop"
        target.parent.mkdir(parents=True, exist_ok=True)
        exe = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        target.write_text(
            "[Desktop Entry]\nType=Application\nName=phasor\n"
            "Comment=Math animations an AI can see, check and fix\n"
            f"Exec={exe} %f\nIcon={ICON}\nTerminal=false\n"
            "Categories=Graphics;Education;Science;\n"
            "MimeType=application/x-phasor+json;\n",
            encoding="utf-8",
        )
        return {"ok": True, "path": str(target)}
    if sys.platform == "win32":
        menu = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs"
        link = menu / "phasor.lnk"
        args = " ".join(f'"{c}"' for c in cmd[1:])
        ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:PHASOR_LNK);"
              "$s.TargetPath = $env:PHASOR_EXE; $s.Arguments = $env:PHASOR_ARGS;"
              "$s.Save()")  # fmt: skip
        env = {**os.environ, "PHASOR_LNK": str(link), "PHASOR_EXE": cmd[0], "PHASOR_ARGS": args}
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env,
                           capture_output=True, text=True, creationflags=NO_WINDOW)  # fmt: skip
        return {"ok": r.returncode == 0, "path": str(link), "error": r.stderr.strip()}
    return {"ok": False, "error": "on macOS, build the app (packaging/README.md) and drag it "
                                  "to Applications"}  # fmt: skip
