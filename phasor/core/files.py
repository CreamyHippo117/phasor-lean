"""Files an object reads (a mesh, a picture), named relative to the scene's folder.

A type lists the parameters that hold such paths in its ``Files`` class variable
(``Files = ("path",)``). Every path is relative and stays inside the folder: an absolute
path or a ``..`` part is a validation error, so a scene (or the AI writing one) can never
read a file from anywhere else. A path naming a file that is not there is the lint issue
``missing_file``; the object then draws nothing instead of failing.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def relative_path(value: str) -> str:
    """``value`` if it is a relative path that stays in its folder, else ``ValueError``.
    Written with forward slashes, so a document reads the same on every system."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a file path must be a non-empty string")
    win, posix = PureWindowsPath(value), PurePosixPath(value.replace("\\", "/"))
    if win.is_absolute() or win.drive or posix.is_absolute():
        raise ValueError(f"{value!r} is absolute; name files relative to the scene's folder")
    if ".." in posix.parts:
        raise ValueError(f"{value!r} leaves the scene's folder; files must stay inside it")
    return posix.as_posix()


def resolve(base_dir: str | Path, value: str) -> Path:
    """The file ``value`` names, inside ``base_dir`` (the current folder when empty).
    ``ValueError`` if it resolves outside the folder (a link pointing out, say)."""
    root = Path(base_dir or ".").resolve()
    path = (root / relative_path(value)).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"{value!r} is outside the scene's folder")
    return path


def exists(base_dir: str | Path, value: str) -> bool:
    """Whether ``value`` names a file inside ``base_dir``."""
    try:
        return resolve(base_dir, value).is_file()
    except ValueError:
        return False
