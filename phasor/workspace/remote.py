"""One live scene, two editors (when the GUI is running it hosts the Engine,
and ``phasor mcp`` attaches to it over a localhost JSON-RPC socket).

``RemoteServer`` serves a ``Workspace``'s tools on 127.0.0.1. ``RemoteWorkspace`` is the
other end: it has every tool of ``Workspace`` with the same signature and docstring (so the
MCP server describes them exactly as before), and each call runs in the host's process,
on the host's engine.

Protocol: one JSON object per line. Request ``{"token", "method", "params"}``, reply
``{"ok": true, "result"}`` or ``{"ok": false, "error", "issues"}``. Only public ``Workspace``
methods can be called; the token (random, written to a handoff file readable only by the
user) keeps other local programs out. Calls are serialized with the host's lock, so a GUI
edit and an AI edit never interleave; every edit still goes through ``Engine.apply``.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import secrets
import socket
import socketserver
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from phasor.core.proc import NO_WINDOW
from phasor.workspace.tools import ToolError, Workspace

HANDOFF = Path(".phasor") / "gui.json"  # inside the workspace root
ENCODING = "utf-8"


def tool_names() -> tuple[str, ...]:
    """The public tools of ``Workspace`` (methods, not properties)."""
    return tuple(
        n for n, v in vars(Workspace).items() if not n.startswith("_") and inspect.isfunction(v)
    )


class RemoteServer:
    def __init__(self, workspace: Workspace, lock: threading.RLock | None = None,
                 *, host: str = "127.0.0.1", port: int = 0) -> None:  # fmt: skip
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the editing socket only listens on this machine")
        self.workspace = workspace
        self.lock = lock or threading.RLock()
        self.token = secrets.token_hex(16)
        self.calls = 0
        allowed = set(tool_names())
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                for line in self.rfile:
                    reply = outer._answer(line, allowed)
                    self.wfile.write((json.dumps(reply, default=str) + "\n").encode(ENCODING))
                    self.wfile.flush()

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server((host, port), Handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        return self._server.server_address[:2]  # type: ignore[return-value]

    def _answer(self, line: bytes, allowed: set[str]) -> dict[str, Any]:
        try:
            req = json.loads(line.decode(ENCODING))
        except (ValueError, UnicodeDecodeError):
            return {"ok": False, "error": "not a JSON request", "issues": []}
        if not isinstance(req, dict) or not secrets.compare_digest(
            str(req.get("token", "")), self.token
        ):
            return {"ok": False, "error": "wrong token", "issues": []}
        method, params = req.get("method"), req.get("params") or {}
        if method not in allowed or not isinstance(params, dict):
            return {"ok": False, "error": f"no tool {method!r}", "issues": []}
        with self.lock:
            before = self._version()
            try:
                result = getattr(self.workspace, method)(**params)
            except ToolError as e:
                return {"ok": False, "error": str(e), "issues": e.issues}
            except TypeError as e:
                return {"ok": False, "error": f"{method}: {e}", "issues": []}
            self.calls += 1
            after = self._version()
            if after != before:  # it changed the scene: the editor shows who did what
                self.workspace.activity.append({"who": "ai", "what": method, "version": after})
        return {"ok": True, "result": result}

    def _version(self) -> int:
        try:
            return self.workspace.engine.version
        except ToolError:
            return -1

    def start(self) -> RemoteServer:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                        name="phasor-remote")  # fmt: skip
        self._thread.start()
        return self

    def write_handoff(self, root: str | Path | None = None) -> Path:
        """Write where to find this server (host, port, token) to ``<root>/.phasor/gui.json``,
        readable only by this user, for ``phasor mcp --attach``."""
        path = Path(root or self.workspace.root) / HANDOFF
        path.parent.mkdir(parents=True, exist_ok=True)
        host, port = self.address
        data = json.dumps({"host": host, "port": port, "token": self.token, "pid": os.getpid()})
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding=ENCODING) as f:
            f.write(data)
        if sys.platform == "win32":  # 0o600 means nothing to Windows: set the ACL instead
            _windows_private(path)
        return path

    def stop(self, root: str | Path | None = None) -> None:
        self._server.shutdown()
        self._server.server_close()
        path = Path(root or self.workspace.root) / HANDOFF
        try:
            if json.loads(path.read_text(encoding=ENCODING)).get("token") == self.token:
                path.unlink()
        except (OSError, ValueError):
            pass


class RemoteWorkspace:
    """Every ``Workspace`` tool, run by the GUI that hosts the scene."""

    def __init__(self, host: str, port: int, token: str, *, timeout: float = 600.0) -> None:
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._file = self._sock.makefile("rwb")
        self._token = token
        self._lock = threading.Lock()

    @classmethod
    def from_handoff(cls, root: str | Path | None = None) -> RemoteWorkspace:
        path = Path(root or Path.cwd()) / HANDOFF
        try:
            data = json.loads(path.read_text(encoding=ENCODING))
        except (OSError, ValueError):
            raise ToolError(
                f"no running phasor gui found ({path} is missing): start `phasor gui` first, "
                "or run `phasor mcp` without --attach"
            ) from None
        try:
            return cls(data["host"], int(data["port"]), data["token"])
        except OSError as e:
            raise ToolError(f"the phasor gui at {data['host']}:{data['port']} does not answer "
                            f"({e}); it may have closed") from None  # fmt: skip

    def call(self, method: str, **params: Any) -> Any:
        req = json.dumps({"token": self._token, "method": method, "params": params})
        with self._lock:
            self._file.write((req + "\n").encode(ENCODING))
            self._file.flush()
            line = self._file.readline()
        if not line:
            raise ToolError("the phasor gui closed the connection")
        reply = json.loads(line.decode(ENCODING))
        if not reply.get("ok"):
            err = ToolError(reply.get("error", "failed"))
            err.issues = list(reply.get("issues") or [])  # already plain dicts
            raise err
        return reply["result"]

    def close(self) -> None:
        self._file.close()
        self._sock.close()

    def __getattr__(self, name: str):
        fn = vars(Workspace).get(name)
        if name.startswith("_") or not inspect.isfunction(fn):
            raise AttributeError(name)
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())[1:]  # without self

        @functools.wraps(fn)
        def forward(*args: Any, **kwargs: Any) -> Any:
            bound = sig.replace(parameters=params).bind(*args, **kwargs)
            return self.call(name, **bound.arguments)

        forward.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
        return forward


def _windows_private(path: Path) -> None:
    """Only the current user may read ``path``: drop inherited permissions and grant this
    user alone (``icacls``, part of Windows). The token in the file lets a program drive
    the editor, so a failure here is an error, not a warning."""
    user = os.environ.get("USERNAME", "")
    domain = os.environ.get("USERDOMAIN", "")
    who = f"{domain}\\{user}" if domain else user
    subprocess.run(
        ["icacls", str(path), "/inheritance:r", "/grant:r", f"{who}:F"],
        check=True, capture_output=True, creationflags=NO_WINDOW,
    )  # fmt: skip
