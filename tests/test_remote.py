"""One live scene, two editors. The GUI hosts the engine and serves the
workspace's tools on localhost; ``phasor mcp --attach`` edits the same scene through it."""

from __future__ import annotations

import inspect
import json
import os
import socket
import sys
import threading

import pytest

from phasor.workspace import ToolError, Workspace
from phasor.workspace.remote import HANDOFF, RemoteServer, RemoteWorkspace, tool_names


@pytest.fixture
def hosted(tmp_path):
    ws = Workspace(tmp_path)
    ws.scene_new("live")
    server = RemoteServer(ws).start()
    server.write_handoff()
    yield ws, server
    server.stop()


def test_an_attached_client_edits_the_hosts_scene(hosted, tmp_path):
    ws, _server = hosted
    remote = RemoteWorkspace.from_handoff(tmp_path)
    r = remote.scene_edit({"op": "add_object", "object": {"id": "p", "type": "dot",
                                                           "params": {"pos": [1, 0, 0]}}})  # fmt: skip
    assert r["ok"] and "p" in ws.engine.state_at(0.0)  # the host's engine changed
    assert remote.inspect(0.0, id="p")["objects"][0]["id"] == "p"  # and reads back through it
    bad = remote.scene_edit({"op": "set_param", "path": "nobody.pos", "value": [0, 0, 0]})
    assert not bad["ok"] and bad["errors"]  # a refused patch is reported, nothing changes
    with pytest.raises(ToolError, match="does not exist"):  # tool errors arrive as ToolError
        remote.scene_open("missing.phasor.json")
    assert remote.undo()["version"] == ws.engine.version
    assert "p" not in ws.engine.state_at(0.0)
    remote.close()


def test_the_proxy_has_every_tool_with_the_same_signature(hosted, tmp_path):
    remote = RemoteWorkspace.from_handoff(tmp_path)
    for name in tool_names():
        a = inspect.signature(getattr(Workspace(tmp_path), name))
        b = inspect.signature(getattr(remote, name))
        assert a == b, name
        assert getattr(remote, name).__doc__ == getattr(Workspace, name).__doc__
    assert {"scene_edit", "inspect", "lint", "storyboard", "undo"} <= set(tool_names())
    with pytest.raises(AttributeError):
        _ = remote._engine
    remote.close()


def test_strangers_and_private_methods_are_refused(hosted, tmp_path):
    ws, server = hosted
    host, port = server.address
    assert host == "127.0.0.1"
    handoff = tmp_path / HANDOFF
    if sys.platform == "win32":  # an ACL with this user alone
        import subprocess

        acl = subprocess.run(["icacls", str(handoff)], capture_output=True, text=True).stdout
        lines = acl.replace(str(handoff), "").splitlines()[:-2]
        # one "domain\name:(F)" per line, the first after the file's own path
        who = [ln.rsplit(":(", 1)[0].strip().lower() for ln in lines if ":(" in ln]
        assert any(os.environ["USERNAME"].lower() in w for w in who), acl  # this user may read it
        # No group of other people may. An elevated (administrator) process also leaves the
        # system, the Administrators group and OWNER RIGHTS on the file; those can read any
        # file anyway, so they are not strangers.
        strangers = {"everyone", "users", "authenticated users", "interactive", "network"}
        assert not [w for w in who if w.rsplit("\\", 1)[-1] in strangers], acl
    else:
        assert oct(handoff.stat().st_mode & 0o777) == "0o600"  # only this user can read it

    def ask(req) -> dict:
        with socket.create_connection((host, port)) as s:
            s.sendall((json.dumps(req) + "\n").encode())
            return json.loads(s.makefile().readline())

    assert ask({"token": "guess", "method": "summary", "params": {}})["error"] == "wrong token"
    for bad in ("_guard", "engine", "__init__", "os.system"):
        r = ask({"token": server.token, "method": bad, "params": {}})
        assert not r["ok"] and "no tool" in r["error"]
    with pytest.raises(ValueError, match="this machine"):
        RemoteServer(ws, host="0.0.0.0")


def test_edits_from_both_sides_are_serialized(hosted, tmp_path):
    """The GUI and the AI editing at the same moment: every edit lands, one at a time."""
    ws, server = hosted
    remote = RemoteWorkspace.from_handoff(tmp_path)

    def ai():
        for k in range(10):
            remote.scene_edit({"op": "add_object", "object": {"id": f"ai_{k}", "type": "dot"}})

    t = threading.Thread(target=ai)
    t.start()
    for k in range(10):
        with server.lock:  # what the GUI does around its own edits
            ws.scene_edit({"op": "add_object", "object": {"id": f"me_{k}", "type": "dot"}})
    t.join()
    ids = {o.id for o in ws.engine.document.objects}
    assert {f"ai_{k}" for k in range(10)} | {f"me_{k}" for k in range(10)} <= ids
    assert server.calls == 10
    remote.close()


def test_attach_without_a_gui_says_what_to_do(tmp_path):
    with pytest.raises(ToolError, match="start `phasor gui`"):
        RemoteWorkspace.from_handoff(tmp_path)


def test_stopping_removes_the_handoff(tmp_path):
    ws = Workspace(tmp_path)
    server = RemoteServer(ws).start()
    path = server.write_handoff()
    assert path.exists()
    server.stop()
    assert not path.exists()


def test_an_attached_mcp_server_describes_its_tools_exactly_as_a_headless_one(hosted, tmp_path):
    asyncio = pytest.importorskip("asyncio")
    pytest.importorskip("mcp")
    from phasor.mcp.server import build_server

    def schema(ws):
        return {t.name: (t.description, t.input_schema)
                for t in asyncio.run(build_server(ws).list_tools())}  # fmt: skip

    remote = RemoteWorkspace.from_handoff(tmp_path)
    assert schema(remote) == schema(Workspace(tmp_path))
    remote.close()
