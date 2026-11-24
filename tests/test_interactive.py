"""Sliders. A scene's controls (document, schema v6) drive a player, a notebook
widget and a baked web page; the exit demo is a Julia explorer with sliders for c."""

from __future__ import annotations

import base64
import io
import json
import re
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, cx
from phasor.core import io as docio
from phasor.core.migrate import migrate
from phasor.core.patch import parse_patch
from phasor.core.schema import SCHEMA_VERSION
from phasor.interactive import Player, export_web, snap
from tests.conftest import gl_available

ROOT = Path(__file__).parents[1]
needs_gl = [pytest.mark.gl, pytest.mark.skipif(not gl_available(), reason="needs OpenGL")]


def julia(passes=None):
    s = Scene("j", passes=passes)
    s.set("camera.height", 2.6)
    s.add(
        cx.Fractal(
            id="set", kind="julia", c=(-0.8, 0.156), pos=(0, 0, 0), size=(4.6, 2.6), max_iter=60
        )
    )
    s.control("set.c", component=0, min=-1.2, max=1.2, step=0.05, label="Re c")
    s.control("set.c", component=1, min=-1.2, max=1.2, step=0.05, label="Im c")
    return s


# ---------------------------------------------------------------- controls in the document


def test_controls_are_validated_against_the_parameter_they_drive():
    s = julia()
    doc = s.engine.document
    assert [c.key for c in doc.controls] == ["set.c[0]", "set.c[1]"]
    bad = [
        ({"path": "ghost.r", "min": 0, "max": 1}, "no parameter"),
        ({"path": "set.c", "min": 0, "max": 1}, "name a component"),  # a vector
        ({"path": "set.c", "component": 5, "min": 0, "max": 1}, "name a component"),
        ({"path": "set.kind", "min": 0, "max": 1}, "not a number"),  # text
        ({"path": "set.max_iter", "component": 0, "min": 0, "max": 1}, "single number"),
    ]
    for ctl, why in bad:
        r = s.engine.apply({"op": "add_control", "control": ctl}, origin="script")
        assert not r.ok and why in r.errors[0].message, (ctl, r.errors)
    r = s.engine.apply(
        {"op": "add_control", "control": {"path": "set.max_iter", "min": 30, "max": 300}},
        origin="script",
    )
    assert r.ok
    with pytest.raises(Exception, match="max must be above min"):
        parse_patch({"op": "add_control", "control": {"path": "set.max_iter", "min": 3, "max": 3}})
    assert s.engine.apply({"op": "remove_control", "path": "set.max_iter"}, origin="script").ok
    assert not s.engine.apply({"op": "remove_control", "path": "set.max_iter"}, origin="script").ok
    # an animated parameter cannot also be a slider
    a = Scene("a", passes=None)
    a.add(cx.Fractal(id="f", kind="julia"))
    a.marker("go", 1.0)
    a.at("go").to("f.max_iter", 200, duration=1.0)
    with pytest.raises(Exception, match="animated by a track"):
        a.control("f.max_iter", min=1, max=300)


def test_controls_survive_saving_and_old_documents_gain_none(tmp_path):
    s = julia()
    path = tmp_path / "j.phasor.json"
    s.engine.save(path)
    assert [c.key for c in docio.load_document(path).controls] == ["set.c[0]", "set.c[1]"]
    old = migrate({"schema_version": 5, "meta": {"title": "old"}})
    assert old["schema_version"] == SCHEMA_VERSION and old["controls"] == []
    a, b = julia().engine.document, julia().engine.document
    assert a == b


def test_a_slider_whose_parameter_goes_away_is_a_lint_error():
    s = julia()
    assert s.engine.lint() == []
    s.engine.apply({"op": "remove_object", "id": "set"}, origin="script")
    codes = {i.code for i in s.engine.lint()}
    assert "orphan_control" in codes


def test_the_diff_names_added_and_removed_controls():
    from phasor.core.diff import diff_documents

    before = julia().engine.document
    s = julia()
    s.engine.apply({"op": "remove_control", "path": "set.c", "component": 1}, origin="script")
    kinds = {
        (d["kind"], d.get("control")) for d in diff_documents(before, s.engine.document)["changes"]
    }
    assert ("control_removed", "set.c[1]") in kinds


# ---------------------------------------------------------------- the player


def test_snapping_clamps_and_rounds_to_the_step():
    c = julia().engine.document.controls[0]
    assert snap(c, 9.0) == 1.2 and snap(c, -9.0) == -1.2
    assert snap(c, 0.07) == pytest.approx(0.05) and snap(c, 0.08) == pytest.approx(0.1)


def test_moving_a_slider_changes_the_player_and_never_the_scene():
    s = julia()
    v0, doc0 = s.engine.version, s.engine.document
    p = Player(s.engine)
    assert p.values == {"set.c[0]": -0.8, "set.c[1]": 0.156}
    assert p.set("set.c[0]", 0.31) == pytest.approx(0.3)  # snapped
    assert p.set("set.c[1]", 99) == 1.2  # clamped
    assert p.values == {"set.c[0]": pytest.approx(0.3), "set.c[1]": 1.2}
    assert p.state_at(0.0)["set"].params.c == pytest.approx((0.3, 1.2))
    assert s.engine.version == v0 and s.engine.document == doc0  # the scene is untouched
    p.reset()
    assert p.values == {"set.c[0]": -0.8, "set.c[1]": pytest.approx(0.156)}
    with pytest.raises(KeyError, match="no control"):
        p.set("set.zoom", 1)


@pytest.mark.parametrize("passes", [pytest.param("auto", marks=needs_gl)])
def test_the_picture_follows_the_sliders(passes):
    p = Player(julia(passes).engine)
    a = p.frame(0.0, (160, 90))
    p.set("set.c[0]", 0.3)
    b = p.frame(0.0, (160, 90))
    assert a.shape == (90, 160, 3) and a.dtype == np.uint8
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 1  # a different Julia set
    p.reset()
    assert (p.frame(0.0, (160, 90)) == a).all()  # and back: pictures are pure


# ---------------------------------------------------------------- the web export


def _frames_of(page: str):
    data = json.loads(re.search(r"const D = (\{.*?\});\nconst pic", page, re.S).group(1))
    return data


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_web_export_bakes_every_combination_into_one_file(tmp_path):
    from PIL import Image

    s = julia("auto")
    info = export_web(s.engine, tmp_path / "page.html", size=(96, 54),
                      steps={"set.c[0]": 3, "set.c[1]": 2})  # fmt: skip
    assert info["frames"] == 6 and [a["steps"] for a in info["axes"]] == [3, 2]
    page = (tmp_path / "page.html").read_text(encoding="utf-8")
    assert "http://" not in page and "https://" not in page  # no network, no server
    d = _frames_of(page)
    assert [a["label"] for a in d["axes"]] == ["Re c", "Im c"] and len(d["frames"]) == 6
    assert d["axes"][0]["values"] == pytest.approx([-1.2, 0.0, 1.2])
    # the picture for (Re c = 0, Im c = 1.2) is what the player draws for those values
    p = Player(s.engine)
    p.set("set.c[0]", 0.0)
    p.set("set.c[1]", 1.2)
    k = 1 * 2 + 1  # axis 0 index 1, axis 1 index 1 (mixed radix, last axis fastest)
    got = np.asarray(Image.open(io.BytesIO(base64.b64decode(d["frames"][k].split(",", 1)[1]))))
    assert (got == p.frame(0.0, (96, 54))).all()
    assert 'addEventListener("pointermove"' in page and 'type = "range"' in page


def test_the_web_export_refuses_a_grid_that_is_too_big(tmp_path):
    with pytest.raises(ValueError, match=r"pictures would be baked \(limit 600\)"):
        export_web(julia().engine, tmp_path / "x.html", steps=30)
    assert not (tmp_path / "x.html").exists()


# ---------------------------------------------------------------- the notebook


def test_the_notebook_widget_drives_the_same_player():
    w = pytest.importorskip("ipywidgets")
    s = julia("auto") if gl_available() else None
    if s is None:
        pytest.skip("needs OpenGL")
    from phasor.interactive import notebook

    box = notebook(s.engine, size=(96, 54))
    assert isinstance(box, w.VBox) and set(box.sliders) == {"set.c[0]", "set.c[1]"}
    before = bytes(box.picture.value)
    box.sliders["set.c[0]"].value = 0.3
    assert box.player.values["set.c[0]"] == pytest.approx(0.3)
    assert bytes(box.picture.value) != before and bytes(box.picture.value)[:4] == b"\x89PNG"


# ---------------------------------------------------------------- the exit demo


def test_the_julia_explorer_example_has_its_sliders():
    s = runpy.run_path(str(ROOT / "examples" / "julia_explorer.py"), run_name="x")["s"]
    p = Player(s.engine)
    assert [c.label for c in p.controls] == ["Re c", "Im c", "iterations"]
    assert s.engine.lint() == []
    assert p.set("set.max_iter", 100) == 90.0  # snapped to the 30-step grid from 30
    assert p.values["set.max_iter"] == 90.0


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_julia_explorer_set_changes_shape_as_c_moves():
    s = runpy.run_path(str(ROOT / "examples" / "julia_explorer.py"), run_name="x")["s"]
    s.engine._passes = "auto"
    p = Player(s.engine)

    def inside(re, im):
        p.set("set.c[0]", re)
        p.set("set.c[1]", im)
        return float((p.frame(0.0, (160, 90)).max(-1) < 40).mean())

    connected, dust = inside(-0.8, 0.15), inside(0.8, 0.8)  # c in the Mandelbrot set / outside it
    assert connected > 0.03 and dust < 0.01
