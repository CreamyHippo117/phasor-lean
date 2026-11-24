"""The reverb story, as a scene the engine can inspect.

The 3D reverb scene is not filmed here (skipped by decision); this is its acceptance
test. It builds the story from the real objects and checks what its lint list asks: the
source and the listener are never hidden by a wall, the impulse-response labels are readable,
the wavefront and arrivals obey the speed of sound, and the pressure field is a cached,
pure simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Scene, Text, ac, expr
from phasor.domains import ac_math
from phasor.inspect.events import problems
from tests.conftest import gl_available

SIZE, SRC, MIC = (8.0, 5.0, 3.0), (2.0, 2.5, 1.5), (6.0, 2.0, 1.5)
MODES = [None, pytest.param("auto", marks=[pytest.mark.gl, pytest.mark.skipif(not gl_available(), reason="gl")])]  # fmt: skip


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_CACHE", str(tmp_path / "cache"))


def reverb(passes, material="curtain"):
    s = Scene("reverb", passes=passes)
    s.use_camera("orbit", azimuth=-62, elevation=28, distance=15, target=(4, 2.5, 1.2), drift=0)
    s.add(ac.Room(id="room", size=SIZE, material=material))
    s.add(ac.Source(id="src", pos=SRC, tags=["key"]))
    s.add(ac.Listener(id="mic", pos=MIC, tags=["key"]))
    s.add(ac.Wavefront(id="wave", center=SRC, radius=expr("343 * (t - 0.002)")))
    s.add(ac.ImageSources(id="ghosts", size=SIZE, source=SRC, order=1, material=material))
    s.add(
        ac.ImpulseResponse(
            id="ir", size=SIZE, source=SRC, listener=MIC, order=10, material=material,
            layer="overlay", pos=(0.15, 0.7, 0.0), extent=(0.9, 0.18), duration=0.3,
        )
    )  # fmt: skip
    s.add(Text(id="ir.label", text="impulse response", layer="overlay", pos=(0.6, 0.45, 0)))
    s.add(Text(id="src.label", text="source"))
    s.add(Text(id="mic.label", text="listener"))
    s.constrain("label(src) near src")
    s.constrain("label(mic) near mic")
    s.constrain("visible_in_3d(src) and visible_in_3d(mic)")
    s.marker("emit", 0.0)
    s.marker("first_hit", 0.006)
    s.marker("tail", 0.02)
    return s


@pytest.mark.parametrize("passes", MODES)
def test_the_source_and_the_listener_are_never_hidden_by_a_wall(passes):
    e = reverb(passes).engine
    for t in (0.0, 0.006, 0.02):
        objs = {o["id"]: o for o in e.inspect(t)["objects"]}
        for name in ("src", "mic"):
            assert objs[name]["on_screen_fraction"] == 1.0
            if passes:  # real pixels: translucent walls must not hide them (a wire wavefront
                # crossing the source may cover a few percent of it, never more)
                assert objs[name]["visible_fraction"] >= 0.95, (t, name, objs[name]["hidden_by"])
                assert not any(h["id"] == "room" for h in objs[name]["hidden_by"])
    codes = {i.code for i in e.lint()}
    assert not codes & {"occluded_label", "unsolved", "offscreen", "contrast"} or passes is None


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_a_solid_wall_would_hide_them_and_the_event_log_says_so():
    s = reverb("auto")
    s.engine.apply(
        {"op": "set_style", "id": "room", "key": "material", "role": "solid"}, origin="script"
    )
    objs = {o["id"]: o for o in s.engine.inspect(0.0)["objects"]}
    assert objs["room"]["visible_fraction"] > 0.95
    hiding = {h["id"] for n in ("src", "mic") for h in objs[n]["hidden_by"]}
    assert "room" in hiding  # now the wall is in the way of one of them


def test_the_wavefront_travels_at_the_speed_of_sound():
    e = reverb(None).engine
    for t in (0.004, 0.01, 0.02):
        st = e.state_at(t)
        assert st["wave"].params.radius == pytest.approx(343.0 * (t - 0.002))
    # it reaches the listener exactly when the direct sound does
    d = float(np.linalg.norm(np.subtract(SRC, MIC)))
    t_hit = d / 343.0 + 0.002
    assert (
        e.state_at(t_hit - 1e-4)["wave"].params.radius
        < d
        < e.state_at(t_hit + 1e-4)["wave"].params.radius
    )
    sem = e.inspect(0.01, id="ir")["objects"][0]["semantic"]
    assert sem["first_arrival_ms"] == pytest.approx(d / 343.0 * 1000, abs=1e-3)
    assert sem["direct_distance"] == pytest.approx(d, abs=1e-3)


def test_the_room_reports_its_physics_and_the_ghosts_are_the_six_walls():
    e = reverb(None).engine
    room = e.inspect(0.0, id="room")["objects"][0]["semantic"]
    assert room["volume"] == 120.0 and room["surface_area"] == 158.0
    assert room["rt60_eyring"] < room["rt60_sabine"]
    ghosts = e.inspect(0.0, id="ghosts")["objects"][0]["semantic"]
    assert ghosts["count"] == 6 and ghosts["weakest_gain"] == pytest.approx(np.sqrt(0.5), abs=1e-4)


def test_more_absorbent_walls_shorten_the_tail_the_plot_reports():
    rt = {}
    for material in ("plaster", "curtain"):
        e = reverb(None, material).engine
        sem = e.inspect(0.0, id="ir")["objects"][0]["semantic"]
        rt[material] = (sem["rt60"], sem["rt60_reliable"], sem["rt60_theory_eyring"])
    assert rt["curtain"][2] < rt["plaster"][2]  # theory: curtains ring less than plaster
    assert rt["curtain"][0] < rt["plaster"][0]  # and so does the fitted decay
    # image sources of order 10 fully describe only 10 * 3 m / c = 87 ms: plaster's tail is
    # longer than that, so the object says its fit is not to be trusted
    assert rt["plaster"][1] is False


@pytest.mark.parametrize("passes", MODES)
def test_the_pressure_field_is_a_cached_pure_simulation(passes, tmp_path):
    def field(t):
        s = Scene("p", passes=passes)
        s.add(ac.PressureSlice(id="f", size=SIZE, source=SRC, time=t, material="curtain"))
        return s.engine.inspect(0.0, id="f")["objects"][0]["semantic"]

    early, later = field(0.006), field(0.05)  # 0.05 s is about 150 steps: three checkpoints
    assert early["peak_pressure"] > 0 and later["peak_pressure"] > 0
    assert early["time"] == 0.006 and early["grid"] == [40, 25]
    assert field(0.05) == later  # same time, same field: a pure function
    assert list((tmp_path / "cache").glob("*/*.npz"))  # checkpoints were cached on disk
    back = field(0.006)  # scrubbing backwards gives the same answer as before
    assert back == early


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_story_draws_and_the_camera_event_log_is_quiet():
    e = reverb("auto").engine
    img = e.still(0.01, (480, 270))
    assert img.shape == (270, 480, 3) and img.std() > 8  # something real was drawn
    log = e.events(0.0, 0.02, hz=4)
    bad = [ev for ev in problems(log) if ev["id"] in ("src", "mic", "src.label", "mic.label")]
    assert bad == [], bad


def test_the_physics_helpers_agree_with_the_objects():
    arr = ac_math.arrivals(SIZE, SRC, MIC, 10, 0.5)
    e = reverb(None).engine
    sem = e.inspect(0.0, id="ir")["objects"][0]["semantic"]
    assert sem["arrivals"] == len(arr["time"])
