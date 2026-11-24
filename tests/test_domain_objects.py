"""The fourier and linalg objects, and the object contract for every new type."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor import Scene, expr, fourier, linalg
from phasor.core.objects import CONTRACT_MEMBERS, registered_types
from tests.conftest import gl_available

gl = [pytest.mark.gl, pytest.mark.skipif(not gl_available(), reason="needs OpenGL")]
NEW = {"room", "source", "listener", "wavefront", "image_sources", "impulse_response",
       "pressure_slice", "epicycles", "partial_sum", "spectrum", "spectrogram",
       "transform_grid", "arrow", "eigen_lines", "matrix"}  # fmt: skip


def test_every_new_type_is_registered_and_keeps_the_object_contract():
    types = registered_types()
    assert set(types) >= NEW
    for name in NEW:
        cls = types[name]
        assert all(hasattr(cls, m) for m in CONTRACT_MEMBERS), name
        assert cls.Style, name  # takes only roles


def heart(n=96):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(0.5 * 16 * math.sin(a) ** 3 / 5, 0.5 * (13 * math.cos(a) - 5 * math.cos(2 * a)) / 5, 0.0)
            for a in t]  # fmt: skip


def test_epicycles_tip_follows_the_curve_and_more_terms_fit_better():
    pts = heart()
    s = Scene("epi", passes=None)
    s.set("camera.height", 6)
    s.add(fourier.Epicycles(id="e", points=pts, terms=40, phase=expr("t / 4")))
    e = s.engine
    for k in (0, 12, 40):
        tip = e.state_at(k / 96 * 4.0)["e"].impl.features(e.state_at(k / 96 * 4.0))["tip"]
        assert tip[:2] == pytest.approx(pts[k][:2], abs=0.05)  # 40 circles: very close
    errs = []
    for terms in (4, 12, 40):
        sc = Scene("e2", passes=None)
        sc.add(fourier.Epicycles(id="e", points=pts, terms=terms))
        errs.append(sc.engine.inspect(0.0, id="e")["objects"][0]["semantic"]["max_error"])
    assert (
        errs[0] > errs[1] >= errs[2]
    )  # the heart is a low-order curve: 12 circles already nail it
    with pytest.raises(Exception, match="points"):
        Scene("bad", passes=None).add(fourier.Epicycles(id="b", points=[(0, 0, 0)] * 2))


def test_partial_sum_spectrum_and_spectrogram_report_their_maths():
    s = Scene("fm", passes=None)
    s.set("camera.height", 8)
    s.add(fourier.PartialSum(id="w", kind="square", terms=30, pos=(0, 2, 0)))
    s.add(fourier.Spectrum(id="on", frequencies=(100.0,), amplitudes=(1.0,), window="rect",
                           samples=256, fs=256.0, pos=(-3, -3, 0)))  # fmt: skip
    s.add(fourier.Spectrum(id="off", frequencies=(100.5,), amplitudes=(1.0,), window="rect",
                           samples=256, fs=256.0, pos=(-3, -3, 0)))  # fmt: skip
    s.add(fourier.Spectrum(id="hann", frequencies=(100.5,), amplitudes=(1.0,), window="hann",
                           samples=256, fs=256.0, pos=(-3, -3, 0)))  # fmt: skip
    s.add(fourier.Spectrogram(id="sg", f0=300, f1=900))
    sem = {o["id"]: o["semantic"] for o in s.engine.inspect(0.0)["objects"]}
    assert 0.085 < sem["w"]["gibbs_overshoot"] < 0.1 and sem["w"]["max_error_away_from_jumps"] < 0.1
    assert sem["on"]["leakage"] < 1e-9 < sem["off"]["leakage"]
    assert sem["hann"]["leakage"] < sem["off"]["leakage"] / 20
    assert sem["on"]["peak_hz"] == pytest.approx(100.0) and sem["on"]["bin_width_hz"] == 1.0
    assert sem["sg"]["backdrop"] is True and sem["sg"]["max_frequency"] == 2000.0
    with pytest.raises(Exception, match="one value per frequency"):
        Scene("x", passes=None).add(
            fourier.Spectrum(id="s", frequencies=(1.0, 2.0), amplitudes=(1.0,))
        )


def test_transform_grid_reports_determinant_eigenvalues_and_svd():
    s = Scene("lg", passes=None)
    s.set("camera.height", 8)
    s.add(linalg.TransformGrid(id="g", matrix=((2.0, 1.0), (1.0, 2.0))))
    s.add(linalg.TransformGrid(id="flip", matrix=((0.0, 1.0), (1.0, 0.0)), pos=(5, 0, 0)))
    s.add(linalg.EigenLines(id="eig", matrix=((2.0, 1.0), (1.0, 2.0))))
    s.add(linalg.EigenLines(id="spin", matrix=((0.0, -1.0), (1.0, 0.0))))
    s.add(linalg.Arrow(id="v", start=(0, 0, 0), end=(3, 4, 0)))
    sem = {o["id"]: o["semantic"] for o in s.engine.inspect(0.0)["objects"]}
    assert sem["g"]["determinant"] == 3.0 and sem["g"]["eigenvalues"] == [[3.0, 0.0], [1.0, 0.0]]
    assert sem["g"]["singular_values"] == [3.0, 1.0] and sem["g"]["rank"] == 2
    assert sem["flip"]["flips_orientation"] is True
    assert sem["eig"]["real_directions"] == 2 and sem["spin"]["real_directions"] == 0
    assert sem["v"]["length"] == 5.0
    feats = s.engine.state_at(0.0)["g"].impl.features(s.engine.state_at(0.0))
    assert feats["basis"][:, :2].tolist() == [[2.0, 1.0], [1.0, 2.0]]  # the images of e1 and e2
    with pytest.raises(Exception, match="2x2 or 3x3"):
        Scene("x", passes=None).add(linalg.TransformGrid(id="b", matrix=((1.0,),)))


def test_a_matrix_object_is_linked_to_the_grid_and_follows_its_animation():
    s = Scene("live", passes=None)
    s.set("camera.height", 8)
    s.add(linalg.TransformGrid(id="g"))
    s.add(linalg.Matrix(id="m", matrix=expr("g.matrix"), pos=(-5, 3, 0)))
    s.marker("go", 1.0)
    s.at("go").morph("g.matrix", ((2.0, 0.0), (0.0, 3.0)))
    e = s.engine
    assert e.state_at(0.0)["m"].params.matrix == ((1.0, 0.0), (0.0, 1.0))
    assert e.state_at(3.0)["m"].params.matrix == ((2.0, 0.0), (0.0, 3.0))  # the numbers follow
    mid = e.state_at(1.5)["m"].params.matrix
    assert 1.0 < mid[0][0] < 2.0 and mid == e.state_at(1.5)["g"].params.matrix  # always equal
    assert e.inspect(3.0, id="m")["objects"][0]["semantic"]["determinant"] == 6.0


def test_the_three_by_three_grid_draws_a_cube_and_has_a_volume():
    s = Scene("cube", passes=None)
    s.use_camera("orbit", azimuth=-50, elevation=25, distance=9, drift=0)
    s.add(linalg.TransformGrid(id="g", matrix=((1.0, 0.4, 0.0), (0.0, 1.0, 0.2), (0.3, 0.0, 1.2))))
    sem = s.engine.inspect(0.0, id="g")["objects"][0]["semantic"]
    m = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.2], [0.3, 0.0, 1.2]])
    assert sem["dimension"] == 3 and sem["determinant"] == pytest.approx(np.linalg.det(m), abs=1e-5)


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_everything_draws_and_a_clean_scene_has_no_lint_errors():
    s = Scene("all", passes="auto")
    s.set("camera.height", 10)
    s.add(fourier.PartialSum(id="w", kind="sawtooth", terms=6, pos=(0, 3, 0), extent=(6, 2)))
    s.add(linalg.TransformGrid(id="g", matrix=((1.0, 0.5), (0.0, 1.0)), pos=(0, -2, 0), extent=2))
    s.add(linalg.Matrix(id="m", matrix=expr("g.matrix"), pos=(-6, -3, 0)))
    s.add(fourier.Epicycles(id="e", points=heart(), terms=8, phase=0.3))
    img = s.engine.still(0.0, (480, 270))
    assert img.std() > 6
    assert not [i for i in s.engine.lint() if i.severity == "error"]
