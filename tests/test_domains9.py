"""The pde, algebra and calc objects, and two scenes as tests: the heat
equation and the dihedral group, driven through the engine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor import Scene, algebra, calc, expr, pde
from phasor.core.objects import CONTRACT_MEMBERS, registered_types
from phasor.domains import algebra_math as am
from tests.conftest import gl_available

NEW = {"pde_profile", "pde_field", "pde_surface", "polygon", "cayley_table", "clock_diagram",
       "graph", "riemann_sum", "taylor", "vector_field", "cayley_graph", "homomorphism",
       "subgroup_lattice", "galois_diagram", "permutation", "pde_modes", "pde_slice", "limit", "loop_integral"}  # fmt: skip


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASOR_CACHE", str(tmp_path / "cache"))


def sem(e, oid, t=0.0):
    return e.inspect(t, id=oid)["objects"][0]["semantic"]


def test_the_new_types_keep_the_object_contract():
    types = registered_types()
    assert set(types) >= NEW
    for name in NEW:
        assert all(hasattr(types[name], m) for m in CONTRACT_MEMBERS), name
        assert types[name].Style, name


# ---------------------------------------------------------------- exit demo: the heat equation


def test_exit_demo_the_heat_equation_through_the_engine():
    s = Scene("heat", passes=None)
    s.set("camera.height", 8)
    s.add(pde.Profile(id="rod", kind="heat", length=2.0, diffusivity=0.5, modes=(1.0, 0.4),
                      time=expr("t / 10"), pos=(-3, -1, 0)))  # fmt: skip
    s.add(pde.Field(id="plate", kind="heat", mode=(0, 0), diffusivity=0.5, time=expr("t / 10"),
                    size=(4, 4), pos=(4, 0, 0)))  # fmt: skip
    e = s.engine
    heat, error = [], []
    for t in (0.0, 1.0, 2.5, 5.0):
        r = sem(e, "rod", t)
        error.append(r["max_error_vs_exact"])
        heat.append(sem(e, "plate", t)["total_heat"])
    assert max(error) < 3e-3  # the numbers sit on the exact solution all the time
    assert heat == sorted(heat, reverse=True) and heat[-1] < heat[0]  # heat only leaves
    peaks = [sem(e, "plate", t)["max"] for t in (0.0, 2.5, 5.0)]
    assert peaks == sorted(peaks, reverse=True)  # a hot spot spreads out and cools
    assert sem(e, "rod", 5.0)["steps"] > sem(e, "rod", 1.0)["steps"] > 0  # real time stepping
    a, b = sem(e, "rod", 3.0), sem(e, "rod", 3.0)
    assert a == b  # scrubbing is pure
    state = e.state_at(2.0)
    f1 = state["rod"].impl.features(state)["curve"]
    again = e.state_at(2.0)
    assert np.array_equal(f1, again["rod"].impl.features(again)["curve"])


def test_the_decay_rate_the_inspector_reports_is_the_theory_rate():
    s = Scene("rate", passes=None)
    s.add(pde.Profile(id="rod", kind="heat", length=1.0, diffusivity=1.0, modes=(1.0,),
                      time=expr("t / 100")))  # fmt: skip
    e = s.engine
    u0, u1 = sem(e, "rod", 0.0)["max"], sem(e, "rod", 5.0)["max"]  # simulated time 0.05
    assert u1 / u0 == pytest.approx(math.exp(-(math.pi**2) * 0.05), rel=0.01)


def test_waves_advection_and_quantum_profiles_report_their_conserved_quantities():
    s = Scene("others", passes=None)
    s.add(pde.Profile(id="string", kind="wave", speed=1.0, modes=(2,), n=200, time=0.25))
    s.add(pde.Profile(id="pulse", kind="advection", speed=0.5, n=200, time=0.4))
    s.add(
        pde.Profile(id="psi", kind="schrodinger", length=20.0, n=256, dt=0.01, time=1.5, ymax=1.0)
    )
    e = s.engine
    assert sem(e, "string")["max_error_vs_exact"] < 0.03
    assert sem(e, "pulse")["max_error_vs_exact"] < 0.06
    assert sem(e, "psi")["norm"] == pytest.approx(1.0, abs=1e-6)  # probability is conserved
    bad = Scene("bad", passes=None)
    with pytest.raises(Exception, match="steps in"):
        bad.add(pde.Profile(id="far", kind="heat", time=1e6))
        bad.engine.inspect(0.0)


def test_2d_objects_surfaces_and_the_reaction_pattern():
    s = Scene("two", passes=None)
    s.use_camera("orbit", azimuth=-50, elevation=30, distance=12, drift=0)
    s.add(pde.Surface(id="drum", kind="wave", mode=(2, 1), time=0.1, size=(4, 4), n=24))
    s.add(
        pde.Field(id="pattern", kind="reaction", time=400, n=40, seed=2, size=(4, 4), pos=(6, 0, 0))
    )
    e = s.engine
    d = sem(e, "drum")
    assert d["equation"] == "wave" and abs(d["max_error_vs_exact"]) < 0.08
    p = sem(e, "pattern")
    assert p["max"] > p["min"] and p["backdrop"] is True
    assert sem(e, "pattern") == p  # same seed, same pattern
    st = e.state_at(0.0)
    assert st["drum"].impl.features(st)["peak"].shape == (3,)


# ---------------------------------------------------------------- exit demo: the dihedral group


def test_exit_demo_the_dihedral_group_acting_on_a_square_through_the_engine():
    """Rotate a square by r, then flip it: the polygon lands on the group element the table says
    r followed by s is, at every marker."""
    n = 4
    group = am.dihedral(n)
    s = Scene("d4", passes=None)
    s.add(algebra.Polygon(id="sq", n=n))
    s.add(algebra.CayleyTable(id="table", group="D4", pos=(4, 0, 0)))
    s.marker("start", 0.0)
    s.marker("turned", 2.0)
    s.marker("flipped", 4.0)
    s.marker("turned_again", 6.0)
    s.at("turned").morph("sq.rotation", 90.0)
    s.at("flipped").morph("sq.flip", 1.0)
    s.at("turned_again").morph("sq.rotation", 180.0)
    e = s.engine
    seen = [sem(e, "sq", t)["element"] for t in (0.0, 3.5, 5.5)]
    assert seen[0] == "e" and seen[1] == "r"
    assert seen[2] == group.labels[n + 1]  # rotate by one step, then flip: the reflection s r
    assert sem(e, "sq", 2.5)["element"] is None  # mid-turn: no symmetry, no name
    assert sem(e, "sq", 3.5)["flipped"] is False and sem(e, "sq", 5.5)["flipped"] is True
    # one more quarter turn after the flip lands on the element the table predicts
    final = sem(e, "sq", 8.0)
    assert final["element"] == group.labels[n + 2] and final["rotation"] == 180.0
    tab = sem(e, "table")
    assert tab["latin_square"] is True and tab["order"] == 8 and tab["abelian"] is False
    assert tab["subgroup_orders"] == [1, 2, 4, 8]


def test_the_polygons_vertices_match_the_matrices_of_the_group():
    n = 5
    for el in range(2 * n):
        rotation, flip = am.polygon_action(n, el)
        s = Scene("v", passes=None)
        s.add(
            algebra.Polygon(
                id="p", n=n, rotation=math.degrees(rotation), flip=1.0 if flip else 0.0, r=2.0
            )
        )
        v = s.engine.state_at(0.0)["p"].impl.features(s.engine.state_at(0.0))["vertices"][:, :2]
        base = (
            np.array(
                [
                    [
                        math.cos(math.pi / 2 + 2 * math.pi * k / n),
                        math.sin(math.pi / 2 + 2 * math.pi * k / n),
                    ]
                    for k in range(n)
                ]
            )
            * 2.0
        )
        want = base @ am.action_matrix(n, el).T
        assert np.allclose(v, want, atol=1e-9), el
        assert sem(s.engine, "p")["element_index"] == el


def test_the_clock_diagram_shows_a_subgroup_and_its_cosets():
    s = Scene("clock", passes=None)
    s.add(algebra.ClockDiagram(id="c", n=12, step=4))
    s.add(algebra.ClockDiagram(id="whole", n=12, step=5, pos=(6, 0, 0)))
    s.add(algebra.ClockDiagram(id="half", n=12, step=6, pos=(12, 0, 0)))
    e = s.engine
    c = sem(e, "c")
    assert c["subgroup"] == [0, 4, 8] and c["index"] == 4 and c["generator_order"] == 3
    assert all(len(x) == 3 for x in c["cosets"]) and sorted(
        x for coset in c["cosets"] for x in coset
    ) == list(range(12))
    assert sem(e, "whole")["subgroup_order"] == 12  # 5 generates everything: gcd(5, 12) = 1
    assert sem(e, "half")["subgroup"] == [0, 6]
    with pytest.raises(Exception, match="unknown group"):
        Scene("x", passes=None).add(algebra.CayleyTable(id="t", group="Q8"))
    with pytest.raises(Exception, match="at most 24"):
        Scene("x", passes=None).add(algebra.CayleyTable(id="t", group="D13"))


# ---------------------------------------------------------------- calc


def test_a_riemann_sum_converges_as_n_is_animated():
    s = Scene("riemann", passes=None)
    s.add(calc.RiemannSum(id="area", function="x**2", a=0.0, b=3.0, n=4.0, rule="left"))
    s.marker("more", 1.0)
    s.at("more").morph("area.n", 64.0)
    e = s.engine
    errs = [sem(e, "area", t)["error"] for t in (0.0, 2.5)]
    assert errs[1] < errs[0] / 8 and sem(e, "area", 2.5)["exact"] == pytest.approx(9.0)
    assert sem(e, "area", 2.5)["pieces"] == 64 and sem(e, "area", 0.0)["pieces"] == 4
    mid = Scene("mid", passes=None)
    mid.add(calc.RiemannSum(id="area", function="x**2", a=0.0, b=3.0, n=10.0, rule="mid"))
    assert sem(mid.engine, "area")["error"] == pytest.approx(
        3 * 0.3**2 / 24 * 2, rel=1e-3
    )  # (b - a) h^2 f'' / 24
    with pytest.raises(Exception, match="greater than a"):
        Scene("bad", passes=None).add(calc.RiemannSum(id="r", a=2.0, b=1.0))
    with pytest.raises(Exception, match="cannot use|unknown name"):
        Scene("bad2", passes=None).add(calc.RiemannSum(id="r", function="__import__('os')"))


def test_taylor_polynomials_improve_with_order_and_graphs_break_at_poles():
    s = Scene("taylor", passes=None)
    s.add(calc.Taylor(id="t", function="exp(x)", center=0.0, order=1.0))
    s.marker("more", 1.0)
    s.at("more").morph("t.order", 6.0)
    e = s.engine
    lo, hi = sem(e, "t", 0.0), sem(e, "t", 3.0)
    assert hi["max_error_within_1"] < lo["max_error_within_1"] / 50
    assert hi["coefficients"][:4] == pytest.approx([1.0, 1.0, 0.5, 1 / 6], abs=1e-6)
    g = Scene("g", passes=None)
    g.add(calc.Graph(id="tan", function="tan(x)", xmin=-4.0, xmax=4.0))
    assert sem(g.engine, "tan")["breaks"] >= 2  # tan jumps at +-pi/2 (and 3 pi / 2 is outside)


def test_vector_fields_report_divergence_and_curl():
    s = Scene("vf", passes=None)
    s.add(calc.VectorField(id="spin", p="-y", q="x"))
    s.add(calc.VectorField(id="out", p="x", q="y", pos=(0, 6, 0)))
    e = s.engine
    assert sem(e, "spin")["curl"] == "2" and sem(e, "spin")["divergence"] == "0"
    assert sem(e, "out")["divergence"] == "2" and sem(e, "out")["curl"] == "0"
    with pytest.raises(Exception, match="unknown name"):
        Scene("bad", passes=None).add(calc.VectorField(id="v", p="z + 1", q="x"))


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_everything_draws():
    s = Scene("all", passes="auto")
    s.set("camera.height", 12)
    s.add(pde.Profile(id="rod", kind="heat", length=2.0, modes=(1.0,), time=0.1, pos=(-6, 2, 0)))
    s.add(pde.Field(id="plate", kind="heat", mode=(0, 0), time=0.1, size=(3, 3), pos=(5, 4, 0)))
    s.add(algebra.Polygon(id="sq", n=5, rotation=72.0, pos=(-5, -2, 0)))
    s.add(algebra.CayleyTable(id="t", group="S3", pos=(0, -2, 0), cell=0.6))
    s.add(algebra.ClockDiagram(id="c", n=8, step=2, pos=(5, -3, 0), cosets=1.0))
    s.add(calc.RiemannSum(id="r", function="sin(x) + 1.5", a=0, b=6, n=12.0, pos=(-3, -5, 0)))
    s.add(calc.VectorField(id="v", p="-y", q="x", pos=(-1, 4, 0), extent=(4, 3)))
    img = s.engine.still(0.0, (480, 270))
    assert img.std() > 8
    assert not [i for i in s.engine.lint() if i.severity == "error"]


# ---------------------------------------------------------------- algebra, the rest of the table


def test_cayley_graph_and_cosets_and_quotient_through_the_engine():
    s = Scene("alg", passes=None)
    s.add(algebra.CayleyGraph(id="g", group="D4", generators="r s"))
    s.add(algebra.CayleyTable(id="t", group="D4", subgroup="r", cosets=1.0, pos=(6, 0, 0)))
    s.add(algebra.CayleyTable(id="u", group="D4", subgroup="s", pos=(-6, 0, 0)))
    e = s.engine
    g = sem(e, "g")
    assert g["edges"] == 16 and g["generators"] == ["r", "s"] and g["relations"] == {"r": 4, "s": 2}
    t = sem(e, "t")
    assert t["index"] == 2 and t["normal"] and t["quotient"]["order"] == 2
    assert t["cosets"][0] == ["e", "r", "r2", "r3"]  # the subgroup itself comes first
    u = sem(e, "u")
    assert not u["normal"] and "quotient" not in u and u["index"] == 4
    with pytest.raises(Exception, match="does not generate"):
        Scene("x", passes=None).add(algebra.CayleyGraph(id="g", group="D4", generators="r"))
    with pytest.raises(Exception, match="no element"):
        Scene("x", passes=None).add(algebra.CayleyTable(id="t", group="D4", subgroup="q"))


def test_homomorphism_lattice_galois_and_permutation_semantics():
    s = Scene("alg", passes=None)
    s.add(algebra.Homomorphism(id="h", n=12, m=4, a=1))
    s.add(algebra.SubgroupLattice(id="l", group="D4", pos=(0, -5, 0)))
    s.add(algebra.GaloisDiagram(id="gal", primes=(2, 3), pos=(0, 5, 0)))
    s.add(algebra.Permutation(id="p", perm=(1, 2, 0, 4, 3), progress=expr("t / 2"),
                              pos=(8, 0, 0)))  # fmt: skip
    e = s.engine
    h = sem(e, "h")
    assert h["kernel"] == [0, 4, 8] and h["surjective"] and not h["injective"]
    lat = sem(e, "l")
    assert len(lat["subgroups"]) == 10
    normal = {x["name"] for x in lat["subgroups"] if x["normal"]}
    assert {"{e}", "<r>", "<r2>", "D4"} <= normal and "<s>" not in normal
    gal = sem(e, "gal")
    assert gal["degree"] == 4 and len(gal["pairs"]) == 5
    assert all(4 // x["degree"] == x["subgroup_order"] for x in gal["pairs"])
    p0, p1 = sem(e, "p", 0.0), sem(e, "p", 2.0)
    assert p0["cycles"] == "(1 2 3)(4 5)" and p0["order"] == 6 and p0["parity"] == "odd"
    assert not p0["done"] and p1["done"]
    with pytest.raises(Exception, match="not a homomorphism"):
        Scene("x", passes=None).add(algebra.Homomorphism(id="h", n=5, m=3, a=1))


def test_permutation_tokens_land_in_their_target_slots():
    from phasor.domains.algebra import Permutation

    perm = (1, 2, 0, 4, 3)
    obj = Permutation(id="p", perm=perm)
    done = obj.Params(perm=perm, progress=1.0)
    start = obj.Params(perm=perm, progress=0.0)
    slots = obj._slots(done)
    np.testing.assert_allclose(obj._where(start), slots, atol=1e-12)
    np.testing.assert_allclose(obj._where(done), slots[list(perm)], atol=1e-12)


# ---------------------------------------------------------------- pde, the rest of the table


def test_boundary_conditions_are_visible_and_reported():
    s = Scene("bc", passes=None)
    s.add(pde.Profile(id="held", kind="heat", modes=(), time=0.05))
    s.add(pde.Profile(id="wall", kind="heat", modes=(), time=0.05, boundary="neumann",
                      method="spectral", pos=(0, 4, 0)))  # fmt: skip
    e = s.engine
    held, wall = sem(e, "held"), sem(e, "wall")
    assert held["boundary"].startswith("u = 0") and wall["boundary"].startswith("no flux")
    assert wall["method"] == "spectral"
    s0 = Scene("bc0", passes=None)
    s0.add(pde.Profile(id="wall", kind="heat", modes=(), boundary="neumann", method="spectral"))
    # insulated: the heat stays; held at 0: it leaves
    assert wall["total_heat"] == pytest.approx(sem(s0.engine, "wall")["total_heat"], rel=1e-6)
    assert held["total_heat"] < sem(s0.engine, "wall")["total_heat"]


def test_modes_show_separation_of_variables():
    s = Scene("modes", passes=None)
    s.add(pde.Modes(id="m", modes=(1.0, 0.0, 0.5), count=5, diffusivity=0.5, length=2.0,
                    time=expr("t / 4")))  # fmt: skip
    s.add(pde.Modes(id="bump", count=15, time=0.0, pos=(0, 4, 0)))
    e = s.engine
    a0, a1 = sem(e, "m", 0.0), sem(e, "m", 2.0)
    assert a0["amplitudes"][:3] == pytest.approx([1.0, 0.0, 0.5], abs=1e-4)
    k = 0.5 * (np.pi / 2.0) ** 2  # mode 1 decays like exp(-k t)
    assert a1["amplitudes"][0] == pytest.approx(math.exp(-k * 0.5), rel=1e-4)
    # mode 3 decays 9 times faster: the high modes go first
    assert a1["amplitudes"][2] / a0["amplitudes"][2] == pytest.approx(
        math.exp(-9 * k * 0.5), rel=1e-3
    )
    assert a1["partial_sum_error"] < 1e-3
    b = sem(e, "bump")
    assert b["amplitudes"][1] == pytest.approx(0.0, abs=1e-9)  # a centred bump is even


def test_slice_sweeps_through_a_heating_cube():
    s = Scene("cube", passes=None)
    s.add(pde.Slice(id="c", mode=(1, 1, 1), n=12, diffusivity=0.1, time=0.2, at=expr("t / 4")))
    e = s.engine
    edge, mid = sem(e, "c", 0.0), sem(e, "c", 2.0)
    assert edge["slice_max"] == pytest.approx(0.0, abs=1e-12)  # the face is held at 0
    assert mid["slice_max"] == pytest.approx(mid["max"], rel=1e-9)  # the middle has the peak
    assert mid["max_error_vs_exact"] < 0.02
    far = Scene("x", passes=None)
    far.add(pde.Slice(id="c", n=64, diffusivity=1.0, time=50.0))
    with pytest.raises(Exception, match="steps in"):
        sem(far.engine, "c")  # the step is chosen stable; too many of them is refused


def test_the_hottest_value_is_the_brightest_color():
    """Bug found by looking: ember is cyclic (back to dark at its end), so heat fields drew
    their hottest spot dark (and the M8 spectrogram its loudest bins). Heat maps run on ``ramp``."""
    from phasor.core.color import ramp
    from phasor.style.theme import load_theme

    lum = np.array([0.2126, 0.7152, 0.0722])
    for name in ("ember", "ocean", "sequential"):
        lut = np.asarray(load_theme("midnight_glow").values["colormap"][name])
        assert (lut[-1] @ lum) < 0.2 * (lut @ lum).max()  # the raw map ends dark
        r = ramp(lut)
        assert np.argmax(r @ lum) == len(r) - 1  # the ramp ends on its brightest
        assert (r[0] @ lum) < 0.05


# ---------------------------------------------------------------- calc, the rest of the table


def test_limit_shrinks_delta_with_epsilon():
    s = Scene("lim", passes=None)
    s.add(calc.Limit(id="l", function="sin(x)/x", a=0.0, epsilon=expr("0.4 / (1 + t)")))
    e = s.engine
    wide, narrow = sem(e, "l", 0.0), sem(e, "l", 3.0)
    assert wide["limit"] == pytest.approx(1.0) and wide["holds"] and narrow["holds"]
    assert narrow["epsilon"] < wide["epsilon"] and 0 < narrow["delta"] < wide["delta"]
    assert wide["worst_inside"] < wide["epsilon"]
    with pytest.raises(Exception, match="no finite limit"):
        Scene("x", passes=None).add(calc.Limit(id="l", function="1/x", a=0.0))


def test_loop_integral_shows_flux_and_greens_theorem():
    s = Scene("green", passes=None)
    s.add(calc.LoopIntegral(id="out", p="x", q="y", radius=1.0, progress=expr("t / 4")))
    s.add(calc.LoopIntegral(id="spin", p="x**2 - y", q="x*y", measure="circulation",
                            center=(0.3, 0.2), radius=0.9, pos=(5, 0, 0)))  # fmt: skip
    e = s.engine
    half = sem(e, "out", 2.0)
    assert half["so_far"] == pytest.approx(math.pi) and half["total"] == pytest.approx(2 * math.pi)
    assert half["inside"] == "2" and half["theorem_holds"]
    spin = sem(e, "spin")
    assert spin["theorem_holds"] and spin["total"] == pytest.approx(spin["area_integral"], abs=1e-6)


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_rest_of_the_tables_example_lints_clean_and_matches_its_golden():
    import os
    import runpy
    from pathlib import Path

    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    root = Path(__file__).parents[1]
    e = runpy.run_path(str(root / "examples" / "more_wave2.py"), run_name="x")["s"].engine
    assert e.lint() == [] and e.critique()["issues"] == []
    path = GOLDEN / e.pass_runner().renderer_class() / "more_wave2.png"
    sheet = e.storyboard([0.0, 4.0], size=(480, 270))["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.98
