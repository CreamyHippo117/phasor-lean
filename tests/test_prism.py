"""The prism look (the reference style): typeset math, equations in colored
parts, a hue running along a curve, warm/cool fringes on the glow, a vignette, and
equations that drift. Each piece is tested on its own, then the demo as a whole."""

from __future__ import annotations

import itertools
import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Equation, Line, Scene, Text
from phasor.core import text as tm
from phasor.render.passes import _Collector
from tests.conftest import gl_available

ROOT = Path(__file__).parents[1]
SUM = r"$\sum_{n=1}^{N} \dfrac{2\sqrt{x}}{\sqrt{1/4 + \gamma_n^2}}$"


def sem(e, oid, t=0.0):
    return e.inspect(t, id=oid)["objects"][0]["semantic"]


# ---------------------------------------------------------------- typeset math


def test_math_strings_are_typeset_and_measured():
    assert tm.is_math(r"$\psi(x)$") and tm.is_math(r"a $x$ b")
    assert not tm.is_math("cost: 5$") and not tm.is_math(r"\$5 and \$6")
    assert tm.box("plain") == 1.0
    assert tm.box(SUM) > 1.5 * tm.box(r"$x$")  # a sum with a fraction is a tall box
    assert tm.aspect(r"$x + y + z$") > tm.aspect(r"$x$")
    assert tm.depth(r"$\gamma_n$") > tm.depth(r"$a$") >= 0  # a descender goes below
    assert tm.missing_glyphs(r"$\psi \gamma \rho \sqrt{x}$") == ()  # mathtext has them all


def test_the_bitmap_fills_exactly_the_quad_that_bounds_report():
    for text in (r"$\psi(x) \approx x - \ln 2\pi$", SUM, "plain label"):
        px = 40
        bmp = tm.rasterize(text, px)
        assert bmp.shape == (int(np.ceil(tm.box(text) * px)), int(np.ceil(tm.aspect(text) * px)))
        assert bmp.max() > 200  # something was drawn
        corner, wv, hv = tm.quad(text, (0, 0, 0), 1.0, "center", np.eye(3)[0], np.eye(3)[1])
        assert wv[0] == pytest.approx(tm.aspect(text)) and hv[1] == pytest.approx(tm.box(text))
        assert corner[1] == pytest.approx(-tm.box(text) / 2)  # centered on the origin


def test_bad_tex_is_refused_with_the_reason():
    with pytest.raises(Exception, match="cannot typeset"):
        Scene("x", passes=None).add(Text(id="t", text=r"$\frac{1}$"))
    with pytest.raises(Exception, match="cannot typeset"):
        Scene("x", passes=None).add(Equation(id="e", parts=(r"$\notacommand{x}$",)))
    s = Scene("x", passes=None)
    s.add(Text(id="t", text=r"$\psi(x)$"))
    assert sem(s.engine, "t")["math"] is True


# ---------------------------------------------------------------- Equation


def _parts(e: Equation, s: Scene):
    return e._layout(s.engine.state_at(0.0))


def test_equation_parts_share_one_baseline_and_relations_get_space():
    s = Scene("eq", passes=None)
    eq = Equation(id="e", parts=(r"$E(x) =$", r"$\psi$", r"$(x) - \gamma_n$"),
                  colors=("primary", "secondary"))  # fmt: skip
    s.add(eq)
    impl = s.engine.state_at(0.0)["e"].impl
    parts, h, total, _, _ = _parts(impl, s)
    baselines = []
    for part, origin, hh in parts:
        hb, d = tm.box(part) * hh, tm.depth(part) * hh
        baselines.append(origin[1] - (hb / 2 - d))
    assert np.ptp(baselines) < 1e-9  # every part sits on the same line
    widths = sum(tm.aspect(p) * h for p, _, _ in parts)
    assert total == pytest.approx(widths + 5 / 18 * h * tm.MATH_EM)  # one thick space after "="
    assert sem(s.engine, "e")["colors"] == ["primary", "secondary", None]
    with pytest.raises(Exception, match="emphasis roles"):
        Scene("x", passes=None).add(Equation(id="e", parts=("$x$",), colors=("red",)))


# ---------------------------------------------------------------- hue along a curve


def _draw(s: Scene, oid: str):
    st = s.engine.state_at(0.0)
    c = _Collector()
    st[oid].impl.draw(c, st, "color")
    return c.commands


def test_a_hue_runs_along_a_line_only_when_asked():
    pts = [(x, np.sin(x), 0.0) for x in np.linspace(0, 6, 50)]
    s = Scene("h", passes=None)
    s.add(Line(id="plain", points=pts))
    s.add(Line(id="hue", points=pts, hue_along=1.0))
    ((_, plain),) = _draw(s, "plain")
    ((_, hued),) = _draw(s, "hue")
    assert plain["colors"] is None
    cols = hued["colors"]
    assert cols.shape == (100, 4)  # two vertices (both sides of the stroke) per point
    assert np.abs(cols[0] - cols[-1]).max() > 0.1 or np.abs(cols[0] - cols[50]).max() > 0.1
    assert np.abs(np.diff(cols[::2], axis=0)).max() < 0.15  # a smooth run, no jumps


# ---------------------------------------------------------------- life: float, drift, hue


def test_life_floats_equations_and_flows_hue():
    s = Scene("live", passes=None, life="lively")
    s.add(Equation(id="e", parts=("$x$",)))
    s.add(Equation(id="still", parts=("$y$",), tags=("exact",)))
    s.add(Line(id="l", points=[(0, 0, 0), (1, 1, 0)], hue_along=1.0))
    e = s.engine
    a, b = e.state_at(1.0), e.state_at(4.0)
    assert a["e"].params.pos != b["e"].params.pos
    assert max(abs(v) for v in a["e"].params.pos) < 0.2  # a drift, not a journey
    assert a["still"].params.pos == b["still"].params.pos == (0.0, 0.0, 0.0)
    assert a["l"].params.hue != b["l"].params.hue
    assert e.state_at(1.0)["e"].params.pos == a["e"].params.pos  # pure in t


# ---------------------------------------------------------------- the look and the lens


def test_the_prism_look_and_theme():
    from phasor.style.looks import available_looks, load_look
    from phasor.style.theme import load_theme

    assert "prism" in available_looks()
    look = load_look("prism")
    assert look.theme == "prism" and look.life == "lively"
    t = load_theme("prism")
    assert t.bloom["fringe"] > 0 and t.bloom["vignette"] > 0
    assert sum(t.background[:3]) < sum(load_theme("midnight_glow").background[:3])  # darker
    assert "fringe" not in load_theme("midnight_glow").bloom  # the default look is unchanged
    s = Scene("p", look="prism", passes=None)
    s.add(Line(id="l", points=[(0, 0, 0), (1, 1, 0)]))
    assert s.engine.state_at(0.0)["l"].params.hue_along > 0  # the look's default


def test_the_lens_leaves_paint_in_the_middle_alone():
    """Fringe works on the glow only and vignette darkens the corners only: plain paint in
    the middle of the frame comes out exactly as the theme defines it (CLAUDE.md)."""
    from phasor.render.post import finish
    from phasor.style.theme import load_theme

    theme = load_theme("prism")
    img = np.zeros((60, 80, 4), dtype=np.float32)
    img[..., :3], img[..., 3] = 0.5, 1
    out = finish(img, theme, np.zeros_like(img), backend="reference")
    assert out[30, 40, :3] == pytest.approx([0.5] * 3, abs=2e-3)
    assert out[0, 0, 0] < 0.45  # the corner is darker
    glowing = np.zeros_like(img)
    glowing[28:32, 30:34, :3] = 3.0
    out = finish(img, theme, glowing, backend="reference")
    halo = out[..., :3] - 0.5
    warm = np.unravel_index(np.argmax(halo[..., 0] - halo[..., 2]), halo.shape[:2])
    cool = np.unravel_index(np.argmax(halo[..., 2] - halo[..., 0]), halo.shape[:2])
    assert warm != cool  # the halo splits into a warm and a cool side


# ---------------------------------------------------------------- the demo


def _demo():
    return runpy.run_path(str(ROOT / "examples" / "prism_primes.py"), run_name="x")


def test_demo_mathematics_the_zeros_rebuild_the_primes():
    ns = _demo()
    err = ns["error"]
    gaps = [np.median(np.abs(-ns["explicit_sum"](ns["x"], n) - err)) for n in ns["COUNTS"]]
    assert all(b < a for a, b in itertools.pairwise(gaps))  # more zeros, closer
    assert gaps[-1] < 0.2 * 1.0 and gaps[0] > 2.5 * gaps[-1]
    assert ns["zeta_zeros"](3)[0] == pytest.approx(14.134725, abs=1e-5)


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_demo_lints_and_critiques_clean_and_matches_its_golden():
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    e = _demo()["s"].engine
    assert e.lint() == [] and e.critique()["issues"] == []
    frame = e.frame(12.0, (320, 180))
    mid = frame[90, 160, :3]
    assert frame[:, :, :3].max() <= 1.0 and mid.max() < 0.3  # dark stage, light only on curves
    path = GOLDEN / e.pass_runner().renderer_class() / "prism_primes.png"
    sheet = e.storyboard([1.5, 4.0, 12.0], size=(480, 270))["image"]
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97


# ---------------------------------------------------------------- the prism look on everything


def test_a_theme_can_start_from_another_and_strokes_are_off_by_default():
    from phasor.style.theme import load_theme

    vivid, prism = load_theme("prism_vivid"), load_theme("prism")
    assert vivid.bloom == prism.bloom and vivid.values == prism.values  # inherited
    assert vivid.strokes["hue_along"] == 0.8 and len(vivid.strokes["lut"]) > 2
    for name in ("midnight_glow", "prism", "paper", "aurora"):
        assert load_theme(name).strokes["hue_along"] == 0.0  # an option, never the default


def _stroke_colors(look: str, glow: str):
    from phasor import calc
    from phasor.render.passes import _stroke_hue

    s = Scene("g", look=look, passes=None)
    s.add(calc.Graph(id="g", function="sin(x)", xmin=-3, xmax=3, style={"glow": glow}))
    st = s.engine.state_at(0.0)
    c = _Collector(_stroke_hue(st, "color"))
    st["g"].impl.draw(c, st, "color")
    return [cmd.get("colors") for kind, cmd in c.commands if kind == "mesh"]


def test_objects_a_verb_makes_get_the_look_too():
    """A ``sweep`` readout and a ``circumscribe`` ring are dressed like objects the script
    added, so the scene's generated code rebuilds the same document."""
    from phasor import Dot, fourier

    s = Scene("v", look="prism_everywhere", passes=None)
    s.add(Dot(id="point", pos=(0, 0)))
    s.add(fourier.PartialSum(id="wave", kind="square", terms=2, pos=(0, -2, 0)))
    s.marker("go", 1.0)
    s.at("go").sweep("wave.terms", 9)
    s.at("go").circumscribe("point")
    objs = {o.id: o for o in s.engine.document.objects}
    assert objs["wave.readout"].style.get("glow") == "soft"  # the look's counter default
    assert objs["point.ring"].style.get("glow") == "soft"  # its callout default


def test_prism_everywhere_runs_the_hue_along_glowing_strokes_only():
    glowing = _stroke_colors("prism_everywhere", "strong")
    assert glowing and glowing[0] is not None
    hues = np.asarray(glowing[0])[:, :3]
    assert np.ptp(hues, axis=0).max() > 0.3  # the color changes along the curve
    assert _stroke_colors("prism_everywhere", "none") == [None]  # plain strokes stay plain
    assert _stroke_colors("prism", "strong") == [None]  # plain prism: only lines run a hue
    assert _stroke_colors("prism_everywhere", "strong")[0].shape == glowing[0].shape


def test_the_hue_drifts_with_time_and_only_in_the_color_pass():
    from phasor.render.passes import _stroke_hue

    s = Scene("g", look="prism_everywhere", passes=None)
    a, b = (
        _stroke_hue(s.engine.state_at(0.0), "color"),
        _stroke_hue(s.engine.state_at(5.0), "color"),
    )
    assert a[2] != b[2]  # shifted by flow x t: a pure function of time
    assert _stroke_hue(s.engine.state_at(0.0), "id") is None


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_prism_everywhere_matches_its_golden():
    import math

    from PIL import Image

    from phasor import calc, fourier, geo
    from phasor.render.output import save_png
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    s = Scene("everywhere", look="prism_everywhere")
    s.set("camera.height", 9)
    s.add(fourier.PartialSum(id="w", kind="square", terms=15, pos=(-3.5, 2, 0), extent=(6, 2.4)))
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    s.add(fourier.Epicycles(id="e", terms=24, points=[
        (3.5 + 1.8 * math.cos(a) ** 3, 2 + 1.6 * math.sin(a) ** 3, 0) for a in t]))  # fmt: skip
    s.add(calc.Graph(id="g", function="sin(3*x)*exp(-x**2/8)", xmin=-7, xmax=7, pos=(0, -2.2, 0)))
    s.add(geo.Axes(id="ax", dims=2, x_range=(-7, 7, 1), y_range=(-1, 1, 1), pos=(0, -2.2, 0)))
    img = s.engine.still(1.0, (480, 270))
    path = GOLDEN / s.engine.pass_runner().renderer_class() / "prism_everywhere.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, img)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(img, np.asarray(Image.open(path).convert("RGB"))) >= 0.97


# ---------------------------------------------------------------- palettes within the prism looks


def test_a_palette_recolors_the_strokes_and_keeps_the_rest_of_the_look():
    from phasor import calc
    from phasor.render.passes import _stroke_hue
    from phasor.style.theme import PALETTES, load_theme

    assert PALETTES[0] == "spectrum"  # the prism rainbow comes first
    luts = {}
    for pal in PALETTES:
        s = Scene("p", look="prism_everywhere", palette=pal, passes=None)
        s.add(Line(id="ring", points=[(0, 0, 0), (1, 1, 0)]))
        s.add(calc.Graph(id="g", function="sin(x)", xmin=-3, xmax=3))
        st = s.engine.state_at(0.0)
        assert s.engine.document.meta.palette == pal
        assert s.engine.document.object("ring").style["colormap"] == pal  # lines follow too
        luts[pal] = _stroke_hue(st, "color")[0]
        assert load_theme(s.engine.document.meta.theme).bloom["vignette"] > 0  # the lens stays
    assert len({lut.tobytes() for lut in luts.values()}) == len(PALETTES)  # all different
    spread = {k: float(np.ptp(v, axis=1).mean()) for k, v in luts.items()}
    assert spread["moonlight"] < spread["spectrum"] / 3  # moonlight is near white, not a rainbow


def test_no_palette_means_the_themes_own_and_a_bad_one_is_refused():
    from phasor.core.issues import DocumentError

    plain = Scene("p", look="prism_everywhere", passes=None)
    plain.add(Line(id="l", points=[(0, 0, 0), (1, 1, 0)]))
    assert plain.engine.document.meta.palette is None
    assert "colormap" not in plain.engine.document.object("l").style  # the type's own default
    with pytest.raises(DocumentError, match="palette"):
        Scene("p", palette="plaid", passes=None)


def test_switching_palette_on_an_open_scene_is_one_undo_step(tmp_path):
    from phasor.workspace import Workspace
    from phasor.workspace.tools import ToolError

    ws = Workspace(tmp_path)
    ws.scene_new("p", look="prism")
    ws.scene_edit({"op": "add_object", "object": {"id": "l", "type": "line",
                   "params": {"points": [[0, 0, 0], [1, 1, 0]]}}})  # fmt: skip
    assert ws.engine.document.object("l").params["hue_along"] == 0.8  # the prism look's line
    r = ws.scene_palette("gold")
    assert r["ok"] and ws.engine.document.meta.palette == "gold"
    assert ws.engine.document.object("l").style["colormap"] == "gold"
    ws.undo()
    assert ws.engine.document.meta.palette is None
    with pytest.raises(ToolError, match="palette must be one of"):
        ws.scene_palette("plaid")


def test_a_version_6_document_opens_with_the_themes_palette():
    from phasor.core.migrate import migrate
    from phasor.core.schema import SCHEMA_VERSION, Document

    raw = {"schema_version": 6, "meta": {"title": "old"}, "objects": []}
    doc = Document.model_validate(migrate(raw))
    assert doc.schema_version == SCHEMA_VERSION and doc.meta.palette is None


@pytest.mark.gl
@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_the_palettes_match_their_golden():
    from PIL import Image

    from phasor import calc
    from phasor.render.output import save_png
    from phasor.style.theme import PALETTES
    from tests.ssim import ssim
    from tests.test_render import GOLDEN

    tiles = []
    for pal in PALETTES:
        s = Scene("palettes", look="prism_everywhere", palette=pal)
        s.set("camera.height", 4)
        s.add(calc.Graph(id="g", function="sin(3*x)*exp(-x**2/4)", xmin=-3.4, xmax=3.4))
        tiles.append(s.engine.still(1.0, (240, 135)))
    sheet = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
    path = GOLDEN / s.engine.pass_runner().renderer_class() / "prism_palettes.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {path.parent.name} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= 0.97
