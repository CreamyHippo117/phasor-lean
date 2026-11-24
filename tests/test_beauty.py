"""The beauty plan: the prism look carried to more of phasor."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from phasor import Scene, cx
from phasor.render.gpu import ComplexFunction, Tile, get
from tests.conftest import gl_available

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden"
needs_gl = pytest.mark.skipif(not gl_available(), reason="needs OpenGL")


def _golden(name: str, sheet: np.ndarray, klass: str, threshold: float = 0.98) -> None:
    from PIL import Image

    from phasor.render.output import save_png
    from tests.ssim import ssim

    path = GOLDEN / klass / f"{name}.png"
    if os.environ.get("PHASOR_UPDATE_GOLDEN"):
        save_png(path, sheet)
    if not path.exists():
        pytest.skip(f"no {klass} golden yet; create with PHASOR_UPDATE_GOLDEN=1")
    assert ssim(sheet, np.asarray(Image.open(path).convert("RGB"))) >= threshold


# ---------------------------------------------------------------- 1. prism domain coloring


def test_the_glow_weight_is_one_at_a_zero_and_half_at_the_radius():
    """For f(z) = z the Newton step |f / f'| is exactly the distance to the zero, so the
    weight is 1 / (1 + (d / r)^2): 1 at the zero, 1/2 at d = r, small far away."""
    k = get("domain_glow").reference
    f = ComplexFunction.parse("z")
    tile = Tile.full(201, 201)  # odd: the middle pixel sits on the zero
    out = k(tile, f, (0.0, 0.0), 4.0, radius=0.5, ambient=0.0, contour=0.0)
    w = out[..., 3]
    px = 4.0 / 201
    assert w[100, 100] == pytest.approx(1.0)
    col = 100 + round(0.5 / px)  # a pixel about r to the right
    d = (col - 100) * px
    assert w[100, col] == pytest.approx(1 / (1 + (d / 0.5) ** 2), rel=1e-3)
    assert w[100, 200] < 0.07
    assert out[..., :3].max() <= 1.0  # paint stays in range; the light comes from the glow


def test_a_pole_glows_like_a_zero():
    """1/f has the same Newton step as f, so a pole lights up just like a zero."""
    k = get("domain_glow").reference
    tile = Tile.full(101, 101)
    zero = k(tile, ComplexFunction.parse("z"), (0.0, 0.0), 2.0)[..., 3]
    pole = k(tile, ComplexFunction.parse("1/z"), (0.0, 0.0), 2.0)[..., 3]
    ring = np.ones_like(zero, dtype=bool)
    ring[45:56, 45:56] = False  # away from the centre pixel itself
    np.testing.assert_allclose(pole[ring], zero[ring], rtol=5e-3)  # finite differences


def test_rays_and_rings_are_a_pixel_wide_at_any_zoom():
    k = get("domain_glow").reference
    f = ComplexFunction.parse("z")
    for height in (4.0, 0.01):
        out = k(Tile.full(240, 240), f, (height / 3, height / 5), height, radius=1e-9,
                ambient=0.0, contour=0.0, lines=8)  # fmt: skip
        lit = out[..., 3] > 0.3
        assert 0.01 < lit.mean() < 0.15, height  # thin lines, not a wash or nothing


def test_the_prism_preset_and_looks_pick_the_glowing_picture():
    s = Scene("p", passes=None)
    s.add(cx.DomainColoring(id="a", function="z", preset="prism"))
    s.add(cx.DomainColoring(id="b", function="z"))
    doc = {o.id: o for o in s.engine.document.objects}
    assert doc["a"].params["radius"] > 0 and doc["a"].params["lines"] > 0
    assert doc["b"].params.get("radius", 0) == 0  # the classic picture stays the default
    for look in ("prism", "prism_everywhere"):
        t = Scene("q", look=look, passes=None)
        t.add(cx.DomainColoring(id="c", function="z"))
        assert t.engine.document.object("c").params["radius"] > 0, look


@needs_gl
def test_the_prism_picture_emits_light_and_the_classic_one_does_not():
    def emission(**kw):
        s = Scene("e")
        s.set("camera.height", 4.5)
        s.add(cx.DomainColoring(id="p", function="z**2 - 1", size=(8, 4.5), **kw))
        r = s.engine.pass_runner()
        _, emit = r.color_and_emission(s.engine.state_at(0.0), (160, 90))
        return emit[..., :3]

    assert emission().max() == 0.0  # as before: kernel pictures emit nothing by default
    glow = emission(preset="prism")
    assert glow.max() > 0.5
    hot = np.argwhere(glow.sum(-1) > glow.sum(-1).max() * 0.5)[:, 1]
    assert ((hot > 50) & (hot < 110)).all()  # the light sits round the zeros at +-1


@needs_gl
@pytest.mark.slow
def test_prism_domain_golden_storyboard():
    e = runpy.run_path(str(ROOT / "examples" / "prism_domain.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([0.6, 2.0, 6.0], size=(320, 180))["image"]
    _golden("prism_domain", sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- 2. heat maps


def _old_heatmap(v, lut, size, pw, ph, center, height):
    """The plain picture as it was drawn before the shared helper (kept to check it)."""
    nx, ny = v.shape
    sx, sy = size
    half_w = height * pw / ph / 2
    xs = center[0] + (np.arange(pw) + 0.5) / pw * 2 * half_w - half_w
    ys = center[1] + (0.5 - (np.arange(ph) + 0.5) / ph) * height
    gx = np.clip(xs / sx * (nx - 1), 0, nx - 1)
    gy = np.clip(ys / sy * (ny - 1), 0, ny - 1)
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
    fx, fy = (gx - x0)[None, :], (gy - y0)[:, None]
    val = (v[x0][:, y0].T * (1 - fx) * (1 - fy) + v[x1][:, y0].T * fx * (1 - fy)
           + v[x0][:, y1].T * (1 - fx) * fy + v[x1][:, y1].T * fx * fy)  # fmt: skip
    return lut[np.clip((val * (len(lut) - 1)).astype(int), 0, len(lut) - 1)]


def test_the_plain_heat_map_is_exactly_what_it_was():
    from phasor.domains.heatmap import heat_pixels, lut_colour

    rng = np.random.default_rng(3)
    v = rng.random((17, 11))
    lut = rng.random((64, 3)).astype(np.float32)
    px = heat_pixels(v, (4.0, 3.0), lut_colour(lut))(80, 60, (2.0, 1.5), 3.0)
    np.testing.assert_array_equal(px[..., :3], _old_heatmap(v, lut, (4.0, 3.0), 80, 60,
                                                            (2.0, 1.5), 3.0))  # fmt: skip
    assert (px[..., 3] == 1).all()


def test_smooth_sampling_passes_through_the_grid_and_has_no_kinks():
    from phasor.domains.heatmap import sample

    x = np.linspace(0, 1, 9)
    grid = np.sin(3 * x)[:, None] * np.ones((1, 4))
    g = np.arange(9.0)
    np.testing.assert_allclose(sample(grid, g, np.zeros(1), "smooth")[0], grid[:, 0])
    fine = np.linspace(0, 8, 400)
    smooth = sample(grid, fine, np.zeros(1), "smooth")[0]
    linear = sample(grid, fine, np.zeros(1), "linear")[0]
    kinks = lambda y: np.abs(np.diff(y, 2)).max()  # noqa: E731
    assert kinks(smooth) < kinks(linear) / 5  # no corners at the grid points
    inner = (fine > 1) & (fine < 7)  # the ends repeat the end value: less exact there
    truth = np.sin(3 * fine / 8)
    assert np.abs(smooth - truth)[inner].max() < np.abs(linear - truth)[inner].max()


def test_isolines_are_about_a_pixel_wide_at_any_scale():
    from phasor.domains.heatmap import isoline_weight

    for n in (100, 400):
        ramp_ = np.tile(np.linspace(0, 1, n), (8, 1))  # 0..1 across n pixels
        lines = isoline_weight(ramp_, 4)[4]
        lit = lines > 0.5
        assert lit.sum() == pytest.approx(3 * 2, abs=3)  # 3 inner levels, about 2 px each
        assert lines[n // 4 - 2 : n // 4 + 3].max() > 0.9


def test_the_prism_heat_preset_and_looks():
    from phasor.domains import pde

    s = Scene("h", passes=None)
    s.add(pde.Field(id="a", kind="heat", preset="prism"))
    s.add(pde.Field(id="b", kind="heat"))
    assert s.engine.document.object("a").params["isolines"] > 0
    assert not s.engine.document.object("b").params.get("smooth", False)
    t = Scene("look", look="prism", passes=None)
    t.add(pde.Field(id="c", kind="wave"))
    c = t.engine.document.object("c")
    assert c.params["smooth"] and c.style["glow"] == "soft"


@needs_gl
def test_a_prism_heat_map_glows_where_hot_and_melts_into_the_background():
    from phasor.domains import pde

    def frame(**kw):
        s = Scene("glow")
        s.set("camera.height", 4.0)
        s.add(pde.Field(id="f", kind="heat", mode=(0, 0), time=0.02, scale=0.3,
                        size=(3.0, 3.0), **kw))  # fmt: skip
        r = s.engine.pass_runner()
        pic, emit = r.color_and_emission(s.engine.state_at(0.0), (120, 90))
        return s, pic[..., :3], emit[..., :3]

    s, _, emit = frame()
    assert emit.max() == 0.0
    s, pic, emit = frame(preset="prism")
    assert emit[45, 60].sum() > 0.1  # the hot middle shines
    corner = pic[10, 30]  # inside the plate, far from the heat
    assert np.abs(corner - np.asarray(s.engine.state_at(0.0).background[:3])).max() < 0.05


@needs_gl
@pytest.mark.slow
def test_prism_heat_golden_storyboard():
    e = runpy.run_path(str(ROOT / "examples" / "prism_heat.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([0.5, 3.0, 7.0], size=(320, 180))["image"]
    _golden("prism_heat", sheet, e.pass_runner().renderer_class())


def _field_pixels(obj, look="prism", t=0.0, size=(200, 200)):
    """What ``obj`` asks ``ctx.field`` to draw, as (pixels, background), over its whole
    rectangle (the camera framing exactly that)."""
    from phasor.render.passes import _Collector

    s = Scene("f", look=look, passes=None)
    s.add(obj)
    st = s.engine.state_at(t)
    c = _Collector()
    st[obj.id].impl.draw(c, st, "color")
    (cmd,) = [cmd for kind, cmd in c.commands if kind == "field"]
    w, h = (np.linalg.norm(cmd["right"]), np.linalg.norm(cmd["up"]))
    px = cmd["pixels"](size[0], size[1], (w / 2, h / 2), h)
    return px, np.asarray(st.background[:3]), s.engine.document.object(obj.id)


def _hue(rgb: np.ndarray) -> float:
    import colorsys

    return colorsys.rgb_to_hsv(*np.clip(rgb, 0, None) / max(float(rgb.max()), 1e-9))[0]


def test_prism_heat_glows_in_spectrum_hues_and_fades_into_the_background():
    from phasor.domains import pde

    px, bg, doc = _field_pixels(pde.Field(id="f", kind="heat", mode=(0, 0), time=0.02,
                                          scale=0.5, size=(3.0, 3.0)))  # fmt: skip
    assert doc.params["hue_cycles"] > 0 and doc.params["fade"] > 0
    assert doc.style["colormap"] == "spectrum"
    assert px[100, 100, 3] > 0.5  # the hot middle emits light (alpha is the glow weight)
    for edge in (px[0, :], px[-1, :], px[:, 0], px[:, -1]):  # no hard rectangle
        assert np.abs(edge[:, :3] - bg).max() < 0.01 and edge[:, 3].max() < 0.01
    quiet = px[30, 30]  # inside the panel, far from the heat: the dark ground
    assert np.abs(quiet[:3] - bg).max() < 0.03


def test_prism_isolines_take_the_hue_of_their_level():
    from phasor.domains import pde

    px, bg, _ = _field_pixels(pde.Field(id="f", kind="heat", mode=(0, 0), time=0.02,
                                        scale=0.5, size=(3.0, 3.0)), size=(400, 400))  # fmt: skip
    row = px[200, 200:, :]  # from the hot middle outwards
    lit = row[:, 3]
    peaks = [i for i in range(1, len(lit) - 1)
             if lit[i] > lit[i - 1] and lit[i] >= lit[i + 1] and lit[i] > 0.2]  # fmt: skip
    hues = [_hue(row[i, :3] - bg) for i in peaks]
    assert len(peaks) >= 3  # several contour lines cross the row
    assert np.ptp(hues) > 0.15  # ... each in its own colour


def test_prism_drum_and_pressure_and_spectrogram_glow_in_many_hues():
    from phasor.domains import acoustics, pde
    from phasor.domains.fourier import Spectrogram

    for obj in (pde.Field(id="d", kind="wave", mode=(2, 3), scale=1.2, time=0.0),
                acoustics.PressureSlice(id="p", time=0.02),
                Spectrogram(id="s", size=(6.0, 3.0))):  # fmt: skip
        px, bg, _ = _field_pixels(obj)
        lit = px[..., 3] > 0.3
        assert lit.any() and px[..., 3].max() <= 1.0
        hues = [_hue(c - bg) for c in px[lit][:: max(1, lit.sum() // 200), :3]]
        bins = set((np.round(np.array(hues) * 12) % 12).tolist())  # twelfths of the wheel
        assert len(bins) >= 5, obj.type  # not two flat tints (the plain drum has 2)
        assert np.abs(px[0, :, :3] - bg).max() < 0.01  # the top edge melts into the ground


def test_the_spectrogram_trace_runs_through_the_spectrum_as_it_rises():
    from phasor.domains.fourier import Spectrogram

    px, bg, _ = _field_pixels(Spectrogram(id="s", size=(6.0, 3.0)), size=(300, 150))
    ridge = [int(np.argmax(px[:, x, 3])) for x in (60, 240)]  # the tone, early and late
    assert ridge[1] < ridge[0]  # it rises (row 0 is the top)
    early, late = (_hue(px[r, x, :3] - bg) for r, x in zip(ridge, (60, 240), strict=True))
    assert abs(early - late) > 0.1


def test_plain_heat_maps_are_unchanged_by_the_prism_colouring():
    from phasor.domains import pde
    from phasor.domains.heatmap import AMBIENT, HeatParams

    p = HeatParams()
    assert (p.hue_cycles, p.fade, p.ambient, p.contrast) == (0.0, 0.0, AMBIENT, 1.0)
    px, _, doc = _field_pixels(pde.Field(id="f", kind="heat", mode=(0, 0), time=0.02),
                               look=None)  # fmt: skip
    assert doc.params.get("hue_cycles", 0) == 0
    assert (px[..., 3] == 1).all()  # no glow weight: drawn exactly as before


def test_prism_heat_maps_take_a_palette_and_drift_with_life():
    from phasor.domains import pde
    from phasor.style.looks import load_look

    assert {"pde_field", "pde_slice", "pressure_slice", "spectrogram"} <= load_look(
        "prism"
    ).palette_targets()
    s = Scene("p", look="prism", palette="glacier", passes=None)
    s.add(pde.Field(id="f", kind="heat"))
    assert s.engine.document.object("f").style["colormap"] == "glacier"
    hues = {round(s.engine.state_at(t)["f"].params.hue, 6) for t in (0.0, 3.0, 7.0)}
    assert len(hues) > 1  # lively: the colours turn a little


def _surface_commands(look):
    from phasor.domains import pde
    from phasor.render.passes import _Collector

    s = Scene("s", look=look, passes=None)
    s.use_camera("orbit")
    s.add(pde.Surface(id="d", kind="wave", mode=(2, 3), scale=1.2, n=32))
    st = s.engine.state_at(0.0)
    c = _Collector()
    st["d"].impl.draw(c, st, "color")
    c.commands.append(("style", st["d"].style))  # for the checks, not a draw
    return c.commands


def test_the_prism_surface_is_dark_under_a_net_of_glowing_lines_in_many_hues():
    (surface,) = [c for k, c in _surface_commands("prism") if k == "surface"]
    strokes = [c for k, c in _surface_commands("prism") if k == "mesh"]
    assert strokes and all(c["glow"] > 1 for c in strokes)  # the net is light
    rgb = np.concatenate([c["colors"][:, :3] for c in strokes])
    hues = {round(_hue(x) * 12) % 12 for x in rgb[:: max(1, len(rgb) // 400)]}
    assert len(hues) >= 5  # the net runs through the spectrum
    lum = surface["colors"][:, :3].max(axis=1)
    assert lum.min() < 0.02 < lum.max()  # dark at rest, coloured where it swings


def test_the_plain_surface_is_unchanged():
    from phasor.core.color import ramp

    *cmds, (_, style) = _surface_commands(None)
    assert [k for k, _ in cmds] == ["surface"]  # no net
    (surface,) = [c for _, c in cmds]
    lut = ramp(style["colormap"])
    assert np.isin(surface["colors"][:, :3].round(6), lut.round(6)).all()  # ramp entries


@needs_gl
@pytest.mark.slow
def test_prism_drum_3d_golden_storyboard():
    e = runpy.run_path(str(ROOT / "examples" / "prism_drum_3d.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([0.5, 3.0, 7.0], size=(320, 180))["image"]
    _golden("prism_drum_3d", sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- 4. prism fractals


def _glow_distance(out: np.ndarray, radius: float) -> np.ndarray:
    """The distance in pixels the glow weight stands for: w = 1 / (1 + (d / r)^2)."""
    with np.errstate(divide="ignore"):
        return radius * np.sqrt(1 / out[..., 3].astype(np.float64) - 1)


def test_the_glow_distance_is_the_distance_to_the_unit_circle():
    """The Julia set of c = 0 is the unit circle, and there the estimate G / |grad G| from
    the escape count is |z| log|z|: the true distance |z| - 1 to first order."""
    tile = Tile.full(201, 201)
    field = get("escape").reference(tile, (0.0, 0.0), 4.0, 200, julia=(0.0, 0.0),
                                    precision="f64")  # fmt: skip
    out = get("fractal_glow").reference(tile, field, np.eye(3, dtype=np.float32), radius=5.0)
    px = 4.0 / 201
    x, y = np.meshgrid((np.arange(201) - 100) * px, (100 - np.arange(201)) * px)
    r = np.hypot(x, y)
    near = (r > 1.04) & (r < 1.25)  # two pixels and more from the circle
    est = _glow_distance(out, 5.0)[near] * px
    truth = r[near] * np.log(r[near])
    np.testing.assert_allclose(est, truth, rtol=0.1)  # a central difference over 2 pixels
    assert np.median(np.abs(est / truth - 1)) < 0.02
    np.testing.assert_allclose(est, (r - 1)[near], rtol=0.15)
    assert (out[..., 3][r < 0.97] == 0).all()  # inside: no light, the background
    assert (out[..., :3][r < 0.97] == 0).all()


def test_the_mandelbrot_glow_matches_the_distance_estimate_and_lights_the_edge():
    """Against the classic estimate |z| log|z| / |dz/dc| with the derivative carried through
    the iteration: the escape count's gradient gives the same distances, within a little."""
    n, height, centre = 160, 3.0, (-0.6, 0.0)
    tile = Tile.full(n, n)
    field = get("escape").reference(tile, centre, height, 500, precision="f64")
    out = get("fractal_glow").reference(tile, field, np.eye(3, dtype=np.float32), radius=2.0)
    from phasor.render.gpu.registry import plane

    x, y = (a.ravel() for a in plane(tile, centre, height))
    c = x + 1j * y
    z, dz = np.zeros_like(c), np.zeros_like(c)
    live = np.ones(c.shape, dtype=bool)
    for _ in range(500):
        dz[live] = 2 * z[live] * dz[live] + 1
        z[live] = z[live] ** 2 + c[live]
        live &= np.abs(z) <= 256
    with np.errstate(all="ignore"):
        de = (np.abs(z) * np.log(np.abs(z)) / np.abs(dz)).reshape(n, n) / (height / n)
    est = _glow_distance(out, 2.0)
    ok = (field > 0) & (de > 3) & (de < 30)  # away from the set, where pixels resolve it
    ratio = est[ok] / de[ok]
    assert ok.sum() > 500 and 0.8 < np.median(ratio) < 1.25
    assert (np.abs(np.log(ratio)) < np.log(1.6)).mean() > 0.9
    w = out[..., 3]
    edge = (field > 0) & (de < 0.5)
    assert w[edge].mean() > 0.6  # the edge is lit
    assert (w[field < 0] == 0).all()  # the inside is not
    assert w[(field > 0) & (de > 40)].max() < 0.01  # nor the far outside


def test_a_deep_field_glows_too():
    """The glow reads only the escape field, so the perturbation path at 1e-20 lights its
    edges like any other."""
    from tests.test_kernels import DEEP_ORBIT

    tile = Tile.full(96, 96)
    field = get("deep").reference(tile, (0.0, 0.0), 1e-20, DEEP_ORBIT, 2000)
    out = get("fractal_glow").reference(tile, field, np.eye(3, dtype=np.float32), radius=2.0)
    w = out[..., 3]
    assert (field > 0).all()  # the spiral's filaments: no pixel wholly inside at this size
    assert w.max() > 0.95 and 0.05 < (w > 0.5).mean() < 0.9  # lit filaments, not a wash


def test_the_prism_fractal_preset_and_looks():
    s = Scene("f", passes=None)
    s.add(cx.Fractal(id="a", preset="prism"))
    s.add(cx.Fractal(id="b"))
    a, b = s.engine.document.object("a"), s.engine.document.object("b")
    assert a.params["radius"] > 0 and a.style["glow"] == "soft"
    assert b.params.get("radius", 0) == 0  # the plain picture stays the default
    for look in ("prism", "prism_everywhere"):
        t = Scene("q", look=look, passes=None)
        t.add(cx.Fractal(id="c", kind="julia"))
        assert t.engine.document.object("c").params["radius"] > 0, look
    p = Scene("p", look="prism", palette="gold", passes=None)
    p.add(cx.Fractal(id="d"))
    assert p.engine.document.object("d").style["colormap"] == "gold"  # palettes still apply


@needs_gl
def test_the_prism_fractal_emits_light_on_its_edge_and_the_plain_one_does_not():
    def frame(**kw):
        s = Scene("e")
        s.set("camera.center", (0.0, 0.0, 0.0))
        s.set("camera.height", 4.5)
        s.add(cx.Fractal(id="f", pos=(0, 0, 0), size=(8, 4.5), center="-0.6 0", zoom=0.25,
                         max_iter=200, **kw))  # fmt: skip
        r = s.engine.pass_runner()
        pic, emit = r.color_and_emission(s.engine.state_at(0.0), (160, 90))
        return s, pic[..., :3], emit[..., :3]

    _, _, emit = frame()
    assert emit.max() == 0.0  # as before: the plain picture emits nothing
    s, pic, emit = frame(preset="prism")
    assert emit.max() > 0.3
    bg = np.asarray(s.engine.state_at(0.0).background[:3])
    assert np.abs(pic[45, 90] - bg).max() < 0.02  # the main cardioid: the background
    assert emit[45, 90].max() < emit.max() * 0.05


# ---------------------------------------------------------------- 5. pictures


def _png(path: Path, w: int = 40, h: int = 20, alpha: bool = False) -> Path:
    from PIL import Image as PILImage

    a = np.zeros((h, w, 4), dtype=np.uint8)
    a[..., 0] = 255  # red on the left half, blue on the right
    a[:, w // 2 :, 0], a[:, w // 2 :, 2] = 0, 255
    a[..., 3] = 255
    if alpha:
        a[: h // 2, :, 3] = 0  # the top half transparent
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(a, "RGBA").save(path)
    return path


def test_a_picture_is_found_next_to_the_scene_and_keeps_its_shape(tmp_path):
    from phasor import Engine, Image
    from phasor.core.schema import Document, Meta

    _png(tmp_path / "art" / "flag.png", 40, 20)
    s = Scene("pic", passes=None)
    s.engine = Engine(Document(meta=Meta(title="pic")), passes=None, base_dir=tmp_path)
    s.add(Image(id="flag", path="art/flag.png", width=4.0))
    sem = s.engine.inspect(0.0, id="flag")["objects"][0]["semantic"]
    assert sem["pixels"] == [40, 20] and sem["size"] == [4.0, 2.0]
    b = s.engine.state_at(0.0)["flag"].impl.world_bounds(s.engine.state_at(0.0))
    np.testing.assert_allclose([b.lo, b.hi], [(-2, -1, 0), (2, 1, 0)])


def test_a_missing_picture_draws_nothing_and_lint_says_so(tmp_path):
    """Contract 3: a file the scene names that is not there is lint
    ``missing_file`` and the object draws nothing; a path outside the folder is refused."""
    from phasor import Engine, Image
    from phasor.core.issues import DocumentError
    from phasor.core.schema import Document, Meta

    s = Scene("pic", passes=None)
    s.engine = Engine(Document(meta=Meta(title="pic")), passes=None, base_dir=tmp_path)
    s.add(Image(id="x", path="nope.png"))
    issues = [i for i in s.engine.lint() if i.code == "missing_file"]
    assert issues and "nope.png" in issues[0].message and issues[0].severity == "warning"
    assert s.engine.inspect(0.0, id="x")["objects"][0]["semantic"]["missing"]
    with pytest.raises(DocumentError, match="notes.txt"):
        s.add(Image(id="y", path="notes.txt"))
    for outside in ("../secret.png", str(tmp_path / "abs.png")):
        with pytest.raises(DocumentError, match="folder|absolute"):
            s.add(Image(id="z", path=outside))


@needs_gl
def test_a_picture_draws_its_pixels_and_transparency_shows_the_background(tmp_path):
    from phasor import Engine, Image
    from phasor.core.schema import Document, Meta

    _png(tmp_path / "half.png", 40, 20, alpha=True)
    s = Scene("draw")
    s.engine = Engine(Document(meta=Meta(title="draw")), base_dir=tmp_path)
    s.set("camera.height", 2.0)
    s.add(Image(id="p", path="half.png", width=4.0, height=2.0))
    pic = s.engine.pass_runner().color(s.engine.state_at(0.0), (160, 90))[..., :3]
    bg = np.asarray(s.engine.state_at(0.0).background[:3])
    left, right = pic[70, 40], pic[70, 120]  # the opaque bottom half
    assert left[0] > 0.8 and left[2] < 0.1 and right[2] > 0.8 and right[0] < 0.1
    np.testing.assert_allclose(pic[20, 40], bg, atol=0.02)  # transparent: the background


def test_the_ai_cannot_show_a_picture_from_outside_its_folder(tmp_path):
    from phasor.workspace import ToolError, Workspace

    _png(tmp_path / "outside.png")
    root = tmp_path / "project"
    _png(root / "inside.png")
    ws = Workspace(root)
    ws.scene_new("pics")
    ok = ws.scene_edit({"op": "add_object", "object": {"id": "a", "type": "image",
                        "params": {"path": "inside.png"}}})  # fmt: skip
    assert ok["ok"]
    # refused by the path rule itself (core/files.py: ".." leaves the folder), and by the
    # workspace check behind it; either way nothing outside is shown
    try:
        r = ws.scene_edit({"op": "add_object", "object": {"id": "b", "type": "image",
                           "params": {"path": "../outside.png"}}})  # fmt: skip
        assert not r["ok"] and "folder" in str(r)
    except ToolError as e:
        assert "outside" in str(e)
    assert [o.id for o in ws.engine.document.objects] == ["a"]  # the edit was taken back


@needs_gl
@pytest.mark.slow
def test_prism_fractal_golden_storyboard():
    e = runpy.run_path(str(ROOT / "examples" / "prism_fractal.py"), run_name="x")["s"].engine
    # the slow drift of c and the dive are raw to() transitions, as in deep_zoom: a verb's
    # timing is the theme's, a second or two
    assert {i.code for i in e.lint(strict=True)} <= {"raw_motion"}
    assert e.lint() == []
    sheet = e.storyboard([1.2, 4.5, 13.5], size=(320, 180))["image"]
    _golden("prism_fractal", sheet, e.pass_runner().renderer_class())


def test_the_domain_constant_a_animates_and_the_poles_follow():
    """``a`` is a parameter like any other: it tweens, and zeros and poles move with it."""
    s = Scene("a", passes=None)
    s.add(cx.DomainColoring(id="p", function="1/(z**2 + a)", a=(1.0, 0.0)))
    s.marker("m", 0.0)
    s.at("m").to("p.a", (4.0, 0.0), duration=1.0)
    e = s.engine
    first = sorted(y for _, y in e.inspect(0.0, id="p")["objects"][0]["semantic"]["poles"])
    last = sorted(y for _, y in e.inspect(1.0, id="p")["objects"][0]["semantic"]["poles"])
    assert np.allclose(first, [-1, 1], atol=1e-6) and np.allclose(last, [-2, 2], atol=1e-6)
    with pytest.raises(Exception, match="DomainColoring only"):
        s.add(cx.ComplexMap(id="m", function="z + a"))


@needs_gl
def test_picture_golden_storyboard():
    e = runpy.run_path(str(ROOT / "examples" / "picture.py"), run_name="x")["s"].engine
    assert e.lint(strict=True) == []
    sheet = e.storyboard([0.8, 2.6, 5.0], size=(320, 180))["image"]
    _golden("picture", sheet, e.pass_runner().renderer_class())


# ---------------------------------------------------------------- the prism default


def test_new_scenes_get_the_prism_look_unless_they_say_otherwise(monkeypatch, tmp_path):
    from phasor.workspace import Workspace

    monkeypatch.delenv("PHASOR_DEFAULT_LOOK", raising=False)  # as outside the test suite
    assert Scene("a", passes=None).engine.document.meta.look == "prism"
    assert Scene("b", look="calm", passes=None).engine.document.meta.look == "calm"
    plain = Scene("c", look="none", passes=None).engine.document.meta
    assert plain.look is None and plain.theme == "midnight_glow"
    themed = Scene("d", theme="paper", passes=None).engine.document.meta
    assert themed.look is None and themed.theme == "paper"  # a chosen theme is kept as is
    ws = Workspace(tmp_path)
    ws.scene_new("e")
    assert ws.engine.document.meta.look == "prism"
    ws.scene_new("f", look="none")
    assert ws.engine.document.meta.look is None


def test_palettes_are_smooth_and_pass_through_their_colours():
    from phasor.core.color import hex_to_linear, linear_to_oklab
    from phasor.style.theme import load_theme

    theme = load_theme("prism")
    for name, stops in (("gold", ["#ffe3a3", "#ffc061", "#ff9a5c", "#ffd27a"]),
                        ("rose", ["#ffb3c7", "#ff7aa2", "#c38bff", "#ff9a8a"])):  # fmt: skip
        lut = np.asarray(theme.values["colormap"][name])
        lab = linear_to_oklab(lut)
        bend = np.abs(np.diff(np.vstack([lab, lab[:2]]), 2, axis=0)).max()
        assert bend < 1e-3, name  # no corners anywhere, the seam included (straight: ~4e-3)
        for k, h in enumerate(stops):  # it passes through every stop
            np.testing.assert_allclose(lut[k * len(lut) // len(stops)], hex_to_linear(h)[:3],
                                       atol=2e-3)  # fmt: skip
