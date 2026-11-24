"""Registry, code generation, tiles, kernels, bridge.

Three levels of evidence, weakest machine first:
- NumPy references, checked against known mathematics (run everywhere);
- the CUDA source itself, compiled for the CPU and compared with the references
  (``tests/cuda_emu.py``; needs a C++ compiler, so Linux CI);
- the real thing on CUDA (``gpu`` marker; skipped without an NVIDIA device).
"""

from __future__ import annotations

import numpy as np
import pytest

from phasor.core.color import luminance, phase_lut
from phasor.domains import fractal_math as fm
from phasor.render import gpu
from phasor.render.gpu import ComplexFunction, FunctionError, KernelError, Tile, plan_tiles
from phasor.render.gpu.kernels.fractal import precision_for
from phasor.render.gpu.registry import active_backend, get, kernel, plane, registered
from tests import cuda_emu

FULL = Tile.full(64, 64)
OFFSET = Tile(8, 4, 48, 40, 64, 48)  # not at the origin, not square: catches index mix-ups
F = ComplexFunction.parse("1/(z**2 + 1)")
FA = ComplexFunction.parse("(z**3 - 1) / (z**2 + a)", constant=True)  # a: a run-time constant
FB = ComplexFunction.parse("(z**5 - 1) / (z - a)", constant=True)  # a blend's second function


def _rgba(seed=0, shape=(24, 32)):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((*shape, 4), dtype=np.float32) * 4.0)


IMAGE = _rgba()
IMAGE_TILE = Tile.full(32, 24)
# Display-range values for the sRGB encode, plus the edges: 0, the curve's linear/power
# switch at 0.0031308, half-way, 1, NaN, negative, above 1 and infinity.
DISPLAY = np.ascontiguousarray(_rgba(3) / np.float32(4.0))
DISPLAY[0, :9, :3] = np.array([0.0, 0.0031308, 0.0031309, 0.5, 1.0, np.nan, -0.2, 1.7, np.inf],
                              dtype=np.float32)[:, None]  # fmt: skip

# Reference orbits for the perturbation kernel: one at the spiral of test_deep_zoom.py (lasts
# every step) and one that escapes after 19, so pixels must rebase onto its start.
DEEP_ORBIT = fm.reference_orbit(fm.parse_point(
    "-0.7756837680090537974694835039347410457282 0.1364673682946901247332744096178487651978"),
    128, 2000)  # fmt: skip
EARLY_ORBIT = fm.reference_orbit(fm.parse_point("-0.75 0.2"), 64, 300)

# Points for the splat kernel: a cloud round the origin, a few far off every tile.
_SPLAT_RNG = np.random.default_rng(3)
SPLAT_POINTS = np.r_[_SPLAT_RNG.normal(0.0, 0.45, (300, 2)), [[5.0, 5.0], [-9.0, 0.0]]]
SPLAT_COLORS = _SPLAT_RNG.random((302, 3)).astype(np.float32)
SPLAT_WEIGHTS = np.r_[_SPLAT_RNG.random(301), 0.0].astype(np.float32)

# Sources for the wave kernel, rows (x, y, amplitude, phase, reach); 1e30: always on.
WAVE_ONE = np.array([[0.3, -0.2, 1.0, 0.0, 1e30]])
WAVE_TWO = np.array([[-0.5, 0.0, 1.0, 0.0, 1e30], [0.5, 0.1, 0.8, 1.0, 1.2]])
WAVE_FOUR = np.array([[0.0, y, 1.0, 0.9 * k, 1e30 if k else 1.5]
                      for k, y in enumerate((-0.6, -0.2, 0.2, 0.6))])  # fmt: skip

# name -> list of (tile, params) cases every backend must agree on
CASES = {
    "escape": [
        (FULL, {"center": (-0.5, 0.0), "height": 2.5, "max_iter": 200}),
        (OFFSET, {"center": (-0.5, 0.0), "height": 2.5, "max_iter": 64, "precision": "f64"}),
        (OFFSET, {"center": (0.0, 0.0), "height": 3.0, "julia": (-0.8, 0.156)}),
        (FULL, {"center": (-0.75, 0.1), "height": 1e-6, "max_iter": 400}),
    ],
    "deep": [
        (FULL, {"center": (0.0, 0.0), "height": 1e-20, "orbit": DEEP_ORBIT, "max_iter": 2000}),
        (OFFSET, {"center": (3e-21, -1e-21), "height": 1e-20, "orbit": DEEP_ORBIT,
                  "max_iter": 2000, "rebase": False}),
        (FULL, {"center": (0.0, -0.1), "height": 0.05, "orbit": EARLY_ORBIT, "max_iter": 300}),
    ],
    "domain": [
        (FULL, {"function": F, "center": (0.0, 0.0), "height": 4.0}),
        (OFFSET, {"function": F, "center": (0.3, -0.2), "height": 3.0, "contour": 0.0,
                  "precision": "f64"}),
        (FULL, {"function": F, "center": (0.0, 0.0), "height": 4.0, "soft": 0.8}),

        (FULL, {"function": FA, "center": (0.0, 0.0), "height": 4.0, "a": 0.3 + 0.7j}),
        (OFFSET, {"function": FA, "center": (0.1, 0.0), "height": 3.0, "a": 0.6j, "to": FB,
                  "blend": 0.4, "precision": "f64"}),
    ],
    "domain_glow": [
        (FULL, {"function": F, "center": (0.0, 0.0), "height": 4.0, "precision": "f64"}),
        (OFFSET, {"function": F, "center": (0.3, 0.8), "height": 1.5, "radius": 0.1,
                  "ambient": 0.0, "contour": 0.4, "precision": "f64"}),
        (FULL, {"function": F, "center": (0.0, 0.0), "height": 4.0}),
        (FULL, {"function": F, "center": (0.0, 0.0), "height": 4.0, "lines": 12,
                "precision": "f64"}),

        (FULL, {"function": FA, "center": (0.0, 0.0), "height": 4.0, "lines": 12,
                "a": -0.2 + 0.6j, "precision": "f64"}),
        (FULL, {"function": FA, "center": (0.0, 0.0), "height": 4.0, "lines": 12,
                "a": 0.6j, "to": FB, "blend": 0.65, "precision": "f64"}),
    ],
    "tonemap": [
        (IMAGE_TILE, {"image": IMAGE, "exposure": 1.7}),
        (IMAGE_TILE, {"image": IMAGE, "bloom": _rgba(5), "bloom_strength": 0.6, "knee": 0.7}),
    ],
    "bright": [(IMAGE_TILE, {"image": IMAGE, "threshold": 1.5})],
    "resize": [
        (Tile.full(16, 12), {"image": IMAGE}),
        (Tile.full(45, 31), {"image": IMAGE}),
        (Tile.full(5, 3), {"image": IMAGE}),
    ],
    "palette": [
        (Tile.full(64, 64), {"field": get("escape").reference(FULL, (-0.5, 0.0), 2.5, 100),
                             "lut": phase_lut(), "density": 0.4, "offset": 0.2,
                             "inside": (0.05, 0.1, 0.2)}),
    ],
    "fractal_glow": [
        (Tile.full(64, 64), {"field": get("escape").reference(FULL, (-0.5, 0.0), 2.5, 100),
                             "lut": phase_lut(), "density": 0.4, "offset": 0.2,
                             "inside": (0.05, 0.1, 0.2), "radius": 1.5, "ambient": 0.1}),
        (Tile.full(48, 40), {"field": get("escape").reference(OFFSET, (0.0, 0.0), 3.0,
                                                              julia=(-0.8, 0.156)),
                             "lut": phase_lut()}),
        (Tile.full(64, 64), {"field": get("deep").reference(FULL, (0.0, 0.0), 1e-20,
                                                            DEEP_ORBIT, 2000),
                             "lut": phase_lut(), "radius": 3.0, "ambient": 0.0}),
    ],
    "fringe": [(IMAGE_TILE, {"image": IMAGE, "amount": 0.01}),
               (IMAGE_TILE, {"image": IMAGE, "amount": 0.0})],
    "vignette": [(IMAGE_TILE, {"image": IMAGE, "strength": 0.6})],
    "srgb8": [(IMAGE_TILE, {"image": DISPLAY}), (IMAGE_TILE, {"image": IMAGE})],
    "accumulate": [  # the first blur sample (float16, as GL stores it), then a running sum
        (IMAGE_TILE, {"image": IMAGE.astype(np.float16), "weight": 0.25}),
        (IMAGE_TILE, {"image": IMAGE.astype(np.float16), "weight": 0.25, "acc": _rgba(7)}),
        (IMAGE_TILE, {"image": IMAGE, "weight": 1.0 / 6, "acc": _rgba(8)}),
        (IMAGE_TILE, {"image": IMAGE[..., :1].copy(), "weight": 0.5}),  # any channel count
    ],
    "gauss": [
        (IMAGE_TILE, {"image": IMAGE, "sigma": 1.5, "axis": 0}),
        (IMAGE_TILE, {"image": IMAGE, "sigma": 2.2, "axis": 1}),
    ],
    "splat": [  # some points fall off the tile, some share pixels, one has no weight
        (FULL, {"points": SPLAT_POINTS, "colors": SPLAT_COLORS, "center": (0.0, 0.0),
                "height": 2.0}),
        (OFFSET, {"points": SPLAT_POINTS, "colors": SPLAT_COLORS, "center": (0.1, -0.05),
                  "height": 1.6, "weights": SPLAT_WEIGHTS, "sigma": 1.4, "gain": 3.0}),
        (OFFSET, {"points": SPLAT_POINTS, "colors": (1.0, 0.5, 0.25), "center": (0.0, 0.0),
                  "height": 2.0, "sigma": 0.7, "curve": False}),
        (FULL, {"points": SPLAT_POINTS, "colors": SPLAT_COLORS, "center": (0.0, 0.0),
                "height": 2.0, "weights": SPLAT_WEIGHTS, "gain": 4.0, "white": 0.8}),
    ],
    "wave_field": [  # one source; two (one switching on, a soft front); four, a hard front
        (FULL, {"sources": WAVE_ONE, "center": (0.0, 0.0), "height": 4.0, "wavenumber": 6.0}),
        (OFFSET, {"sources": WAVE_TWO, "center": (0.2, -0.1), "height": 3.0,
                  "wavenumber": 9.0, "omega_t": 1234.5, "core": 0.1, "edge": 0.3}),
        (FULL, {"sources": WAVE_FOUR, "center": (0.0, 0.0), "height": 5.0, "wavenumber": 7.5,
                "omega_t": -0.7, "precision": "f64"}),
    ],
}  # fmt: skip
ALL_CASES = [(n, t, p) for n, cases in CASES.items() for t, p in cases]
needs_compiler = pytest.mark.skipif(cuda_emu.compiler() is None, reason="no C++ compiler")


def _close(name: str, got: np.ndarray, want: np.ndarray, *, strict: bool) -> None:
    """Agreement between a kernel and its reference. Color kernels may differ on the few
    pixels that sit on a discontinuity (a contour edge, a branch cut); real GPUs also fuse
    multiply-adds, which moves fractal pixels on the set's boundary. Byte outputs (sRGB)
    may differ by one step where pow's last bit tips the rounding, and nowhere else."""
    if want.dtype == np.uint8:
        assert got.shape == want.shape and got.dtype == np.uint8
        diff = np.abs(got.astype(np.int16) - want)
        assert diff.max() <= 1 and (diff > 0).mean() <= 1e-4, (name, int(diff.max()))
        return
    assert got.shape == want.shape and got.dtype == want.dtype == np.float32
    err = np.abs(got.astype(np.float64) - want)
    if got.ndim == 3:
        err = err.max(axis=-1)
    tol, allowed = (1e-4, 0.0) if strict else (2e-2, 0.02)
    if name in ("domain", "domain_glow"):
        allowed = max(allowed, 0.01)
    assert (err > tol).mean() <= allowed, (name, float(err.max()), float((err > tol).mean()))


# ---------------------------------------------------------------- registry


def test_every_kernel_has_a_reference_and_a_signature():
    ks = registered()
    assert sorted(ks) == sorted(CASES)
    assert sorted(ks) == [
        "accumulate",
        "bright",
        "deep",
        "domain",
        "domain_glow",
        "escape",
        "fractal_glow",
        "fringe",
        "gauss",
        "palette",
        "resize",
        "splat",
        "srgb8",
        "tonemap",
        "vignette",
        "wave_field",
    ]
    assert all(callable(k.reference) and "->" in k.signature for k in ks.values())


def test_registration_fails_without_a_matching_reference():
    with pytest.raises(KernelError, match="no NumPy reference"):
        kernel("orphan", "() -> x")(lambda gpu, tile: None)
    with pytest.raises(KernelError, match="reference takes"):
        kernel("skewed", "() -> x", reference=lambda tile, a: None)(lambda gpu, tile, b: None)
    with pytest.raises(KernelError, match="already registered"):
        kernel("escape", "() -> x", reference=lambda tile: None)(lambda gpu, tile: None)
    assert "orphan" not in registered() and "skewed" not in registered()
    with pytest.raises(KernelError, match="no kernel 'nope'"):
        get("nope")


def test_run_falls_back_to_the_reference_and_cuda_can_be_demanded():
    tile, params = CASES["escape"][0]
    want = get("escape").reference(tile, **params)
    np.testing.assert_array_equal(gpu.run("escape", tile, backend="reference", **params), want)
    if gpu.default_gpu() is None:  # auto is the CPU backend when a compiler is found
        auto = active_backend()
        same = want if auto == "reference" else gpu.run("escape", tile, backend=auto, **params)
        np.testing.assert_array_equal(gpu.run("escape", tile, **params), same)
        with pytest.raises(KernelError, match="no CUDA device"):
            gpu.run("escape", tile, backend="cuda", **params)
    with pytest.raises(ValueError, match="backend"):
        gpu.run("escape", tile, backend="opencl", **params)


# ---------------------------------------------------------------- the same source, three ways


@needs_compiler
@pytest.mark.parametrize(("name", "tile", "params"), ALL_CASES, ids=lambda v: str(v)[:24])
def test_cuda_source_matches_reference_when_compiled_for_cpu(name, tile, params):
    k = get(name)
    got = k.launch(cuda_emu.EmuGpu(), tile, **params)
    _close(name, got, k.reference(tile, **params), strict=True)


@pytest.mark.gpu
@pytest.mark.parametrize(("name", "tile", "params"), ALL_CASES, ids=lambda v: str(v)[:24])
def test_cuda_kernel_matches_reference_on_the_gpu(name, tile, params):
    from phasor.render.gpu.bridge import to_host

    got = to_host(gpu.run(name, tile, backend="cuda", **params))
    _close(name, got, get(name).reference(tile, **params), strict=False)


# ---------------------------------------------------------------- view and tiles


def test_plane_maps_pixel_centers_with_y_up():
    x, y = plane(Tile.full(4, 2), (10.0, -1.0), 2.0)  # 4 x 2 world units, centered there
    np.testing.assert_allclose(x[0], [8.5, 9.5, 10.5, 11.5])
    np.testing.assert_allclose(y[:, 0], [-0.5, -1.5])  # row 0 is the top
    xt, yt = plane(Tile(2, 1, 2, 1, 4, 2), (10.0, -1.0), 2.0)
    np.testing.assert_array_equal(xt, x[1:, 2:])
    np.testing.assert_array_equal(yt, y[1:, 2:])


@pytest.mark.parametrize(
    ("w", "h", "bpp", "budget"),
    [
        (3840, 2160, 16, 64 * 2**20),
        (1920, 1080, 64, 10 * 2**20),
        (100, 7, 16, 16 * 30),
        (5, 5, 4, 1),
    ],
)
def test_tile_plans_cover_the_image_once_within_budget(w, h, bpp, budget):
    tiles = plan_tiles(w, h, bpp, budget)
    seen = np.zeros((h, w), dtype=np.int32)
    for t in tiles:
        seen[t.y0 : t.y0 + t.h, t.x0 : t.x0 + t.w] += 1
        assert t.n * bpp <= max(budget, bpp)
    assert (seen == 1).all()
    heights = {t.h for t in tiles}
    assert max(heights) - min(heights) <= max(1, max(heights) // 2)  # no sliver tiles


def test_plan_defaults_and_bad_input():
    assert plan_tiles(64, 64, 16, 10**9) == [Tile.full(64, 64)]
    assert len(plan_tiles(64, 64, 16)) == 1  # the default budget is large
    with pytest.raises(ValueError):
        plan_tiles(0, 10, 16)
    with pytest.raises(ValueError, match="does not fit"):
        Tile(60, 0, 10, 10, 64, 64)
    assert Tile(1, 2, 3, 4, 8, 8).scaled(2) == Tile(2, 4, 6, 8, 16, 16)


@pytest.mark.parametrize("name", ["escape", "domain"])
def test_tiles_stitch_into_exactly_the_full_image(name):
    _, params = CASES[name][0]
    whole = gpu.render(name, 64, 48, backend="reference", **params)
    pieces = gpu.render(name, 64, 48, backend="reference", budget=16 * 64 * 5, **params)
    np.testing.assert_array_equal(whole, pieces)
    assert whole.shape[:2] == (48, 64)


def test_supersampling_averages_each_pixel():
    params = {"function": F, "center": (0.0, 0.0), "height": 4.0, "backend": "reference"}
    fine = gpu.render("domain", 32, 32, **params)
    base = gpu.render("domain", 16, 16, **params)
    soft = gpu.render("domain", 16, 16, supersample=2, budget=16 * 4 * 16 * 3, **params)
    np.testing.assert_allclose(soft, fine.reshape(16, 2, 16, 2, 4).mean(axis=(1, 3)), atol=1e-6)
    assert soft.shape == base.shape and not np.allclose(soft, base)
    with pytest.raises(ValueError):
        gpu.render("domain", 16, 16, supersample=0, **params)


# ---------------------------------------------------------------- fractals


def _escape(c, max_iter=100):
    """The smooth escape count at one point (a one-pixel image centered on it)."""
    return float(get("escape").reference(Tile.full(1, 1), c, 1e-3, max_iter, None, "f64")[0, 0])


def test_mandelbrot_membership_and_smooth_counts():
    for inside in [(0.0, 0.0), (-1.0, 0.0), (-0.2, 0.3), (0.25, 0.0), (-1.75488, 0.0)]:
        assert _escape(inside) == -1.0
    assert 0 < _escape((2.0, 2.0)) < _escape((0.5, 0.5)) < 100  # nearer the set: slower
    # Smooth means no integer steps: along a line outside the set the count changes a
    # little per sample, where a plain iteration count would sit still and then jump by 1.
    line = np.array([_escape((x, 0.6)) for x in np.linspace(1.0, 3.0, 201)])
    steps = -np.diff(line)
    assert (steps > 0).all() and steps.max() < 0.05 and line[0] - line[-1] > 1
    assert np.abs(line - np.round(line)).max() > 0.4  # fractional, not whole iterations
    field = get("escape").reference(FULL, (-0.5, 0.0), 2.5, 200)
    np.testing.assert_allclose(field, field[::-1], atol=1e-3)  # symmetric about the real axis
    assert 0.2 < (field < 0).mean() < 0.4


def test_julia_set_of_zero_is_the_unit_disc():
    k = get("escape").reference
    field = k(FULL, (0.0, 0.0), 3.0, 60, (0.0, 0.0))
    x, y = plane(FULL, (0.0, 0.0), 3.0)
    r = np.hypot(x, y)
    assert (field[r < 0.97] == -1).all() and (field[r > 1.03] >= 0).all()


def test_precision_policy():
    assert precision_for(2.5) == "f32" and precision_for(1e-5) == "f32"
    assert precision_for(9e-6) == "f64"
    deep = {"center": (-0.75, 0.1), "height": 1e-6, "max_iter": 300}
    k = get("escape").reference
    auto, f64, f32 = (
        k(FULL, **deep),
        k(FULL, **deep, precision="f64"),
        k(FULL, **deep, precision="f32"),
    )
    np.testing.assert_array_equal(auto, f64)
    # float32 cannot tell neighboring pixels apart at this depth: it renders in blocks.
    assert (np.diff(f32, axis=1) == 0).mean() > 0.5 > 0.05 > (np.diff(f64, axis=1) == 0).mean()


# ---------------------------------------------------------------- domain coloring and codegen


def test_phase_wheel_is_cyclic_even_and_in_gamut():
    lut = phase_lut()
    assert lut.shape == (256, 3) and lut.dtype == np.float32
    assert lut.min() >= 0 and lut.max() <= 1
    steps = np.abs(np.diff(np.vstack([lut, lut[:1]]), axis=0)).max()
    assert steps < 0.03  # no seam, including from the last entry back to the first
    lum = np.array([luminance(c) for c in lut])
    assert lum.std() / lum.mean() < 0.12  # no hue is much brighter than another (HSV: ~0.5)
    assert len({tuple(np.round(c, 2)) for c in lut[::32]}) == 8


def test_domain_coloring_reads_phase_and_modulus():
    k = get("domain").reference
    lut = phase_lut()
    ident = ComplexFunction.parse("z")
    one = Tile.full(1, 1)

    def color(z, contour=0.0, f=ident):
        return k(one, f, (z.real, z.imag), 1e-9, contour, lut)[0, 0]

    np.testing.assert_allclose(color(1 + 0j), [*lut[0], 1], atol=1e-6)  # phase 0
    np.testing.assert_allclose(color(1j), [*lut[64], 1], atol=1e-5)  # phase pi/2
    np.testing.assert_allclose(color(-1 + 0j), [*lut[128], 1], atol=1e-5)
    np.testing.assert_allclose(color(3 - 3j), [*lut[224], 1], atol=1e-5)  # modulus ignored
    # Contours: brightness climbs within a band and drops when |f| doubles.
    shades = [color(complex(m, 0), contour=0.5)[0] / lut[0][0] for m in (1.0, 1.4, 1.99, 2.0)]
    assert shades == pytest.approx([0.5, 0.5 + 0.5 * np.log2(1.4), 0.5 + 0.5 * np.log2(1.99), 0.5],
                                   abs=1e-4)  # fmt: skip
    np.testing.assert_array_equal(color(1j, f=F), [1, 1, 1, 1])  # the pole of 1/(z^2+1): white
    assert color(0j)[3] == 1 and np.isfinite(color(0j)).all()  # a zero is finite and opaque


@pytest.mark.parametrize(
    ("source", "at", "value"),
    [
        ("1/(z**2 + 1)", 2.0, 0.2),
        ("z**3 - 1", 1j, -1 - 1j),
        ("exp(i*pi*z)", 1.0, -1.0),
        ("sqrt(z)", -4.0, 2j),
        ("(z - 1)/(z + 1)", 3.0, 0.5),
        ("conj(z)*z + abs(z)", 3 + 4j, 30.0),
        ("-z + 2.5e0", 1.0, 1.5),
        ("z**(1/2) * e**z", 0.0, 0.0),
        ("cosh(z)**2 - sinh(z)**2", 0.3 - 2j, 1.0),
        ("3", 5j, 3.0),
    ],
)
def test_functions_parse_and_evaluate(source, at, value):
    f = ComplexFunction.parse(source)
    assert f.numpy()(np.array([at]))[0] == pytest.approx(value, abs=1e-12)
    assert f.latex() and f.c()


@pytest.mark.parametrize(
    "source",
    ["__import__('os').system('true')", "z.real", "w + 1", "f(z)", "z if z else 1", "exp(z, 2)",
     "[z]", "'z'", "1/0", "z +", "lambda: z"],
)  # fmt: skip
def test_only_formulas_are_accepted(source):
    with pytest.raises(FunctionError):
        ComplexFunction.parse(source)


@needs_compiler
@pytest.mark.parametrize(
    "source",
    ["z", "z**3 - 1", "exp(z)", "sin(z)/z", "sqrt(z)", "log(z)", "(z - 1)/(z + 1)", "tanh(z)",
     "conj(z)*z + abs(z)", "z**(1 + i)", "cos(z) + cosh(z) - e", "pi*z**-2", "tan(z) - sinh(z)",
     "z**(-1/2) + z**2.5"],
)  # fmt: skip
def test_generated_c_computes_what_numpy_computes(source):
    """The C that codegen emits (through complex.cuh) against SymPy's own NumPy function:
    two independent routes from one expression."""
    f = ComplexFunction.parse(source)
    k = get("domain")
    params = {"function": f, "center": (0.2, -0.1), "height": 5.0, "contour": 0.4,
              "precision": "f64"}  # fmt: skip
    _close("domain", k.launch(cuda_emu.EmuGpu(), OFFSET, **params),
           k.reference(OFFSET, **params), strict=True)  # fmt: skip


# ---------------------------------------------------------------- post-processing


def test_tonemap_keeps_paint_and_rolls_light_off_towards_white():
    ramp = np.zeros((1, 7, 4), dtype=np.float32)
    ramp[0, :, :3] = np.array([0, 0.18, 0.9, 1, 4, 1e4, np.nan], dtype=np.float32)[:, None]
    ramp[..., 3] = 0.25
    out = get("tonemap").reference(Tile.full(7, 1), ramp)
    v = out[0, :, 0]
    assert v[0] == 0 and v[6] == 0 and v[1] == np.float32(0.18) and v[2] == np.float32(0.9)
    assert (np.diff(v[:5]) > 0).all() and v[5] == pytest.approx(1.0) and v.max() <= 1.0
    assert (out[..., 3] == 0.25).all()
    eps = np.float32(1e-3)  # slope 1 on both sides of the knee: no visible crease
    above = get("tonemap").reference(Tile.full(1, 1), np.full((1, 1, 4), 0.9 + eps, np.float32))
    assert (above[0, 0, 0] - 0.9) / eps == pytest.approx(1.0, abs=2e-2)
    brighter = get("tonemap").reference(Tile.full(7, 1), ramp, exposure=2.0)
    assert (brighter[0, 1:4, 0] > v[1:4]).all()  # (4.0 is already at white)


def test_gauss_keeps_flat_images_and_spreads_an_impulse_evenly():
    k = get("gauss").reference
    flat = np.full((9, 9, 4), 0.7, dtype=np.float32)
    np.testing.assert_allclose(k(Tile.full(9, 9), flat, 2.0, 0), flat, atol=1e-6)
    dot = np.zeros((21, 21, 4), dtype=np.float32)
    dot[10, 10] = 1.0
    t = Tile.full(21, 21)
    blur = k(t, k(t, dot, 1.5, 0), 1.5, 1)
    assert blur.sum(axis=(0, 1)) == pytest.approx([1, 1, 1, 1], abs=1e-5)  # energy kept
    np.testing.assert_allclose(blur, blur[::-1, ::-1], atol=1e-7)
    np.testing.assert_allclose(blur, blur.transpose(1, 0, 2), atol=1e-7)
    assert blur[10, 10, 0] == blur.max() and blur[10, 12, 0] == pytest.approx(
        blur[10, 10, 0] * np.exp(-0.5 * 4 / 1.5**2), rel=1e-5
    )
    only_x = k(t, dot, 1.5, 0)
    assert (only_x[10] > 0).any() and only_x[9].sum() == 0  # axis 0 blurs along x


def test_post_kernels_reject_tiles_and_wrong_images():
    with pytest.raises(KernelError, match="whole image"):
        get("gauss").reference(Tile(0, 0, 16, 12, 32, 24), IMAGE[:12, :16].copy(), 1.0, 0)
    with pytest.raises(KernelError, match="float32"):
        get("tonemap").reference(IMAGE_TILE, IMAGE.astype(np.float64))
    with pytest.raises(KernelError, match="knee"):
        get("tonemap").launch(None, IMAGE_TILE, IMAGE, knee=1.0)
    with pytest.raises(KernelError, match="axis"):
        get("gauss").launch(None, IMAGE_TILE, IMAGE, 1.0, 2)


# ---------------------------------------------------------------- the bridge


@pytest.mark.gl
def test_copy_bridge_round_trips_a_frame_and_reuses_its_texture():
    from phasor.render.glcontext import on_gl_thread, shared_context
    from phasor.render.gpu.bridge import CopyBridge

    ctx = shared_context()
    bridge = CopyBridge(ctx)
    frame = _rgba(3, (18, 40))
    frame[0, 0] = [9.5, 0.0, 0.25, 1.0]  # HDR values and a known corner survive
    on_gl_thread(_round_trip, ctx, bridge, frame)


def _round_trip(ctx, bridge, frame):
    with ctx:
        tex = bridge.upload(frame)
        assert tex.size == (40, 18) and tex.components == 4 and tex.dtype == "f2"
        np.testing.assert_allclose(bridge.read_back(), frame, rtol=1e-3, atol=1e-4)  # float16
        np.testing.assert_allclose(bridge.read_back()[0, 0], [9.5, 0.0, 0.25, 1.0], atol=1e-6)
        assert bridge.upload(frame * 0.5) is tex  # same size: the texture is reused
        np.testing.assert_allclose(bridge.read_back(), frame * 0.5, rtol=1e-3, atol=1e-4)
        assert bridge.upload(_rgba(4, (8, 8))) is not tex
        with pytest.raises(ValueError, match="RGBA"):
            bridge.upload(np.zeros((4, 4, 3), dtype=np.float32))


@pytest.mark.gl
def test_a_kernel_frame_reaches_opengl_through_the_bridge():
    from phasor.render.glcontext import on_gl_thread, shared_context
    from phasor.render.gpu.bridge import CopyBridge

    frame = gpu.render("domain", 96, 54, function=F, center=(0, 0), height=4.0)
    ctx = shared_context()
    bridge = CopyBridge(ctx)

    def check():
        with ctx:
            bridge.upload(frame)
            np.testing.assert_allclose(bridge.read_back(), frame, rtol=1e-3, atol=1e-4)

    on_gl_thread(check)


def test_soft_contours_remove_the_jump_and_keep_the_bands():
    k = get("domain").reference
    lut = phase_lut()
    ident = ComplexFunction.parse("z")
    one = Tile.full(1, 1)

    def shade(m, soft):
        return k(one, ident, (m, 0.0), 1e-9, 0.5, lut, "f32", soft)[0, 0, 0] / lut[0][0]

    just_below, just_above = 1.999999, 2.0
    assert abs(shade(just_below, 0.0) - shade(just_above, 0.0)) > 0.4  # hard: a step
    assert abs(shade(just_below, 0.8) - shade(just_above, 0.8)) < 1e-3  # feathered: continuous
    assert shade(1.2, 0.8) == pytest.approx(shade(1.2, 0.0))  # the rest of the band is untouched


# ---------------------------------------------------------------- lens (the prism look)


def test_fringe_splits_a_glow_into_warm_and_cool_rims_and_keeps_its_light():
    img = np.zeros((60, 80, 4), dtype=np.float32)
    img[28:32, 38:42, :3] = 4.0  # a white light in the middle
    img[..., 3] = 1
    out = get("fringe").reference(Tile.full(80, 60), img, 0.02)
    np.testing.assert_array_equal(out[..., 1], img[..., 1])  # green stays
    ry, rx = np.unravel_index(np.argmax(out[..., 0]), out.shape[:2])
    by, bx = np.unravel_index(np.argmax(out[..., 2]), out.shape[:2])
    assert (rx, ry) != (bx, by)  # red and blue land in different places
    assert out[..., 0].sum() == pytest.approx(img[..., 0].sum(), rel=0.05)  # moved, not lost
    np.testing.assert_array_equal(get("fringe").reference(Tile.full(80, 60), img, 0.0), img)


def test_vignette_leaves_the_middle_and_darkens_the_corners():
    img = np.ones((40, 60, 4), dtype=np.float32)
    out = get("vignette").reference(Tile.full(60, 40), img, 0.5)
    assert out[20, 30, 0] == pytest.approx(1.0, abs=1e-3)
    assert out[0, 0, 0] < 0.6 and out[0, 0, 0] > 0.4  # 1 - 0.5 r^4 with r near 1
    assert (out[..., 3] == 1).all()


def test_the_constant_a_is_a_kernel_argument_not_part_of_the_source():
    """Animating ``a`` must not recompile: the CUDA source is the same for every value, and
    the reference with ``a`` set matches the function with that number written in."""
    from phasor.render.gpu.kernels.domain_coloring import domain_reference, source_for

    fixed = ComplexFunction.parse("(z**3 - 1) / (z**2 + (0.3+0.7j))")
    assert "(a," in FA.c() and "(a," not in ComplexFunction.parse("z + 1").c()
    assert source_for(FA) == source_for(FA)  # one source, whatever a is
    tile = Tile.full(24, 16)
    got = domain_reference(tile, FA, (0.0, 0.0), 4.0, a=0.3 + 0.7j)
    want = domain_reference(tile, fixed, (0.0, 0.0), 4.0)
    assert np.allclose(got, want, atol=1e-5)
    with pytest.raises(FunctionError):
        ComplexFunction.parse("z + a")  # only where the caller allows the constant


@pytest.mark.parametrize("name", ["domain", "domain_glow"])
def test_a_blend_ends_exactly_on_each_function_and_mixes_linearly_between(name):
    """``to`` + ``blend``: at 0 the pixels are the first function's, at 1 the second's (to the
    bit, poles included); in between, w = (1 - b) f + b g, the straight line between them."""
    from phasor.render.gpu.kernels import domain_coloring as dc

    k = get(name).reference
    tile, kw = Tile.full(40, 30), {"center": (0.0, 0.0), "height": 4.0, "a": 0.6j}
    np.testing.assert_array_equal(k(tile, FA, to=FB, blend=0.0, **kw), k(tile, FA, **kw))
    np.testing.assert_array_equal(k(tile, FA, to=FB, blend=1.0, **kw), k(tile, FB, **kw))
    z = np.array([0.3 + 0.2j, -1.1 + 0.4j])
    w = dc.blended(FA, FB, 0.25)(z, 0.6j)
    assert np.allclose(w, 0.75 * FA.numpy()(z, 0.6j) + 0.25 * FB.numpy()(z, 0.6j))
    # without ``to`` the source is the one-function source, as before; the blend value is a
    # kernel argument, never part of the source (an animated blend never recompiles)
    one = dc.source_for(FA) if name == "domain" else dc.glow_source_for(FA)
    assert "blend" not in one and "phasor_f(cplx z, cplx a) { return " + FA.c() in one
    two = dc.source_for(FA, FB) if name == "domain" else dc.glow_source_for(FA, FB)
    assert "double blend" in two and "0.25" not in two
