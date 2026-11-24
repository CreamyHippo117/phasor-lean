"""Finished frames as sRGB bytes on the card.

``finish(..., output="srgb8")`` runs the ``srgb8`` kernel where the frame already is, so a
render copies 6 MB of bytes back per 1080p frame instead of 33 MB of float32. The bytes
must be the ones the host encode (``render.output.encode_srgb8``) gives: at most one step
apart, on almost no pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Dot, Scene, Text
from phasor.render import gpu
from phasor.render.gpu import Tile
from phasor.render.gpu.bridge import to_host
from phasor.render.output import encode_srgb8
from phasor.render.post import finish
from phasor.style.theme import available_themes, load_theme


def _hdr(seed: int, h: int = 54, w: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((h, w, 4), dtype=np.float32) * 3)


def _close_bytes(got: np.ndarray, want: np.ndarray) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype == np.uint8
    diff = np.abs(got.astype(np.int16) - want)
    assert diff.max() <= 1 and (diff > 0).mean() <= 1e-4, (int(diff.max()), (diff > 0).mean())


def test_finish_gives_bytes_and_the_reference_path_is_the_host_encode():
    theme = load_theme("midnight_glow")
    image, glow = _hdr(1), _hdr(2)
    linear = finish(image, theme, glow, backend="reference")
    got = finish(image, theme, glow, backend="reference", output="srgb8")
    assert got.dtype == np.uint8 and got.shape == (*image.shape[:2], 3)
    np.testing.assert_array_equal(got, encode_srgb8(linear))  # the same code: bit-identical


def test_finish_refuses_an_unknown_output():
    with pytest.raises(ValueError):
        finish(_hdr(1), load_theme("midnight_glow"), output="png")


def _paint() -> np.ndarray:
    """Every paint color of every theme, as linear RGBA pixels (one row)."""
    colors = []
    for name in available_themes():
        for role in load_theme(name).values["emphasis"].values():
            colors.append(role)
    return np.ascontiguousarray(np.array([colors], dtype=np.float32))


def test_theme_paint_survives_the_tone_map_and_encode_on_every_backend():
    """Paint below the knee comes out as its exact sRGB bytes (CLAUDE.md: paint is unchanged
    by the tone map), on the card the same as on the host."""
    px = _paint()
    tile = Tile.full(px.shape[1], 1)
    knee = load_theme("midnight_glow").bloom["knee"]
    below = (px[..., :3] <= knee).all(axis=-1)[0]
    want = encode_srgb8(px)
    for backend in ("reference", "auto"):
        mapped = gpu.run("tonemap", tile, image=px, knee=knee, backend=backend)
        got = to_host(gpu.run("srgb8", tile, image=mapped, backend=backend))
        np.testing.assert_array_equal(got[0, below], want[0, below], err_msg=backend)


@pytest.mark.gpu
def test_srgb8_on_the_card_matches_the_host_encode_on_whole_frames():
    theme = load_theme("midnight_glow")
    for seed in range(3):
        image, glow = _hdr(seed, 540, 960), _hdr(seed + 10, 540, 960)
        _close_bytes(finish(image, theme, glow, output="srgb8"),
                     encode_srgb8(finish(image, theme, glow)))  # fmt: skip


@pytest.mark.gpu
@pytest.mark.gl
def test_rendered_frames_through_the_card_equal_the_old_host_path():
    """The strict check of step 1: real frames (strokes, text, glow, motion blur) through
    the new path are the old path's bytes, at most one step apart on almost no pixels."""
    s = Scene("display")
    s.add(Dot(id="d", pos=(-2.0, 0.0), style={"color": "highlight"}))
    s.add(Text(id="t", text=r"$e^{i\pi} + 1 = 0$", pos=(0.0, 1.5)))
    s.at(0).to("d.pos", (2.0, -1.0, 0.0), duration=2.0)
    e = s.engine
    for t in (0.0, 1.0, 2.0):
        _close_bytes(e.frame(t, (480, 270), output="srgb8"), encode_srgb8(e.frame(t, (480, 270))))
        blurred = e.frame_blurred(t, 60.0, size=(480, 270), output="srgb8")
        _close_bytes(blurred, encode_srgb8(e.frame_blurred(t, 60.0, size=(480, 270))))
