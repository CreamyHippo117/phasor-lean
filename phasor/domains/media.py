"""Things brought in from outside: pictures and sound files.

``Image`` puts a PNG or JPEG file in the scene, as a flat rectangle placed, sized and
animated like any object (it fades, moves, flashes, sits in 3D). The file's path is relative
to the scene's folder; the document stores the path, never the pixels, and compiling checks
the file is there (issue ``missing_file``).
"""

from __future__ import annotations

import functools
import os
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, field_validator

from phasor.core import files
from phasor.core.color import srgb_to_linear
from phasor.core.objects import Bounds, DrawContext, ObjectBase, ParamsBase, Pass, register
from phasor.core.state import SceneState
from phasor.core.values import Vec3
from phasor.domains.strokes import polyline

MAX_SOURCE_PX = 8192  # larger pictures are scaled down once, when read


@functools.lru_cache(maxsize=16)
def _read(path: str, mtime: float) -> np.ndarray:
    """The picture as linear RGBA float32, (h, w, 4), row 0 = top. Cached per file version."""
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        im = im.convert("RGBA")
        if max(im.size) > MAX_SOURCE_PX:
            im.thumbnail((MAX_SOURCE_PX, MAX_SOURCE_PX))
        a = np.asarray(im, dtype=np.float32) / 255.0
    rgb = np.asarray(srgb_to_linear(a[..., :3]), dtype=np.float32)
    return np.concatenate([rgb, a[..., 3:]], axis=-1)


@functools.lru_cache(maxsize=8)
def read_wav(path: str, mtime: float) -> tuple[np.ndarray, float]:
    """A .wav file as mono float64 in -1..1 and its sample rate. Cached per file version."""
    from scipy.io import wavfile

    fs, data = wavfile.read(path)
    if data.dtype.kind == "u":  # 8-bit PCM is unsigned, centred on 128
        x = (data.astype(np.float64) - 128.0) / 128.0
    elif data.dtype.kind == "i":
        x = data.astype(np.float64) / float(-np.iinfo(data.dtype).min)
    else:
        x = data.astype(np.float64)
    return (x.mean(axis=1) if x.ndim > 1 else x), float(fs)


def picture(state: SceneState, path: str) -> np.ndarray | None:
    """The picture ``path`` names in the scene's folder (core/files.py), re-read when the file
    changes; None while it is missing or unreadable (lint ``missing_file``: draws nothing)."""
    try:
        file = files.resolve(state.base_dir, path)
        return _read(str(file), os.path.getmtime(file))
    except (OSError, ValueError):
        return None


@register
class Image(ObjectBase):
    """A picture from a file (PNG, JPEG, anything Pillow reads), ``width`` world units wide
    (``height`` 0: keep the picture's shape), centred on ``pos``. Transparent parts show the
    scene's background colour. The ``glow`` style role makes it shine (bright parts most);
    ``frame`` draws a thin outline of light round it."""

    type: ClassVar[str] = "image"
    Files: ClassVar = ("path",)  # in the scene's folder (core/files.py); missing: lint
    Style: ClassVar = {"glow": ("glow", "none"), "frame": ("emphasis", "muted"),
                       "stroke": ("stroke", "hairline")}  # fmt: skip
    Presets: ClassVar = {
        "plain": {"params": {"frame": False}, "style": {"glow": "none"}},
        "framed": {"params": {"frame": True}, "style": {"frame": "primary"}},
        "luminous": {"params": {"frame": True}, "style": {"glow": "soft", "frame": "highlight"}},
    }

    class Params(ParamsBase):
        path: str = Field(min_length=1)  # relative to the scene's folder, and inside it
        pos: Vec3 = (0.0, 0.0, 0.0)  # centre
        width: float = Field(4.0, gt=0)
        height: float = Field(0.0, ge=0)  # 0: from the picture's aspect ratio
        frame: bool = False  # an outline of light round the picture

        @field_validator("path")
        @classmethod
        def _image_file(cls, v: str) -> str:
            ext = v.lower().rsplit(".", 1)[-1] if "." in v else ""
            if ext not in {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"}:
                raise ValueError(f"{v!r} is not a picture file (png, jpg, webp, ...)")
            return files.relative_path(v)

    def _size(self, state: SceneState) -> tuple[float, float]:
        p = self.me(state).params
        if p.height > 0:
            return p.width, p.height
        src = picture(state, p.path)
        if src is None:  # missing: a 16:9 placeholder rectangle (nothing is drawn in it)
            return p.width, p.width * 9 / 16
        h, w = src.shape[:2]
        return p.width, p.width * h / w

    def _rect(self, state: SceneState):
        p = self.me(state).params
        w, h = self._size(state)
        corner = np.asarray(p.pos, dtype=np.float64) - np.array([w / 2, h / 2, 0.0])
        return corner, np.array([w, 0.0, 0.0]), np.array([0.0, h, 0.0])

    def world_bounds(self, state: SceneState) -> Bounds:
        c, r, u = self._rect(state)
        return Bounds.of_points([c, c + r + u])

    def features(self, state: SceneState) -> dict[str, np.ndarray]:
        c, r, u = self._rect(state)
        return {"center": c + (r + u) / 2, "corners": np.array([c, c + r, c + r + u, c + u])}

    def semantics(self, state: SceneState) -> dict[str, Any]:
        p = self.me(state).params
        src = picture(state, p.path)
        h, w = src.shape[:2] if src is not None else (0, 0)
        return {"kind": "image", "path": p.path, "pixels": [w, h], "missing": src is None,
                "size": [round(v, 6) for v in self._size(state)]}  # fmt: skip

    def draw(self, ctx: DrawContext, state: SceneState, pass_: Pass) -> None:
        me = self.me(state)
        p = me.params
        corner, right, up = self._rect(state)
        src = picture(state, p.path)
        if src is None:
            return  # missing file: lint missing_file says so; nothing to draw
        bg = np.asarray(state.background[:3], dtype=np.float32)
        glow = float(me.style.get("glow", 1.0))
        rw, uh = float(right[0]), float(up[1])

        def pixels(pw: int, ph: int, center: tuple[float, float], height: float) -> np.ndarray:
            sh, sw = src.shape[:2]
            half_w = height * pw / ph / 2
            # the visible part of the picture, in its own pixels (row 0 = top)
            x0, x1 = (center[0] - half_w) / rw * sw, (center[0] + half_w) / rw * sw
            y0, y1 = (
                (1 - (center[1] + height / 2) / uh) * sh,
                (1 - (center[1] - height / 2) / uh) * sh,
            )
            xs = np.clip(x0 + (np.arange(pw) + 0.5) / pw * (x1 - x0) - 0.5, 0, sw - 1)
            ys = np.clip(y0 + (np.arange(ph) + 0.5) / ph * (y1 - y0) - 0.5, 0, sh - 1)
            img = _area_sample(src, xs, ys, (x1 - x0) / pw, (y1 - y0) / ph)
            a = img[..., 3:]
            rgb = img[..., :3] * a + bg * (1 - a)  # transparency shows the background
            out = np.empty((ph, pw, 4), dtype=np.float32)
            out[..., :3] = rgb
            out[..., 3] = (rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
                           if glow > 1 else 1.0)  # fmt: skip
            return out

        ctx.field(corner, right, up, pixels=pixels, opacity=p.opacity, glow=glow)
        if p.frame and pass_ == "color":
            cam = state.camera_for(me)
            ring = [corner, corner + right, corner + right + up, corner + up, corner]
            w = me.style["stroke"] * float(cam.units_per_height(corner + (right + up) / 2))
            polyline(ctx, cam, np.asarray(ring), w,
                     color=(*me.style["frame"][:3], p.opacity), glow=2.5)  # fmt: skip


def _area_sample(src: np.ndarray, xs: np.ndarray, ys: np.ndarray, sx: float, sy: float):
    """Bilinear samples of ``src`` at ``xs`` x ``ys``; when a screen pixel covers several
    picture pixels (shrinking), the picture is first box-averaged down to about that scale, so
    a large photo does not shimmer or alias."""
    k = int(max(1, np.floor(min(abs(sx), abs(sy)))))
    if k > 1:
        h, w = (src.shape[0] // k) * k, (src.shape[1] // k) * k
        src = src[:h, :w].reshape(h // k, k, w // k, k, 4).mean(axis=(1, 3))
        xs, ys = (xs + 0.5) / k - 0.5, (ys + 0.5) / k - 0.5
        xs, ys = np.clip(xs, 0, src.shape[1] - 1), np.clip(ys, 0, src.shape[0] - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, src.shape[1] - 1), np.minimum(y0 + 1, src.shape[0] - 1)
    fx, fy = (xs - x0)[None, :, None], (ys - y0)[:, None, None]
    return (src[y0][:, x0] * (1 - fx) * (1 - fy) + src[y0][:, x1] * fx * (1 - fy)
            + src[y1][:, x0] * (1 - fx) * fy + src[y1][:, x1] * fx * fy)  # fmt: skip
