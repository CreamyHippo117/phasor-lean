"""Color math. Everything inside phasor is linear RGBA; sRGB exists only at the edges
(theme files in, final frames out). Contract 4."""

from __future__ import annotations

import numpy as np


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_to_linear(hex_color: str) -> tuple[float, float, float, float]:
    """``#rrggbb`` or ``#rrggbbaa`` (sRGB) -> linear RGBA."""
    h = hex_color.lstrip("#")
    if len(h) not in (6, 8):
        raise ValueError(f"bad hex color: {hex_color!r}")
    rgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    a = int(h[6:8], 16) / 255 if len(h) == 8 else 1.0
    r, g, b = (float(x) for x in srgb_to_linear(rgb))
    return (r, g, b, a)


def luminance(rgb) -> float:
    """Relative luminance of a linear RGB color (Rec. 709 weights)."""
    r, g, b = (float(x) for x in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def over(top, bottom) -> tuple[float, float, float, float]:
    """Composite linear RGBA ``top`` over an opaque ``bottom``."""
    a = float(top[3])
    mixed = (float(t) * a + float(b) * (1 - a) for t, b in zip(top[:3], bottom[:3], strict=True))
    return (*mixed, 1.0)


def contrast_ratio(a, b) -> float:
    """WCAG contrast ratio between two opaque linear colors, from 1 (same) to 21."""
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def mark_color(index: int) -> tuple[int, int, int]:
    """A distinct sRGB color for object index ``index`` (the value in the id pass), used to
    draw set-of-mark outlines. Golden-angle hue steps keep neighbors far apart."""
    import colorsys

    r, g, b = colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.75, 1.0)
    return (round(r * 255), round(g * 255), round(b * 255))


def phase_lut(n: int = 256, lightness: float = 0.74, chroma: float = 0.125):
    """A cyclic, perceptually even color wheel for complex phase: ``(n, 3)`` float32 linear
    RGB. Entry k is the color of phase ``2*pi*k/n``; it walks the OKLCH hue circle at
    constant lightness and chroma, so no hue looks brighter than another (unlike HSV)."""
    hue = 2 * np.pi * np.arange(n) / n + np.pi / 6  # phase 0 (positive reals) is warm red
    a, b = chroma * np.cos(hue), chroma * np.sin(hue)
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = np.stack(
        [
            4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
            -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
            -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
        ],
        axis=1,
    )
    return np.ascontiguousarray(np.clip(rgb, 0.0, 1.0), dtype=np.float32)


def ramp(colormap, dtype=np.float64) -> np.ndarray:
    """A colormap as a ramp for values 0..1. A cyclic one (ember, ocean: made for fractals,
    they rise to their brightest and come back to dark) is cut at its brightest entry, so the
    hottest value is the brightest color and not dark again."""
    lut = np.asarray(colormap, dtype=np.float64)
    if np.abs(lut[0] - lut[-1]).max() < 0.05:
        lum = lut @ np.array([0.2126, 0.7152, 0.0722])
        lut = lut[: int(np.argmax(lum)) + 1]
    return lut.astype(dtype)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Linear sRGB (..., 3) to OKLab (..., 3): a space where equal steps look equal."""
    rgb = np.asarray(rgb, dtype=np.float64)
    m = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                  [0.2119034982, 0.6806995451, 0.1073969566],
                  [0.0883024619, 0.2817188376, 0.6299787005]])  # fmt: skip
    lms = np.cbrt(rgb @ m.T)
    o = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                  [1.9779984951, -2.4285922050, 0.4505937099],
                  [0.0259040371, 0.7827717662, -0.8086757660]])  # fmt: skip
    return lms @ o.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    """OKLab (..., 3) back to linear sRGB (..., 3), clipped to 0..1."""
    lab = np.asarray(lab, dtype=np.float64)
    o = np.array([[1.0, 0.3963377774, 0.2158037573],
                  [1.0, -0.1055613458, -0.0638541728],
                  [1.0, -0.0894841775, -1.2914855480]])  # fmt: skip
    lms = (lab @ o.T) ** 3
    m = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                  [-1.2684380046, 2.6097574011, -0.3413193965],
                  [-0.0041960863, -0.7034186147, 1.7076147010]])  # fmt: skip
    return np.clip(lms @ m.T, 0.0, 1.0)


def smooth_cycle(stops: np.ndarray, n: int) -> np.ndarray:
    """``n`` colours running once round the cyclic list of linear-RGB ``stops``, through
    each stop, eased (smoothstep) between neighbours in OKLab: no corners at the stops, no
    dull greyish mixtures between them (as straight blends in RGB give), and never outside
    the stops' own colours (a spline would overshoot and tint a near-white palette)."""
    lab = linear_to_oklab(np.asarray(stops, dtype=np.float64))
    k = len(lab)
    x = np.arange(n) / n * k
    i = np.floor(x).astype(int)
    t = (x - i)[:, None]
    t = t * t * (3 - 2 * t)
    return oklab_to_linear(lab[i % k] * (1 - t) + lab[(i + 1) % k] * t)
