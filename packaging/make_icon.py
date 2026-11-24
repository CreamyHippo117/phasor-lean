"""Draw the app icon (a phasor: a glowing unit circle with its rotating vector) into
``phasor/gui/assets``: icon.png (512 px), icon.ico (Windows), icon.icns is made on macOS
by the packaging step. Run: uv run python packaging/make_icon.py"""

import colorsys
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

S = 512
OUT = Path(__file__).parents[1] / "phasor" / "gui" / "assets"


def draw(size: int = S) -> Image.Image:
    k = 4  # draw big, then shrink: smooth edges
    n = size * k
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    bg = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    ImageDraw.Draw(bg).rounded_rectangle([0, 0, n - 1, n - 1], radius=n * 0.22,
                                         fill=(9, 11, 17, 255))  # fmt: skip
    ink = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(ink)
    c, r, w = n / 2, n * 0.30, max(2, int(n * 0.028))
    for i in range(360):  # the circle, its hue running round it (phase as hue)
        a0, a1 = math.radians(i), math.radians(i + 1.6)
        rgb = colorsys.hsv_to_rgb((i / 360 + 0.55) % 1, 0.65, 1.0)
        d.line([(c + r * math.cos(a0), c - r * math.sin(a0)),
                (c + r * math.cos(a1), c - r * math.sin(a1))],
               fill=(*[int(255 * v) for v in rgb], 255), width=w)  # fmt: skip
    a = math.radians(38)
    tip = (c + r * math.cos(a), c - r * math.sin(a))
    d.line([(c, c), tip], fill=(255, 196, 110, 255), width=int(w * 1.15))
    d.ellipse([tip[0] - w * 1.6, tip[1] - w * 1.6, tip[0] + w * 1.6, tip[1] + w * 1.6],
              fill=(255, 226, 170, 255))  # fmt: skip
    d.ellipse([c - w, c - w, c + w, c + w], fill=(230, 235, 245, 255))
    glow = ink.filter(ImageFilter.GaussianBlur(n * 0.03))
    img = Image.alpha_composite(bg, glow)
    img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, ink)
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=n * 0.22, fill=255)
    img.putalpha(mask)
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    icon = draw()
    icon.save(OUT / "icon.png", optimize=True)
    icon.save(OUT / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                       (128, 128), (256, 256)])  # fmt: skip
    print("wrote", OUT / "icon.png", OUT / "icon.ico")
