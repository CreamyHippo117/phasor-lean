"""Text metrics and rasterization on the CPU, shared by bounds (inspector, layout) and the
renderer so they can never disagree about how big a label is.

Plain text uses the scalable font bundled inside Pillow (Aileron), identical on every OS.

Math: a string written as ``$...$`` (one or more ``$`` pairs) is typeset by matplotlib's
mathtext in Computer Modern (its own bundled fonts, so also identical everywhere), e.g.
``"$\\psi(x) \\approx x - \\ln 2\\pi$"``. A math string is sized by the same
``height`` (line height) as text of its role, its em being ``MATH_EM`` of it; the box it
occupies can be taller (a sum, a fraction) and is what ``quad`` reports, so layout sees the
real size. No TeX install is needed
(mathtext covers the common subset).

Either dialect works on either engine: every math string passes through
``texconv.to_mathtext`` or ``texconv.to_latex`` first (``\\tfrac`` and ``\\le`` work on
mathtext; Unicode ``π`` and ``50%`` work on LaTeX).
"""

from __future__ import annotations

import functools
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from phasor.core.proc import NO_WINDOW
from phasor.core.texconv import to_latex, to_mathtext

_REF = 64
MATH_EM = 0.8  # a math string's em, as a share of the line height of text with the same role
_MATH_REF = 96  # px per em when measuring math


def is_math(text: str) -> bool:
    """Whether ``text`` is typeset as math: it holds at least one pair of unescaped ``$``."""
    return text.replace("\\$", "").count("$") >= 2


ENGINES = ("mathtext", "latex", "auto")
_engine = os.environ.get("PHASOR_MATH", "mathtext")


def set_math_engine(name: str) -> None:
    """How ``$...$`` is typeset: ``mathtext`` (built in, the default: identical on every
    machine), ``latex`` (a real TeX: ``latex`` and ``dvipng`` on the PATH, e.g. MiKTeX or TeX
    Live; an error when they are missing) or ``auto`` (latex when present, else mathtext).
    Also set by the environment variable ``PHASOR_MATH``. Pictures differ between engines, so
    keep the same one for a project's golden frames."""
    global _engine
    if name not in ENGINES:
        raise ValueError(f"math engine must be one of {ENGINES}, got {name!r}")
    _engine = name
    aspect.cache_clear()
    math_box.cache_clear()
    rasterize.cache_clear()


def math_engine() -> str:
    """The engine in effect: ``mathtext`` or ``latex``."""
    if _engine == "mathtext":
        return "mathtext"
    if _engine == "auto":
        return "latex" if latex_available() else "mathtext"
    if not latex_available():
        raise ValueError(
            "math engine 'latex' needs `latex` and `dvipng` on the PATH (install MiKTeX or "
            "TeX Live), or use set_math_engine('mathtext') / PHASOR_MATH=mathtext"
        )
    return "latex"


def latex_available() -> bool:
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


_TEX_DOC = (
    "\\documentclass[10pt]{article}\\usepackage{amsmath,amssymb}\\pagestyle{empty}\n"
    "\\begin{document}%s\\end{document}\n"
)


def _tex(program: str) -> str:
    """The full path of a TeX program: Windows starts only ``.exe`` files by bare name."""
    return shutil.which(program) or program


def _latex(text: str, em_px: int):
    """(coverage uint8 (h, w), depth in px) of a math string through TeX: ``latex`` makes a
    DVI, ``dvipng`` draws it white on transparent at ``em_px`` pixels per (10 pt) em."""
    from PIL import Image as _Image

    dpi = max(8.0, em_px * 72.27 / 10.0)
    with tempfile.TemporaryDirectory(prefix="phasor_tex_") as d:
        work = Path(d)
        (work / "m.tex").write_text(_TEX_DOC % text, encoding="utf-8")
        run = subprocess.run(
            [_tex("latex"), "-interaction=nonstopmode", "-halt-on-error", "m.tex"],
            cwd=work, capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW,
        )  # fmt: skip
        if run.returncode != 0 or not (work / "m.dvi").exists():
            tail = [ln for ln in run.stdout.splitlines() if ln.startswith("!")][:1]
            raise ValueError(f"cannot typeset {text!r}: {tail[0] if tail else 'latex failed'}")
        png = subprocess.run(
            [_tex("dvipng"), "-q", "-T", "tight", "--depth", "-bg", "Transparent", "-fg",
             "rgb 1.0 1.0 1.0", "-D", f"{dpi:.2f}", "-o", "m.png", "m.dvi"],
            cwd=work, capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW,
        )  # fmt: skip
        if png.returncode != 0 or not (work / "m.png").exists():
            raise ValueError(f"cannot typeset {text!r}: dvipng failed")
        found = re.search(r"depth=(-?\d+)", png.stdout)
        depth = float(found.group(1)) if found else 0.0
        img = np.asarray(_Image.open(work / "m.png").convert("RGBA"))[..., 3].copy()
    return img, depth


@functools.lru_cache(maxsize=1024)
def _math(text: str, em_px: int, engine: str | None = None):
    """(coverage uint8 (h, w), depth in px) of a math string at ``em_px`` pixels per em."""
    if (engine or math_engine()) == "latex":
        return _latex(to_latex(text), em_px)
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib import rc_context
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser

    with rc_context({"mathtext.fontset": "cm"}):
        try:
            r = MathTextParser("agg").parse(to_mathtext(text), dpi=72,
                                            prop=FontProperties(size=em_px))  # fmt: skip
        except ValueError as e:
            raise ValueError(f"cannot typeset {text!r}: {str(e).splitlines()[-1]}") from None
    img = np.asarray(r.image, dtype=np.uint8)
    return img, float(r.depth)


def check_math(text: str) -> None:
    """Raise ``ValueError`` (naming the problem) when a math string cannot be typeset."""
    if is_math(text):
        _math(text, 16, math_engine())


@functools.lru_cache(maxsize=4096)
def math_box(text: str) -> tuple[float, float, float]:
    """(width, box height, depth below the baseline), each in ems, of a math string."""
    img, depth = _math(text, _MATH_REF, math_engine())
    h, w = img.shape
    return w / _MATH_REF, h / _MATH_REF, depth / _MATH_REF


@functools.cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


@functools.lru_cache(maxsize=4096)
def aspect(text: str) -> float:
    """Width / height of one line of text, where height is the full line (ascent + descent).
    For math: width / the line height of its role (the box may be taller; see ``box``)."""
    if is_math(text):
        return math_box(text)[0] * MATH_EM
    f = _font(_REF)
    ascent, descent = f.getmetrics()
    return max(f.getlength(text), 1.0) / (ascent + descent)


@functools.lru_cache(maxsize=512)
def rasterize(text: str, px_height: int) -> np.ndarray:
    """Coverage bitmap (uint8, row 0 = top), exactly ``px_height`` tall and ``aspect`` wide.

    Glyphs are drawn at the nearest whole font size, whose line can come out a little wider
    or narrower than ``aspect`` predicts; the drawing is then resampled to the exact width, so
    no glyph is cut off, nothing is left blank at the end, and the bitmap fills exactly the
    quad that bounds report.
    """
    px_height = max(int(px_height), 2)
    if is_math(text):  # ``px_height`` is the role's line height; the bitmap is the whole box
        em = max(2, round(px_height * MATH_EM))
        img = Image.fromarray(_math(text, em, math_engine())[0])
        w = max(1, math.ceil(aspect(text) * px_height))
        h = max(1, math.ceil(box(text) * px_height))
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        out = np.asarray(img, dtype=np.uint8)
        out.setflags(write=False)
        return out
    ascent, descent = _font(_REF).getmetrics()
    f = _font(max(1, round(_REF * px_height / (ascent + descent))))
    width = max(1, math.ceil(aspect(text) * px_height))
    drawn = max(1, math.ceil(f.getlength(text)))
    img = Image.new("L", (drawn, px_height), 0)
    ImageDraw.Draw(img).text((0, 0), text, fill=255, font=f)
    if img.width != width:  # a few percent either way; keeps centered text centered
        img = img.resize((width, px_height), Image.Resampling.LANCZOS)
    out = np.asarray(img, dtype=np.uint8)
    out.setflags(write=False)
    return out


_ANCHOR_SHIFT = {"center": -0.5, "left": 0.0, "right": -1.0}


def depth(text: str) -> float:
    """How far the box reaches below the baseline, as a multiple of the role's line height
    (math: its real depth; plain text: the font's descent share)."""
    if is_math(text):
        return math_box(text)[2] * MATH_EM
    ascent, descent = _font(_REF).getmetrics()
    return descent / (ascent + descent)


def box(text: str) -> float:
    """Height of the box ``text`` occupies, as a multiple of its role's line height (1 for
    plain text; a math string with a sum or a fraction is taller)."""
    return math_box(text)[1] * MATH_EM if is_math(text) else 1.0


def quad(text: str, origin, height: float, anchor: str, right, up):
    """The world-space rectangle a line of text occupies: (lower-left corner, width vector,
    height vector), centered vertically on ``origin``. Bounds and rendering both use this, so
    they always agree."""
    w = aspect(text) * height
    hb = box(text) * height
    corner = np.asarray(origin, dtype=np.float64) + _ANCHOR_SHIFT[anchor] * w * right
    return corner - (hb / 2) * up, w * right, hb * up


@functools.cache
def _tofu() -> bytes:
    return bytes(_font(_REF).getmask("\U000f0000"))  # a private-use code point: never drawn


@functools.lru_cache(maxsize=4096)
def missing_glyphs(text: str) -> tuple[str, ...]:
    """Characters the font cannot draw (they would show as empty boxes), in order, once each.
    Whitespace is never missing."""
    if is_math(text):
        return ()  # mathtext draws its own glyphs (an unknown command fails validation)
    font, tofu, out = _font(_REF), _tofu(), []
    for ch in dict.fromkeys(text):
        if not ch.isspace() and bytes(font.getmask(ch)) == tofu:
            out.append(ch)
    return tuple(out)
