"""LaTeX as an option: ``$...$`` goes through matplotlib's mathtext by default; a real TeX
(``latex`` + ``dvipng``, MiKTeX or TeX Live) can be chosen, or used when present ('auto').
Without TeX installed, the plumbing is tested with stand-in programs on the PATH."""

from __future__ import annotations

import stat
import sys

import numpy as np
import pytest

from phasor import Equation, Scene, Text
from phasor.core import text as tm

FAKE_LATEX = """#!{py}
import sys, pathlib
src = pathlib.Path("m.tex").read_text()
if "FAIL" in src:
    print("! Undefined control sequence.")
    sys.exit(1)
pathlib.Path("m.dvi").write_bytes(b"dvi")
"""
FAKE_DVIPNG = """#!{py}
import sys, numpy as np
from PIL import Image
a = np.zeros((20, 40, 4), dtype=np.uint8); a[5:15, 5:35] = 255
Image.fromarray(a).save("m.png")
print("depth=3")
"""


@pytest.fixture(autouse=True)
def _reset_engine():
    tm.set_math_engine("mathtext")
    yield
    tm.set_math_engine("mathtext")


@pytest.fixture
def fake_tex(tmp_path, monkeypatch):
    for name, body in (("latex", FAKE_LATEX), ("dvipng", FAKE_DVIPNG)):
        path = tmp_path / name
        path.write_text(body.format(py=sys.executable), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        if sys.platform == "win32":  # Windows runs programs by extension, not by #!
            (tmp_path / f"{name}.cmd").write_text(f'@"{sys.executable}" "{path}" %*\r\n',
                                                  encoding="utf-8")  # fmt: skip
    monkeypatch.setenv("PATH", str(tmp_path))
    tm._math.cache_clear()
    yield
    tm._math.cache_clear()


def test_the_default_is_the_built_in_engine_and_choices_are_checked():
    assert tm.math_engine() == "mathtext"
    with pytest.raises(ValueError, match="one of"):
        tm.set_math_engine("word")


def test_latex_without_tex_says_what_to_install_and_auto_falls_back(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert not tm.latex_available()
    tm.set_math_engine("auto")
    assert tm.math_engine() == "mathtext"
    assert tm.aspect(r"$x^2$") > 0  # still typesets
    tm.set_math_engine("latex")
    with pytest.raises(ValueError, match="MiKTeX or TeX Live"):
        tm.math_engine()
    with pytest.raises(Exception, match="MiKTeX or TeX Live"):
        Scene("x", passes=None).add(Text(id="t", text=r"$x$"))


def test_a_tex_install_is_used_when_chosen(fake_tex):
    assert tm.latex_available()
    tm.set_math_engine("auto")
    assert tm.math_engine() == "latex"
    tm.set_math_engine("latex")
    w, h, depth = tm.math_box(r"$\int_0^1 x\,dx$")
    assert (w, h) == pytest.approx((40 / tm._MATH_REF, 20 / tm._MATH_REF))  # the fake's bitmap
    assert depth == pytest.approx(3 / tm._MATH_REF)
    bmp = tm.rasterize(r"$a$", 40)
    assert bmp.max() == 255 and bmp.shape[0] == int(np.ceil(tm.box(r"$a$") * 40))
    s = Scene("x", passes=None)
    s.add(Equation(id="e", parts=(r"$a =$", r"$b$")))  # validated through TeX too
    assert s.engine.inspect(0, id="e")["objects"][0]["semantic"]["parts"] == ["$a =$", "$b$"]


def test_tex_errors_are_refused_with_the_reason(fake_tex):
    tm.set_math_engine("latex")
    with pytest.raises(Exception, match=r"cannot typeset.*Undefined control sequence"):
        Scene("x", passes=None).add(Text(id="t", text=r"$FAIL$"))


@pytest.mark.skipif(not tm.latex_available(), reason="no TeX installed on this machine")
def test_real_tex_typesets_and_differs_from_mathtext():
    text = r"$\sum_{n=1}^{N} \frac{1}{n^2}$"
    tm.set_math_engine("mathtext")
    a = tm.math_box(text)
    tm.set_math_engine("latex")
    b = tm.math_box(text)
    assert b[0] > 0 and b[1] > b[2] >= 0
    assert abs(a[0] - b[0]) / a[0] < 0.4  # the same formula, a similar size
    assert tm.rasterize(text, 48).max() > 200
