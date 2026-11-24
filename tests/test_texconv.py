"""Maths in both dialects: LaTeX that mathtext lacks, mathtext that LaTeX lacks, and the
text around them. The default engine stays mathtext."""

from __future__ import annotations

import numpy as np
import pytest

from phasor import Equation, Scene, Text
from phasor.core import text as tm
from phasor.core.texconv import join_latex, mathtext_losses, split_math, to_latex, to_mathtext
from tests.conftest import gl_available

SAMPLES = [
    r"$\tfrac{1}{2} \le x$",
    r"area 50% of $\pi r^2$ & more",
    r"$\lvert z\rvert \coloneqq \sqrt{x^2}$",
    r"$a \equiv b \pmod{n}$",
    r"$\boxed{x}\color{red}y$",
    r"$f \xrightarrow{g} h$",
    "π and $x$",
    r"$\big(\sum\limits_{k} k\big)$",
    r"cost \$5 for $x$",
    r"$\mathdefault{d}x°$",
    "$θ = 2π$",
    "one $ dollar",
    r"$\displaystyle\int_0^1 \textbf{v}\,\mathrm{d}t \implies \operatorname*{lim}$",
]


def test_the_default_engine_is_mathtext():
    assert tm.ENGINES[0] == "mathtext"
    if "PHASOR_MATH" not in __import__("os").environ:
        assert tm.math_engine() == "mathtext"


@pytest.mark.parametrize("s", SAMPLES)
def test_both_conversions_are_idempotent(s):
    assert to_mathtext(to_mathtext(s)) == to_mathtext(s)
    assert to_latex(to_latex(s)) == to_latex(s)


@pytest.mark.parametrize("s", SAMPLES)
def test_whatever_dialect_is_written_mathtext_can_typeset_it(s):
    if tm.is_math(s):
        tm.check_math(s)  # raises if mathtext cannot set it


def test_latex_only_commands_become_mathtext():
    assert to_mathtext(r"$\tfrac{1}{2} \le \lvert z \rvert$") == r"$\frac{1}{2} \leq | z |$"
    assert to_mathtext(r"$\left(\big(x\big)\right)$") == r"$\left((x)\right)$"
    assert r"\overset{g}{\longrightarrow}" in to_mathtext(r"$f \xrightarrow{g} h$")
    assert to_mathtext(r"$\textbf{v}$") == r"$\mathbf{v}$"
    # \leq, \left and friends are not mistaken for \le
    assert to_mathtext(r"$\leq \left( \leftarrow$") == r"$\leq \left( \leftarrow$"
    # text around the maths is left alone (mathtext shows it as it is)
    assert to_mathtext("50% of $x$") == "50% of $x$"


def test_what_mathtext_cannot_show_is_named():
    assert mathtext_losses(r"$\boxed{x}\color{red}y$") == [r"\boxed", r"\color"]
    assert mathtext_losses(r"$\tfrac{1}{2}$") == []
    assert to_mathtext(r"$\underbrace{a+b}_{n}$") == r"${a+b}_{n}$"


def test_mathtext_only_writing_becomes_latex():
    assert to_latex("50% of $2π r$") == r"50\% of $2\pi r$"
    assert to_latex("$θ ≤ ∞$") == r"$\theta \leq \infty$"
    assert to_latex("$πr$") == r"$\pi r$"  # a space keeps \pi from swallowing r
    assert to_latex(r"$\mathdefault{d}x$") == r"$\mathrm{d}x$"
    assert to_latex("π then $x$") == r"\ensuremath{\pi} then $x$"
    assert to_latex("a_b & c #1 {x} ~ ^") == (
        r"a\_b \& c \#1 \{x\} \textasciitilde{} \textasciicircum{}"
    )
    assert to_latex("one $ dollar") == r"one \$ dollar"
    assert to_latex(r"cost \$5") == r"cost \$5"


def test_splitting_respects_escaped_dollars():
    assert split_math(r"a \$ b $x$ c") == [(False, r"a \$ b "), (True, "x"), (False, " c")]
    assert split_math("$a$$b$") == [(True, "a"), (True, "b")]


def test_equation_parts_join_into_one_latex_formula():
    assert join_latex(["$e^{i\\pi}$", "$+1$", "$=0$"]) == r"$e^{i\pi}+1=0$"


def test_round_trip_through_latex_renders_the_same_on_mathtext():
    s = r"$\frac{1}{2} \leq \sqrt{x^2+y^2}$"
    a = tm._math(to_mathtext(s), 32, "mathtext")[0]
    b = tm._math(to_mathtext(to_latex(s)), 32, "mathtext")[0]
    assert np.array_equal(a, b)


@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_text_and_equation_report_their_latex():
    s = Scene("both")
    s.add(Text(id="half", text=r"$\tfrac{1}{2} \le x$"))
    s.add(Text(id="pct", text="50% of $2π$"))
    s.add(Equation(id="euler", parts=("$e^{i\\pi}$", "$+1=0$")))
    e = s.engine
    sem = {i: e.inspect(0.0, id=i)["objects"][0]["semantic"] for i in ("half", "pct", "euler")}
    assert sem["half"]["latex"] == r"$\tfrac{1}{2} \le x$"
    assert sem["pct"]["latex"] == r"50\% of $2\pi$"
    assert sem["euler"]["latex"] == r"$e^{i\pi}+1=0$"
    assert e.still(0.0, (160, 90)).shape[:2] == (90, 160)
