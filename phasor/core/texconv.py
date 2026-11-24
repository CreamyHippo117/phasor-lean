"""Maths between the two typesetters: the built-in one (matplotlib mathtext, the default)
and real LaTeX (``set_math_engine("latex")``).

A scene writes maths once, as ``$...$`` in either dialect, and it renders on both engines:
``core.text`` passes every math string through ``to_mathtext`` or ``to_latex`` before
typesetting it. The two functions are also public, to show or export the other form
(Text and Equation report ``latex`` in their semantics).

- ``to_mathtext(s)``: LaTeX that mathtext lacks becomes the nearest mathtext (``\\tfrac``
  -> ``\\frac``, ``\\le`` -> ``\\leq``, ``\\lvert`` -> ``|``, ``\\textbf`` -> ``\\mathbf``,
  ``\\big(`` -> ``(``, ``\\pmod{n}`` -> ``(\\mathrm{mod}\\ n)``, ...). What has no
  equivalent (a color, a box, a brace under a term) is dropped, keeping its content;
  ``mathtext_losses(s)`` names what was dropped. Environments (``cases``, ``pmatrix``) and
  line breaks have no mathtext form and stay, so typesetting them reports the error.
- ``to_latex(s)``: mathtext that LaTeX lacks becomes LaTeX (Unicode letters and symbols ->
  commands, ``\\mathdefault`` -> ``\\mathrm``, ``\\degree`` -> ``^{\\circ}``), and the plain
  text around the maths is escaped (``50%`` -> ``50\\%``).

Both are pure, leave text they do not know alone and are idempotent.
"""

from __future__ import annotations

import re
from collections.abc import Callable

__all__ = ["join_latex", "mathtext_losses", "split_math", "to_latex", "to_mathtext"]


def split_math(s: str) -> list[tuple[bool, str]]:
    """``s`` cut into (is_math, piece) runs at unescaped ``$`` (the ``$`` are dropped); an
    unpaired last ``$`` stays in the text."""
    out: list[tuple[bool, str]] = []
    buf: list[str] = []
    math = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            buf.append(s[i : i + 2])
            i += 2
            continue
        if ch == "$":
            out.append((math, "".join(buf)))
            buf, math = [], not math
        else:
            buf.append(ch)
        i += 1
    if math:  # unpaired: the last "$" was text after all
        prev = out.pop()
        out.append((False, prev[1] + "$" + "".join(buf)))
    else:
        out.append((False, "".join(buf)))
    # merge neighbors of the same kind, drop empty text runs
    merged: list[tuple[bool, str]] = []
    for m, p in out:
        if merged and not m and not merged[-1][0]:
            merged[-1] = (False, merged[-1][1] + p)
        elif m or p:
            merged.append((m, p))
    return merged


def _join(runs: list[tuple[bool, str]]) -> str:
    return "".join(f"${p}$" if m else p for m, p in runs)


def _cmd(name: str) -> re.Pattern[str]:
    return re.compile(r"\\" + name + r"(?![A-Za-z])")


def _group(s: str, i: int) -> tuple[str, int]:
    """The argument starting at ``s[i]`` (spaces skipped): a braced group's content or one
    token. Returns (content, index after it)."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            if s[j] == "\\":
                j += 2
                continue
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1 : j], j + 1
            j += 1
        return s[i + 1 :], len(s)
    if s[i] == "\\":
        m = re.match(r"\\([A-Za-z]+|.)", s[i:])
        assert m is not None
        return m.group(0), i + m.end()
    return s[i], i + 1


def _with_args(s: str, name: str, nargs: int, make: Callable[..., str]) -> str:
    """Replace ``\\name{a}{b}`` by ``make(a, b)`` everywhere (innermost content converted
    first, since ``make`` sees already-converted arguments when called again)."""
    pat = _cmd(name)
    out, pos = [], 0
    while (m := pat.search(s, pos)) is not None:
        out.append(s[pos : m.start()])
        args, j = [], m.end()
        for _ in range(nargs):
            a, j = _group(s, j)
            args.append(_with_args(a, name, nargs, make))
        out.append(make(*args))
        pos = j
    out.append(s[pos:])
    return "".join(out)


# ---------------------------------------------------------------- LaTeX -> mathtext

# Same meaning, another name (mathtext lacks the first).
_RENAME = {
    "tfrac": r"\frac", "cfrac": r"\dfrac", "le": r"\leq", "ge": r"\geq",
    "lvert": "|", "rvert": "|", "lVert": r"\Vert", "rVert": r"\Vert",
    "coloneqq": ":=", "eqqcolon": "=:", "iff": r"\Leftrightarrow",
    "implies": r"\Rightarrow", "impliedby": r"\Leftarrow", "land": r"\wedge",
    "lor": r"\vee", "lnot": r"\neg", "gets": r"\leftarrow", "square": r"\Box",
    "textbf": r"\mathbf", "textit": r"\mathit", "emph": r"\mathit", "textrm": r"\text",
    "mbox": r"\text", "textsf": r"\mathsf", "texttt": r"\mathtt", "stackrel": r"\overset",
    "bmod": r"\,\mathrm{mod}\,", "dotsc": r"\ldots", "dotsb": r"\cdots", "dotsm": r"\cdots",
    "dotsi": r"\cdots", "operatorname*": r"\operatorname",
}  # fmt: skip
# Only change how big or how set something is: dropped, the content stays.
_DROP = ("displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle", "limits",
         "nolimits", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl",
         "biggr", "Biggl", "Biggr", "bigm", "Bigm", "nonumber", "notag")  # fmt: skip
# Commands with arguments mathtext cannot show: (arguments, what replaces them, lossy).
_ARGS: dict[str, tuple[int, Callable[..., str], bool]] = {
    "pmod": (1, lambda n: r"\ (\mathrm{mod}\ " + n + ")", False),
    "xrightarrow": (1, lambda f: r"\overset{" + f + r"}{\longrightarrow}", False),
    "xleftarrow": (1, lambda f: r"\overset{" + f + r"}{\longleftarrow}", False),
    "hspace": (1, lambda _w: r"\quad", True),
    "color": (1, lambda _c: "", True),
    "textcolor": (2, lambda _c, x: "{" + x + "}", True),
    "boxed": (1, lambda x: "{" + x + "}", True),
    "underbrace": (1, lambda x: "{" + x + "}", True),
    "overbrace": (1, lambda x: "{" + x + "}", True),
    "label": (1, lambda _x: "", True),
    "tag": (1, lambda _x: "", True),
}


def _math_to_mathtext(m: str, lost: list[str] | None = None) -> str:
    for name, (n, make, lossy) in _ARGS.items():
        if _cmd(name).search(m):
            if lossy and lost is not None:
                lost.append("\\" + name)
            m = _with_args(m, name, n, make)
    m = m.replace(r"\operatorname*", r"\operatorname")
    for name, new in _RENAME.items():
        if not name.endswith("*"):
            m = _cmd(name).sub(lambda _x, new=new: new, m)
    for name in _DROP:
        m = _cmd(name).sub("", m)
    return m


def to_mathtext(s: str) -> str:
    """``s`` with every ``$...$`` rewritten so mathtext can typeset it (see the module)."""
    return _join([(m, _math_to_mathtext(p) if m else p) for m, p in split_math(s)])


def mathtext_losses(s: str) -> list[str]:
    """The LaTeX commands in ``s`` that ``to_mathtext`` drops (keeping their content)."""
    lost: list[str] = []
    for m, p in split_math(s):
        if m:
            _math_to_mathtext(p, lost)
    return sorted(set(lost))


# ---------------------------------------------------------------- mathtext -> LaTeX

_UNICODE = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon",
    "ϵ": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "ϑ": r"\vartheta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi",
    "π": r"\pi", "ϖ": r"\varpi", "ρ": r"\rho", "ϱ": r"\varrho", "σ": r"\sigma",
    "ς": r"\varsigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\varphi", "ϕ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega", "Γ": r"\Gamma", "Δ": r"\Delta",
    "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma",
    "Υ": r"\Upsilon", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    "∞": r"\infty", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx",
    "≡": r"\equiv", "∼": r"\sim", "±": r"\pm", "∓": r"\mp", "×": r"\times", "÷": r"\div",
    "·": r"\cdot", "∘": r"\circ", "→": r"\to", "←": r"\leftarrow", "↦": r"\mapsto",
    "⇒": r"\Rightarrow", "⇔": r"\Leftrightarrow", "∈": r"\in", "∉": r"\notin",
    "⊂": r"\subset", "⊆": r"\subseteq", "∪": r"\cup", "∩": r"\cap", "∅": r"\varnothing",
    "∀": r"\forall", "∃": r"\exists", "∂": r"\partial", "∇": r"\nabla", "∑": r"\sum",
    "∏": r"\prod", "∫": r"\int", "∮": r"\oint", "√": r"\surd", "ℓ": r"\ell", "ℏ": r"\hbar",
    "ℝ": r"\mathbb{R}", "ℂ": r"\mathbb{C}", "ℤ": r"\mathbb{Z}", "ℕ": r"\mathbb{N}",
    "ℚ": r"\mathbb{Q}", "′": "'", "…": r"\ldots", "⋯": r"\cdots", "°": r"^{\circ}",
    "⟨": r"\langle", "⟩": r"\rangle", "−": "-",
}  # fmt: skip
_MT_ONLY = {"mathdefault": r"\mathrm", "degree": r"^{\circ}", "cdotp": r"\cdot",
            "AA": r"\text{\AA}", "dag": r"\dagger", "ddag": r"\ddagger"}  # fmt: skip
_TEXT_ESCAPE = {"$": r"\$", "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}  # fmt: skip


def _unicode_cmds(m: str) -> str:
    out = []
    for i, ch in enumerate(m):
        rep = _UNICODE.get(ch)
        if rep is None:
            out.append(ch)
            continue
        nxt = m[i + 1] if i + 1 < len(m) else ""
        out.append(rep + (" " if rep[-1].isalpha() and nxt.isalpha() else ""))
    return "".join(out)


def _math_to_latex(m: str) -> str:
    for name, new in _MT_ONLY.items():
        m = _cmd(name).sub(lambda _x, new=new: new, m)
    return _unicode_cmds(m)


def _text_to_latex(t: str) -> str:
    out, i = [], 0
    while i < len(t):
        ch = t[i]
        if ch == "\\" and i + 1 < len(t) and t[i + 1] in "$%&#_{}":
            out.append(t[i : i + 2])  # already escaped
            i += 2
            continue
        if ch == "\\" and (cmd := re.match(r"\\[A-Za-z]+(\{[^{}]*\})?", t[i:])):
            out.append(cmd.group(0))  # a text command (``\textbackslash{}``): already LaTeX
            i += cmd.end()
            continue
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in _TEXT_ESCAPE:
            out.append(_TEXT_ESCAPE[ch])
        elif ch in _UNICODE:
            out.append(r"\ensuremath{" + _UNICODE[ch] + "}")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def to_latex(s: str) -> str:
    """``s`` as LaTeX source: maths rewritten where mathtext differs, text escaped."""
    return "".join(f"${_math_to_latex(p)}$" if m else _text_to_latex(p)
                   for m, p in split_math(s))  # fmt: skip


def join_latex(parts: list[str]) -> str:
    """Pieces written to sit side by side (an Equation's parts) as one LaTeX string:
    neighboring maths join into one ``$...$`` (``$a$`` + ``$=b$`` -> ``$a=b$``)."""
    runs: list[tuple[bool, str]] = []
    for part in parts:
        for m, p in split_math(part):
            if runs and m and runs[-1][0]:
                runs[-1] = (True, runs[-1][1] + p)
            else:
                runs.append((m, p))
    return to_latex(_join(runs))
