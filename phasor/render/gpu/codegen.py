"""One expression, three outputs: CUDA C for the kernel, a NumPy
function for the reference, and LaTeX for the label.

    f = ComplexFunction.parse("1/(z**2 + 1)")
    f.c()       -> "c_powi(c_add(c_powi(z, 2), c_make(R(1.0), R(0.0))), -1)"
    f.numpy()   -> callable on complex arrays
    f.latex()   -> "\\frac{1}{z^{2} + 1}"

The text is read with a whitelist (names, numbers, + - * / **, known functions) and built
into a SymPy expression node by node; it is never evaluated as Python.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import sympy as sp

Z = sp.Symbol("z")
A = sp.Symbol("a")  # a complex constant the kernel takes as an argument, so it can animate
_NAMES = {"z": Z, "i": sp.I, "I": sp.I, "j": sp.I, "pi": sp.pi, "e": sp.E}
_FUNCS = {
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos,
    "tan": sp.tan, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "conj": sp.conjugate, "abs": sp.Abs,
}  # fmt: skip
_C_FUNCS = {
    sp.exp: "c_exp", sp.log: "c_log", sp.sin: "c_sin", sp.cos: "c_cos", sp.tan: "c_tan",
    sp.sinh: "c_sinh", sp.cosh: "c_cosh", sp.tanh: "c_tanh", sp.conjugate: "c_conj",
}  # fmt: skip
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}


class FunctionError(ValueError):
    pass


def _build(node: ast.AST, source: str, names: dict | None = None) -> sp.Expr:
    match node:
        case ast.Expression(body=body):
            return _build(body, source, names)
        case ast.Constant(value=int() | float() as v) if not isinstance(v, bool):
            return sp.Integer(v) if isinstance(v, int) else sp.Float(v)
        case ast.Constant(value=complex() as v):
            return sp.Float(v.imag) * sp.I
        case ast.Name(id=name):
            table = _NAMES if names is None else names
            if name not in table:
                raise FunctionError(
                    f"unknown name {name!r} in {source!r}; use {', '.join(sorted(table))} "
                    f"and {sorted(_FUNCS)}"
                )
            return table[name]
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_build(operand, source, names)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return _build(operand, source, names)
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINOPS:
            return _BINOPS[type(op)](_build(left, source, names), _build(right, source, names))
        case ast.Call(func=ast.Name(id=name), args=[arg], keywords=[]) if name in _FUNCS:
            return _FUNCS[name](_build(arg, source, names))
    raise FunctionError(f"cannot use {ast.unparse(node)!r} in a function of z ({source!r})")


def _real(value: sp.Expr) -> str:
    return f"R({float(value)!r})"


def _emit(e: sp.Expr) -> str:
    if e == Z:
        return "z"
    if e == A:
        return "a"
    if e.is_number and e.is_real and e.is_finite:
        return f"c_make({_real(e)}, R(0.0))"
    if e == sp.I:
        return "c_make(R(0.0), R(1.0))"
    if isinstance(e, sp.Add | sp.Mul):
        fn = "c_add" if isinstance(e, sp.Add) else "c_mul"
        terms = [_emit(a) for a in e.args]
        out = terms[0]
        for term in terms[1:]:
            out = f"{fn}({out}, {term})"
        return out
    if isinstance(e, sp.Pow):
        base, exponent = e.args
        if exponent.is_Integer:
            return f"c_powi({_emit(base)}, {int(exponent)})"
        if exponent == sp.Rational(1, 2):
            return f"c_sqrt({_emit(base)})"
        if exponent == sp.Rational(-1, 2):
            return f"c_powi(c_sqrt({_emit(base)}), -1)"
        return f"c_pow({_emit(base)}, {_emit(exponent)})"
    if isinstance(e, sp.Abs):
        return f"c_make(c_abs({_emit(e.args[0])}), R(0.0))"
    if type(e) in _C_FUNCS:
        return f"{_C_FUNCS[type(e)]}({_emit(e.args[0])})"
    raise FunctionError(f"no kernel code for {e!r}")


def parse_expression(source: str, variables: tuple[str, ...]) -> sp.Expr:
    """A SymPy expression in the named ``variables`` (plus ``pi``, ``e``, ``i`` and the known
    functions), read with the same whitelist as ``ComplexFunction`` and never evaluated as
    Python. For domains whose functions are not of ``z`` (calculus: ``x``, ``x, y``)."""
    names = {**{k: v for k, v in _NAMES.items() if k not in ("z",)},
             **{v: sp.Symbol(v, real=True) for v in variables}}  # fmt: skip
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as e:
        raise FunctionError(f"cannot read {source!r}: {e.msg}") from e
    expr = sp.sympify(_build(tree, source, names))
    if expr.has(sp.zoo, sp.nan, sp.oo):
        raise FunctionError(f"{source!r} is not finite anywhere")
    return expr


@dataclass(frozen=True)
class ComplexFunction:
    """A function of one complex variable ``z``."""

    source: str
    expr: sp.Expr

    @classmethod
    def parse(cls, source: str, *, constant: bool = False) -> ComplexFunction:
        """``constant=True`` also allows the name ``a``: a complex constant given at run
        time (``c()`` then reads a ``cplx a`` in scope, ``numpy()`` takes ``a=``)."""
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except SyntaxError as e:
            raise FunctionError(f"cannot read {source!r}: {e.msg}") from e
        expr = sp.sympify(_build(tree, source, {**_NAMES, "a": A} if constant else None))
        if expr.has(sp.zoo, sp.nan, sp.oo):
            raise FunctionError(f"{source!r} is not finite anywhere")
        return cls(source, expr)

    def c(self) -> str:
        """A C expression in ``z`` (type ``cplx``) using ``complex.cuh``."""
        return _emit(self.expr)

    def numpy(self):
        """``f(z)`` for complex NumPy arrays; poles give inf or nan, never a warning. A
        function of the constant ``a`` takes it as ``f(z, a=...)`` (default 0)."""
        fn = sp.lambdify((Z, A), self.expr, "numpy")

        def call(z: np.ndarray, a: complex = 0j) -> np.ndarray:
            with np.errstate(all="ignore"):
                w = fn(np.asarray(z, dtype=np.complex128), complex(a))
            return np.broadcast_to(np.asarray(w, dtype=np.complex128), np.shape(z))

        return call

    def latex(self) -> str:
        return sp.latex(self.expr)
