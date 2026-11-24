"""Safe parameter expressions: ``dot.pos + vec(0, 0.4, 0)``, ``2 * loop.r + sin(t)``.

Parsed with ``ast`` against a whitelist and compiled to closures; ``eval`` is never used.
Dependencies are known statically, which is what lets the parameter graph be sorted and
cycles be rejected before anything is evaluated (contract 3).
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

Env = Mapping[str, Any]
Fn = Callable[[Env, float], Any]


class ExprError(ValueError):
    pass


def _broadcast(op: Callable[[float, float], float], a: Any, b: Any) -> Any:
    ta, tb = isinstance(a, tuple), isinstance(b, tuple)
    if ta and tb:
        if len(a) != len(b):
            raise ExprError(f"vector lengths differ: {len(a)} and {len(b)}")
        return tuple(_broadcast(op, x, y) for x, y in zip(a, b, strict=True))
    if ta:
        return tuple(_broadcast(op, x, b) for x in a)
    if tb:
        return tuple(_broadcast(op, a, y) for y in b)
    if isinstance(a, bool | str) or isinstance(b, bool | str):
        raise ExprError("arithmetic needs numbers or vectors")
    return op(a, b)


def _map(fn: Callable[[float], float]) -> Callable[[Any], Any]:
    def apply(v: Any) -> Any:
        return tuple(apply(x) for x in v) if isinstance(v, tuple) else fn(v)

    return apply


def _length(v: tuple) -> float:
    return math.sqrt(sum(x * x for x in v))


def _normalize(v: tuple) -> tuple:
    n = _length(v)
    return tuple(x / n for x in v) if n else v


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_MATH = ("sin", "cos", "tan", "exp", "log", "sqrt", "floor", "ceil")
CONSTANTS = {"pi": math.pi, "tau": math.tau, "e": math.e}
FUNCTIONS: dict[str, Callable[..., Any]] = {
    **{n: _map(getattr(math, n)) for n in _MATH},
    "abs": _map(abs),
    "min": min,
    "max": max,
    "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
    "lerp": lambda a, b, u: _broadcast(lambda x, y: x + (y - x) * u, a, b),
    "vec": lambda *xs: tuple(float(x) for x in xs),
    "length": _length,
    "normalize": _normalize,
    "dot": lambda a, b: sum(x * y for x, y in zip(a, b, strict=True)),
    "midpoint": lambda a, b: _broadcast(lambda x, y: (x + y) / 2, a, b),
}  # fmt: skip


@dataclass(frozen=True)
class CompiledExpr:
    source: str
    deps: frozenset[str]  # parameter paths this expression reads
    uses_t: bool
    fn: Fn

    def __call__(self, env: Env, t: float) -> Any:
        try:
            return self.fn(env, t)
        except ExprError:
            raise
        except Exception as e:
            raise ExprError(f"{self.source!r}: {type(e).__name__}: {e}") from e


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and parts:
        return ".".join([node.id, *reversed(parts)])
    return None


def compile_expr(source: str) -> CompiledExpr:
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"{source!r}: syntax error at column {e.offset}") from e
    deps: set[str] = set()
    uses_t = False

    def bad(node: ast.AST, what: str) -> ExprError:
        return ExprError(f"{source!r}: {what} at column {getattr(node, 'col_offset', 0) + 1}")

    def build(node: ast.AST) -> Fn:
        nonlocal uses_t
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise bad(node, "only numbers are allowed as literals")
            value = float(node.value)
            return lambda env, t: value
        if isinstance(node, ast.Name):
            if node.id == "t":
                uses_t = True
                return lambda env, t: t
            if node.id in CONSTANTS:
                c = CONSTANTS[node.id]
                return lambda env, t: c
            raise bad(node, f"unknown name {node.id!r} (parameters are written <id>.<param>)")
        if isinstance(node, ast.Attribute):
            path = _dotted(node)
            if path is None:
                raise bad(node, "bad parameter reference")
            deps.add(path)
            return lambda env, t: env[path]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            op, left, right = _BINOPS[type(node.op)], build(node.left), build(node.right)
            return lambda env, t: _broadcast(op, left(env, t), right(env, t))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
            operand = build(node.operand)
            return lambda env, t: _broadcast(operator.mul, sign, operand(env, t))
        if isinstance(node, ast.Tuple | ast.List):
            items = [build(e) for e in node.elts]
            return lambda env, t: tuple(i(env, t) for i in items)
        if isinstance(node, ast.Subscript):
            idx = node.slice
            if not (isinstance(idx, ast.Constant) and isinstance(idx.value, int)):
                raise bad(node, "index must be an integer literal")
            base, i = build(node.value), idx.value
            return lambda env, t: base(env, t)[i]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                name = getattr(node.func, "id", "?")
                raise bad(node, f"unknown function {name!r}; available: {sorted(FUNCTIONS)}")
            if node.keywords:
                raise bad(node, "keyword arguments are not supported")
            f, args = FUNCTIONS[node.func.id], [build(a) for a in node.args]
            return lambda env, t: f(*(a(env, t) for a in args))
        raise bad(node, f"{type(node).__name__} is not allowed in expressions")

    fn = build(tree.body)  # fills deps and uses_t as it walks; must run before they are read
    return CompiledExpr(source, frozenset(deps), uses_t, fn)
