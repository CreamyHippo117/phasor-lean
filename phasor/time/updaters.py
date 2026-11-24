"""Pure updaters (ADR 0005): ``obj.always(fn)`` with ``fn(t, look) -> {param: value}``.

An updater is a node of the parameter graph (contract 3), like an expression: each
parameter it sets is an ``UpdaterOut`` whose dependencies are the parameters it reads with
``look``. Those reads are found by running it once when the document is compiled, so a
cycle is a validation error as for expressions, and at any other time it may read only what
it read then: reading something new is an error, never a silent stale value. It is called
once per evaluation; the parameters it does not return keep their own value.

Purity is checked, not assumed: lint ``impure_updater`` (``inspect/lint.py``) calls every
updater again, twice, in another order and at other times, and compares.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from phasor.time.expr import CompiledExpr, ExprError

MEMO = "#updater:"  # env keys that hold one evaluation's updater results (removed after)


class Params:
    """What ``look(id)`` returns: the object's parameters at ``t``, read as attributes
    (``look("dot").pos``) or items (``look("dot")["pos"]``). Read-only."""

    __slots__ = ("_id", "_read")

    def __init__(self, object_id: str, read: Callable[[str], Any]) -> None:
        object.__setattr__(self, "_id", object_id)
        object.__setattr__(self, "_read", read)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return self._read(f"{self._id}.{name}")

    __getitem__ = __getattr__

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"look({self._id!r}) is read-only; return the values to set")

    def __repr__(self) -> str:
        return f"look({self._id!r})"


def _plain(v: Any) -> Any:
    """NumPy arrays and scalars as plain lists and floats, so parameters validate."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, list | tuple):
        return [_plain(x) for x in v]
    return v


@dataclass(frozen=True, eq=False)
class Updater:
    ref: str  # "<module>:<name>", as the document names it
    object_id: str
    fn: Callable[..., Mapping[str, Any]]
    params: frozenset[str]  # what it may set: the object's parameters
    reads: frozenset[str] = frozenset()  # parameter paths it reads (its edges in the graph)

    def call(self, env: Mapping[str, Any], t: float, reads: set[str] | None = None) -> dict:
        """``fn(t, look)`` with ``look`` reading ``env``. ``reads``: collect every path read
        (discovery); without it a read outside ``self.reads`` is an error."""

        def read(path: str) -> Any:
            if reads is not None:
                if path not in env:
                    raise ExprError(f"updater {self.ref}: look read unknown parameter {path!r}")
                reads.add(path)
            elif path not in self.reads:
                raise ExprError(f"updater {self.ref} read {path!r} at t={t:g}, which it did not "
                                "read when the scene was compiled; read it at every t")  # fmt: skip
            return env[path]

        try:
            out = self.fn(float(t), lambda object_id: Params(object_id, read))
        except ExprError:
            raise
        except Exception as e:
            raise ExprError(f"updater {self.ref}: {type(e).__name__}: {e}") from e
        if not isinstance(out, Mapping):
            raise ExprError(f"updater {self.ref} must return a dict {{param: value}}, got {out!r}")
        bad = sorted(set(out) - self.params)
        if bad:
            raise ExprError(f"updater {self.ref}: {self.object_id!r} has no parameter(s) {bad}")
        return {k: _plain(v) for k, v in out.items()}


@dataclass(frozen=True, eq=False)
class UpdaterOut(CompiledExpr):
    """One parameter an updater sets, as a node of the parameter graph. ``base`` is the
    parameter's own node: its value whenever the updater leaves the parameter out."""

    updater: Updater = field(default=None)  # type: ignore[assignment]
    name: str = ""
    base: Any = None
    model: type[BaseModel] | None = None
    coerce: Callable[[type[BaseModel], str, Any], Any] | None = None

    def __call__(self, env: Mapping[str, Any], t: float) -> Any:
        key = f"{MEMO}{self.updater.object_id}:{self.updater.ref}"
        out = env.get(key)
        if out is None:
            out = self.updater.call(env, t)
            env[key] = out  # type: ignore[index]  (evaluate's own dict; cleared after)
        if self.name not in out:
            return base_value(self.base, env, t)
        try:
            return self.coerce(self.model, self.name, out[self.name])
        except ValueError as e:
            raise ExprError(f"updater {self.updater.ref}: {self.name}: {e}") from e


def base_value(node: Any, env: Mapping[str, Any], t: float) -> Any:
    """The value a parameter's own node (constant, track or expression) has at ``t``."""
    if hasattr(node, "value_at"):
        return node.value_at(t)
    if isinstance(node, CompiledExpr):
        return node(env, t)
    return node.value
