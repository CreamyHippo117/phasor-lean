"""Parameter values: bool, float, str, or (nested) tuples of floats. Nothing else.

Keeping the value universe this small is what makes ``state_at(t)`` cheap to compute,
hash, compare and serialize.
"""

from __future__ import annotations

import math
import numbers
from typing import Annotated, Any

from pydantic import BeforeValidator

Value = Any  # bool | float | str | tuple[Value, ...]


def normalize(v: Any) -> Value:
    """Canonical form: ints -> float, lists -> tuples, recursively. Rejects everything else."""
    t = type(v)
    if t is float:  # the common case, ahead of the isinstance chain (called ~1e6 times a load)
        if v - v != 0.0:  # nan or inf
            raise ValueError(f"non-finite number: {v!r}")
        return v
    if t is str or t is bool:
        return v
    if isinstance(v, bool | str):
        return v
    if isinstance(v, numbers.Real):
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"non-finite number: {v!r}")
        return f
    if isinstance(v, list | tuple):
        return tuple(normalize(x) for x in v)
    if hasattr(v, "tolist"):  # numpy arrays and scalars
        return normalize(v.tolist())
    raise ValueError(f"unsupported parameter value: {v!r}")


def same_shape(a: Value, b: Value) -> bool:
    if isinstance(a, tuple) or isinstance(b, tuple):
        return (
            isinstance(a, tuple)
            and isinstance(b, tuple)
            and len(a) == len(b)
            and all(same_shape(x, y) for x, y in zip(a, b, strict=True))
        )
    return type(a) is type(b)


def lerp(a: Value, b: Value, u: float) -> Value:
    """Interpolate numbers and tuples; bools and strings switch when ``u`` reaches 1."""
    if isinstance(a, tuple):
        return tuple(lerp(x, y, u) for x, y in zip(a, b, strict=True))
    if isinstance(a, bool | str):
        return b if u >= 1.0 else a
    return a + (b - a) * u


def _pad3(v: Any) -> Any:
    if isinstance(v, list | tuple) and len(v) == 2:
        return (*v, 0.0)
    return v


# A world-space point. 2-tuples are accepted and padded with z = 0 (contract 4:
# a 2D scene is a 3D scene on the plane z = 0).
Vec3 = Annotated[tuple[float, float, float], BeforeValidator(_pad3)]
