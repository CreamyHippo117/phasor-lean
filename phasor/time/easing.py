"""Easing curves: functions from progress u in [0, 1] to eased progress, with f(0)=0, f(1)=1.

``back`` and ``spring`` overshoot (go past 1 and return), so use them on positions and sizes,
not on values with hard limits such as opacity. ``log:<name>`` runs any curve in log space
(``value = a (b / a)^curve(u)``, see ``core.values.geo_lerp``): equal times give equal
factors, so a 600x camera dive moves evenly; both ends must be positive. Easings are named,
never Python functions in a document: a document must mean the same thing on every machine
(contract 3).
"""

from __future__ import annotations

import math
from collections.abc import Callable

OVERSHOOT = {"back", "spring"}


def _back(u: float, s: float = 1.70158) -> float:
    """Ease out with one overshoot of about 10%."""
    v = u - 1
    return 1 + v * v * ((s + 1) * v + s)


def _spring(u: float) -> float:
    """A damped spring released at u = 0 that settles exactly at u = 1."""
    return 1 - (1 - u) * math.exp(-4 * u) * math.cos(3 * math.pi * u)


EASINGS: dict[str, Callable[[float], float]] = {
    "linear": lambda u: u,
    "smooth": lambda u: u * u * (3 - 2 * u),
    "smoother": lambda u: u * u * u * (u * (6 * u - 15) + 10),
    "in": lambda u: u * u * u,
    "out": lambda u: 1 - (1 - u) ** 3,
    "in_out": lambda u: 4 * u * u * u if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2,
    "back": _back,
    "spring": _spring,
}

LOG = "log:"  # prefix: the same curve, moving through the values by factors


def ease_named(name: str) -> tuple[Callable[[float], float], bool] | None:
    """``(curve, geometric)`` for an ease name (``"smooth"``, ``"log:smooth"``), or None."""
    geometric = name.startswith(LOG)
    curve = EASINGS.get(name.removeprefix(LOG))
    return None if curve is None else (curve, geometric)
