"""Tracks and the TimeMap (contract 3).

A track is a base value plus ordered, non-overlapping segments. Its value at t is a pure
function: before the first segment it is the base, inside a segment it is eased between the
value the segment started from and its target (in log space for a ``log:`` ease), and
between segments it holds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from phasor.core.schema import MarkerRef, TimeRef
from phasor.core.values import geo_lerp, lerp


class TimeMap:
    """Marker name -> seconds. Re-recording narration replaces only this."""

    def __init__(self, markers: Mapping[str, float]):
        self.markers = dict(markers)

    def resolve(self, ref: TimeRef) -> float:
        if isinstance(ref, MarkerRef):
            if ref.marker not in self.markers:
                raise KeyError(ref.marker)
            return self.markers[ref.marker] + ref.offset
        return float(ref)

    def nearest(self, t: float) -> str | None:
        if not self.markers:
            return None
        return min(self.markers, key=lambda m: (abs(self.markers[m] - t), m))


@dataclass(frozen=True)
class ResolvedSegment:
    t0: float
    t1: float
    to: Any
    ease: Callable[[float], float]
    geometric: bool = False  # ``log:`` easing: move by factors (``geo_lerp``)


@dataclass(frozen=True)
class ResolvedTrack:
    initial: Any
    segments: tuple[ResolvedSegment, ...]

    @property
    def end(self) -> float:
        return self.segments[-1].t1 if self.segments else 0.0

    def value_at(self, t: float) -> Any:
        v = self.initial
        for s in self.segments:
            if t < s.t0:
                return v
            if t >= s.t1:
                v = s.to
                continue
            mix = geo_lerp if s.geometric else lerp
            return mix(v, s.to, s.ease((t - s.t0) / (s.t1 - s.t0)))
        return v
