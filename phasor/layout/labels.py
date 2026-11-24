"""Stage 2 of layout: the label placer (contract 7).

For each label: 8 compass slots around its anchor, plus 8 leader-line slots further out.

    cost = overlap area  +  occlusion  +  distance from anchor  +  movement since last key

A label leaves the slot it had at the previous key time only when the best alternative is
more than 20% cheaper (hysteresis), so labels never jitter between two near-equal places.

Pure geometry in screen heights (see ``geometry``); occlusion comes in as a ``Coverage``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from phasor.layout.geometry import Box, Coverage

# slot -> direction. The order is the tie-break order: sides before corners.
SLOTS: dict[str, tuple[int, int]] = {
    "n": (0, 1),
    "e": (1, 0),
    "s": (0, -1),
    "w": (-1, 0),
    "ne": (1, 1),
    "se": (1, -1),
    "sw": (-1, -1),
    "nw": (-1, 1),
}
LEADER = "+leader"
HYSTERESIS = 0.20
TIE = 1e-9  # costs closer than this are equal; slot order decides, not rounding noise


# The weights are chosen so that a slot covered by more than a few percent always loses to a
# clean compass slot despite hysteresis, and a leader slot wins only when no compass slot
# is cleaner than about 7% covered.
@dataclass(frozen=True)
class Weights:
    overlap: float = 25.0  # per label-area covered by another label or cut by the frame
    occlusion: float = 25.0  # per label-area covered by other objects
    distance: float = 0.5  # per label height between label center and anchor center
    movement: float = 0.1  # per label height moved relative to the anchor since last key
    leader: float = 0.5  # flat price of needing a leader line


@dataclass(frozen=True)
class Request:
    id: str
    w: float
    h: float
    anchor: Box
    gap: float
    leader_len: float | None = None  # None: this object cannot draw a leader line
    # What hides this label, when that differs per label (3D: depends on its depth).
    coverage: Coverage | None = None


@dataclass(frozen=True)
class Placement:
    slot: str
    center: tuple[float, float]
    leader: bool
    cost: float
    blocked: float  # share of the label covered or cut, 0 when the slot is clean

    def box(self, req: Request) -> Box:
        return Box.around(self.center, req.w, req.h)


@dataclass(frozen=True)
class Previous:
    slot: str
    offset: tuple[float, float]  # label center minus anchor center at the previous key


def candidates(req: Request) -> list[tuple[str, tuple[float, float], bool]]:
    """(slot name, label center, needs leader) for every slot of one request."""
    ax, ay = req.anchor.center
    out = []
    for extra, suffix in ((0.0, ""), (req.leader_len, LEADER)):
        if extra is None:
            continue
        for name, (dx, dy) in SLOTS.items():
            x = ax + dx * (req.anchor.w / 2 + req.gap + extra + req.w / 2)
            y = ay + dy * (req.anchor.h / 2 + req.gap + extra + req.h / 2)
            out.append((name + suffix, (x, y), bool(suffix)))
    return out


def place(
    requests: Sequence[Request],
    *,
    frame: Box,
    coverage: Coverage | None = None,
    obstacles: Sequence[Box] = (),
    previous: Mapping[str, Previous] | None = None,
    weights: Weights = Weights(),
) -> dict[str, Placement]:
    """Choose a slot for every request. Deterministic; order of ``requests`` breaks ties."""
    previous = previous or {}
    cands = {r.id: candidates(r) for r in requests}
    chosen: dict[str, Placement] = {}

    def score(req: Request, slot: str, center, leader: bool) -> Placement:
        box = Box.around(center, req.w, req.h)
        area = max(box.area, 1e-12)
        cut = 1.0 - box.overlap(frame) / area
        hit = sum(box.overlap(o) for o in obstacles)
        hit += sum(box.overlap(p.box(q)) for q in requests
                   if q.id != req.id and (p := chosen.get(q.id)))  # fmt: skip
        overlap = cut + hit / area
        cov = req.coverage or coverage
        occlusion = cov.fraction(box) if cov else 0.0
        ax, ay = req.anchor.center
        off = (center[0] - ax, center[1] - ay)
        cost = (
            weights.overlap * overlap
            + weights.occlusion * occlusion
            + weights.distance * math.hypot(*off) / req.h
            + (weights.leader if leader else 0.0)
        )
        if (prev := previous.get(req.id)) is not None:
            moved = math.hypot(off[0] - prev.offset[0], off[1] - prev.offset[1])
            cost += weights.movement * moved / req.h
        return Placement(slot, center, leader, cost, min(1.0, overlap + occlusion))

    def choose(req: Request) -> Placement:
        scored = [score(req, *c) for c in cands[req.id]]
        lowest = min(p.cost for p in scored)
        best = next(p for p in scored if p.cost <= lowest + TIE)  # first of (near) equals
        prev = previous.get(req.id)
        held = next((p for p in scored if prev and p.slot == prev.slot), None)
        if held is not None and not best.cost < (1 - HYSTERESIS) * held.cost:
            return held
        current = chosen.get(req.id)
        if current is not None:
            now = next(p for p in scored if p.slot == current.slot)
            if not best.cost < now.cost - TIE:
                return now
        return best

    for _ in range(4):  # a few sweeps let neighbors make room for each other
        before = {k: v.slot for k, v in chosen.items()}
        for req in requests:
            chosen[req.id] = choose(req)
        if before == {k: v.slot for k, v in chosen.items()}:
            break
    return chosen
