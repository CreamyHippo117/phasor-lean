"""The label placer's occlusion oracle, backed by the real ``id`` pass (ADR 0001).

``layout`` defines the protocol; this implements it on top of a ``PassRunner`` so the
placer avoids the pixels objects actually cover (a diagonal line is a thin stripe, not its
bounding box).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from phasor.core.state import SceneState
from phasor.inspect.passes import PassRunner, inspect_size


class PassOracle:
    def __init__(self, runner: PassRunner) -> None:
        self.runner = runner

    def coverage(self, state: SceneState, exclude: frozenset[str]) -> np.ndarray:
        rest = tuple(o for o in state.objects if o.id not in exclude)
        others = dataclasses.replace(state, objects=rest)
        return self.runner.id_buffer(others, inspect_size(state)) != 0

    def depth(self, state: SceneState, exclude: frozenset[str]) -> np.ndarray:
        rest = tuple(o for o in state.objects if o.id not in exclude)
        others = dataclasses.replace(state, objects=rest)
        size = inspect_size(state)
        depth = self.runner.depth_buffer(others, size).copy()
        overlay = {o.index for o in rest if o.layer == "overlay"}
        if overlay:
            ids = self.runner.id_buffer(others, size)
            depth[np.isin(ids, list(overlay))] = -np.inf
        return depth
