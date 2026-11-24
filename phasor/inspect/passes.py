"""What the inspector needs from a renderer, as a protocol (ADR 0001).

``render`` implements this; ``Engine`` injects it. With no implementation (tier C) the
inspector falls back to analytic geometry and says so in its output.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from phasor.core.state import SceneState

INSPECT_WIDTH = 480  # the inspector's passes run at 480x270 for 16:9, whatever the output is


def inspect_size(state: SceneState) -> tuple[int, int]:
    return INSPECT_WIDTH, max(1, round(INSPECT_WIDTH * state.height / state.width))


class PassRunner(Protocol):
    def color(self, state: SceneState, size: tuple[int, int] | None = None) -> np.ndarray:
        """(h, w, 4) float32 linear RGBA, row 0 = top: the picture before bloom and tone
        mapping. The linter reads it to see what is really behind a label."""
        ...

    def id_buffer(
        self, state: SceneState, size: tuple[int, int], force: frozenset[str] = frozenset()
    ) -> np.ndarray:
        """(h, w) uint32, row 0 = top: ``ObjectState.index`` of the nearest occluding surface,
        0 for background. Objects in ``force`` write even if they normally would not occlude."""
        ...

    def depth_buffer(self, state: SceneState, size: tuple[int, int]) -> np.ndarray:
        """(h, w) float64, row 0 = top: distance along the camera's view direction to the
        nearest occluding world surface; ``inf`` where there is none. Overlay objects are
        left out (they have no depth in the world)."""
        ...

    def solo_mask(
        self, state: SceneState, size: tuple[int, int], object_id: str, zoom_out: float = 1.0
    ) -> np.ndarray:
        """(h, w) bool, row 0 = top: the object's full silhouette as if nothing hid it.
        ``zoom_out`` k > 1 shrinks the view about the frame center so a region k times the
        frame fits in the buffer; the real frame is then the central 1/k of it."""
        ...
