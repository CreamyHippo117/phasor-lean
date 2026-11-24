"""Wave fields of point sources: pure NumPy, no drawing.

A source k at p_k with amplitude a_k and phase phi_k sends out
a_k cos(k |x - p_k| - w t + phi_k) / sqrt(|x - p_k|) (2D spreading), k = 2 pi / wavelength.
The sum is evaluated per pixel by the ``wave_field`` kernel; ``amplitude`` here is its
NumPy specification, for points anywhere (tests, semantics).

A phased array steers its beam by delaying its elements: for the far field in direction u
from the array centre c, |x - p_k| ~ R - (p_k - c).u, so the waves add up in step when
phi_k = k (p_k - c).u (``steering_phases``).
"""

from __future__ import annotations

import math

import numpy as np

from phasor.render.gpu.kernels.wave import ALWAYS, amplitude

__all__ = ["ALWAYS", "amplitude", "beamwidth", "direction", "elements", "source_table",
           "steering_phases", "wavelength"]  # fmt: skip


def wavelength(speed: float, frequency: float) -> float:
    return speed / frequency


def direction(degrees: float) -> np.ndarray:
    """The unit vector at ``degrees`` counterclockwise from +x."""
    a = math.radians(degrees)
    return np.array([math.cos(a), math.sin(a)])


def elements(layout: str, count: int, spacing: float, axis: float, center) -> np.ndarray:
    """(count, 2) element positions. ``line``: ``spacing`` apart along the direction
    ``axis`` (degrees), centred on ``center``; ``ring``: on a circle round ``center`` with
    ``spacing`` between neighbours along it, the first at ``axis``."""
    c = np.asarray(center, dtype=np.float64)[:2]
    if layout == "line":
        along = (np.arange(count) - (count - 1) / 2) * spacing
        return c + along[:, None] * direction(axis)[None, :]
    if layout == "ring":
        radius = count * spacing / (2 * math.pi)
        ang = math.radians(axis) + 2 * math.pi * np.arange(count) / count
        return c + radius * np.c_[np.cos(ang), np.sin(ang)]
    raise ValueError(f"layout must be line or ring, got {layout!r}")


def steering_phases(points, center, wavenumber: float, degrees: float) -> np.ndarray:
    """Phases (radians) that point the beam of elements ``points`` (n, 2) at ``degrees``
    (counterclockwise from +x) as seen from ``center``."""
    d = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)[:2]
    return wavenumber * (d @ direction(degrees))


def beamwidth(count: int, spacing: float, wavelength_: float, steer: float) -> float:
    """The half-power width (degrees) of a line array's main lobe steered ``steer`` degrees
    off broadside: about 0.886 wavelength / (count spacing cos steer) radians."""
    c = max(math.cos(math.radians(steer)), 1e-3)
    return math.degrees(0.886 * wavelength_ / (count * spacing * c))


def source_table(points, amplitudes, phases, reaches) -> np.ndarray:
    """The ``wave_field`` kernel's (n, 5) rows: x, y, amplitude, phase (radians), reach
    (``ALWAYS``: always on)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.ascontiguousarray(np.c_[pts, amplitudes, phases, reaches], dtype=np.float64)
