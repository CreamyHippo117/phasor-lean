"""phasor.domains: object types, grouped by field. Importing this package registers the
built-in types; Engine does that once at startup."""

from phasor.domains import (
    acoustics,
    algebra,
    basic,
    calc,
    cloud,
    convolution,
    cx,
    dynamics,
    filters,
    fourier,
    geometry,
    linalg,
    media,
    ndlattice,
    net,
    network,
    pde,
    sweep,
    unfold,
)  # fmt: skip

__all__ = ["acoustics", "algebra", "basic", "calc", "cloud", "convolution", "cx", "dynamics",
           "filters", "fourier", "geometry", "linalg", "media", "ndlattice", "net", "network",
           "pde", "sweep", "unfold"]  # fmt: skip
