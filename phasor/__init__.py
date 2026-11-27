"""phasor: GPU-rendered math animations that an AI can see, check and fix.

This module stays import-light on purpose: importing ``phasor`` must never pull in
CuPy, ModernGL or PyAV, and must never import a layer as a side effect (that would
hide layer violations from import-linter and create import cycles). Public names
are resolved lazily.
"""

from typing import Any

__version__ = "0.0.1"

# name -> module that defines it; resolved on first access.
_LAZY = {
    "Scene": "phasor.scene",
    "expr": "phasor.scene",
    "load": "phasor.scene",
    "Engine": "phasor.engine",
    "Dot": "phasor.domains.basic",
    "Line": "phasor.domains.basic",
    "Text": "phasor.domains.basic",
    "Equation": "phasor.domains.basic",
    "Tracer": "phasor.domains.basic",
    "Image": "phasor.domains.media",
    "Callout": "phasor.domains.basic",
    "Counter": "phasor.domains.basic",
    "Value": "phasor.domains.basic",
    "HingedCopy": "phasor.domains.unfold",
    "to_latex": "phasor.core.texconv",
    "to_mathtext": "phasor.core.texconv",
}
_MODULES = {  # domain modules: ``from phasor import cx``
    "cx": "phasor.domains.cx",
    "ac": "phasor.domains.acoustics",
    "fourier": "phasor.domains.fourier",
    "linalg": "phasor.domains.linalg",
    "pde": "phasor.domains.pde",
    "algebra": "phasor.domains.algebra",
    "calc": "phasor.domains.calc",
    "geo": "phasor.domains.geometry",
    "cloud": "phasor.domains.cloud",
    "nw": "phasor.domains.network",
    "dyn": "phasor.domains.dynamics",
    "conv": "phasor.domains.convolution",
    "filt": "phasor.domains.filters",
    "sweep": "phasor.domains.sweep",
    "nd": "phasor.domains.ndlattice",
    "net": "phasor.domains.net",
}

__all__ = ["__version__", *_LAZY, *_MODULES]


def __getattr__(name: str) -> Any:
    import importlib

    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    if name in _MODULES:
        return importlib.import_module(_MODULES[name])
    raise AttributeError(f"module 'phasor' has no attribute {name!r}")
