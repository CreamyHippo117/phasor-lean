"""Static guards for the core contracts. They are crude on purpose: each one names the single
place a thing may happen, so a second place shows up as a failing test, not as drift."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "phasor"
SOURCES = sorted(PKG.rglob("*.py"))


def _offenders(pattern: str, allowed: set[str]) -> list[str]:
    rx = re.compile(pattern)
    hits = []
    for path in SOURCES:
        rel = path.relative_to(PKG).as_posix()
        if rel in allowed:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line) and not line.lstrip().startswith(("#", '"', "``")):
                hits.append(f"{rel}:{n}: {line.strip()}")
    return hits


@pytest.mark.parametrize(
    ("why", "pattern", "allowed"),
    [
        (
            "contract 1: documents are only rebuilt by patches, and only Engine switches them",
            r"\.model_copy\(|\b_snap\s*=",
            {"core/patch.py", "engine.py"},
        ),
        (
            "one GL context, created in one place",
            r"\bcreate_context\(|\bcreate_standalone_context\(",
            {"render/glcontext.py", "render/doctor.py"},
        ),
        (
            "contract 3: state depends on (document, t), never on the clock or global randomness",
            r"\btime\.time\(|\bperf_counter\(|datetime\.now\(|\brandom\.\w+\(|np\.random\.",
            # bench and timing read the clock only to report where time went;
            # nothing they measure flows back into a scene's state.
            {"core/rng.py", "render/bench.py", "core/timing.py"},
        ),
        (
            "contract 4: sRGB encoding happens once, in render/output.py",
            r"\blinear_to_srgb\(",
            {"core/color.py", "render/output.py"},
        ),
        (
            "contract 7: one linear solver, and layout is solved only inside the bake",
            r"\bkiwisolver\b|\bsolver\.solve\(|\blabels\.place\(",
            {"layout/solver.py", "layout/bake.py"},
        ),
        (
            "contract 7: renderers and inspector read the bake; only Engine runs or applies it",
            r"(?<![\w.])bake\(|\bbaked_state\(",
            {"layout/bake.py", "engine.py"},
        ),
        (
            "one kernel path: only the registry (and the doctor's probe) creates a RawKernel",
            r"\bRawKernel\(|\bElementwiseKernel\(|\bnumba\b",
            {"render/gpu/registry.py", "render/doctor.py"},
        ),
        (
            "nothing outside the bridge moves pixels between CUDA and GL",
            r"\basnumpy\(|\.get\(\)\s*$",
            {"render/gpu/bridge.py", "render/doctor.py"},
        ),
        (
            "expressions are compiled from a whitelist, never evaluated as Python",
            r"(?<![\w.])(eval|exec)\(",
            set(),
        ),
    ],
)
def test_single_place_rules(why, pattern, allowed):
    assert _offenders(pattern, allowed) == [], why
