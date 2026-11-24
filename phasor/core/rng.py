"""The only source of randomness (contract 3): same (object_id, seed) -> same stream."""

from __future__ import annotations

import hashlib

import numpy as np


def rng(object_id: str, seed: int = 0) -> np.random.Generator:
    digest = hashlib.sha256(f"{object_id}\x00{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))
