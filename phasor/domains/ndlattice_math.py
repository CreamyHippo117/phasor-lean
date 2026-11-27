"""The numerics of ``nd.StarLattice`` (pure NumPy, no drawing): the points n of N^K, drawn
in 3D as sum_i n_i v_i, and when each one arrives.

The picture is a feedback-delay network with K loops. Loop i
delays the sound by d_i; an echo that went n_i times round loop i, for every i, arrives at
t = sum_i n_i d_i. So the echoes are the lattice N^K, and the ones heard by time t are the
lattice points under the hyperplane sum_i n_i d_i = t: about t^K / (K! prod d_i) of them, so
with 16 loops they outrun a 3D room's t^3 almost at once.

To see N^K in 3D, each loop gets an arm, a vector v_i in 3D (``star_directions``: the
first three are the axes, so K = 3 is the plain grid; later arms fan out between them), and
the echo n sits at sum_i n_i v_i (``project``). Arms can fade in one by one (``arm_weights``):
a weight w_i in [0, 1] per arm, an echo shown with the product of the weights of the arms
it uses, its position sliding out along the new arm as the weight grows. Everything is a
pure function of its inputs.

``lattice_points`` returns ``points`` (M, 3), ``arrival`` (M,) and ``values`` (M,). The flash
itself is shared with every lattice of lights: ``flash`` is ``sweep_math.envelope`` fed the
echo's own arrival time (the sweep here is a hyperplane in N-D, not a surface in 3D), and the
object draws through ``strokes.flash_points``.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from phasor.domains import sweep_math as sm

GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
MAX_LOOPS = 32
MAX_POINTS = 1_000_000  # points enumerated at most (the object's ``max_points`` cap)
MAX_ORDER = 10_000  # laps in all, at most (no limit in practice)
CAP_RADIANS = math.radians(72.0)  # "grid"/"cap" arms: within this of (1, 1, 1)
CANDIDATES = 4096  # directions the arms are chosen from (a Fibonacci net)
EXPAND = 4  # enumeration gives up past EXPAND * max_points points and lowers the time


def default_delays(k: int) -> tuple[float, ...]:
    """K loop delays (in seconds when time is): 1, then spread in [1, 1.5) by the golden
    ratio, so no two loops are in a simple ratio and echoes rarely arrive together."""
    if not 1 <= k <= MAX_LOOPS:
        raise ValueError(f"loops must be 1 to {MAX_LOOPS}, got {k}")
    return tuple(1.0 + 0.5 * ((i * GOLDEN) % 1.0) for i in range(k))


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    a = 2.0 * np.pi * GOLDEN * np.arange(n)
    return np.c_[r * np.cos(a), r * np.sin(a), z]


@functools.lru_cache(maxsize=16)
def _star(k: int, fan: str) -> np.ndarray:
    cand = _fibonacci_sphere(CANDIDATES)
    if fan == "sphere":
        chosen = [np.array([0.0, 0.0, 1.0])]
    else:
        c = np.ones(3) / math.sqrt(3.0)
        cand = cand[cand @ c >= math.cos(CAP_RADIANS)]
        chosen = list(np.eye(3)) if fan == "grid" else [c]
    # farthest point first: each new arm as far (in angle) from every earlier one as it can
    # be, so every prefix of the list is well spread and an added arm never moves the others
    best = np.max(cand @ np.array(chosen).T, axis=1)  # cos of the angle to the nearest arm
    while len(chosen) < k:
        j = int(np.argmin(best))
        chosen.append(cand[j])
        best = np.maximum(best, cand @ cand[j])
    out = np.array(chosen[:k], dtype=np.float64)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    out.flags.writeable = False
    return out


def star_directions(k: int, fan: str = "grid") -> np.ndarray:
    """(K, 3) unit vectors, one arm per loop. ``fan``: ``grid`` (the x, y and z axes first,
    so three loops are the cubic grid, then arms spread between them inside a cone of 72
    degrees round (1, 1, 1)), ``cap`` (the same cone, starting from its axis) or ``sphere``
    (spread over every direction, starting from z: the points pile up round the origin).
    The list for K arms starts with the list for fewer, so adding loops never moves an arm."""
    if not 1 <= k <= MAX_LOOPS:
        raise ValueError(f"loops must be 1 to {MAX_LOOPS}, got {k}")
    if fan not in ("grid", "cap", "sphere"):
        raise ValueError(f"fan must be grid, cap or sphere, got {fan!r}")
    return _star(k, fan)


def arm_vectors(directions, delays, spacing: float = 1.0, by_delay: bool = True) -> np.ndarray:
    """(K, 3) arms: each direction ``spacing`` long, times its delay over the mean delay when
    ``by_delay`` (so how far an echo sits from the origin follows when it arrives)."""
    u = np.asarray(directions, dtype=np.float64)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    d = np.asarray(delays, dtype=np.float64)
    length = spacing * (d / d.mean() if by_delay else np.ones_like(d))
    return u * length[:, None]


def lattice_count(k: int, max_order: int) -> int:
    """How many n in N^K have n_1 + ... + n_K <= ``max_order``: C(max_order + K, K)."""
    return math.comb(max_order + k, k)


def _expand(d: np.ndarray, t: float, max_order: int, cap: int):
    """Every n with sum n_i d_i <= t and sum n_i <= max_order, arm by arm; None when there
    are more than ``cap`` (the count after each arm is at most the final count)."""
    eps = 1e-9 * max(t, 1.0)
    arrival = np.zeros(1)
    laps = np.zeros(1, dtype=np.int64)
    index = np.zeros((1, 0), dtype=np.int16)
    for di in d:
        top = np.minimum(np.floor((t + eps - arrival) / di).astype(np.int64), max_order - laps)
        counts = top + 1
        total = int(counts.sum())
        if total > cap:
            return None
        rows = np.repeat(np.arange(len(arrival)), counts)
        start = np.repeat(np.cumsum(counts) - counts, counts)
        n = np.arange(total, dtype=np.int64) - start
        arrival = arrival[rows] + n * di
        laps = laps[rows] + n
        index = np.c_[index[rows], n.astype(np.int16)]
    return index, arrival, laps


@functools.lru_cache(maxsize=16)
def _enumerate(delays: tuple, max_time: float, max_order: int, max_points: int):
    d = np.asarray(delays, dtype=np.float64)
    k = len(d)
    t = float(max_time)
    shrunk = False
    while True:
        got = _expand(d, t, max_order, EXPAND * max_points)
        if got is not None:
            break
        shrunk = True  # the count grows like t^K: step the time down until it fits
        t *= min(0.9, 0.5 ** (1.0 / k))
    index, arrival, laps = got
    order = np.lexsort((*index.T[::-1], arrival))  # by arrival, ties by index: one order
    index, arrival, laps = index[order], arrival[order], laps[order]
    complete = t
    if len(arrival) > max_points:
        cut = arrival[max_points]
        keep = arrival < cut - 1e-9 * max(cut, 1.0)
        index, arrival, laps = index[keep], arrival[keep], laps[keep]
        complete, shrunk = float(cut), True
    for a in (index, arrival, laps):
        a.flags.writeable = False
    return index, arrival, laps, float(complete), shrunk


def enumerate_lattice(delays, max_time: float, max_order: int = MAX_ORDER,
                      max_points: int = 20_000) -> dict:  # fmt: skip
    """Every echo n in N^K (K = len(delays)) with arrival sum n_i d_i <= ``max_time`` and
    n_1 + ... + n_K <= ``max_order``, sorted by arrival (ties by index), at most
    ``max_points`` of them. When there are more, the latest are dropped *whole arrival
    times at a time*: the result is every echo arriving before ``complete_until`` (then
    ``truncated`` is True), never a ragged sample. Returns ``index`` (M, K) int16,
    ``arrival`` (M,), ``order`` (M,) total laps, ``complete_until`` and ``truncated``.
    Cached; the arrays are read-only."""
    d = tuple(float(x) for x in delays)
    if not 1 <= len(d) <= MAX_LOOPS:
        raise ValueError(f"1 to {MAX_LOOPS} delays, got {len(d)}")
    if min(d) <= 0:
        raise ValueError("delays must be positive")
    if not 1 <= max_points <= MAX_POINTS:
        raise ValueError(f"max_points must be 1 to {MAX_POINTS}")
    if max_time < 0 or max_order < 0:
        raise ValueError("max_time and max_order must be at least 0")
    index, arrival, laps, complete, truncated = _enumerate(d, float(max_time), int(max_order),
                                                           int(max_points))  # fmt: skip
    return {"index": index, "arrival": arrival, "order": laps,
            "complete_until": complete, "truncated": truncated}  # fmt: skip


def arm_weights(k: int, arms: float) -> np.ndarray:
    """(K,) how much of each arm is shown when ``arms`` of them are (a fraction: the last
    one part of the way in): arm i has weight clip(arms - i, 0, 1)."""
    return np.clip(float(arms) - np.arange(k), 0.0, 1.0)


smoothstep = sm.smoothstep  # the same ramp as the flash's attack


def point_weights(index: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(M,) each echo's share of light: the product of the weights of the arms it uses (an
    echo that never went round a loop does not care whether that loop is there yet)."""
    out = np.ones(len(index))
    for i in np.flatnonzero(np.asarray(weights) < 1.0):
        out = np.where(index[:, i] > 0, out * weights[i], out)
    return out


def project(index: np.ndarray, arms: np.ndarray, weights: np.ndarray | None = None,
            origin=(0.0, 0.0, 0.0)) -> np.ndarray:  # fmt: skip
    """(M, 3) the echoes in 3D: ``origin`` + sum_i n_i v_i. With ``weights``, each arm is
    scaled by smoothstep(w_i), so as an arm fades in its echoes slide out of the ones that
    do not use it (at weight 0 they sit on them)."""
    v = np.asarray(arms, dtype=np.float64)
    if weights is not None:
        v = v * smoothstep(weights)[:, None]
    return np.asarray(index, dtype=np.float64) @ v + np.asarray(origin, dtype=np.float64)


def flash(tau, rise: float = 0.05, decay: float = 0.6, afterglow: float = 0.35):
    """(light, burst) of an echo ``tau`` seconds after it arrived: ``sweep_math.envelope``
    (the one flash every lattice of lights shares) with nothing before arrival, ``rise`` its
    attack and ``afterglow`` its after. Light rises from 0 to 1, then settles to ``afterglow``
    with time constant ``decay``; burst (1 at the top, falling to 0 the same way) is the
    flash above the afterglow. Pure and continuous in tau (rise 0: a step at arrival)."""
    return sm.envelope(tau, attack=rise, decay=max(decay, 1e-9), before=0.0, after=afterglow)


def values(kind: str, index: np.ndarray, arrival: np.ndarray, max_time: float,
           arm_values=()) -> np.ndarray:  # fmt: skip
    """(M,) the number an echo is coloured by: ``arrival`` (its time over ``max_time``, 0-1),
    ``order`` (its laps in all over the most shown, 0-1), ``arm`` (which loop it went round
    most, i / K: a hue per loop) or ``value`` (sum_i n_i a_i with ``arm_values`` a, e.g. the
    gain in dB each lap costs; not scaled)."""
    index = np.asarray(index)
    k = index.shape[1]
    if kind == "arrival":
        return np.asarray(arrival, dtype=np.float64) / max(float(max_time), 1e-12)
    if kind == "order":
        laps = index.sum(axis=1).astype(np.float64)
        return laps / max(float(laps.max(initial=0.0)), 1.0)
    if kind == "arm":
        return np.argmax(index, axis=1) / k
    if kind == "value":
        a = np.asarray(arm_values, dtype=np.float64)
        if a.shape != (k,):
            raise ValueError(f"{k} loops but {len(a)} arm values")
        return index.astype(np.float64) @ a
    raise ValueError(f"unknown colour value {kind!r}")


def lattice_points(delays, max_time: float, *, directions=None, fan: str = "grid",
                   spacing: float = 1.0, by_delay: bool = True, arms: float | None = None,
                   max_order: int = MAX_ORDER, max_points: int = 20_000,
                   color_by: str = "arrival", arm_values=(),
                   origin=(0.0, 0.0, 0.0)) -> dict:  # fmt: skip
    """Everything at once: the echoes up to ``max_time`` as ``points`` (M, 3), ``arrival``
    (M,), ``values`` (M,) and ``weight`` (M,) (the light from ``arms``; all 1 by default),
    plus what ``enumerate_lattice`` returns and the ``arms`` vectors (K, 3)."""
    k = len(delays)
    u = star_directions(k, fan) if directions is None else directions
    v = arm_vectors(u, delays, spacing, by_delay)
    lat = enumerate_lattice(delays, max_time, max_order, max_points)
    w = arm_weights(k, k if arms is None else arms)
    return {**lat, "arms": v, "points": project(lat["index"], v, w, origin),
            "values": values(color_by, lat["index"], lat["arrival"], max_time, arm_values),
            "weight": point_weights(lat["index"], w)}  # fmt: skip
