"""The prism look (the reference style): primes heard through the zeta zeros.

Chebyshev's psi(x) adds ln p at every prime power, so E(x) = psi(x) - x + ln 2 pi is a jagged
error that jumps at each prime. The explicit formula rebuilds it from the zeros of the zeta
function: each zero 1/2 + i gamma_n contributes one wave, and with more zeros the waves add
up to the jumps. The true error glows with a hue running along it; the sum over the first
N zeros (1, 5, 20, then 60) sharpens toward it; the equations drift.

Run:  uv run python examples/prism_primes.py      (prints how close each sum gets)
"""

import functools
import math

import numpy as np

from phasor import Equation, Line, Scene

X0, X1 = 2.0, 50.0  # the stretch of x shown
SAMPLES = 1400
COUNTS = (1, 5, 20, 60)  # how many zeros each approximation uses


@functools.cache
def _all_zeros() -> tuple[float, ...]:
    import mpmath

    return tuple(float(mpmath.zetazero(k).imag) for k in range(1, max(COUNTS) + 1))


def zeta_zeros(n: int) -> tuple[float, ...]:
    """Imaginary parts gamma_1..gamma_n of the first nontrivial zeros of zeta (each zero is
    found once, however many approximations use it)."""
    return _all_zeros()[:n]


def chebyshev_psi(x: np.ndarray) -> np.ndarray:
    """psi(x) = sum of ln p over prime powers p^k <= x."""
    top = int(x.max()) + 1
    sieve = np.ones(top + 1, dtype=bool)
    sieve[:2] = False
    for k in range(2, int(top**0.5) + 1):
        if sieve[k]:
            sieve[k * k :: k] = False
    jumps = np.zeros(top + 1)
    for p in np.flatnonzero(sieve):
        q = p
        while q <= top:
            jumps[q] += math.log(p)
            q *= p
    return np.cumsum(jumps)[np.floor(x).astype(int)]


def explicit_sum(x: np.ndarray, n: int) -> np.ndarray:
    """S(x) over the first n zeros; the error is about -S(x)."""
    s = np.zeros_like(x)
    for g in zeta_zeros(n):
        s += 2 * np.sqrt(x) / math.sqrt(0.25 + g * g) * np.cos(g * np.log(x) - math.atan(2 * g))
    return s


x = np.linspace(X0, X1, SAMPLES)
error = chebyshev_psi(x) - x + math.log(2 * math.pi)


def curve(y: np.ndarray, lift: float) -> list[tuple[float, float, float]]:
    """Plot coordinates: x across the frame, the error scaled, raised by ``lift``."""
    return [
        (float((a - 26.0) / 3.4), float(b * 0.42 + lift), 0.0) for a, b in zip(x, y, strict=True)
    ]


s = Scene("prism_primes", look="prism")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 9.0)
s.add(Line(id="truth", points=curve(error, -1.6), style={"color": "highlight"}))
for n in COUNTS:
    s.add(Line(id=f"zeros_{n}", points=curve(-explicit_sum(x, n), -1.6), hue_along=0.0,
               style={"color": "secondary", "glow": "soft", "stroke": "hairline"}))  # fmt: skip
s.add(Equation(id="error_eq", pos=(0.0, 2.9, 0.0),
               parts=("$E(x) =$", "$\\psi$", "$(x) - x + \\ln 2\\pi$"),
               colors=("primary", "secondary", "primary")))  # fmt: skip
SUM = (r"$E(x) \approx -\sum_{n=1}^{N} \dfrac{2\sqrt{x}}{\sqrt{1/4 + \gamma_n^2}}"
       r"\cos(\gamma_n \ln x - \arctan 2\gamma_n)$")  # fmt: skip
s.add(Equation(id="sum_eq", pos=(0.0, 1.65, 0.0), parts=(SUM,)))
s.add(Equation(id="zeros_eq", pos=(0.0, 0.55, 0.0),
               parts=("$\\gamma_n = 14.1347,\\ 21.0220,\\ 25.0109,\\ \\ldots$",),
               colors=("highlight",), style={"size": "label"}))  # fmt: skip

s.marker("truth", 0.5)
s.marker("one_zero", 3.0)
s.marker("five", 5.5)
s.marker("twenty", 8.0)
s.marker("sixty", 10.5)
s.at("truth").reveal("error_eq")
s.at("truth", 0.6).trace("truth")
s.at("one_zero").reveal("sum_eq", "zeros_eq")
# one sum gives way to the next: the old one leaves, then the new one arrives
for (a, b), m in zip(
    zip((None, *COUNTS), COUNTS, strict=False), ("one_zero", "five", "twenty", "sixty"), strict=True
):
    if a is not None:
        s.at(m).exit(f"zeros_{a}")
    s.at(m, 0.6).reveal(f"zeros_{b}")
s.marker("end", 13.0)

if __name__ == "__main__":
    for n in COUNTS:
        gap = np.abs(-explicit_sum(x, n) - error)
        print(f"{n:3d} zeros: median distance to the true error {np.median(gap):.3f}")
