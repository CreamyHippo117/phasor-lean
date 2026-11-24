"""Linear algebra numerics: pure NumPy."""

from __future__ import annotations

import numpy as np
import scipy.linalg

TOL = 1e-9


def matrix(m) -> np.ndarray:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] not in (2, 3):
        raise ValueError(f"a matrix here is 2x2 or 3x3, got shape {a.shape}")
    return a


def determinant(m) -> float:
    """Signed area (2x2) or volume (3x3) scale factor; negative flips orientation."""
    return float(np.linalg.det(matrix(m)))


def eigen(m) -> dict[str, np.ndarray]:
    """Eigenvalues (complex) and unit eigenvectors (columns), sorted by decreasing |lambda|
    then by decreasing real part, so the order is stable."""
    a = matrix(m)
    vals, vecs = np.linalg.eig(a)
    order = sorted(range(len(vals)), key=lambda i: (-abs(vals[i]), -vals[i].real, vals[i].imag))
    vals, vecs = vals[order], vecs[:, order]
    for i in range(vecs.shape[1]):  # fix the sign: the largest component positive
        k = int(np.argmax(np.abs(vecs[:, i])))
        if vecs[k, i].real < 0 or (abs(vecs[k, i].real) < TOL and vecs[k, i].imag < 0):
            vecs[:, i] = -vecs[:, i]
    return {"values": vals, "vectors": vecs}


def svd(m) -> dict[str, np.ndarray]:
    """``m = U diag(S) Vt``: rotate/reflect by Vt, stretch by S along the axes, rotate/reflect
    by U. ``stages`` are the matrices to apply in turn: Vt, then diag(S) @ Vt, then m."""
    a = matrix(m)
    u, s, vt = np.linalg.svd(a)
    return {"U": u, "S": s, "Vt": vt, "stages": [vt, np.diag(s) @ vt, u @ np.diag(s) @ vt]}


def rank(m) -> int:
    return int(np.linalg.matrix_rank(matrix(m)))


def null_space(m) -> np.ndarray:
    """An orthonormal basis (columns) of the vectors the matrix sends to zero."""
    a = matrix(m)
    _, s, vt = np.linalg.svd(a)
    k = int((s > TOL * max(s.max(), 1.0)).sum())
    return vt[k:].T


def gram_schmidt(vectors) -> np.ndarray:
    """Orthonormalize the columns of ``vectors`` (classical Gram-Schmidt, dependent columns
    are skipped). Returns the orthonormal columns."""
    v = np.asarray(vectors, dtype=np.float64)
    out: list[np.ndarray] = []
    for k in range(v.shape[1]):
        w = v[:, k].copy()
        for q in out:
            w -= (q @ w) * q
        n = np.linalg.norm(w)
        if n > 1e-10:
            out.append(w / n)
    return np.array(out).T if out else np.zeros((v.shape[0], 0))


def project(v, onto) -> np.ndarray:
    """The projection of ``v`` onto the line through ``onto``."""
    v, d = np.asarray(v, dtype=np.float64), np.asarray(onto, dtype=np.float64)
    return (v @ d) / (d @ d) * d


def expm(m, t: float = 1.0) -> np.ndarray:
    """The matrix exponential e^(t m): the flow of x' = m x for time ``t`` (it carries every
    start point to where it is after ``t``; its determinant is e^(trace t))."""
    return scipy.linalg.expm(float(t) * matrix(m))
