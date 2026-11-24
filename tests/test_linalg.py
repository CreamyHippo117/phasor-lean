"""Linear algebra numerics and objects."""

from __future__ import annotations

import numpy as np
import pytest

from phasor.domains import linalg_math as la


def test_determinant_is_signed_area_and_volume():
    assert la.determinant([[2, 0], [0, 3]]) == pytest.approx(6.0)
    assert la.determinant([[0, 1], [1, 0]]) == pytest.approx(-1.0)  # a reflection flips
    assert la.determinant([[1, 2], [2, 4]]) == pytest.approx(0.0)  # squashed flat
    assert la.determinant(np.diag([1, 2, 3])) == pytest.approx(6.0)
    # the image of the unit square has the determinant's area (shoelace formula)
    m = np.array([[1.0, 0.5], [0.2, 2.0]])
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    p = (m @ sq).T
    area = 0.5 * abs(
        sum(p[i, 0] * p[(i + 1) % 4, 1] - p[(i + 1) % 4, 0] * p[i, 1] for i in range(4))
    )
    assert area == pytest.approx(abs(la.determinant(m)))
    with pytest.raises(ValueError, match="2x2 or 3x3"):
        la.determinant(np.eye(4))


def test_eigenvectors_stay_on_their_line_and_are_stable():
    m = np.array([[2.0, 1.0], [1.0, 2.0]])
    e = la.eigen(m)
    assert e["values"].real.tolist() == pytest.approx([3.0, 1.0])
    for i in range(2):
        v = e["vectors"][:, i]
        assert m @ v == pytest.approx(e["values"][i] * v)  # A v = lambda v
        assert np.linalg.norm(v) == pytest.approx(1.0)
    again = la.eigen(m)
    assert np.array_equal(again["vectors"], e["vectors"])  # deterministic signs and order
    rot = la.eigen([[0.0, -1.0], [1.0, 0.0]])  # a quarter turn: no real direction survives
    assert np.allclose(rot["values"].real, 0) and np.abs(rot["values"].imag).tolist() == [1.0, 1.0]


def test_svd_is_rotate_stretch_rotate():
    m = np.array([[3.0, 1.0], [1.0, 2.0]]) @ np.array([[0.0, -1.0], [1.0, 0.0]])
    d = la.svd(m)
    assert d["U"] @ np.diag(d["S"]) @ d["Vt"] == pytest.approx(m)
    assert d["S"][0] >= d["S"][1] >= 0
    for r in (d["U"], d["Vt"]):
        assert r @ r.T == pytest.approx(np.eye(2))  # orthogonal: lengths and angles kept
    first, second, last = d["stages"]
    assert last == pytest.approx(m) and first @ first.T == pytest.approx(np.eye(2))
    assert np.linalg.svd(second)[1] == pytest.approx(d["S"])  # the middle stage only stretches


def test_rank_null_space_projection_and_gram_schmidt():
    m = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    assert la.rank(m) == 2
    n = la.null_space(m)
    assert n.shape == (3, 1) and np.allclose(m @ n, 0)
    assert la.null_space(np.eye(2)).shape == (2, 0)
    assert la.project([3.0, 4.0], [1.0, 0.0]).tolist() == [3.0, 0.0]
    q = la.gram_schmidt(np.array([[1.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]))
    assert q.shape == (3, 2)  # the dependent third column is dropped
    assert q.T @ q == pytest.approx(np.eye(2))
