"""Finite groups (the dihedral group is the exit demo)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phasor.domains import algebra_math as am


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
def test_dihedral_groups_are_groups_of_order_2n(n):
    d = am.dihedral(n)
    d.verify()
    assert d.order == 2 * n and d.name == f"D{n}"
    assert not d.is_abelian()
    r, s = 1, n  # the generating rotation and a reflection
    assert d.element_order(r) == n and d.element_order(s) == 2
    assert all(d.element_order(n + k) == 2 for k in range(n))  # every reflection is its own inverse
    assert d.mul(s, r) == d.mul(d.inverse(r), s) and d.mul(r, s) != d.mul(s, r)  # r s = s r^-1
    assert d.generated([r, s]) == frozenset(range(2 * n))  # two generators give everything


def test_the_cayley_table_is_a_latin_square():
    d = am.dihedral(4)
    t = np.array(d.table)
    for k in range(d.order):
        assert sorted(t[k]) == list(range(d.order)) and sorted(t[:, k]) == list(range(d.order))
    assert d.labels[:4] == ("e", "r", "r2", "r3") and d.labels[4] == "s"


def test_dihedral_elements_really_act_as_rotations_and_reflections_of_the_polygon():
    n = 5
    d = am.dihedral(n)
    for a in range(d.order):
        m = am.action_matrix(n, a)
        assert abs(np.linalg.det(m) - (-1.0 if a >= n else 1.0)) < 1e-12
        for b in range(d.order):  # the matrices multiply like the group does
            assert np.allclose(am.action_matrix(n, d.mul(a, b)), m @ am.action_matrix(n, b)), (a, b)
    # a rotation by one step maps the first vertex onto the next
    v = np.array([0.0, 1.0])
    ang = am.action_matrix(n, 1) @ v
    assert math.atan2(ang[1], ang[0]) == pytest.approx(math.pi / 2 + 2 * math.pi / n)


def test_subgroups_lagrange_and_cosets():
    d = am.dihedral(4)
    subs = d.subgroups()
    assert len(subs) == 10  # the known count for D4
    assert all(d.order % len(h) == 0 for h in subs)  # Lagrange's theorem
    rotations = d.generated([1])
    assert len(rotations) == 4 and d.is_normal(rotations)  # index 2 is always normal
    cosets = d.left_cosets(rotations)
    assert len(cosets) == 2 and set().union(*cosets) == set(range(8))  # they partition the group
    assert not d.is_normal(d.generated([4]))  # a single reflection is not normal in D4
    assert d.is_normal(d.generated([2]))  # the centre {e, r2} is
    centre = [a for a in range(8) if all(d.mul(a, b) == d.mul(b, a) for b in range(8))]
    assert centre == [0, 2]


def test_cyclic_and_symmetric_groups():
    z = am.cyclic(12)
    z.verify()
    assert z.is_abelian() and z.element_order(5) == 12 and z.element_order(4) == 3
    assert sorted(len(h) for h in z.subgroups()) == [1, 2, 3, 4, 6, 12]  # one per divisor
    s3 = am.symmetric(3)
    s3.verify()
    assert s3.order == 6 and not s3.is_abelian()
    assert sorted(len(h) for h in s3.subgroups()) == [1, 2, 2, 2, 3, 6]
    assert sorted(len(h) for h in am.dihedral(3).subgroups()) == sorted(
        len(h) for h in s3.subgroups()
    )  # D3 = S3
    with pytest.raises(ValueError, match="n >= 3"):
        am.dihedral(2)
    with pytest.raises(ValueError, match="1 to 5"):
        am.symmetric(9)


def test_verify_catches_a_table_that_is_not_a_group():
    bad = am.Group("bad", ((0, 1, 2), (1, 2, 0), (2, 0, 0)), ("a", "b", "c"))
    with pytest.raises(ValueError, match="inverse|associative"):
        bad.verify()


# ---------------------------------------------------------------- the rest of the module table


def test_cayley_graph_has_one_edge_per_element_and_generator_and_is_connected():
    d = am.dihedral(4)
    edges = am.cayley_edges(d, [1, 4])  # r and s
    assert len(edges) == 2 * d.order
    assert all(d.mul(x, g) == y for x, g, y in edges)
    reach, frontier = {0}, [0]
    while frontier:
        x = frontier.pop()
        for a, _, b in edges:
            if a == x and b not in reach:
                reach.add(b)
                frontier.append(b)
    assert reach == set(range(d.order))  # r and s generate D4: the graph is connected
    with pytest.raises(ValueError, match="does not generate"):
        am.cayley_edges(d, [1])  # rotations alone miss the flips


def test_quotient_by_a_normal_subgroup_is_a_group_and_lagrange_holds():
    d = am.dihedral(4)
    rot = d.generated([1])  # the rotations: index 2, normal
    q = am.quotient(d, rot)
    q.verify()
    assert q.order == 2 and q.order * len(rot) == d.order
    center = d.generated([2])  # {e, r2}: the center, normal; D4 / Z(D4) is the Klein group
    k = am.quotient(d, center)
    k.verify()
    assert k.order == 4 and k.is_abelian() and all(k.element_order(a) <= 2 for a in range(4))
    with pytest.raises(ValueError, match="not normal"):
        am.quotient(d, d.generated([4]))  # {e, s} is not normal


def test_homomorphisms_between_clocks_obey_the_first_isomorphism_theorem():
    for n, m, a in [(12, 4, 1), (6, 9, 3), (8, 8, 3), (12, 18, 3)]:
        h = am.clock_hom(n, m, a)
        assert am.is_homomorphism(am.cyclic(n), am.cyclic(m), h.images)
        assert n == len(h.kernel) * len(h.image)  # |G| = |ker| |im|
        assert all(h.images[k] == 0 for k in h.kernel)
    with pytest.raises(ValueError, match="not a homomorphism"):
        am.clock_hom(5, 3, 1)  # 5 * 1 is not 0 mod 3
    assert not am.is_homomorphism(am.cyclic(4), am.cyclic(4), (0, 1, 1, 0))


def test_subgroup_lattice_of_d4_has_ten_subgroups_and_covering_edges():
    d = am.dihedral(4)
    subs, edges = am.subgroup_lattice(d)
    assert len(subs) == 10 and len(subs[0]) == 1 and len(subs[-1]) == 8
    for lo, hi in edges:
        assert subs[lo] < subs[hi]  # strictly contained
        assert not any(subs[lo] < subs[k] < subs[hi] for k in range(len(subs)))  # covers
    # every subgroup but the whole group sits under something
    assert {lo for lo, _ in edges} == set(range(len(subs) - 1))


def test_galois_correspondence_of_square_root_towers():
    gal = am.galois_sqrt((2, 3))
    assert gal.group.order == 4 and gal.group.is_abelian()  # the Klein four-group
    assert len(gal.fields) == len(gal.subgroups) == 5
    for f, h in zip(gal.fields, gal.subgroups, strict=True):
        # the fundamental theorem: [L : F] = |H|, and H fixes exactly F's generators
        assert gal.degree // f.degree == len(h)
        for sigma in h:
            assert all(gal.fixes(sigma, d) for d in f.radicands)
    assert gal.degree == 4 and {f.name for f in gal.fields} >= {"Q", "Q(√2)", "Q(√6)", "Q(√2, √3)"}
    # sympy agrees on the degrees
    import sympy as sp

    x = sp.Symbol("x")
    assert sp.degree(sp.minimal_polynomial(sp.sqrt(2) + sp.sqrt(3) + sp.sqrt(5), x), x) == 8
    assert am.galois_sqrt((2, 3, 5)).degree == 8
    with pytest.raises(ValueError, match="square-free"):
        am.galois_sqrt((2, 8))


def test_permutations_cycles_sign_and_order():
    p = (1, 2, 0, 4, 3)  # 0->1->2->0 and 3<->4
    assert am.cycles(p) == [(0, 1, 2), (3, 4)]
    assert am.sign(p) == -1 and am.perm_order(p) == 6
    assert am.cycles((0, 1, 2)) == [] and am.sign((0, 1, 2)) == 1
    with pytest.raises(ValueError, match="permutation"):
        am.cycles((0, 0, 1))
