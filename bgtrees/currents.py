"""
    Low-level vectorized current builders.

    By using bgtrees.compute_current_j_mu (from __init__.py) the right function
    will be used automagically.
"""

import functools

import numpy
import tensorflow as tf

from bgtrees.finite_gpufields import operations as op
from bgtrees.finite_gpufields.finite_fields_tf import FiniteField

from .metric_and_vertices import V3g, V4g, new_V3g, η
from .settings import settings


# @gpu_function
def J_μ(lmoms, lpols, put_propagator=True, depth=0, verbose=False, einsum=numpy.einsum):
    """Recursive vectorized current builder. End of recursion is polarization tensors."""

    assert lmoms.shape[:2] == lpols.shape[:2]
    replicas, multiplicity, D = lmoms.shape
    Ds = lpols.shape[-1]
    assert D == Ds

    @functools.lru_cache
    def _J_μ(a, b, put_propagator=True):
        if verbose:
            print(f"Calling currents: a = {a}, b = {b}.")
        multiplicity = b - a
        if multiplicity == 1:
            return einsum("mn,rn->rm", η(D), lpols[:, a])
        else:
            if put_propagator:
                tot_moms = einsum("rid->rd", lmoms[:, a:b])
                propagators = 1 / einsum("rm,mn,rn->r", tot_moms, η(D), tot_moms)
            Jrμ = sum(
                [
                    einsum(
                        "rmno,rn,ro->rm",
                        V3g(einsum("rim->rm", lmoms[:, a : a + i, :]), einsum("rim->rm", lmoms[:, a + i : b, :])),
                        _J_μ(a, a + i),
                        _J_μ(a + i, b),
                    )
                    for i in range(1, multiplicity)
                ]
            ) + sum(
                [
                    einsum("mnop,rn,ro,rp->rm", V4g(D), _J_μ(a, a + i), _J_μ(a + i, a + j), _J_μ(a + j, b))
                    for i in range(1, multiplicity - 1)
                    for j in range(i + 1, multiplicity)
                ]
            )
            Jr_μ = einsum("mn,rn->rm", η(D), Jrμ)
            return einsum("r,rm->rm", propagators, Jr_μ) if put_propagator else Jr_μ

    return _J_μ(0, multiplicity, put_propagator=put_propagator)


def forest(lmoms, lpols, verbose=False, einsum=numpy.einsum):
    return einsum("rm,rm->r", lpols[:, 0], J_μ(lmoms[:, 1:], lpols[:, 1:], put_propagator=False, verbose=verbose))


@tf.function(reduce_retracing=False)
def _compute_propagator(mom_slice, eta):
    """Compute the propagator for a slice of momentum in FF.

    Parameters
    ----------
        mom_slice: (events, particles, 4vec)
        eta: (D,D)
    """
    tot_moms = tf.reduce_sum(mom_slice, axis=1)
    prop_lhs = op.ff_dot_product_single_batch(tot_moms, eta, rank_x=3, rank_y=2)
    prop_den = op.ff_dot_product(prop_lhs, tot_moms, rank_x=2, rank_y=2)  # rn, nr -> r
    return (1.0 / prop_den).reshape_ff((-1, 1))


@tf.function(reduce_retracing=False, jit_compile=True)
def _vert_3_gluon(slice_left, slice_right):
    """Computes the 3g vertex for the two slices coming in."""
    moms_sl = tf.reduce_sum(slice_left, axis=1)
    moms_sr = tf.reduce_sum(slice_right, axis=1)
    return new_V3g(moms_sl, moms_sr)


@tf.function(reduce_retracing=False)
def _contract_v3_current(vertex, jnu, jo):
    """Contract the vertex with the two associated currents:

    V3g   J    J -> J
    rmno, rn, ro -> rm
    """
    tmp = op.ff_dot_product_tris(vertex, jo, rank_x=4, rank_y=2)  # rmno, ro -> rmn
    return op.ff_dot_product(tmp, jnu, rank_x=3, rank_y=2)  # rmn, rn -> rm


@tf.function(reduce_retracing=False)
def _contract_v4_current(vertex, jnu, jo, jrho):
    """Contract the vertex with the 3 associated currents:

    V4g   J    J   J -> J
    mnop, rn, ro, rp -> rm
    """
    # Abuse the axes of vg4_c (D, D, D, D)
    # to fake later the product: rp, ponm -> ronm
    D = vertex.shape[0]
    if D is None:
        D = settings.D
    v4 = vertex.reshape_ff((D, D * D * D))

    tmp_1 = op.ff_dot_product_single_batch(jrho, v4, rank_x=2, rank_y=2)  # rp, pN -> rN
    tmp_1 = tmp_1.reshape_ff((-1, D, D, D))  # rN -> ronm
    tmp_1 = tmp_1.transpose_ff((0, 3, 2, 1))
    #     tmp_1 = op.ff_index_permutation("ronm->rmno", tmp_1)

    tmp_2 = op.ff_dot_product_tris(tmp_1, jo, rank_x=4, rank_y=2)  # rmno, ro -> rmn
    return op.ff_dot_product(tmp_2, jnu, rank_x=3, rank_y=2)  # rmn, rn -> rm


def another_j(lmoms, lpols, put_propagator=True, verbose=False):
    """
    Compute the current for an input array of shape
        (replicas, multiplicity, dimension)
    """
    events, multiplicity, D = lmoms.shape
    # Set up the dimensionality
    settings.D = D
    pprime = lpols.p
    mmatrix = FiniteField(tf.transpose(η(D)), pprime)
    vg4_c = FiniteField(tf.transpose(V4g(D)), pprime)
    f_eta = FiniteField(η(D), lpols.p)

    @functools.cache
    def _internal_j(a, b, put_propagator=True):
        """a and b are the extreme indexes of the particles
        being considered for this instance of the current

        Parameters
        ---------
            a: int
            b: int
        """
        # Compute the current current multiplicity
        mul = b - a
        if mul == 1:
            ret = op.ff_dot_product_single_batch(lpols[:, a], mmatrix, rank_x=2, rank_y=2)
            return ret

        propagators = 1.0
        if put_propagator:
            propagators = _compute_propagator(lmoms[:, a:b], f_eta)  # (r, 1)

        jrmu = FiniteField(tf.zeros((events, D)), pprime)
        for i in range(1, mul):
            v3val = _vert_3_gluon(lmoms[:, a : a + i], lmoms[:, a + i : b])  # (r,

            jnu = _internal_j(a, a + i)
            jo_3g = _internal_j(a + i, b)
            # rmno, rn, ro -> rm
            jrmu += _contract_v3_current(v3val, jnu, jo_3g)

            if i == mul - 1:
                continue
            for j in range(i + 1, mul):
                jo_4g = _internal_j(a + i, a + j)
                jrho_4g = _internal_j(a + j, b)
                jrmu += _contract_v4_current(vg4_c, jnu, jo_4g, jrho_4g)

        jr_submu = op.ff_dot_product_single_batch(jrmu, mmatrix, rank_x=2, rank_y=2)
        return jr_submu * propagators

    ret = _internal_j(0, multiplicity, put_propagator=put_propagator)
    # Clean the cache on the way out
    _internal_j.cache_clear()
    return ret
