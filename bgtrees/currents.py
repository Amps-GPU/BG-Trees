import functools

import numpy
import tensorflow as tf

from bgtrees.finite_gpufields import operations as op
from bgtrees.finite_gpufields.finite_fields_tf import FiniteField

from .metric_and_verticies import V3g, V4g, η


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
                        V3g(
                            einsum("rim->rm", lmoms[:, a : a + i, :]),
                            einsum("rim->rm", lmoms[:, a + i : b, :]),
                        ),
                        _J_μ(a, a + i),
                        _J_μ(a + i, b),
                    )
                    for i in range(1, multiplicity)
                ]
            ) + sum(
                [
                    einsum(
                        "mnop,rn,ro,rp->rm",
                        V4g(D),
                        _J_μ(a, a + i),
                        _J_μ(a + i, a + j),
                        _J_μ(a + j, b),
                    )
                    for i in range(1, multiplicity - 1)
                    for j in range(i + 1, multiplicity)
                ]
            )
            Jr_μ = einsum("mn,rn->rm", η(D), Jrμ)
            return einsum("r,rm->rm", propagators, Jr_μ) if put_propagator else Jr_μ

    return _J_μ(0, multiplicity, put_propagator=put_propagator)


def forest(lmoms, lpols, verbose=False, einsum=numpy.einsum):
    return einsum(
        "rm,rm->r",
        lpols[:, 0],
        J_μ(lmoms[:, 1:], lpols[:, 1:], put_propagator=False, verbose=verbose),
    )


def another_j(lmoms, lpols, put_propagator=True, verbose=False):
    """
    Compute the current for an input array of shape
        (replicas, multiplicity, dimension)
    """

    _, multiplicity, D = lmoms.shape
    mmatrix = FiniteField(tf.transpose(η(D)), lpols.p)
    vg4_c = FiniteField(tf.transpose(V4g(D)), lpols.p)

    @functools.lru_cache
    def _internal_j(a, b, put_propagator=True):
        """a and b are the extreme indexes of the particles
        being considered for this instance of the current

        Parameters
        ---------
            a: int
            b: int
        """
        if verbose:
            print(f"Calling currents: a = {a}, b = {b}.")
        # Compute the current current multiplicity
        mul = b - a
        if mul == 1:
            ret = op.ff_dot_product_single_batch(lpols[:, a], mmatrix)
            return ret

        propagators = 1.0
        if put_propagator:
            tot_moms = tf.reduce_sum(lmoms[:, a:b], axis=1)
            prop_lhs_raw = tf.einsum("rm, mn->rn", tot_moms.values, η(D))
            prop_lhs = FiniteField(prop_lhs_raw, tot_moms.p)
            prop_den = op.ff_dot_product(prop_lhs, tot_moms)  # rn, nr -> r
            propagators = (1.0 / prop_den).reshape_ff((-1, 1))

        first_list = []
        for i in range(1, mul):
            moms_sl = tf.reduce_sum(lmoms[:, a : a + i], axis=1)
            moms_sr = tf.reduce_sum(lmoms[:, a + i : b], axis=1)
            v3val = V3g(moms_sl, moms_sr, einsum=op.ff_tensor_product)

            jmu_one = _internal_j(a, a + i)
            jmu_two = _internal_j(a + i, b)

            tmp_lhs = op.ff_dot_product_tris(v3val, jmu_two)  # rmno, ro -> rmn
            ret = op.ff_dot_product(tmp_lhs, jmu_one)  # rmn, rn -> rm
            first_list.append(ret)

        second_list = []
        for i in range(1, mul - 1):
            for j in range(i + 1, mul):
                jmu_1 = _internal_j(a, a + i)
                jmu_2 = _internal_j(a + i, a + j)
                jmu_3 = _internal_j(a + j, b)

                # Abuse the axes of vg4_c (D, D, D, D)
                # to fake the product: rp, ponm -> ronm
                v1 = vg4_c.reshape_ff((D, D * D * D))
                v2 = op.ff_dot_product_single_batch(jmu_3, v1)  # rp, pN -> rN
                tmp_1 = v2.reshape_ff((-1, D, D, D))  # rN -> ronm

                tmp_1 = op.ff_index_permutation("ronm->rmno", tmp_1)
                tmp_2 = op.ff_dot_product_tris(tmp_1, jmu_2)  # rmno, ro -> rmn
                ret = op.ff_dot_product(tmp_2, jmu_1)  # rmn, rn -> rm
                second_list.append(ret)

        jrmu = sum(first_list) + sum(second_list)

        jr_submu = op.ff_dot_product_single_batch(jrmu, mmatrix)
        return jr_submu * propagators

    return _internal_j(0, multiplicity, put_propagator=put_propagator)
