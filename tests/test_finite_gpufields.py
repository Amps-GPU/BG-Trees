"""
    Tests the finite_gpufields module
"""

import functools
import operator

import numpy as np
from pyadic.finite_field import ModP
import pytest
import tensorflow as tf

from bgtrees.finite_gpufields import FiniteField
from bgtrees.tools import _oinsum as oinsum
from bgtrees.finite_gpufields.operations import (
    ff_dot_product,
    ff_dot_product_single_batch,
    ff_dot_product_tris,
    ff_dot_product_tris_single_batch,
    ff_tensor_product,
)


def oinsum(eq, *arrays):
    """A ``einsum`` implementation for ``numpy`` object arrays."""
    lhs, output = eq.split("->")
    inputs = lhs.split(",")

    sizes = {}
    for term, array in zip(inputs, arrays):
        for k, d in zip(term, array.shape):
            sizes[k] = d

    out_size = tuple(sizes[k] for k in output)
    out = np.empty(out_size, dtype=object)

    inner = [k for k in sizes if k not in output]
    inner_size = [sizes[k] for k in inner]

    for coo_o in np.ndindex(*out_size):
        coord = dict(zip(output, coo_o))

        def gen_inner_sum():
            for coo_i in np.ndindex(*inner_size):
                coord.update(dict(zip(inner, coo_i)))

                locs = []
                for term in inputs:
                    locs.append(tuple(coord[k] for k in term))

                elements = []
                for array, loc in zip(arrays, locs):
                    elements.append(array[loc])

                yield functools.reduce(operator.mul, elements)

        tmp = functools.reduce(operator.add, gen_inner_sum())
        out[coo_o] = tmp

    # if the output is made of finite fields, take them out
    if isinstance(tmp, FiniteField) and len(out_size) == 0:
        out = tmp
    elif isinstance(tmp, FiniteField):
        p = tmp.p

        def unff(x):
            if isinstance(x, FiniteField):
                return x.n.numpy()
            return x

        vunff = np.vectorize(unff)

        new_out = vunff(out)
        out = FiniteField(new_out, p)

    return out


# p-value for tests
p = 2**31 - 19
NBATCH = 10


def _create_example(shape=(2, 3, 4)):
    """Creates a pair of a random int array between 0 and p and its FiniteField representation"""
    np_arr = np.random.randint(0, p, size=shape)
    ff_arr = FiniteField(np_arr, p)
    return np_arr, ff_arr


# Helper functions
def _array_to_pyadic(r):
    """Make an array of numbers into an array of pyadic ModP objects"""
    clist = []
    if hasattr(r, "shape") and r.shape:
        for i in r:
            clist.append(_array_to_pyadic(i).tolist())
    else:
        clist.append(ModP(r, p))

    ret = np.array(clist)
    # Ensure that the output shape is equal to the input
    return ret.reshape(r.shape)


def compare(tfff, pyff):
    """Compare two arrays of finite fields (which might not be the same type)"""
    if not isinstance(tfff, FiniteField) and isinstance(pyff, FiniteField):
        return compare(pyff, tfff)

    a = tfff.values.numpy()
    b = np.vectorize(lambda x: x.n)(pyff)
    return np.testing.assert_allclose(a, b)


def test_ff_primitives_against_pyadic():
    """Test all FF primitives against pyadic"""
    r1, f1 = _create_example()
    r2, f2 = _create_example()

    p1 = _array_to_pyadic(r1)
    p2 = _array_to_pyadic(r2)

    a = p // 2
    compare(-f1, -p1)

    compare(f1 + a, p1 + a)

    compare(f1 + f2, p1 + p2)

    compare(f1 - f2, p1 - p2)

    compare(f1 * a, p1 * a)

    compare(f1 * f2, p1 * p2)

    compare(f1 / a, p1 / a)

    compare(f1 / f2, p1 / p2)

    compare(a / f2, a / p2)

    compare(f1**5, p1**5)


def test_kernel_batched_dot_product():
    """Test the batched dot product
    i.e., rij * rjk -> rik
    for a batch r of matrices
    Whether it will run on CPU or GPU depends on the underlying machine
    """
    _, fd1 = _create_example(shape=(NBATCH, 2, 3))
    _, fd2 = _create_example(shape=(NBATCH, 3, 4))

    dot_str = "rij,rjk->rik"
    res_object = oinsum(dot_str, fd1, fd2)
    fres_cuda = ff_dot_product(fd1, fd2)
    compare(fres_cuda, res_object)

    # Now test the special 1-index-missing cases
    _, fd3 = _create_example(shape=(NBATCH, 3))
    dot_str = "rj,rjk->rk"
    res_object = oinsum(dot_str, fd3, fd2)
    fres_cuda = ff_dot_product(fd3, fd2)
    compare(fres_cuda, res_object)

    _, fd4 = _create_example(shape=(NBATCH, 3))
    dot_str = "rj,rj->r"
    res_object = oinsum(dot_str, fd3, fd4)
    fres_cuda = ff_dot_product(fd3, fd4)
    compare(fres_cuda, res_object)

    dot_str = "rij,rj->ri"
    res_object = oinsum(dot_str, fd1, fd4)
    fres_cuda = ff_dot_product(fd1, fd4)
    compare(fres_cuda, res_object)

    _, fdB = _create_example(shape=(NBATCH, 1, 1, 1))
    with pytest.raises(ValueError):
        ff_dot_product(fd1, fdB)

    with pytest.raises(ValueError):
        ff_dot_product(fdB, fd2)


def test_kernel_single_batched_dot_product():
    """Test the single batched dot product
    i.e., rij * jk -> rik
    for a batch r of matrices
    Whether it will run on CPU or GPU depends on the underlying machine
    """
    _, fd1 = _create_example(shape=(NBATCH, 2, 3))
    _, fd2 = _create_example(shape=(3, 4))

    dot_str = "rij,jk->rik"
    res_object = oinsum(dot_str, fd1, fd2)
    fres_cuda = ff_dot_product_single_batch(fd1, fd2)
    compare(fres_cuda, res_object)

    # Now test the special 1-index-missing cases
    _, fd3 = _create_example(shape=(NBATCH, 3))
    dot_str = "rj,jk->rk"
    res_object = oinsum(dot_str, fd3, fd2)
    fres_cuda = ff_dot_product_single_batch(fd3, fd2)
    compare(fres_cuda, res_object)

    _, fd4 = _create_example(shape=(3,))
    dot_str = "rj,j->r"
    res_object = oinsum(dot_str, fd3, fd4)
    fres_cuda = ff_dot_product_single_batch(fd3, fd4)
    compare(fres_cuda, res_object)

    dot_str = "rij,j->ri"
    res_object = oinsum(dot_str, fd1, fd4)
    fres_cuda = ff_dot_product_single_batch(fd1, fd4)
    compare(fres_cuda, res_object)

    _, fdB = _create_example(shape=(1, 1, 1))
    with pytest.raises(ValueError):
        ff_dot_product_single_batch(fd1, fdB)

    _, fdB2 = _create_example(shape=(NBATCH, 1, 1, 1))
    with pytest.raises(ValueError):
        ff_dot_product_single_batch(fdB2, fd2)


def test_kernel_batched_dot_product_tris():
    """Test the batched dot product of 3x3 tensors
    i.e., riaj * rjkb -> riakb for a batch r of matrices
    """
    _, fd1 = _create_example(shape=(NBATCH, 2, 5, 3))
    _, fd2 = _create_example(shape=(NBATCH, 3, 4, 6))

    dot_str = "riaj,rjkb->riakb"
    res_object = oinsum(dot_str, fd1, fd2)
    fres_cuda = ff_dot_product_tris(fd1, fd2)
    compare(fres_cuda, res_object)

    # Now test the special 1-index-missing cases
    _, fd3 = _create_example(shape=(NBATCH, 5, 3))
    dot_str = "raj,rjkb->rakb"
    res_object = oinsum(dot_str, fd3, fd2)
    fres_cuda = ff_dot_product_tris(fd3, fd2)
    compare(fres_cuda, res_object)

    _, fd4 = _create_example(shape=(NBATCH, 3, 5))
    dot_str = "riaj,rjb->riab"
    res_object = oinsum(dot_str, fd1, fd4)
    fres_cuda = ff_dot_product_tris(fd1, fd4)
    compare(fres_cuda, res_object)

    _, fdB = _create_example(shape=(NBATCH, 1, 1, 1, 1))
    with pytest.raises(ValueError):
        ff_dot_product_tris(fd1, fdB)

    with pytest.raises(ValueError):
        ff_dot_product_tris(fdB, fd2)


def test_kernel_single_batched_dot_product_tris():
    """Test the batched dot product of 3x3 tensors
    i.e., riaj * jkb -> riakb for a batch r of matrices
    """
    _, fd1 = _create_example(shape=(NBATCH, 2, 5, 3))
    _, fd2 = _create_example(shape=(3, 4, 6))

    dot_str = "riaj,jkb->riakb"
    res_object = oinsum(dot_str, fd1, fd2)
    fres_cuda = ff_dot_product_tris_single_batch(fd1, fd2)
    compare(fres_cuda, res_object)

    # Now test the special 1-index-missing cases
    _, fd3 = _create_example(shape=(NBATCH, 5, 3))
    dot_str = "raj,jkb->rakb"
    res_object = oinsum(dot_str, fd3, fd2)
    fres_cuda = ff_dot_product_tris_single_batch(fd3, fd2)
    compare(fres_cuda, res_object)

    _, fd4 = _create_example(shape=(3, 5))
    dot_str = "riaj,jb->riab"
    res_object = oinsum(dot_str, fd1, fd4)
    fres_cuda = ff_dot_product_tris_single_batch(fd1, fd4)
    compare(fres_cuda, res_object)

    _, fdB = _create_example(shape=(1, 1, 1, 1))
    with pytest.raises(ValueError):
        ff_dot_product_tris_single_batch(fd1, fdB)

    _, fdB2 = _create_example(shape=(NBATCH, 1, 1, 1, 1))
    with pytest.raises(ValueError):
        ff_dot_product_tris_single_batch(fdB2, fd2)


def test_einsum_permutation():
    """Test a permutation of index using einsum strings"""
    np1, ff1 = _create_example(shape=(2, 3, 4, 5))
    ein_str = "ijkl->kilj"
    np_res = _array_to_pyadic(np.einsum(ein_str, np1))
    ff_res = ff1.transpose_ff((2, 0, 3, 1))
    compare(ff_res, np_res)


def test_einsum_tensorproduct():
    """Test a tensor product between two arrays which can or not be batched"""
    np1, ff1 = _create_example(shape=(7, 1))
    np2, ff2 = _create_example(shape=(2, 5, 6))
    np3, ff3 = _create_example(shape=(NBATCH, 1, 4))
    np4, ff4 = _create_example(shape=(NBATCH, 9))

    # Unbatched - unbatched
    ein_str = "ij,klm->ijklm"
    np_res1 = _array_to_pyadic(np.einsum(ein_str, np1, np2))
    ff_res1 = ff_tensor_product(ein_str, ff1, ff2)
    compare(np_res1, ff_res1)

    # Batched - unbatched
    ein_str = "ij,rlm->rijlm"
    np_res2 = _array_to_pyadic(np.einsum(ein_str, np1, np3))
    ff_res2 = ff_tensor_product(ein_str, ff1, ff3)
    compare(np_res2, ff_res2)

    # Batched - batched
    ein_str = "rij,rk->rijk"
    np_res3 = _array_to_pyadic(np.einsum(ein_str, np3, np4))
    ff_res3 = ff_tensor_product(ein_str, ff3, ff4)
    compare(np_res3, ff_res3)

    # Now check for errors
    with pytest.raises(ValueError):
        ff_tensor_product("ij,rik->rjk", ff1, ff3)
    with pytest.raises(ValueError):
        ff_tensor_product("ii,rik->rk", ff1, ff3)


@pytest.mark.parametrize("mode", ["prod", "sum"])
def test_reduce(mode):
    """Checks that reduce_sum works as intended"""
    r1, f1 = _create_example(shape=(2, 2, 3))
    p1 = _array_to_pyadic(r1)

    if mode == "prod":
        reduce_function = tf.reduce_prod
        np_reduce = np.prod
    elif mode == "sum":
        reduce_function = tf.reduce_sum
        np_reduce = np.sum

    # For a selected axis
    sum_ff = reduce_function(f1, axis=1)
    sum_pp = np_reduce(p1, axis=1)
    compare(sum_ff, sum_pp)

    # For a selected axis
    sum_ff = reduce_function(f1, axis=1, keepdims=True)
    sum_pp = np_reduce(p1, axis=1, keepdims=True)
    compare(sum_ff, sum_pp)

    # For all axis
    sum_f2 = reduce_function(f1)
    sum_p2 = np_reduce(p1)
    compare(sum_f2, sum_p2)

    # And for a selection
    sum_f3 = reduce_function(f1, axis=(0, 2))
    sum_p3 = np_reduce(p1, axis=(0, 2))
    compare(sum_f3, sum_p3)
