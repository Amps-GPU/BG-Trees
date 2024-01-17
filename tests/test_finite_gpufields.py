"""
    Tests the finite_gpufields module
"""

import numpy as np
from pyadic.finite_field import ModP

from bgtrees.finite_gpufields import FiniteField
from bgtrees.finite_gpufields.finite_fields_tf import oinsum
from bgtrees.finite_gpufields.operations import ff_dot_product, ff_index_permutation

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
    return np.array(clist).squeeze()


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


def test_kernel_dot_product():
    """Test the batched dot product
    i.e., ij * jk -> ik
    for a batch r of matrices
    Whether it will run on CPU or GPU depends on the underlying machine
    """
    _, fd1 = _create_example(shape=(NBATCH, 2, 3))
    _, fd2 = _create_example(shape=(NBATCH, 3, 4))

    dot_str = "rij,rjk->rik"
    res_object = oinsum(dot_str, fd1, fd2)
    fres_cuda = ff_dot_product(fd1, fd2)

    compare(fres_cuda, res_object)


def test_einsum_permutation():
    """Test a permutation of index using einsum strings"""
    np1, ff1 = _create_example(shape=(2, 3, 4, 5))
    ein_str = "ijkl->kilj"
    np_res = _array_to_pyadic(np.einsum(ein_str, np1))
    ff_res = ff_index_permutation(ein_str, ff1)
    compare(ff_res, np_res)
