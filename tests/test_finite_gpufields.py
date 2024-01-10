"""
    Tests the finite_gpufields module
"""

import numpy as np
from pyadic.finite_field import ModP

from bgtrees.finite_gpufields import FiniteField
from bgtrees.finite_gpufields.finite_fields_tf import oinsum
from bgtrees.finite_gpufields.operations import ff_dot_product

# p-value for tests
p = 2**31 - 19


# Helper functions
def array_to_pyadic(r):
    """Make an array of numbers into an array of pyadic ModP objects"""
    clist = []
    if hasattr(r, "shape") and r.shape:
        for i in r:
            clist.append(array_to_pyadic(i))
    else:
        clist.append(ModP(r, p))
    return clist


def compare(tfff, pyff):
    """Compare two arrays of finite fields (which might not be the same type)"""
    if not isinstance(tfff, FiniteField) and isinstance(pyff, FiniteField):
        return compare(pyff, tfff)

    a = tfff.values.numpy()
    b = np.vectorize(lambda x: x.n)(pyff)
    return np.testing.assert_allclose(a, b)


def test_ff_primitives_against_pyadic():
    """Test all FF primitives against pyadic"""
    r1 = (p * np.random.rand(2, 3, 4)).astype(int)
    r2 = (p * np.random.rand(2, 3, 4)).astype(int)

    f1 = FiniteField(r1, p)
    f2 = FiniteField(r2, p)

    p1 = np.array(array_to_pyadic(r1)).squeeze()
    p2 = np.array(array_to_pyadic(r2)).squeeze()

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
    N = 10
    dd1 = (p * np.random.rand(N, 2, 3)).astype(int)
    dd2 = (p * np.random.rand(N, 3, 4)).astype(int)
    fd1 = FiniteField(dd1, p)
    fd2 = FiniteField(dd2, p)

    dot_str = "rij,rjk->rik"
    res_object = oinsum(dot_str, fd1, fd2)
    fres_cuda = ff_dot_product(fd1, fd2)

    compare(fres_cuda, res_object)
