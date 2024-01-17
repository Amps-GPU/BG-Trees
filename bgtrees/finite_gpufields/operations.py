"""
    Collections of operation that can be performed with batched finite fields.
    In all cases, the batch dimension is denoted as `r` in the einsum-string.

    When the operation can be passed through tf.einsum safely
    (i.e., a simple permutation of indices), it will be done in this way.
"""

from collections import Counter

from .cuda_operators import wrapper_dot_product
from .finite_fields_tf import FiniteField
import tensorflow as tf


def ff_index_permutation(einstr, x):
    """Uses tf.einsum to permute the index of the tensor x
    Since this is simply an index permutation, it goes transparently to tf.einsum
    """
    ret = tf.einsum(einstr, x.values)
    return FiniteField(ret, x.p)


def ff_dot_product(x, y):
    """Perform a dot product between two batched Finite Fields
    Uses a CUDA kernel underneath

    Equivalent einsum string: "rij,rjk->rik"
    """
    ret = wrapper_dot_product(x.values, y.values)
    return FiniteField(ret, x.p)


def ff_tensor_product(einstr, x, y=None):
    """
    Wrapper to apply the tensor product for Finite Fields
        A_ijk...B_lmn... = C_ijk...lmn...
    It allows batched operation, i.e.,
        A_rijk...B_lmn... = C_rijk...lmn...
    or
        A_rijk...B_rlmn... = C_rijk...lmn...

    It uses tf.einsum underneath, but checks that it is a tensor product
    and hence it is safe to be used with finite fields.
    The main use case is to perform a tensor product where one (or both)
    tensors are actually batched
    """
    # TODO: check that this is truly a tensor product
    in_str, out_str = einstr.split("->")
    n_in = len(in_str.split(","))

    if (y is None and n_in != 1) or (y is not None and n_in != 2):
        raise ValueError(f"Number of inputs and string {einstr} don't match")

    # Now look at repeated indices and check that there are no contractions
    for i, c in Counter(in_str).items():
        if i == ",":
            pass
        if c > 2:
            raise ValueError(f"Index {i} appears more than twice in the input")
        elif c == 2:
            # Check that it is also in the output (confirming it as a batch dimension)
            if i not in out_str:
                raise ValueError(f"Index {i} is contracted. Not allowed")

    if y is None:
        ret = tf.einsum(einstr, x.values)
    else:
        ret = tf.einsum(einstr, x.values, y.values)

    return FiniteField(ret, x.p)
