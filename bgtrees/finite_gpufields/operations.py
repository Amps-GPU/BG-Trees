"""
    Collections of operation that can be performed with batched finite fields.
    In all cases, the batch dimension is denoted as `r` in the einsum-string.

    When the operation can be passed through tf.einsum safely
    (i.e., a simple permutation of indices), it will be done in this way.
"""

from collections import Counter

import tensorflow as tf

from .cuda_operators import wrapper_dot_product
from .finite_fields_tf import FiniteField


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
    x_vals = x.values
    y_vals = y.values

    dims_to_squeeze = []
    if len(x.shape) == 2:
        # If rank is 2, then it _must_ correspond, for the first input to
        # (batch, contracted_index); for now don't think about special cases
        x_vals = tf.expand_dims(x_vals, axis=1)
        dims_to_squeeze.append(1)

    if len(y.shape) == 2:
        y_vals = tf.expand_dims(y_vals, axis=2)
        dims_to_squeeze.append(2)

    ret = wrapper_dot_product(x_vals, y_vals)
    if dims_to_squeeze:
        ret = tf.squeeze(ret, dims_to_squeeze)
    return FiniteField(ret, x.p)


def ff_tensor_product(einstr, x, y):
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
    # Perform a quick check to ensure there are no contractions in the string
    in_str, out_str = einstr.split("->")

    for i, c in Counter(in_str).items():
        if i == ",":
            continue
        if c > 2:
            raise ValueError(f"Index {i} appears more than twice in the input?")
        if i not in out_str:
            # Don't allow ff_tensor_product to be used as a substitute for squeeze() or sum
            raise ValueError(f"Index {i} is contracted. Not allowed")

    ret = tf.einsum(einstr, x.values, y.values)
    return FiniteField(ret, x.p)
