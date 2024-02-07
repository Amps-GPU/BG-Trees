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
