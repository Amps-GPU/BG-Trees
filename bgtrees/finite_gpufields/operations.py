"""
    Collections of operation that can be performed with batched finite fields.
    In all cases, the batch dimension is denoted as `r` in the einsum-string.

    When the operation can be passed through tf.einsum safely
    (i.e., a simple permutation of indices), it will be done in this way.
"""

from collections import Counter

import tensorflow as tf

from .cuda_operators import wrapper_dot_product, wrapper_dot_product_single_batch
from .finite_fields_tf import FiniteField


def ff_einsum_generic(einstr, *args):
    """Tries to automagically select the right operation
    given the einstr
    currently only works for 
        ff_tensor_product
        ff_index_permutation
    """
    if len(args) == 1:
        return ff_index_permutation(einstr, *args)
    elif len(args) == 2:
        return ff_tensor_product(einstr, *args)
    raise NotImplementedError(f"Automatic understanding of contractions not implemented for {einstr}")


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
    y_vals = y
    x_vals = x
    p = None

    if isinstance(y, FiniteField):
        y_vals = y.values
        p = y.p

    if isinstance(x, FiniteField):
        x_vals = x.values
        p = x.p

    if p is None:
        # You should not be using this function if this is the case
        raise ValueError("Wrong call of ff_dot_product")

    if len(x.shape) > 3 or len(y.shape) > 3:
        raise ValueError("This function cannot deal with more than 2 axes")

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
    return FiniteField(ret, p)


def ff_dot_product_single_batch(x, y):
    """Perform a dot product between one batch (x) and unbatched (y) Finite Fields
    Uses a CUDA kernel underneath

    Equivalent einsum string: "rij,jk->rik"
    """
    x_vals = x.values
    y_vals = y.values

    if len(x.shape) > 3 or len(y.shape) > 2:
        raise ValueError("This function cannot deal with more than 2 axes")

    dims_to_squeeze = []
    if len(x.shape) == 2:
        # If rank is 2, then it _must_ correspond, for the first input to
        # (batch, contracted_index); for now don't think about special cases
        x_vals = tf.expand_dims(x_vals, axis=1)
        dims_to_squeeze.append(1)

    if len(y.shape) == 1:
        y_vals = tf.expand_dims(y_vals, axis=1)
        dims_to_squeeze.append(2)

    ret = wrapper_dot_product_single_batch(x_vals, y_vals)
    if dims_to_squeeze:
        ret = tf.squeeze(ret, dims_to_squeeze)
    return FiniteField(ret, x.p)


def ff_dot_product_tris(x, y):
    """Wrapper for a product rijk->rklmn with k contracted
    reshapes the input and then applies wrapper_dot_product
    transitional function during development
    TODO: make all functions into one that is able to dispatch the right operation
    upon receiving a eisum string

    It assumes both x and y are batched, i.e., the equivalent operation is
        rijk, rklm -> rijlm
    it works by collapsing the ij and lm axes, performing the operation rAk, rkB -> rAB
    and then unrolling back A and B
    """
    shape_back = list(x.shape)[:-1] + list(y.shape)[2:]

    if len(x.shape) > 4 or len(y.shape) > 4:
        raise ValueError("This function cannot deal with more than 3 axes")

    if len(x.shape) == 4:
        # Reshape this to collapse the intermediate ij axis
        new_x_shape = (x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = x.reshape_ff(new_x_shape)

    if len(y.shape) == 4:
        # Collapse the intermediate lm axis
        new_y_shape = (y.shape[0], y.shape[1], y.shape[2] * y.shape[3])
        y = y.reshape_ff(new_y_shape)

    ret = ff_dot_product(x, y)
    return ret.reshape_ff(shape_back)


def ff_dot_product_tris_single_batch(x, y):
    """Single batched version of ff_dot_product_tris
    TODO: it should eventually go as well
    """
    shape_back = list(x.shape)[:-1] + list(y.shape)[1:]

    if len(x.shape) > 4 or len(y.shape) > 3:
        raise ValueError("This function cannot deal with more than 3 axes")

    if len(x.shape) == 4:
        # Reshape this to collapse the intermediate ij axis
        new_x_shape = (x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = x.reshape_ff(new_x_shape)

    if len(y.shape) == 3:
        # Collapse the intermediate lm axis
        new_y_shape = (y.shape[0], y.shape[1] * y.shape[2])
        y = y.reshape_ff(new_y_shape)

    ret = ff_dot_product_single_batch(x, y)
    return ret.reshape_ff(shape_back)


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

    y_vals = y
    x_vals = x
    p = None

    if isinstance(y, FiniteField):
        y_vals = y.values
        p = y.p

    if isinstance(x, FiniteField):
        x_vals = x.values
        p = x.p

    if p is None:
        # You should not be using this function if this is the case
        raise ValueError("Wrong call of ff_tensor_product")
        
    ret = tf.einsum(einstr, x_vals, y_vals)
    return FiniteField(ret, p)
