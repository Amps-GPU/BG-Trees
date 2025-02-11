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


@tf.function(reduce_retracing=False, jit_compile=True)
def ff_einsum_generic(einstr, *args):
    """
    Tries to automagically select the right operation given the einstr.

    Currently only works for:
    - ff_tensor_product
    - ff_index_permutation
    """
    if len(args) == 2:
        return ff_tensor_product(einstr, *args)
    raise NotImplementedError(f"Automatic understanding of contractions not implemented for {einstr}")


@tf.function(reduce_retracing=False)
def ff_dot_product(x, y, rank_x=None, rank_y=None):
    """Perform a dot product between two batched Finite Fields
    Uses a CUDA kernel underneath

    Equivalent einsum string: "rij,rjk->rik"
    """
    if rank_x is None:
        rank_x = len(x.shape)
    if rank_y is None:
        rank_y = len(y.shape)

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

    if rank_x > 3 or rank_y > 3:
        raise ValueError("This function cannot deal with more than 2 axes")

    dims_to_squeeze = []
    if rank_x == 2:
        # If rank is 2, then it _must_ correspond, for the first input to
        # (batch, contracted_index); for now don't think about special cases
        x_vals = tf.expand_dims(x_vals, axis=1)
        dims_to_squeeze.append(1)

    if rank_y == 2:
        y_vals = tf.expand_dims(y_vals, axis=2)
        dims_to_squeeze.append(2)

    ret = wrapper_dot_product(x_vals, y_vals)
    if dims_to_squeeze:
        ret = tf.squeeze(ret, dims_to_squeeze)
    return FiniteField(ret, p)


@tf.function(reduce_retracing=False)
def ff_dot_product_single_batch(x, y, rank_x=None, rank_y=None):
    """Perform a dot product between one batch (x) and unbatched (y) Finite Fields
    Uses a CUDA kernel underneath

    Equivalent einsum string: "rij,jk->rik"
    """
    if rank_x is None:
        rank_x = len(x.shape)
    if rank_y is None:
        rank_y = len(y.shape)

    x_vals = x.values
    y_vals = y.values

    if rank_x > 3 or rank_y > 2:
        raise ValueError("This function cannot deal with more than 2 axes")

    dims_to_squeeze = []
    if rank_x == 2:
        # If rank is 2, then it _must_ correspond, for the first input to
        # (batch, contracted_index); for now don't think about special cases
        x_vals = tf.expand_dims(x_vals, axis=1)
        dims_to_squeeze.append(1)

    if rank_y == 1:
        y_vals = tf.expand_dims(y_vals, axis=1)
        dims_to_squeeze.append(2)

    ret = wrapper_dot_product_single_batch(x_vals, y_vals)
    if dims_to_squeeze:
        ret = tf.squeeze(ret, dims_to_squeeze)
    return FiniteField(ret, x.p)


@tf.function(reduce_retracing=False)
def ff_dot_product_tris(x, y, rank_x=None, rank_y=None):
    """
    Wrapper for a product rijk->rklmn with k contracted.

    This function reshapes the input and then applies wrapper_dot_product. It is a
    transitional function during development.

    TODO: Make all functions into one that is able to dispatch the right operation
    upon receiving an einsum string.

    It assumes both x and y are batched, i.e., the equivalent operation is:
        rijk, rklm -> rijlm

    It works by:
    - collapsing the ij and lm axes,
    - performing the operation rAk, rkB -> rAB,
    - and then unrolling back A and B.
    """
    if rank_x is None:
        rank_x = len(x.shape)
    if rank_y is None:
        rank_y = len(y.shape)

    shape_back = list(x.shape)[:-1] + list(y.shape)[2:]

    if rank_x > 4 or rank_y > 4:
        raise ValueError("This function cannot deal with more than 3 axes")

    if rank_x == 4:
        # Reshape this to collapse the intermediate ij axis
        new_x_shape = (-1, x.shape[1] * x.shape[2], x.shape[3])
        x = x.reshape_ff(new_x_shape)
        rank_x -= 1

    if rank_y == 4:
        # Collapse the intermediate lm axis
        new_y_shape = (-1, y.shape[1], y.shape[2] * y.shape[3])
        y = y.reshape_ff(new_y_shape)
        rank_y -= 1

    shape_back[0] = -1
    ret = ff_dot_product(x, y, rank_x=rank_x, rank_y=rank_y)
    return ret.reshape_ff(shape_back)


@tf.function(reduce_retracing=False)
def ff_dot_product_tris_single_batch(x, y, rank_x=None, rank_y=None):
    """Single batched version of ff_dot_product_tris
    """
    if rank_x is None:
        rank_x = len(x.shape)
    if rank_y is None:
        rank_y = len(y.shape)
    shape_back = list(x.shape)[:-1] + list(y.shape)[1:]

    if rank_x > 4 or rank_y > 3:
        raise ValueError("This function cannot deal with more than 3 axes")

    if rank_x == 4:
        # Reshape this to collapse the intermediate ij axis
        new_x_shape = (x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = x.reshape_ff(new_x_shape)
        rank_x -= 1

    if rank_y == 3:
        # Collapse the intermediate lm axis
        new_y_shape = (y.shape[0], y.shape[1] * y.shape[2])
        y = y.reshape_ff(new_y_shape)
        rank_y -= 1

    ret = ff_dot_product_single_batch(x, y, rank_x=rank_x, rank_y=rank_y)
    return ret.reshape_ff(shape_back)


@tf.function(reduce_retracing=False, jit_compile=True)
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
        raise ValueError("Wrong call of ff_tensor_product, non FF type being used")

    ret = tf.einsum(einstr, x_vals, y_vals)
    return FiniteField(ret, p)
