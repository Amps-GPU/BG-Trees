"""
    The finite_gpufields module provides primitives to utilize the FiniteField type.
"""

import tensorflow as tf

from .finite_fields_tf import FiniteField


def tensordot(a, b, axes=None):
    """Perform a tensordot operation with the internal arrays of the ff
    and return a Finite Field
    Note: this will only produce the correct result if all intermediate operations are such that x < p
    """
    raise ValueError("Should not be used yet")
    result = tf.tensordot(a.n, b.n, axes=axes)
    return FiniteField(result, p=a.p)
