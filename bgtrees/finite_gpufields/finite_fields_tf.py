"""
    Tensorflow extention to have a Finite Fields type that works in GPU
    port of https://github.com/GDeLaurentis/pyadic/blob/main/pyadic/finite_field.py

    The TensorFlow ExtensionType FiniteField act as a container.
    Any operation between two FiniteField will result in another FiniteField

    Note: all operations between two FiniteField assume that p is going to always be prime
    and the same. Any checks should have occur before getting to this function.
"""

import functools
import operator
from time import time

import numpy as np
from pyadic.finite_field import ModP, finite_field_sqrt
import tensorflow as tf
from tensorflow import experimental

from .cuda_operators import wrapper_inverse

# EAGER_MODE = True
# # eager mode must be true since `extended_euclidean_algorithm` is not compilable yet
# tf.config.run_functions_eagerly(EAGER_MODE)

# To inspect the memory usage (it doesn't work well in Titan V)
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

DTYPE = tf.int64


@functools.lru_cache
def get_imaginary_for(p):
    """Get the value of sqrt(ModP(-1, p))"""
    modi = finite_field_sqrt(ModP(-1, p))
    if not isinstance(modi, ModP):
        raise ValueError(f"i is not in F({p=})")
    return modi.n


# TF extended euclidean algorithm
@tf.function
def loop_check(r, *args):
    return tf.math.reduce_any(r != 0)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPE),
    ]
)
def loop_body(r, s, ro, so):
    # Stop the computation for the entries for which r == 0
    stop = r == 0

    # Compute the quotient, but protecting it from 0s
    divisor = tf.where(stop, tf.cast(1, dtype=tf.int64), r)
    qu = ro // divisor

    rnew = tf.where(stop, r, ro - qu * r)
    snew = tf.where(stop, s, so - qu * s)

    ro = tf.where(stop, ro, r)
    so = tf.where(stop, so, s)

    return rnew, snew, ro, so


@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=DTYPE), tf.TensorSpec(shape=[], dtype=DTYPE)])
def extended_euclidean_algorithm(n, p):
    """Extended euclidean algorithm implementation (copying from pyadic).
    Note that this function is not compilable so it need to be decorated
    with `py_function`.
    """
    so, s = tf.ones_like(n), tf.zeros_like(n)
    ro = n
    r = tf.ones_like(ro) * p

    # https://github.com/GDeLaurentis/linac-dev/blob/master/linac/row_reduce.cu
    #     start = time()
    r, s, ro, so = tf.while_loop(loop_check, loop_body, (r, s, ro, so), parallel_iterations=1)
    #     end = time()
    #     print(f"-> {end-start}s")
    #     print(n)

    #     if ro.numpy().any() != 1:
    #         raise ZeroDivisionError("Inverse cannot be taken")

    return so


class FiniteField(experimental.ExtensionType):
    """TF implementation of Finite Fields
    Finite Fields are integer tensor which override certain methods
    """

    n: tf.Tensor
    p: int

    def __init__(self, n, p=None):
        if p is None or isinstance(n, FiniteField):
            # Then the input n is already a Finite Field
            self.n = n.n
            self.p = n.p
        # Complex numbers at the moment cannot be compiled
        elif tf.executing_eagerly() and np.iscomplex(n).any():
            a = FiniteField(tf.math.real(n), p=p)
            b = FiniteField(tf.math.imag(n), p=p)
            self.n = (a + b * get_imaginary_for(p)).n
            self.p = p
        else:
            n = tf.cast(n, dtype=tf.int64)
            self.p = p
            self.n = tf.math.floormod(n, tf.cast(p, dtype=tf.int64))

    def __validate__(self):
        assert self.n.dtype.is_integer, "FiniteFields must be integers"

    def _inv(self):
        s = wrapper_inverse(self.n)
        return self.__class__(s, self.p)

    # Primitive operations
    def __neg__(self):
        return self.__class__(self.p - self.n, self.p)

    def __pos__(self):
        return self

    def __add__(self, b):
        if not isinstance(b, FiniteField):
            b = FiniteField(b, self.p)
        return self.__class__(self.n + b.n, self.p)

    def __sub__(self, b):
        if not isinstance(b, FiniteField):
            b = FiniteField(b, self.p)
        return self.__class__(self.n - b.n, self.p)

    def __mul__(self, b):
        if not isinstance(b, FiniteField):
            b = FiniteField(b, self.p)
        return FiniteField(self.n * b.n, self.p)

    def __rtruediv__(self, b):
        return b * self._inv()

    def __truediv__(self, b):
        if not isinstance(b, FiniteField):
            b = self.__class__(b, self.p)
        return self * b._inv()

    def __pow__(self, n: int):
        if n < 0:
            return 1 / (self**-n)
        elif n == 0:
            return self.__class__(tf.ones_like(self.n), self.p)
        elif n % 2 == 0:
            root_2 = self ** (n // 2)
            return root_2 * root_2
        return self * (self ** (n - 1))

    # Experimental
    def __getitem__(self, idx):
        return self.__class__(self.n[idx], self.p)

    def zero_panning(self, front=True, m=4):
        """Pan with 0s the finite Field
        this can only be used (hence the experimental) when
        the shape of the values is (n, 1) or (1, n)
        and n < m, the panning happens in the `n` dimension up to m
        """
        panshape = []
        axis = -1
        for a, i in enumerate(self.shape):
            if i == 1:
                panshape.append(i)
            else:
                panshape.append(m - i)
                axis = a
        zeros = tf.zeros(panshape, dtype=tf.int64)
        if front:
            new_values = tf.concat([zeros, self.n], axis=axis)
        else:
            new_values = tf.concat([self.n, zeros], axis=axis)
        return self.__class__(new_values, self.p)

    def reshape_ff(self, new_shape):
        """Reshape the tensor contained in this FF"""
        new_values = tf.reshape(self.n, new_shape)
        return self.__class__(new_values, self.p)

    # Mirror version
    def __radd__(self, b):
        return self + b

    def __rsub__(self, b):
        return -(self - b)

    def __rmul__(self, b):
        return self * b

    # For Tensorflow's benefit
    @property
    def shape(self):
        return self.n.shape

    @property
    def values(self):
        return self.n


def test_me(a, b):
    """This function takes two finite fields
    (which in principle might be coming from C++)
    performs and operaton on them and then converts the result to a numpy array
    so that it can be easily parsed by TensorFlow
    """
    tmp = (a * b).n.numpy()
    return tmp


def generate_finite_field(list_of_ints, cardinality):
    """This function generates a finite field given a list of ints
    and the cardinality of the field
    Note that `list_of_ints` could also be a list of longs coming from C++
    in principle `np.array` should be able to deal with both
    Returns a `FiniteField` which in C++ can be taken as a PyObject* and passed around
    """
    # np.array should be able to interact with c-objects
    treat_input = np.array(list_of_ints)
    return FiniteField(treat_input, cardinality)


# Dispatchers
@experimental.dispatch_for_unary_elementwise_apis(FiniteField)
def finite_field_unary_operations(api_func, x):
    val = api_func(x.n)
    return FiniteField(val, x.p)


# @experimental.dispatch_for_binary_elementwise_apis(FiniteField, FiniteField)
# def finite_field_binary_operations(api_func, x, y):
#     val = api_func(x.n, y.n)
#     return FiniteField(val, x.p)


# Overrides for array transformations (no generic way of doing this?)
@experimental.dispatch_for_api(tf.unstack, {"value": FiniteField})
def finite_field_unstack(value, num=None, axis=0, name="unstack"):
    """Override the unstack function to act on a FiniteField container
    Returns a list of FiniteFields where the value of the FF is unstacked"""
    ret_int = tf.unstack(value.n, num=num, axis=axis, name=name)
    return [FiniteField(i, value.p) for i in ret_int]


@experimental.dispatch_for_api(tf.expand_dims, {"input": FiniteField})
def finite_field_expand_dims(input, axis, name=None):
    """Override expand dims for FiniteField containers"""
    # note: the name `input` for this input is tensorflow's fault...
    new_n = tf.expand_dims(input.n, axis=axis)
    return FiniteField(new_n, input.p)


@experimental.dispatch_for_api(tf.squeeze, {"input": FiniteField})
def finite_field_squeeze(input, axis, name=None):
    """Override squeeze for FiniteField containers"""
    new_n = tf.squeeze(input.n, axis=axis)
    return FiniteField(new_n, input.p)


######
@experimental.dispatch_for_api(tf.reduce_sum, {"input_tensor": FiniteField})
def finite_field_reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
    """Override the reduce_sum operation for a FiniteField container"""
    return FiniteField(tf.reduce_sum(input_tensor.n, axis=axis, keepdims=keepdims), input_tensor.p)


@experimental.dispatch_for_api(tf.reduce_prod, {"input_tensor": FiniteField})
def finite_field_reduce_prod(input_tensor, axis=None, keepdims=False, name=None):
    """Override the reduce_sum operation for a FiniteField container"""
    return FiniteField(tf.reduce_prod(input_tensor.n, axis=axis, keepdims=keepdims), input_tensor.p)


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
