"""
    Tensorflow extention to have a Finite Fields type that works in GPU
    port of https://github.com/GDeLaurentis/pyadic/blob/main/pyadic/finite_field.py

    Note: all operations between two finite fields assume that the size is the same
    and that p is going to always be prime. Any checks should have occur before getting to
    this function.

    The `isinstance` conditional in practice will create two versions of the function
    depending on the input type.
"""
import functools
import operator
import string

import numpy as np
from pyadic.finite_field import ModP, finite_field_sqrt
import tensorflow as tf
from tensorflow import experimental

# tf.config.run_functions_eagerly(True)


@functools.lru_cache
def get_imaginary_for(p):
    """Get the value of sqrt(ModP(-1, p))"""
    modi = finite_field_sqrt(ModP(-1, p))
    if not isinstance(modi, ModP):
        raise ValueError(f"i is not in F({p=})")
    return modi.n


def extended_euclidean_algorithm(n, p):
    """Extended euclidean algorithm implementation (copying from pyadic)"""
    p = tf.cast(p, dtype=tf.int64)
    so, s = tf.ones_like(n), tf.zeros_like(n)
    to, t = tf.zeros_like(n), tf.ones_like(p)
    ro, r = n, p

    ones = tf.ones_like(r)

    # Note that this loop won't end until coefficients are found for the entire tensor
    while tf.math.reduce_any(r != 0):
        # Stop the computation for the entries for which r == 0
        stop = r == 0

        # Compute the quotient, but protecting it from 0s
        divisor = tf.where(stop, ones, r)
        qu = ro // divisor

        rnew = tf.where(stop, r, ro - qu * r)
        snew = tf.where(stop, s, so - qu * s)
        tnew = tf.where(stop, t, to - qu * t)

        ro = tf.where(stop, ro, r)
        so = tf.where(stop, so, s)
        to = tf.where(stop, to, t)

        r, s, t = rnew, snew, tnew

    return so, to, ro


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
        elif np.iscomplex(n).any():
            a = FiniteField(tf.math.real(n), p=p)
            b = FiniteField(tf.math.imag(n), p=p)

            self.n = (a + b * get_imaginary_for(p)).n
            self.p = p
        else:
            n = tf.cast(n, dtype=tf.int64)
            self.p = p
            self.n = tf.math.floormod(n, p)

    def __validate__(self):
        assert self.n.dtype.is_integer, "FiniteFields must be integers"

    def _inv(self):
        s, _, c = extended_euclidean_algorithm(self.n, self.p)
        if c.numpy().any() != 1:
            raise ZeroDivisionError(f"{self} has no inverse")
        return FiniteField(s, self.p)

    # Primitive operations
    def __neg__(self):
        return FiniteField(self.p - self.n, self.p)

    def __pos__(self):
        return self

    def __add__(self, b):
        if not isinstance(b, FiniteField):
            b = FiniteField(b, self.p)
        return FiniteField(self.n + b.n, self.p)

    def __sub__(self, b):
        if not isinstance(b, FiniteField):
            b = FiniteField(b, self.p)
        return FiniteField(self.n - b.n, self.p)

    def __mul__(self, b):
        if not isinstance(b, FiniteField):
            b = FiniteField(b, self.p)
        return FiniteField(self.n * b.n, self.p)

    def __rtruediv__(self, b):
        return b * self._inv()

    def __truediv__(self, b):
        if not isinstance(b, FiniteField):
            b = FiniteField(b, self.p)
        return self * b._inv()

    def __pow__(self, n: int):
        if n < 0:
            return 1 / (self**-n)
        elif n == 0:
            return FiniteField(tf.ones_like(self.n), self.p)
        elif n % 2 == 0:
            root_2 = self ** (n // 2)
            return root_2 * root_2
        return self * (self ** (n - 1))

    # Experimental
    def __getitem__(self, idx):
        return FiniteField(self.n[idx], self.p)

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
        return FiniteField(new_values, self.p)

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


@experimental.dispatch_for_binary_elementwise_apis(FiniteField, FiniteField)
def finite_field_binary_operations(api_func, x, y):
    val = api_func(x.n, y.n)
    return FiniteField(val, x.p)


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


# @experimental.dispatch_for_api(tf.tensordot)
# Actually it doesn't make sense to change the api here
# unless a version able to handle also other types is generated
def ff_tensordot(a, b, axes):
    """Implementation of tensordot for operations involving finite fields"""
    lhs = len(a.shape)
    rhs = len(b.shape)
    i1 = string.ascii_lowercase[:lhs]
    i2 = string.ascii_lowercase[-rhs:]
    if axes > 0:
        if lhs < rhs:
            raise NotImplementedError("Not implemented yet for y.shape > x.shape")
        i2 = i1[::-1][:axes] + i2[axes:]
    elif axes == 0:
        pass
    else:
        raise NotImplementedError(f"Not implemented yet for {axes=}")

    ret = "".join(sorted(set(i1).symmetric_difference(i2)))
    eq_final = f"{i1},{i2}->{ret}"
    print(eq_final)
    return oinsum(eq_final, a, b)


if __name__ == "__main__":
    print("Testing TF finite fields and comparing them to pyadic")

    p = 2**31 - 19

    r1 = (p * np.random.rand(2, 3, 4)).astype(int)
    r2 = (p * np.random.rand(2, 3, 4)).astype(int)

    f1 = FiniteField(r1, p)
    f2 = FiniteField(r2, p)

    def artor(r):
        clist = []
        if hasattr(r, "shape") and r.shape:
            for i in r:
                clist.append(artor(i))
        else:
            clist.append(ModP(r, p))
        return clist

    def compare(tfff, pyff):
        if not isinstance(tfff, FiniteField) and isinstance(pyff, FiniteField):
            return compare(pyff, tfff)

        a = tfff.values.numpy()
        b = np.vectorize(lambda x: x.n)(pyff)
        if np.allclose(a, b):
            return "equal!"
        return "ERROR!"

    p1 = np.array(artor(r1)).squeeze()
    p2 = np.array(artor(r2)).squeeze()

    a = p // 2

    t = compare(-f1, -p1)
    print(f"Comparing operation: -, {t}")

    t = compare(f1 + a, p1 + a)
    print(f"Comparing operation: FF + a, {t}")

    t = compare(f1 + f2, p1 + p2)
    print(f"Comparing operation: FF + FF, {t}")

    t = compare(f1 - f2, p1 - p2)
    print(f"Comparing operation: FF - FF, {t}")

    t = compare(f1 * a, p1 * a)
    print(f"Comparing operation: FF * a, {t}")

    t = compare(f1 * f2, p1 * p2)
    print(f"Comparing operation: FF * FF, {t}")

    t = compare(f1 / a, p1 / a)
    print(f"Comparing operation: FF / a, {t}")

    t = compare(f1 / f2, p1 / p2)
    print(f"Comparing operation: FF / FF, {t}")

    t = compare(a / f2, a / p2)
    print(f"Comparing operation: FF / FF, {t}")

    t = compare(f1**5, p1**5)
    print(f"Comparing operation: FF**n, {t}")

    # Test the dot product
    N = 10
    dd1 = (p * np.random.rand(N, 2, 3)).astype(int)
    dd2 = (p * np.random.rand(N, 3, 4)).astype(int)
    fd1 = FiniteField(dd1, p)
    fd2 = FiniteField(dd2, p)

    from operations import ff_dot_product, ff_tensor_product

    dot_str = "rij,rjk->rik"
    res_object = oinsum(dot_str, fd1, fd2)
    fres_cuda = ff_dot_product(fd1, fd2)
    t = compare(fres_cuda, res_object)
    print(f"Comparing dot product {dot_str}: {t}")

    ein_str = "rij,rlk->riklj"
    res_object = oinsum(ein_str, fd1, fd2)
    fres_cuda = ff_tensor_product(ein_str, fd1, fd2)
    t = compare(fres_cuda, res_object)
    print(f"Comparing tensor product {ein_str}: {t}")
