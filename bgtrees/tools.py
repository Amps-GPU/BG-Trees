import functools
import operator

import numpy
import tensorflow

from .finite_gpufields.finite_fields_tf import FiniteField
from .finite_gpufields.operations import ff_einsum_generic
from .settings import settings

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def tsr(x):
    return tensorflow.constant(x)


# TODO:
# Enable automatic setting of gpu / cpu constant only when everything is working
# Actually, probably it should not be necessary at all, the container should be the one dispatching
# either CPU or GPU objects


def gpu_constant(func):
    """Turns constants into gpu constants if needed."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if settings.use_gpu:
            return tsr(res)
        else:
            return res

    return wrapper


def gpu_function(func):
    """Passes additional arguments to run on gpu if needed.
    Dispatchs a different function depending on whether the arguments are finite fields or not
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], FiniteField):
            # If we are dealign with FiniteField type, use the ff_einsum
            kwargs["einsum"] = ff_einsum_generic
        else:
            if settings.use_gpu:
                kwargs["einsum"] = tensorflow.einsum
            else:
                # kwargs['tensordot'] = numpy.tensordot
                kwargs["einsum"] = numpy.einsum
                # kwargs['block'] = ...
        return func(*args, **kwargs)

    return wrapper


def _oinsum(eq, *arrays):
    """A ``einsum`` implementation for ``numpy`` object arrays."""
    lhs, output = eq.split("->")
    inputs = lhs.split(",")

    sizes = {}
    for term, array in zip(inputs, arrays):
        for k, d in zip(term, array.shape):
            sizes[k] = d

    out_size = tuple(sizes[k] for k in output)
    out = numpy.empty(out_size, dtype=object)

    inner = [k for k in sizes if k not in output]
    inner_size = [sizes[k] for k in inner]

    for coo_o in numpy.ndindex(*out_size):
        coord = dict(zip(output, coo_o))

        def gen_inner_sum():
            for coo_i in numpy.ndindex(*inner_size):
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

        vunff = numpy.vectorize(unff)

        new_out = vunff(out)
        out = FiniteField(new_out, p)

    return out
