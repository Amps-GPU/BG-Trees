import functools

import numpy
import tensorflow

from .settings import settings

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def tsr(x):
    return tensorflow.constant(x)


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
    """Passes additional arguments to run on gpu if needed."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if settings.use_gpu:
            kwargs['tensordot'] = tensorflow.tensordot
            kwargs['einsum'] = tensorflow.einsum
        else:
            kwargs['tensordot'] = numpy.tensordot
            kwargs['einsum'] = numpy.einsum
        return func(*args, **kwargs)

    return wrapper
