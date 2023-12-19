import tensorflow
import functools
import numpy

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
        else:
            kwargs['tensordot'] = numpy.tensordot
        return func(*args, **kwargs)
    return wrapper
