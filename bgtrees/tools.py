import functools

import numpy

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
    return func

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

    return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if settings.use_gpu:
            kwargs["einsum"] = ff_einsum_generic
        else:
            # kwargs['tensordot'] = numpy.tensordot
            kwargs["einsum"] = numpy.einsum
            # kwargs['block'] = ...
        return func(*args, **kwargs)

    return wrapper
