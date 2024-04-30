import functools

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
