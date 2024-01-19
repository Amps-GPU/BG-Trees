import numpy
import tensorflow

from .tools import gpu_constant, gpu_function

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@gpu_constant
def MinkowskiMetric(D):
    """D-dimensional Minkowski metric in the mostly negative convention."""
    return numpy.diag([1] + [-1] * (D - 1))


@gpu_constant
def V4g(D):
    """4-gluon vertex, upper indices μνρσ"""
    return (
        2 * numpy.einsum("ln,mo->lmno", MinkowskiMetric(D), MinkowskiMetric(D))
        - numpy.einsum("lm,no->lmno", MinkowskiMetric(D), MinkowskiMetric(D))
        - numpy.einsum("lo,mn->lmno", MinkowskiMetric(D), MinkowskiMetric(D))
    )


@gpu_function
def V3g(lp1, lp2, tensordot=tensorflow.tensordot, einsum=tensorflow.einsum):
    """3-gluon vertex, upper indices μνρ, D-dimensional"""
    D = lp1.shape[1]
    return (
        einsum("mn,il->ilmn", MinkowskiMetric(D), (lp1 - lp2))
        + 2 * einsum("nl,im->ilmn", MinkowskiMetric(D), lp2)
        - 2 * einsum("lm,in->ilmn", MinkowskiMetric(D), lp1)
    )
