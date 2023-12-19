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
        2 * numpy.einsum("lnmo->lmno", numpy.tensordot(MinkowskiMetric(D), MinkowskiMetric(D), axes=0)) -
        numpy.einsum("lmno->lmno", numpy.tensordot(MinkowskiMetric(D), MinkowskiMetric(D), axes=0)) -
        numpy.einsum("lomn->lmno", numpy.tensordot(MinkowskiMetric(D), MinkowskiMetric(D), axes=0))
    )


@gpu_function
def V3g(p1, p2, tensordot=tensorflow.tensordot, einsum=tensorflow.einsum, ):
    """3-gluon vertex, upper indices μνρ, D-dimensional"""
    D = p1.shape[0]
    return (
        einsum("mnl->lmn", tensordot(MinkowskiMetric(D), (p1 - p2), axes=0)) +
        2 * einsum("nlm->lmn", tensordot(MinkowskiMetric(D), p2, axes=0)) -
        2 * einsum("lmn->lmn", tensordot(MinkowskiMetric(D), p1, axes=0))
    )
