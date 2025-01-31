from lips.tools import Pauli, Pauli_bar
import numpy
import tensorflow as tf

from .finite_gpufields.finite_fields_tf import FiniteField
from .settings import settings
from .tools import gpu_constant, gpu_function

Gamma = γμ = numpy.block([[numpy.zeros((4, 2, 2)), Pauli_bar], [Pauli, numpy.zeros((4, 2, 2))]])
Gamma5 = γ5 = numpy.block([[numpy.identity(2), numpy.zeros((2, 2))], [numpy.zeros((2, 2)), -numpy.identity(2)]])


@gpu_constant
def MinkowskiMetric(D):
    """D-dimensional Minkowski metric in the mostly negative convention."""
    return numpy.diag([1] + [-1] * (D - 1)).astype(settings.dtype)
    # The metric can be changed here to a different convention,
    # e.g. to match the alternating metric used in Caravel.
    # NB: be careful with the definition of polarization states.
    # return numpy.diag([1, -1] * (D // 2)).astype(settings.dtype)


η = MinkowskiMetric


@gpu_constant
def V4g(D):
    """4-gluon vertex, upper indices μνρσ"""
    return (
        2 * numpy.einsum("ln,mo->lmno", η(D), η(D))
        - numpy.einsum("lm,no->lmno", η(D), η(D))
        - numpy.einsum("lo,mn->lmno", η(D), η(D))
    )


@gpu_function
@tf.function(reduce_retracing=True)
def V3g(lp1, lp2, einsum=numpy.einsum):
    """3-gluon vertex, upper indices μνρ, D-dimensional"""
    D = lp1.shape[1]
    if D is None:
        D = settings.D
    mm = η(D)
    return (
        einsum("mn,rl->rlmn", mm, (lp1 - lp2)) + 2 * einsum("nl,rm->rlmn", mm, lp2) - 2 * einsum("lm,rn->rlmn", mm, lp1)
    )


@tf.function(reduce_retracing=True)
def new_V3g(lp1, lp2):
    """3-gluon vertex, upper indices μνρ, D-dimensional. Reduce tensor products."""
    D = lp1.shape[1]
    if D is None:
        D = settings.D

    mm = tf.linalg.diag_part(η(D))
    r1 = tf.tensordot(lp1.n, mm, 0)
    r2 = tf.tensordot(lp2.n, mm, 0)

    r1 = tf.linalg.diag(r1)
    r2 = tf.linalg.diag(r2)

    a1 = FiniteField(r1, lp1.p)
    a2 = FiniteField(r2, lp1.p)

    return (a1 - a2) + 2.0 * a2.transpose_ff((0, 3, 1, 2)) - 2.0 * a1.transpose_ff((0, 2, 3, 1))
