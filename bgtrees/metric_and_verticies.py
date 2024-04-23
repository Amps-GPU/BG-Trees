from lips.tools import Pauli, Pauli_bar
import numpy

from .tools import gpu_constant, gpu_function

Gamma = γμ = numpy.block([[numpy.zeros((4, 2, 2)), Pauli_bar], [Pauli, numpy.zeros((4, 2, 2))]])
Gamma5 = γ5 = numpy.block(
    [[numpy.identity(2), numpy.zeros((2, 2))], [numpy.zeros((2, 2)), -numpy.identity(2)]]
)


@gpu_constant
def MinkowskiMetric(D):
    """D-dimensional Minkowski metric in the mostly negative convention."""
    return numpy.diag([1] + [-1] * (D - 1))


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
def V3g(lp1, lp2, einsum=numpy.einsum):
    """3-gluon vertex, upper indices μνρ, D-dimensional"""
    D = lp1.shape[1]
    mm = η(D)
    return (
        einsum("mn,rl->rlmn", mm, (lp1 - lp2))
        + 2 * einsum("nl,rm->rlmn", mm, lp2)
        - 2 * einsum("lm,rn->rlmn", mm, lp1)
    )
