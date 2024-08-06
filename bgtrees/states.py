import numpy

from .metric_and_verticies import Gamma, η


def μ2(momD, d=None):
    """
    The 4D mass of a massless D momentum (d=None).
    For μ^2_d = k^2_4 + ... + k^2_d-1
    """
    if d is None:
        d = momD.shape[0]
    return - momD[4:d] @ η[4:d, 4:d] @ momD[4:d]


def momflat(momD, momχ):
    """Massless (`flat`) projection of 4D massive momentum onto a reference direction momχ."""
    return momD[:4] - μ2(momD) * momχ / (2 * momχ @ η[:4, :4] @ momD[:4])


def εm(oParticles, index, einsum=numpy.einsum):
    """Negative polarization vector: ε⁻^μ = <i|γ^μ|q]/(√2[iq])"""
    # Warning: not vectorized over replicas.
    # <i|γ^μ
    εm = einsum("xa,mab->mb", numpy.block([oParticles[index].r_sp_u, numpy.zeros((1, 2))]), Gamma)
    # <i|γ^μ|q]
    εm = einsum("mb,bx->m", εm, numpy.block([[numpy.zeros((2, 1))], [oParticles.oRefVec.l_sp_u]]))
    # ε⁻^μ = <i|γ^μ|q]/(√2[iq]) - removed √2 from denominator
    εm = εm / (einsum("xa,ay->", oParticles[index].l_sp_d, oParticles.oRefVec.l_sp_u))
    return εm


ε1 = εm


def εp(oParticles, index, einsum=numpy.einsum):
    """Positive polarization vector: ε⁺^μ = <q|γ^μ|i]/(√2<qi>)"""
    # Warning: not vectorized over replicas.
    # <q|γ^μ
    εp = einsum("xa,mab->mb", numpy.block([oParticles.oRefVec.r_sp_u, numpy.zeros((1, 2))]), Gamma)
    # <q|γ^μ|i]
    εp = einsum("mb,bx->m", εp, numpy.block([[numpy.zeros((2, 1))], [oParticles[index].l_sp_u]]))
    # ε⁺^μ = <q|γ^μ|i]/(<qi>) - removed √2 from denominator
    εp = εp / (einsum("xa,ay->", oParticles.oRefVec.r_sp_u, oParticles[index].r_sp_d))
    return εp


ε2 = εp
