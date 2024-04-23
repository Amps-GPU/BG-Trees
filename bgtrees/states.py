import numpy

from .metric_and_verticies import Gamma


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
