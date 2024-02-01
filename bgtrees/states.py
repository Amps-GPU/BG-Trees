from .tools import gpu_function

import tensorflow
import numpy


@gpu_function
def εp(lParticles, index, einsum=tensorflow.einsum, ):
    """Positive polarization vectors: ε⁺^μ = <q|γ^μ|i]/(√2<qi>)"""
    # <q|γ^μ
    εp = einsum("xa,mab->mb", numpy.block([lParticles.oRefVec.r_sp_u, numpy.zeros((1, 2))]), Gamma)
    # <q|γ^μ|i]
    εp = einsum("mb,ibx->im", εp, numpy.block([[numpy.zeros((len(lParticles), 2, 1))], [lParticles.ll_sp_us[:, index - 1, :, :]]]))
    # ε⁺^μ = <q|γ^μ|i]/(√2<qi>) - removed √2 from denominator
    εp = εp / (einsum("xa,iay->i", lParticles.oRefVec.r_sp_u, lParticles.lr_sp_ds[:, index - 1, :, :]))[:, None]
    return εp


@gpu_function
def εm(lParticles, index, einsum=tensorflow.einsum, ):
    """Negative polarization vectors: ε⁻^μ = <i|γ^μ|q]/(√2[iq])"""
    # <i|γ^μ|
    εm = einsum("ixa,mab->imb", numpy.block([lParticles.lr_sp_us[:, index - 1, :, :], numpy.zeros((len(lParticles), 1, 2))]), Gamma)
    # <i|γ^μ|q]
    εm = einsum("imb,bx->im", εm, numpy.block([[numpy.zeros((2, 1))], [lParticles.oRefVec.l_sp_u]]))
    # ε⁻^μ = <i|γ^μ|q]/(√2[iq]) - removed √2 from denominator
    εm = εm / (einsum("ixa,ay->i", lParticles.ll_sp_ds[:, index - 1, :, :], lParticles.oRefVec.l_sp_u))[:, None]
    return εm
