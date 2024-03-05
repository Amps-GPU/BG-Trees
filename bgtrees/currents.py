from .metric_and_verticies import η, V3g, V4g

import functools
import numpy


# @gpu_function
def J_μ(lmoms, lpols, put_propagator=True, depth=0, verbose=False, einsum=numpy.einsum):
    """Recursive vectorized current builder. End of recursion is polarization tensors."""

    assert lmoms.shape[:2] == lpols.shape[:2]
    replicas, multiplicity, D = lmoms.shape
    Ds = lpols.shape[-1]
    assert D == Ds

    @functools.lru_cache
    def _J_μ(a, b, put_propagator=True, ):
        if verbose:
            print(f"Calling currents: a = {a}, b = {b}.")
        multiplicity = b - a
        if multiplicity == 1:
            return einsum("mn,rn->rm", η(D), lpols[:, a])
        else:
            if put_propagator:
                tot_moms = einsum("rid->rd", lmoms[:, a:b])
                propagators = 1 / einsum("rm,mn,rn->r", tot_moms, η(D), tot_moms)
            Jrμ = (sum([einsum("rmno,rn,ro->rm",
                               V3g(einsum("rim->rm", lmoms[:, a:a + i, :]),
                                   einsum("rim->rm", lmoms[:, a + i:b, :])),
                               _J_μ(a, a + i, ),
                               _J_μ(a + i, b, )
                               ) for i in range(1, multiplicity)]) +
                   sum([einsum("mnop,rn,ro,rp->rm",
                               V4g(D),
                               _J_μ(a, a + i, ),
                               _J_μ(a + i, a + j, ),
                               _J_μ(a + j, b, )
                               ) for i in range(1, multiplicity - 1) for j in range(i + 1, multiplicity)])
                   )
            Jr_μ = einsum("mn,rn->rm", η(D), Jrμ)
            return einsum("r,rm->rm", propagators, Jr_μ) if put_propagator else Jr_μ

    return _J_μ(0, multiplicity, put_propagator=put_propagator, )


def forest(lmoms, lpols, verbose=False, einsum=numpy.einsum):
    return einsum("rm,rm->r", lpols[:, 0],
                  J_μ(lmoms[:, 1:], lpols[:, 1:], put_propagator=False, verbose=verbose, ))
