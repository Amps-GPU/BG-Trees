from .metric_and_verticies import η, V3g, V4g

import numpy


# @functools.lru_cache
# @gpu_function
def J_μ(lmoms, lpols, put_propagator=True, depth=0, verbose=False, einsum=numpy.einsum):
    """Recursive vectorized current builder. End of recursion is polarization tensors."""
    if verbose:
        print(f"Calling currents: {depth} - {lmoms.shape}")
    assert lmoms.shape[:2] == lpols.shape[:2]
    replicas, multiplicity, D = lmoms.shape
    Ds = lpols.shape[-1]
    assert D == Ds
    if multiplicity == 1:
        return numpy.einsum("mn,rn->rm", η(D), lpols[:, 0])
    else:
        if put_propagator:
            tot_moms = numpy.einsum("rid->rd", lmoms)
            propagators = 1 / numpy.einsum("rm,mn,rn->r", tot_moms, η(D), tot_moms)
        Jrμ = (sum([einsum("rmno,rn,ro->rm",
                           V3g(numpy.einsum("rim->rm", lmoms[:, :i, :]),
                               numpy.einsum("rim->rm", lmoms[:, i:, :])),
                           J_μ(lmoms[:, :i], lpols[:, :i], depth=depth + 1, verbose=verbose),
                           J_μ(lmoms[:, i:], lpols[:, i:], depth=depth + 1, verbose=verbose)
                           ) for i in range(1, multiplicity)]) +
               sum([einsum("mnop,rn,ro,rp->rm",
                           V4g(D),
                           J_μ(lmoms[:, :i], lpols[:, :i], depth=depth + 1, verbose=verbose),
                           J_μ(lmoms[:, i:j], lpols[:, i:j], depth=depth + 1, verbose=verbose),
                           J_μ(lmoms[:, j:], lpols[:, j:], depth=depth + 1, verbose=verbose))
                    for i in range(1, multiplicity - 1) for j in range(i + 1, multiplicity)])
               )
        Jr_μ = einsum("mn,rn->rm", η(D), Jrμ)
        return einsum("r,rm->rm", propagators, Jr_μ) if put_propagator else Jr_μ


def forest(lmoms, lpols, verbose=False, einsum=numpy.einsum):
    return einsum("rm,rm->r", lpols[:, 0],
                  J_μ(lmoms[:, 1:], lpols[:, 1:], put_propagator=False, verbose=verbose, ))
