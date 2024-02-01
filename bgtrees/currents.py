from .tools import gpu_function
from .metric_and_verticies import η, V3g, V4g
from .states import εp, εm

import numpy
import functools
import tensorflow


@functools.lru_cache
@gpu_function
def J_μ(lParticles, put_propagator=True, depth=0, verbose=False, einsum=tensorflow.einsum):
    """Recursive vectorized current builder. End of recursion is polarization tensors."""
    if verbose:
        print(f"Calling currents: {depth} - {lParticles.shape} - {lParticles.helconf}")
    if lParticles.shape[1] == 1:
        if lParticles.helconf == "p":
            return einsum("mn,rn->rm", η, εp(lParticles, 1))
        elif lParticles.helconf == "m":
            return einsum("mn,rn->rm", η, εm(lParticles, 1))
        else:
            raise Exception
    else:
        if put_propagator:
            propagators = numpy.array([[1 / sum(oParticles).mass] for oParticles in lParticles])
        Jiμ = (sum([einsum("rmno,rn,ro->rm",
                                 V3g(numpy.sum(lParticles.lfour_moms[:, :i, :], axis=1),
                                     numpy.sum(lParticles.lfour_moms[:, i:, :], axis=1)),
                                 J_μ(lParticles[:, :i], depth=depth+1, verbose=verbose),
                                 J_μ(lParticles[:, i:], depth=depth+1, verbose=verbose)
                                ) for i in range(1, len(lParticles[0]))]) +
               sum([einsum("mnop,rn,ro,rp->rm", 
                                 V4g(), 
                                 J_μ(lParticles[:, :i], depth=depth+1, verbose=verbose),
                                 J_μ(lParticles[:, i:j], depth=depth+1, verbose=verbose),
                                 J_μ(lParticles[:, j:], depth=depth+1, verbose=verbose))
                    for i in range(1, len(lParticles[0]) - 1) for j in range(i + 1, len(lParticles[0]))])
              )
        Ji_μ = einsum("mn,rn->rm", η, Jiμ)
        return propagators * Ji_μ if put_propagator else Ji_μ
