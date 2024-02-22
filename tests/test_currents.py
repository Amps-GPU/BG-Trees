import numpy
import lips

from lips import Particles
from syngular import Field

from bgtrees.metric_and_verticies import η
from bgtrees.states import εp, εm
from bgtrees.currents import J_μ


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_ward_identity():
    chosen_field = Field("finite field", 2 ** 31 - 19, 1)
    # chosen_field = Field("mpc", 0, 32)
    lips.spinor_convention = "asymmetric"

    helconf = "ppmmmm"
    lPs = [Particles(len(helconf), field=chosen_field, seed=i) for i in range(25)]
    for oPs in lPs:
        oPs.helconf = helconf

    lmoms = numpy.array([numpy.array([numpy.block([oParticle.four_mom, numpy.array([0, 0, 0, 0])])
                                      for oParticle in oParticles]) for oParticles in lPs])
    print(lmoms.shape)  # e.g.: (25 replicas, 6 gluon momenta, 8 dimensions)

    lpols = numpy.array([numpy.array([numpy.block([εp(oParticles, index + 1), numpy.array([0, 0, 0, 0])])
                                      if oParticles.helconf[index] == "p" else
                                      numpy.block([εm(oParticles, index + 1), numpy.array([0, 0, 0, 0])])
                                      if oParticles.helconf[index] == "m" else
                                      numpy.block([numpy.array([0, 0, 0, 0, 0, 0, 1, 0])])
                                      if oParticles.helconf[index] == "x" else
                                      None for index in range(len(helconf))]) for oParticles in lPs])
    print(lpols.shape)

    D = lmoms.shape[-1]

    # momentum is conserved
    assert numpy.all(numpy.einsum("rim->rm", lmoms) == 0)
    # polarization . momentum is zero
    assert numpy.all(numpy.einsum("rim,mn,rin->ri", lmoms, η(D), lpols) == 0)
    # polarization . current is zero
    assert numpy.all(numpy.einsum(
        "rm,rm->r", lmoms[:, 0],
        J_μ(lmoms[:, 1:], lpols[:, 1:], put_propagator=False, verbose=True, )
    ) == 0)
