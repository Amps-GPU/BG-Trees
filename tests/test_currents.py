import lips
from lips import Particles
import numpy
from syngular import Field

from bgtrees.currents import J_μ, another_j
from bgtrees.metric_and_verticies import η
from bgtrees.states import εm, εp

from bgtrees.finite_gpufields.finite_fields_tf import FiniteField
from bgtrees.finite_gpufields.operations import ff_dot_product

lips.spinor_convention = "asymmetric"
chosenP = 2**31 - 19
N_test = 2500

def _generate_input(chosen_field, helconf, n=25):
    """Generate the momentum and polarization arrays using lips
    Returns the momentum and polarization as numpy array and the
    list of lips particle_lists
    """
    lmoms = []
    lpols = []
    lPs = []

    for seed in range(n):
        particle_list = Particles(len(helconf), field=chosen_field, seed=seed)
        particle_list.helconf = helconf

        # Prepare hte momentum array
        lm = []
        for oParticle in particle_list:
            lm.append(numpy.block([oParticle.four_mom, numpy.array([0, 0, 0, 0])]))
        lmoms.append(numpy.array(lm))

        # Prepare the polarization array
        lp = []
        for index, helconf_index in enumerate(helconf):
            if helconf_index == "p":
                tmp = [εp(particle_list, index + 1), numpy.array([0, 0, 0, 0])]
            elif helconf_index == "m":
                tmp = [εm(particle_list, index + 1), numpy.array([0, 0, 0, 0])]
            elif helconf_index == "x":
                tmp = [numpy.array([0, 0, 0, 0, 0, 0, 1, 0])]
            else:
                tmp = None
            lp.append(numpy.block(tmp))

        lpols.append(numpy.array(lp))
        lPs.append(particle_list)

    lmoms = numpy.array(lmoms)
    lpols = numpy.array(lpols)

    return lmoms, lpols, lPs


def test_ward_identity():
    chosen_field = Field("finite field", chosenP, 1)
    # chosen_field = Field("mpc", 0, 32)
    helconf = "ppmmmm"

    lmoms, lpols, _ = _generate_input(chosen_field, helconf, N_test)

    D = lmoms.shape[-1]

    # momentum is conserved
    assert numpy.all(numpy.einsum("rim->rm", lmoms) == 0)
    # polarization . momentum is zero
    assert numpy.all(numpy.einsum("rim,mn,rin->ri", lmoms, η(D), lpols) == 0)
    # polarization . current is zero
    assert numpy.all(
        numpy.einsum(
            "rm,rm->r",
            lmoms[:, 0],
            J_μ(lmoms[:, 1:], lpols[:, 1:], put_propagator=False, verbose=True),
        )
        == 0
    )


def test_MHV_amplitude_in_D_eq_4():
    chosen_field = Field("finite field", chosenP, 1)
    helconf = "ppmmmm"

    lmoms, lpols, lPs = _generate_input(chosen_field, helconf, N_test)

    D = lmoms.shape[-1]

    # momentum is conserved
    assert numpy.all(numpy.einsum("rim->rm", lmoms) == 0)
    # polarization . momentum is zero
    assert numpy.all(numpy.einsum("rim,mn,rin->ri", lmoms, η(D), lpols) == 0)

    # polarization . current is zero
    assert numpy.all(
        numpy.einsum(
            "rm,rm->r",
            lpols[:, 0],
            J_μ(lmoms[:, 1:], lpols[:, 1:], put_propagator=False, verbose=True),
        )
        == numpy.array([oPs("(32[12]^4)/([12][23][34][45][56][61])") for oPs in lPs])
    )


def test_MHV_amplitude_in_GPU(verbose=False):
    """Same test as above using the Finite Field container"""
    chosen_field = Field("finite field", 2**31 - 19, 1)
    helconf = "ppmmmm"
    lmoms, lpols, lPs = _generate_input(chosen_field, helconf, N_test)

    actual_val = numpy.array([oPs("(32[12]^4)/([12][23][34][45][56][61])") for oPs in lPs])

    # Now make it into finite Finite Fields containers
    def _make_to_container(array_of_arrays, p):
        """Make any array of arrays or list of list into a finite field container"""
        return FiniteField(array_of_arrays.astype(int), p)

    ff_moms = _make_to_container(lmoms, chosenP)
    ff_pols = _make_to_container(lpols, chosenP)

    ret = another_j(ff_moms[:, 1:], ff_pols[:, 1:], put_propagator=False, verbose=verbose)
    actual_ret = ff_dot_product(ff_pols[:, 0], ret)

    numpy.testing.assert_allclose(actual_ret.values.numpy(), actual_val.astype(int))

# from time import time
# start = time()
# test_MHV_amplitude_in_GPU(True)
# end = time()
# print(f"With container took: {end-start}")
# 
# start = time()
# test_MHV_amplitude_in_D_eq_4()
# end = time()
# print(f"With default took: {end-start}")
