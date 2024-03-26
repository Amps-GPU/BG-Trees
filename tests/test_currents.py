from math import sqrt

import lips
from lips import Particles
import numpy
import pytest
from syngular import Field

from bgtrees.currents import J_μ, another_j
from bgtrees.finite_gpufields.finite_fields_tf import FiniteField
from bgtrees.finite_gpufields.operations import ff_dot_product
from bgtrees.metric_and_verticies import η
from bgtrees.settings import settings
from bgtrees.states import εm, εp

lips.spinor_convention = "asymmetric"
chosenP = 2**31 - 19
NTEST = 25
settings.run_tf_eagerly()


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


@pytest.mark.parametrize("field", [Field("finite field", chosenP, 1), Field("mpc", 0, 16)])
def test_ward_identity(field, n_test=NTEST):
    helconf = "ppmmmm"

    lmoms, lpols, _ = _generate_input(field, helconf, n_test)

    D = lmoms.shape[-1]

    # momentum is conserved
    assert numpy.all(numpy.vectorize(abs)(numpy.einsum("rim->rm", lmoms)) <= sqrt(field.tollerance))
    # polarization . momentum is zero
    assert numpy.all(numpy.vectorize(abs)(numpy.einsum("rim,mn,rin->ri", lmoms, η(D), lpols)) <= sqrt(field.tollerance))
    # polarization . current is zero
    assert numpy.all(
        numpy.vectorize(abs)(
            numpy.einsum("rm,rm->r", lmoms[:, 0], J_μ(lmoms[:, 1:], lpols[:, 1:], put_propagator=False, verbose=True))
        )
        <= sqrt(field.tollerance)
    )


def _run_test_MHV_amplitude_in_D_eq_4(field, lmoms, lpols, target_result, verbose=False):
    D = lmoms.shape[-1]

    # momentum is conserved
    assert numpy.all(numpy.vectorize(abs)(numpy.einsum("rim->rm", lmoms)) <= sqrt(field.tollerance))
    # polarization . momentum is zero
    assert numpy.all(numpy.vectorize(abs)(numpy.einsum("rim,mn,rin->ri", lmoms, η(D), lpols)) <= sqrt(field.tollerance))
    # polarization . current is zero
    final_result = numpy.einsum(
        "rm,rm->r", lpols[:, 0], J_μ(lmoms[:, 1:], lpols[:, 1:], put_propagator=False, verbose=verbose)
    )
    diff = numpy.vectorize(abs)(final_result - target_result)
    assert numpy.all(diff <= sqrt(field.tollerance))


@pytest.mark.parametrize("field", [Field("finite field", chosenP, 1), Field("mpc", 0, 16)])
def test_MHV_amplitude_in_D_eq_4(field, verbose=False, n_test=NTEST):
    helconf = "ppmmmm"

    lmoms, lpols, lPs = _generate_input(field, helconf, n_test)
    target_result = numpy.array([oPs("(32[12]^4)/([12][23][34][45][56][61])") for oPs in lPs])
    _run_test_MHV_amplitude_in_D_eq_4(field, lmoms, lpols, target_result, verbose=verbose)


def _run_test_mhv_amplitude_in_gpu(lmoms, lpols, target_result, verbose=False):
    # This function can only be run with the GPU enabled
    prev_setting = settings.use_gpu
    settings.use_gpu = True

    # Now make it into finite Finite Fields containers
    def _make_to_container(array_of_arrays, p):
        """Make any array of arrays or list of list into a finite field container"""
        return FiniteField(array_of_arrays.astype(int), p)

    ff_moms = _make_to_container(lmoms, chosenP)
    ff_pols = _make_to_container(lpols, chosenP)

    ret = another_j(ff_moms[:, 1:], ff_pols[:, 1:], put_propagator=False, verbose=verbose)
    actual_ret = ff_dot_product(ff_pols[:, 0], ret)

    numpy.testing.assert_allclose(actual_ret.values.numpy(), target_result.astype(int))
    settings.use_gpu = prev_setting


def test_MHV_amplitude_in_GPU(verbose=False, nt=NTEST):
    """Same test as above using the Finite Field container"""
    chosen_field = Field("finite field", 2**31 - 19, 1)
    helconf = "ppmmmm"
    lmoms, lpols, lPs = _generate_input(chosen_field, helconf, nt)
    target_result = numpy.array([oPs("(32[12]^4)/([12][23][34][45][56][61])") for oPs in lPs])
    _run_test_mhv_amplitude_in_gpu(lmoms, lpols, target_result, verbose=verbose)


# test_MHV_amplitude_in_GPU()

# chosen_field = Field("finite field", 2**31 - 19, 1)
# helconf = "ppmmmm"
#
# timings = []
#
# for exp_n in range(1, 6):
#     test_n = numpy.power(10, exp_n)
#     lmoms, lpols, lPs = _generate_input(chosen_field, helconf, test_n)
#     target_result = numpy.array([oPs("(32[12]^4)/([12][23][34][45][56][61])") for oPs in lPs])
#     # Compile
#     test_MHV_amplitude_in_GPU(False, nt=2)
#     from time import time
#
#     #
#     print("Begin calculation...")
#     start = time()
#     _run_test_mhv_amplitude_in_gpu(lmoms, lpols, target_result)
#     end = time()
#     print(f"With container took: {end-start}")
#     timings.append((test_n, end - start))
# #
# #     start = time()
# #     _run_test_MHV_amplitude_in_D_eq_4(lmoms, lpols, target_result)
# #     end = time()
# #     print(f"With default took: {end-start}")
#
# for n, ela in timings:
#     print(f"{n}   {ela:.5}")1G
