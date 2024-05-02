import lips
from lips import Particles
import numpy as np
from syngular import Field

from bgtrees.states import εm, εp

lips.spinor_convention = "asymmetric"


def generate_input(chosen_field, helconf, n=25):
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
            lm.append(np.block([oParticle.four_mom, np.array([0, 0, 0, 0])]))
        lmoms.append(np.array(lm))

        # Prepare the polarization array
        lp = []
        for index, helconf_index in enumerate(helconf):
            if helconf_index == "p":
                tmp = [εp(particle_list, index + 1), np.array([0, 0, 0, 0])]
            elif helconf_index == "m":
                tmp = [εm(particle_list, index + 1), np.array([0, 0, 0, 0])]
            elif helconf_index == "x":
                tmp = [np.array([0, 0, 0, 0, 0, 0, 1, 0])]
            else:
                tmp = None
            lp.append(np.block(tmp))

        lpols.append(np.array(lp))
        lPs.append(particle_list)

    lmoms = np.array(lmoms)
    lpols = np.array(lpols)

    return lmoms, lpols, lPs


chosen_p = 2**31 - 19
chosen_field = Field("finite field", chosen_p, 1)
helconf = "ppmmmm"
n_generate = int(1e5)

# Generate the arrays
lmoms, lpols, lPs = generate_input(chosen_field, helconf, n_generate)
# Make them into integers
lmoms_int = lmoms.astype(int)
lpols_int = lpols.astype(int)
# Create target results for the string below
target_results = np.array([oPs("(32[12]^4)/([12][23][34][45][56][61])") for oPs in lPs]).astype(int)

# Now save all information
np.savez("benchmark_data.npz", lmoms=lmoms_int, lpols=lpols_int, target=target_results, p=chosen_p)
