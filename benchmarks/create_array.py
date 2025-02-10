from argparse import ArgumentParser

import lips
from lips import Particles
import numpy as np
from syngular import Field

from bgtrees.states import ε2 as εm, ε1 as εp

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--legs", type=int, default=6)
    parser.add_argument("--events", type=int, default=int(1e5))
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    chosen_p = 2**31 - 19
    chosen_field = Field("finite field", chosen_p, 1)

    helconf = "pp" + "m" * (args.legs - 2)
    nparticles = len(helconf)

    # Generate the arrays
    lmoms, lpols, lPs = generate_input(chosen_field, helconf, args.events)
    # Make them into integers
    lmoms_int = lmoms.astype(int)
    lpols_int = lpols.astype(int)

    # Create target results
    parke_taylor_den = ""
    for i in range(nparticles):
        n1 = i + 1
        n2 = i + 2
        if n2 > nparticles:
            n2 = 1
        parke_taylor_den += f"[{n1}{n2}]"

    fact = -((-2) ** (nparticles - 1))
    parke_taylor_str = f"({fact}[12]^4)/({parke_taylor_den})"
    target_results = np.array([oPs(parke_taylor_str) for oPs in lPs]).astype(int)

    # Now save all information
    if args.output is None:
        file_out = f"benchmark_data_n{nparticles}.npz"
    else:
        file_out = f"{args.output}"
        if not file_out.endswith(".npz"):
            file_out += ".npz"
    np.savez(file_out, lmoms=lmoms_int, lpols=lpols_int, target=target_results, p=chosen_p)
    print(f"Results saved to {file_out}")
