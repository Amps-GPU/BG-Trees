#!/usr/bin/env python
"""
    Timing benchmarks

    To turn off tensorflow logging use
    export TF_CPP_MIN_LOG_LEVEL="3"
"""
from argparse import ArgumentParser
from time import time

import numpy as np

from bgtrees.currents import another_j
from bgtrees.finite_gpufields.finite_fields_tf import FiniteField
from bgtrees.finite_gpufields.operations import ff_dot_product
from bgtrees.settings import settings

settings.use_gpu = True

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "data_file", help="A npz data file with the right information (use `create_array` to run this benchmark)"
    )
    parser.add_argument(
        "-n",
        "--n_events",
        help="How many events to run, by default it will run for [10, 100, 1000]",
        nargs="+",
        type=int,
        default=[10, 100, 1000],
    )
    args = parser.parse_args()

    load_info = np.load(args.data_file)

    p = int(load_info["p"])
    lmoms_int = load_info["lmoms"]
    lpols_int = load_info["lpols"]

    max_n = lmoms_int.shape[0]
    list_of_n = sorted(args.n_events)
    if list_of_n[-1] > max_n:
        raise ValueError(f"Only {max_n} events available, can't run more than that")

    # Make the arrays into FiniteField containers which are GPU-ready
    ff_moms = FiniteField(lmoms_int, p)
    ff_pols = FiniteField(lpols_int, p)

    D = ff_moms.shape[2]
    mul = ff_moms.shape[1]

    # Run a bit just to activate the JIT compilation
    _ = another_j(ff_moms[:2, 1:], ff_pols[:2, 1:], put_propagator=False, verbose=False)

    print("Starting the run...")
    batch_size = 20000

    for nev in list_of_n:
        start = time()

        for from_ev in range(0, nev, batch_size):
            to_ev = np.minimum(from_ev + batch_size, nev)
            momenta = ff_moms[from_ev:to_ev]
            polariz = ff_pols[from_ev:to_ev]

            ret = another_j(momenta, polariz, put_propagator=False, verbose=False)
            final_result = ff_dot_product(momenta[:, :0], ret)

        end = time()
        print(f"n = {nev} took {end-start:.5}s")
#
#     # np.testing.assert_allclose(final_result.values.numpy(), load_info["target_result"])
