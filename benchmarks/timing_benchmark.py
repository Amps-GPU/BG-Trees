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
    parser.add_argument("-a", "--average", help="Run <average> times and take the average", type=int, default=1)
    parser.add_argument("-o", "--output", help="Output file for results as <n> <events>", type=str)
    parser.add_argument("--profile", action="store_true")
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

    print("Starting the run...")
    batch_size = 5 * int(10**5)

    res_per_n = {}

    # Run a bit just to activate the JIT compilation
    if not settings.executing_eagerly():
        _ = another_j(ff_moms[:10, 1:], ff_pols[:10, 1:], put_propagator=False, verbose=False)

    if args.profile:
        import tensorflow as tf

        logdir_path = "profiling_here"
        options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=3, python_tracer_level=1, device_tracer_level=1
        )
        tf.profiler.experimental.start(logdir_path, options=options)

    for nev in list_of_n:

        timing_raw = 0

        for nrun in range(args.average):
            start = time()
            total_final_results = []

            for from_ev in range(0, nev, batch_size):
                to_ev = np.minimum(from_ev + batch_size, nev)
                momenta = ff_moms[from_ev:to_ev]
                polariz = ff_pols[from_ev:to_ev]

                ret = another_j(momenta[:, 1:], polariz[:, 1:], put_propagator=False, verbose=False)
                final_result = ff_dot_product(polariz[:, 0], ret)
                total_final_results.append(final_result)

            end = time()
            if nrun == 0 and args.average > 1:
                # Don't count the first run
                continue
            timing_raw += end - start

        timing = timing_raw / np.maximum(args.average - 1, 1)
        res_per_n[nev] = timing

        print(f"n = {nev} took {timing:.5}s")

    #         finres = np.concatenate([i.values.numpy() for i in total_final_results])
    #         np.testing.assert_allclose(finres, load_info["target"][:nev])

    if args.profile:
        tf.profiler.experimental.stop()

    if args.output is not None:
        res_as_str = "\n".join([f"{i} {j}" for i, j in res_per_n.items()])
        res_as_str += "\n"
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(res_as_str)
        print(f"Results written to {args.output}")
