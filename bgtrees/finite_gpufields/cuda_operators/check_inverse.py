#!/usr/bin/env python3

import time

import numpy as np
import tensorflow as tf

PMOD = 2**31 - 19

inverse_module = tf.load_op_library("./inverse.so")


@tf.function
def wrapper_inverse(x):
    return inverse_module.inverse(x)


def check_galois(x, pmod=PMOD, nmax=1000):
    try:
        import galois
    except ModuleNotFoundError:
        return None, None

    FF = galois.GF(pmod)
    fx = FF(x[:nmax])
    return np.array(FF(1) / fx)


if __name__ == "__main__":
    N = int(1e7)
    maxval = PMOD
    x = np.random.randint(maxval, size=6 * N).reshape(N, 2, 3) + 1
    tfx = tf.constant(x, dtype=tf.int64)

    start = time.time()
    res = wrapper_inverse(tfx) % PMOD
    end = time.time()
    op_time = end - start
    print(f"OP, took {op_time:.4}s")

    test_n = min(1000, N)
    st = time.time()
    res_gal = check_galois(x, nmax=test_n)
    ft = time.time()
    time_gal = ft - st
    print(f"The Galois loop, took {time_gal:.4}s ({test_n/N*100}% of the events)")
    print(np.allclose(res_gal, res[:test_n]))
