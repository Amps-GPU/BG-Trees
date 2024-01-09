#!/usr/bin/env python3
"""
    Script to test and benchmark the dot_product kernels

    The C++ version is competitive with numpy (probably doing the same thing under the hood)
    when using @, when using einsum, the C++ version is about 3 or 4 times faster.

    The cuda version is faster than numpy (scales with number of elements)
        1e5: 2 times
        1e6: 10 times
        1e7: 60 times
    note that it is an unfair comparison (for us) since in numpy the % operation is done only at the end
    The same factor of 3-4 can be multiplied to these numbers when using einsum with numpy

    There exist some overhead in our operations that takes as long as computing ~1e4 events.
    It is unclear how this would scale in a situation in which there are _many_ operations,
    if the overhead is not per-operation (i.e., once the events are in-device they remain there)
    this might not be a problem
"""
import time

import numpy as np
import tensorflow as tf

PMOD = 2**31 - 19

dot_product_module = tf.load_op_library('./dot_product.so')


@tf.function
def wrapper_dot_product(x, y):
    ret = dot_product_module.dot_product(x, y)
    return ret


def fully_python_dot_product(x, y):
    ret = np.einsum("bij,bjk->bik", x, y)
    return ret


def check_galois(x, y, pmod=PMOD, nmax=1000):
    try:
        import galois
    except ModuleNotFoundError:
        return None, None

    FF = galois.GF(pmod)
    fx = FF(x[:nmax])
    fy = FF(y[:nmax])

    st = time.time()
    res = [f1 @ f2 for f1, f2 in zip(fx, fy)]
    ft = time.time()

    return np.array(res), ft - st


if __name__ == "__main__":
    N = int(1e7)
    maxval = PMOD
    x = np.random.randint(maxval, size=6 * N).reshape(N, 2, 3)
    y = np.random.randint(maxval, size=12 * N).reshape(N, 3, 4)

    #     print(x)
    #     print(y)

    tfx = tf.constant(x)
    tfy = tf.constant(y)

    # Compile the operation beforehand
    _ = wrapper_dot_product(tfx[0:2], tfy[0:2])

    start = time.time()
    res = wrapper_dot_product(tfx, tfy)
    end = time.time()
    op_time = end - start

    print(f"OP, took {op_time:.4}s")

    start = time.time()
    # Do a naive pmod to the numpy result
    res_py = fully_python_dot_product(x, y) % PMOD
    end = time.time()
    py_time = end - start
    print(f"py, took {py_time:.4}s")

    print(f"Do they agree? {np.allclose(res.numpy(), res_py)}")
    print(f"The operator was {py_time/op_time:.1} times faster")

    # Galois cannot do batches (or I don't know how?, so test only the first N)
    test_n = 1000
    res_gal, time_gal = check_galois(x, y, nmax=test_n)

    if res_gal is not None:
        print(f"The Galois loop, took {time_gal:.4}s ({test_n/N*100}% of the events)")
        print(f"Does it agree? {np.allclose(res.numpy()[:test_n], np.array(res_gal))}")
