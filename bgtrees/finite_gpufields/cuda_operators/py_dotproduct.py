#!/usr/bin/env python3
"""
Script to test and benchmark the dot_product kernels.

The C++ version is competitive with NumPy (probably doing the same thing under the hood)
when using the @ operator. When using einsum, the C++ version is about 3 to 4 times faster.

The CUDA version is faster than NumPy (scales with the number of elements):
- 1e5: 2 times faster
- 1e6: 10 times faster
- 1e7: 60 times faster

Note that it is an unfair comparison (for us) since in NumPy the % operation is done only at the end.
The same factor of 3-4 can be multiplied to these numbers when using einsum with NumPy.

There exists some overhead in our operations that takes as long as computing ~1e4 events.
It is unclear how this would scale in a situation in which there are _many_ operations.
If the overhead is not per-operation (i.e., once the events are in-device they remain there),
this might not be a problem.
"""
import time

import numpy as np
import tensorflow as tf

PMOD = 2**31 - 19

dot_product_module = tf.load_op_library("./dot_product.so")


@tf.function
def wrapper_dot_product(x, y):
    ret = dot_product_module.dot_product(x, y)
    return ret


@tf.function
def wrapper_dot_product_single_batch(x, y):
    ret = dot_product_module.dot_product_single_batch(x, y)
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
    if len(y.shape) == 3:
        fy = FF(y[:nmax])
    else:
        fy = [FF(y)] * nmax

    st = time.time()
    res = [f1 @ f2 for f1, f2 in zip(fx, fy)]
    ft = time.time()

    return np.array(res), ft - st


if __name__ == "__main__":
    N = 2  # int(1e7)
    maxval = 10  # PMOD
    x = np.random.randint(maxval, size=6 * N).reshape(N, 2, 3)
    y = np.random.randint(maxval, size=12 * N).reshape(N, 3, 4)
    ysb = np.random.randint(maxval, size=12).reshape(3, 4)

    #     print(x)
    #     print(y)

    tfx = tf.constant(x)
    tfy = tf.constant(y)
    tfy_sb = tf.constant(ysb)

    # Compile the operation beforehand
    c1 = wrapper_dot_product(tfx[0:2], tfy[0:2])
    c2 = wrapper_dot_product_single_batch(tfx[0:2], tfy_sb)
    #     print(x @ ysb)
    #     print(c2)
    #     import ipdb; ipdb.set_trace()

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
    test_n = min(1000, N)
    res_gal, time_gal = check_galois(x, y, nmax=test_n)

    test_n = min(1000, N)
    res_sb = wrapper_dot_product_single_batch(tfx, tfy_sb)
    res_gal_sb, _ = check_galois(x, ysb, nmax=test_n)

    if res_gal is not None:
        print(" > Testing with galois the double batched results (rx . ry = rz)")
        print(f"The Galois loop, took {time_gal:.4}s ({test_n/N*100}% of the events)")
        print(f"Does it agree? {np.allclose(res.numpy()[:test_n], np.array(res_gal))}")

        print(" > Testing with galois the single batched numbers (rx . y = rz)")
        print(f"Does it agree? {np.allclose(res_sb.numpy()[:test_n], np.array(res_gal_sb))}")
