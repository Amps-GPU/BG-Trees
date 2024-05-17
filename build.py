# Build script ran by poetry to compile the cuda kernels before installation
from pathlib import Path
import shutil
import subprocess as sp

import tensorflow as tf

CUDA_FOLDER = Path("bgtrees") / "finite_gpufields" / "cuda_operators"


def fake_cuda(op_name):
    """Create an empty file to avoid the compilation of the cuda kernel"""
    tmp = CUDA_FOLDER / f"{op_name}.cuo"
    tmp.write_text("")


def operator_compilation():
    """Try to compile the cuda kernels before installation
    Note, these commands will all fail silently
    """
    # TODO: make them fail loudly (check=True)
    # TODO: make them fallback to CPU-only compilation instead of loudly failing
    kerdef = ""
    ops_to_compile = ["dot_product", "inverse"]

    # Check whether nvcc is available
    nvcc_available = shutil.which("nvcc") is not None
    tf_cuda = tf.test.is_built_with_cuda()
    # clean the directory
    sp.run(["make", "clean"], cwd=CUDA_FOLDER)

    if nvcc_available and tf_cuda:# and False:
        kerdef = "KERNEL_DEF='-D GOOGLE_CUDA=1'"
    else:
        kerdef = "KERNEL_DEF=''"
        for op_name in ops_to_compile:
            fake_cuda(op_name)

    for op_name in ops_to_compile:
        # sp.run(["make", "-n", f"{op_name}.so", kerdef], cwd=CUDA_FOLDER)
        sp.run(f"make {op_name}.so {kerdef}", cwd=CUDA_FOLDER, shell=True)


if __name__ == "__main__":
    operator_compilation()
