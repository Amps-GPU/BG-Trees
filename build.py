# Build script ran by poetry to compile the cuda kernels before installation
from pathlib import Path
import subprocess as sp

CUDA_FOLDER = Path("bgtrees") / "finite_gpufields" / "cuda_operators"


def cuda_compile():
    """Try to compile the cuda kernels before installation
    Note, these commands will all fail silently
    """
    # TODO: make them fail loudly (check=True)
    # TODO: make them fallback to CPU-only compilation instead of loudly failing
    # clean the directory
    sp.run(["make", "clean"], cwd=CUDA_FOLDER, shell=True)
    # compile dot_product
    sp.run(["make", "dot_product.so"], cwd=CUDA_FOLDER, shell=True)


if __name__ == "__main__":
    cuda_compile()
