# Berends-Giele Recursion for Trees on the GPU

[![Continuous Integration Status](https://github.com/Amps-GPU/BG-Trees/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/Amps-GPU/BG-Trees/actions)
[![Coverage](https://img.shields.io/badge/Coverage-76%25-yellow?labelColor=2a2f35)](https://github.com/Amps-GPU/BG-Trees/actions)

## Installation

The code in this repository can be installed with

```python
pip install .
```

or, in development mode:

```python
pip install -e .[tests]
```

the `tests` extra will install several utilities useful for testing.
In addition it is recommended to install tools like `black` or `isort` to standardize the code.

## Installation of Cuda kernels
The script `build.py`, which should be run during the installation with pip, tries to automatically compile the cuda kernels.
However it is not guaranteed that it will work (in particular `nvcc` needs to be installed in the system).

If it doesn't work, it is possible to install (and hopefully debug) the compilation of the kernels which are found in `bgtrees/finite_gpufields/cuda_operators/`.

### Note for developers
1. The dependency `tensorflow[and-cuda]` is a build dependency, but it takes a while to install, so at the moment is commented out.
2. The `build.py` script is there mainly for future reference, it works, but during development is orders of magnitude faster going into the cuda folder `bgtrees/finite_gpufields/cuda_operators/` and running manually `make`
3. The `build.py` script is written at the moment to fail silently so that it doesn't interfere with a python-only installation


## Finite Fields `finite_gpufields`

The `FiniteField` type defined in `bgtrees/finite_gpufields` powers the hardware agnostic part of this package.
The `FiniteField` type is a wrapper of GPU-aware arrays and the prime number `p` defining the cardinality of the field.
Operations involving `FiniteField` will try to use the GPU whenever possible, with a fallback to CPU when the operation is not possible or not implemented.
