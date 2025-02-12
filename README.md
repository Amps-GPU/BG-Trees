# Berends-Giele Recursion for Trees on the GPU

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14859785.svg)](https://doi.org/10.5281/zenodo.14859785)
[![arxiv](https://img.shields.io/badge/arXiv-hep--ph%2F2502.07060-%23B31B1B.svg)](https://arxiv.org/abs/2502.07060)


[![CI Lint](https://github.com/Amps-GPU/BG-Trees/actions/workflows/ci_lint.yml/badge.svg)](https://github.com/Amps-GPU/BG-Trees/actions/workflows/ci_lint.yml)
[![CI Test](https://github.com/Amps-GPU/BG-Trees/actions/workflows/ci_test_gpu.yml/badge.svg)](https://github.com/Amps-GPU/BG-Trees/actions/workflows/ci_test_gpu.yml)
[![Coverage](https://img.shields.io/badge/Coverage-74%25-yellow?labelColor=2a2f35)](https://github.com/Amps-GPU/BG-Trees/actions)
[![Docs](https://github.com/Amps-GPU/BG-Trees/actions/workflows/cd_docs.yml/badge.svg?label=Docs)](https://amps-gpu.github.io/BG-Trees/)

## Installation

The code in this repository can be installed with

```python
pip install .
```

or, in development mode:

```python
pip install -e .[tests]
```

the `tests` extra will install several utilities useful for testing and generation of phase space number.

## Example
A minimal example with the computation of currents for different numerical types, dimensions and multiplicities are given in the `examples` folder in the form of a python script and a Jupyter Notebook.

## Installation of Cuda kernels
The script `build.py`, which should be run during the installation with pip, tries to automatically compile the cuda kernels.
However it is not guaranteed that it will work (in particular `nvcc` needs to be installed in the system).

If it doesn't work, it is possible to install (and hopefully debug) manually the compilation of the kernels which are found in `bgtrees/finite_gpufields/cuda_operators/`.

### Note for developers
1. The dependency `tensorflow[and-cuda]` is a build dependency, but it takes a while to install, so at the moment is commented out.
2. The `build.py` script is there mainly for future reference, it works, but during development is orders of magnitude faster going into the cuda folder `bgtrees/finite_gpufields/cuda_operators/` and running manually `make`
3. The `build.py` script is written at the moment to fail silently so that it doesn't interfere with a python-only installation

In addition it is recommended to install tools like `black` or `isort` to standardize the code.

## Finite Fields `finite_gpufields`

The `FiniteField` type defined in `bgtrees/finite_gpufields` powers the hardware agnostic part of this package.
The `FiniteField` type is a wrapper of GPU-aware arrays and the prime number `p` defining the cardinality of the field.
Operations involving `FiniteField` will try to use the GPU whenever possible, with a fallback to CPU when the operation is not possible or not implemented.

## Citation policy

If this code has been useful in your research please cite the paper 

```
@article{Cruz-Martinez:2025kwa,
    author = "Cruz-Martinez, Juan M. and De Laurentis, Giuseppe and Pellen, Mathieu",
    title = "{Accelerating Berends-Giele recursion for gluons in arbitrary dimensions over finite fields}",
    eprint = "2502.07060",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "CERN-TH-2025-017, FR-PHENO-2025-001",
    month = "2",
    year = "2025"
}
```

as well as the [Zenodo entry](10.5281/zenodo.14859784)
