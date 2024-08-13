BG Trees on the GPU
========================================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Welcome to the documentation of BG Trees!

BG Trees is a flexible and modular library that makes it easier to compute scattering amplitudes using the Berends-Giele recursion algorithm.
As higher-order calculations get more complex and resource-intensive, new strategies need to be developed.
Being able to compute amplitudes with various types of arithmetic and in an arbitrary number of dimension can be proven crucial for loop calculations.

By tapping into the power of GPUs, BG Trees helps cut down on the normally high costs of these calculations, making it much more feasible to tackle advanced problems.


Installation
------------

The code, library and CUDA kernels can be installed with.

.. code::

  pip install git+https://github.com/Amps-GPU/BG-Trees

    
This command will automatically install the ``tensorflow[and-cuda]`` package
and it will attempt to compile the CUDA kernels if the ``nvcc`` compiler is available.


Quick start
-----------


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :caption: Modules Documentation
   :maxdepth: 2

   modules
