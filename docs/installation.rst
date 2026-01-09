Installation
============

adam requires Python 3.10 or later. Install it using your preferred package manager.

Quickstart
----------

Install with your preferred backend(s):

.. code-block:: bash

    # Single backend
    pip install adam-robotics[jax]        # JAX backend
    pip install adam-robotics[casadi]     # CasADi backend
    pip install adam-robotics[pytorch]    # PyTorch backend
    
    # All backends
    pip install adam-robotics[all]

Or with conda:

.. code-block:: bash

    conda install -c conda-forge adam-robotics-casadi
    conda install -c conda-forge adam-robotics-jax
    conda install -c conda-forge adam-robotics-pytorch

Which Backend?
--------------

See the :doc:`guides/backend_selection` guide to choose the right backend for your use case.

Development Install
--------------------

Clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/ami-iit/adam.git
    cd adam
    pip install -e .[all,test]  # Install all backends + test dependencies

This enables development mode with all backends and testing tools.

GPU Support (JAX & PyTorch)
---------------------------

**JAX with GPU**

Follow the `JAX GPU installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_ for GPU support.

**PyTorch with GPU**

PyTorch is automatically GPU-compatible. Check also `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.
