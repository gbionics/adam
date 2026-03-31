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

    # Optional model interfaces
    pip install adam-robotics[mujoco]     # MuJoCo model loading
    pip install adam-robotics[usd]        # OpenUSD model loading/conversion
    pip install adam-robotics[visualization]  # viser-based visualization

    # All backends
    pip install adam-robotics[all]        # jax + casadi + pytorch

Or with conda:

.. code-block:: bash

    conda install -c conda-forge adam-robotics-casadi
    conda install -c conda-forge adam-robotics-jax
    conda install -c conda-forge adam-robotics-pytorch
    conda install -c conda-forge mujoco
    conda install -c conda-forge openusd

Which Backend?
--------------

See the :doc:`guides/backend_selection` guide to choose the right backend for your use case.

Development Install
--------------------

Clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/ami-iit/adam.git
    cd adam
    pip install -e .[all,mujoco,usd,test]

This enables development mode with all backends, model interfaces, and testing tools.

GPU Support (JAX & PyTorch)
---------------------------

**JAX with GPU**

Follow the `JAX GPU installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_ for GPU support.

**PyTorch with GPU**

PyTorch is automatically GPU-compatible. Check also `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.
