.. adam documentation master file, created by
   sphinx-quickstart on Fri Jun 28 14:10:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


adam - Rigid-Body Dynamics for Floating-Base Robots
========================================================

**Automatic Differentiation for rigid-body-dynamics AlgorithMs**

**adam** computes rigid-body dynamics for floating-base robots. Choose from multiple backends depending on your use case:

- 🔥 `JAX <https://github.com/google/jax>`_ – JIT compilation, batched operations, and differentiation with XLA
- 🎯 `CasADi <https://web.casadi.org/>`_ – Symbolic computation for optimization and control
- 🔦 `PyTorch <https://github.com/pytorch/pytorch>`_ – GPU acceleration, batched operations and differentiation
- 🐍 `NumPy <https://numpy.org/>`_ – Simple numerical evaluation

All backends share the same API and produce numerically consistent results, letting you pick the tool that fits your workflow.

**Model Loading**
  - URDF files – standard robot description format (see :doc:`quickstart/index`)
  - MuJoCo models – direct integration with ``MjModel`` objects (see :doc:`guides/mujoco`)

Core Features
-------------

**Kinematics & Geometry**
  - Forward kinematics for any frame
  - Jacobians for any frame
  - Jacobian time derivatives

**Dynamics**
  - Mass matrix computed via Composite Rigid Body Algorithm (CRBA)
  - Bias forces (Coriolis and centrifugal forces + gravity term) computed via Recursive Newton-Euler Algorithm (RNEA)
  - Articulated Body Algorithm (ABA)

**Centroidal Dynamics**
  - Centroidal momentum matrix via the Composite Rigid Body Algorithm
  - Center of mass position and Jacobian

**Automatic Differentiation**
  - Gradients with JAX and PyTorch
  - Symbolic computation with CasADi

**Advanced Features**
  - Parametric models for shape/inertia optimization
  - Inverse kinematics (CasADi)
  - MuJoCo integration
  - Batch processing (PyTorch and JAX)

Philosophy
----------

Built on **Roy Featherstone's Rigid Body Dynamics Algorithms**, adam provides a composable interface across multiple backends. Consistency is guaranteed through extensive testing against `iDynTree <https://github.com/robotology/idyntree>`_.

Resources
---------

- **Examples**: See the `examples folder <https://github.com/ami-iit/adam/tree/main/examples>`_ for notebooks and scripts
- **Tests**: The `tests folder <https://github.com/ami-iit/adam/tree/main/tests>`_ contains comprehensive usage patterns

License
-------

BSD 3-Clause License – `view license <https://choosealicense.com/licenses/bsd-3-clause/>`_



.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart/index

.. toctree::
   :maxdepth: 2
   :caption: Guides:

   guides/concepts
   guides/backend_selection
   guides/mujoco
   guides/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules/index
