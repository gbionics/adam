Jax inteface
================

This module provides the Jax implementation of the Rigid Body Dynamics algorithms.

.. tip::

   We suggest to ``jax.jit`` the functions as it will make them run faster!

.. tip::

   The functions in this module natively support batching via broadcasting, or can be explicitly vectorized with ``jax.vmap`` for advanced patterns.

.. note::

   The first time you run a ``jax.jit``-ed function, it will take a bit longer to execute as they are being compiled by Jax.

.. note::

   If the GPU support for ``JAX`` is needed, follow the instructions in the `Jax documentation <https://jax.readthedocs.io/en/latest/installation.html#conda-community-supported>`_.


----------------

.. currentmodule:: adam.jax

.. automodule:: adam.jax.computations
   :members:
   :undoc-members:
   :show-inheritance:
