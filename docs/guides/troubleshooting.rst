Troubleshooting & FAQ
====================

Common Issues and Solutions
---------------------------

JAX: Slow First Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** The first call to a JAX function is very slow.

**Cause:** JAX needs to compile the function with XLA.

**Solution:** Use ``jit`` to compile once:

.. code-block:: python

    from jax import jit
    
    @jit
    def compute_mass_matrix(w_H_b, joints):
        return kinDyn.mass_matrix(w_H_b, joints)
    
    M = compute_mass_matrix(w_H_b, joints)  # Slow (compilation)
    M = compute_mass_matrix(w_H_b, joints)  # Fast (cached)

**See also:** :doc:`backend_selection`


JAX: Array Value in Traced Context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** ``ValueError: array value being traced cannot be used in branching``

**Cause:** You're using Python conditionals on JAX arrays inside jitted code.

**Solution:** Use JAX control flow:

.. code-block:: python

    # ❌ Wrong
    @jit
    def bad_function(x):
        if x > 0:  # Can't do this!
            return x
        return -x
    
    # ✅ Correct
    @jit
    def good_function(x):
        return jnp.where(x > 0, x, -x)


JAX: Loss of Precision
^^^^^^^^^^^^^^^^^^^^^^

**Problem:** Results don't match NumPy, differences larger than expected.

**Cause:** JAX defaults to 32-bit floats.

**Solution:** Enable 64-bit precision:

.. code-block:: python

    from jax import config
    config.update("jax_enable_x64", True)


PyTorch: Device or Dtype Mismatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** ``RuntimeError: Expected all tensors to be on the same device``

**Cause:** Mixing CPU and GPU tensors, or different dtypes.

**Solution:** Ensure consistent device and dtype:

.. code-block:: python

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    w_H_b = torch.tensor(w_H_b, device=device, dtype=torch.float64)
    joints = torch.tensor(joints, device=device, dtype=torch.float64)
    
    kinDyn = KinDynComputations(model_path, joints_list)
    M = kinDyn.mass_matrix(w_H_b, joints)


PyTorch Batch: Wrong Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** Batched operations fail with shape errors.

**Cause:** The unified ``KinDynComputations`` API expects batch dimension first for batched operations.

**Solution:** Shape inputs as ``(batch_size, ...)``:

.. code-block:: python

    batch_size = 32
    
    # ❌ Wrong: (4, 4, batch) or (batch_size, 4) and (4)
    w_H_b = np.random.randn(4, 4, batch_size)
    joints = np.random.randn(batch_size)
    
    # ✅ Correct: (batch, 4, 4) and (batch, n_dof)
    w_H_b = torch.tensor(np.tile(np.eye(4), (batch_size, 1, 1)), dtype=torch.float64)
    joints = torch.randn(batch_size, n_dof, dtype=torch.float64)
    
    M = kinDyn.mass_matrix(w_H_b, joints)  # Shape: (batch, 6+n, 6+n)


CasADi: Numeric vs Symbolic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** Unsure when to use ``_fun()`` methods vs direct methods.

**Solution:**

- Use direct methods for immediate numeric evaluation:
  
  .. code-block:: python
  
      M = kinDyn.mass_matrix(w_H_b, joints)
  
- Use ``_fun()`` for reusable compiled functions or symbolic computation:
  
  .. code-block:: python
  
      M_fun = kinDyn.mass_matrix_fun()
      M_numeric = M_fun(w_H_b_numeric, joints_numeric)
      M_symbolic = M_fun(w_H_b_symbolic, joints_symbolic)



Numerical Issues
----------------

Results Don't Match Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** adam results differ from physics simulator (Gazebo, MuJoCo, etc).

**Likely causes:**

1. **Different gravity direction** – Set explicitly:
   
   .. code-block:: python
   
       kinDyn.g = np.array([0, 0, -9.81, 0, 0, 0])

2. **Different velocity representation** – Ensure consistency:
   
   .. code-block:: python
   
       kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

3. **Joint ordering mismatch** – Verify ``joints_name_list`` order matches simulator

4. **Model differences** – Ensure same URDF or model definition


Accuracy Tolerance
^^^^^^^^^^^^^^^^^^

**Question:** How accurate are the computations?

**Answer:** adam is validated against `iDynTree <https://github.com/robotology/idyntree>`_ with tolerance ``1e-5`` to ``1e-4``.

For your use case:


Getting Help
------------

1. **Check the tests** – Browse `tests/ <https://github.com/ami-iit/adam/tree/main/tests>`_ for working examples
2. **Read some useful theory** – See idyntree `theory.md <https://github.com/robotology/idyntree/blob/main/doc/theory.md>`_
3. **Open an issue** – Report bugs on `GitHub <https://github.com/ami-iit/adam/issues>`_
4. **Review examples** – Look at `examples/ <https://github.com/ami-iit/adam/tree/main/examples>`_
