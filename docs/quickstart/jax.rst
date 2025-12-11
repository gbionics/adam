JAX Backend
===========

The JAX backend excels at automatic differentiation and batching. Use it for gradient-based optimization and research prototypes.

Key Features
------------

- **JIT Compilation** – First call is slow, subsequent calls are fast
- **Automatic Differentiation** – Compute gradients, Jacobians, Hessians
- **Native Batching** – Process batches of configurations
- **GPU Support** – Runs on GPU with proper JAX installation

Basic Usage
-----------

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    import adam
    from adam.jax import KinDynComputations
    from jax import jit, grad
    import icub_models
    
    # Load model
    model_path = icub_models.get_model_file("iCubGazeboV2_5")
    joints_name_list = [
        'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
        'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
        'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
        'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
        'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
    ]
    
    kinDyn = KinDynComputations(model_path, joints_name_list)
    kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    
    # Define state
    w_H_b = jnp.eye(4)
    joints = jnp.ones(len(joints_name_list)) * 0.1
    
    # Compute (slow, no compilation)
    M = kinDyn.mass_matrix(w_H_b, joints)
    print(f"Mass matrix shape: {M.shape}")


JIT Compilation
---------------

Wrap your functions with ``@jit`` for speed:

.. code-block:: python

    from jax import jit
    
    @jit
    def compute(w_H_b, joints):
        M = kinDyn.mass_matrix(w_H_b, joints)
        J = kinDyn.jacobian('l_sole', w_H_b, joints)
        return M, J
    
    # First call: slow (compilation)
    M, J = compute(w_H_b, joints)
        
    # Subsequent calls: fast (cached)
    M, J = compute(w_H_b, joints)

.. warning::

    Frame names must remain as strings (not traced). Wrap them in a closure:

    .. code-block:: python

        # ✅ Correct
        def make_fk_fn(frame_name):
            @jit
            def fk(w_H_b, joints):
                return kinDyn.forward_kinematics(frame_name, w_H_b, joints)
            return fk
        
        fk_l_sole = make_fk_fn('l_sole')

Automatic Differentiation
--------------------------

Compute gradients easily:

.. code-block:: python

    from jax import grad
    
    # Gradient of mass matrix trace w.r.t. joint positions
    def mass_matrix_trace(w_H_b, joints):
        M = kinDyn.mass_matrix(w_H_b, joints)
        return jnp.trace(M)
    
    grad_fn = grad(mass_matrix_trace, argnums=1)  # Gradient w.r.t. joints
    grad_joints = grad_fn(w_H_b, joints)
    print(f"Gradient shape: {grad_joints.shape}")

**Higher-order derivatives:**

.. code-block:: python

    from jax import grad, hessian
    
    hess_fn = hessian(mass_matrix_trace, argnums=1)
    hess_joints = hess_fn(w_H_b, joints)
    print(f"Hessian shape: {hess_joints.shape}")


Native Batching
----------------

JAX automatically broadcasts batched operations:

.. code-block:: python

    # Batch size 1024
    batch_size = 1024
    w_H_b_batch = jnp.tile(jnp.eye(4), (batch_size, 1, 1))  # Shape: (1024, 4, 4)
    joints_batch = jnp.tile(joints, (batch_size, 1))  # Shape: (1024, n_dof)
    
    # Just pass batched tensors - JAX handles batching automatically
    M_batch = kinDyn.mass_matrix(w_H_b_batch, joints_batch)  # Shape: (1024, 6+n, 6+n)
    J_batch = kinDyn.jacobian('l_sole', w_H_b_batch, joints_batch)  # Shape: (1024, 6, 6+n)
    print(f"Mass matrix shape: {M_batch.shape}")

**Combine JIT with Native Batching:**

.. code-block:: python

    # JIT for maximum speed
    @jit
    def jit_batched_compute(w_H_b_batch, joints_batch):
        M = kinDyn.mass_matrix(w_H_b_batch, joints_batch)
        J = kinDyn.jacobian('l_sole', w_H_b_batch, joints_batch)
        return M, J
    
    # First call compiles, subsequent calls are fast
    M_batch, J_batch = jit_batched_compute(w_H_b_batch, joints_batch)


Optimization Example
--------------------

Use gradients for optimization:

.. code-block:: python

    import optax  # pip install optax
    from jax import grad, jit
    
    def objective(joints):
        """Minimize mass matrix trace"""
        M = kinDyn.mass_matrix(w_H_b, joints)
        return jnp.trace(M)
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(joints)
    
    # JIT the step
    @jit
    def step(joints, opt_state):
        loss, grads = jax.value_and_grad(objective)(joints)
        updates, opt_state = optimizer.update(grads, opt_state)
        joints = optax.apply_updates(joints, updates)
        return joints, opt_state, loss
    
    # Optimize
    for i in range(100):
        joints, opt_state, loss = step(joints, opt_state)
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss:.6f}")


Tips and Tricks
---------------

**Enable 64-bit precision** (recommended for robustness):

.. code-block:: python

    from jax import config
    config.update("jax_enable_x64", True)

**Disable JIT temporarily for debugging:**

.. code-block:: python

    from jax import config
    config.update("jax_disable_jit", True)

Loading from MuJoCo
-------------------

Load models from MuJoCo and leverage JAX's JIT and autodiff:

.. code-block:: python

    import mujoco
    from robot_descriptions.loaders.mujoco import load_robot_description
    from adam.jax import KinDynComputations
    import jax.numpy as jnp
    from jax import jit, grad
    
    # Load MuJoCo model
    mj_model = load_robot_description("g1_mj_description")
    
    # Create KinDynComputations from MuJoCo model
    kinDyn = KinDynComputations.from_mujoco_model(mj_model)
    kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    
    # Use with JIT and autodiff
    @jit
    def compute_mass_trace(w_H_b, joints):
        M = kinDyn.mass_matrix(w_H_b, joints)
        return jnp.trace(M)
    
    w_H_b = jnp.eye(4)
    joints = jnp.zeros(kinDyn.NDoF)
    
    trace_val = compute_mass_trace(w_H_b, joints)
    grad_fn = grad(compute_mass_trace, argnums=1)
    grad_val = grad_fn(w_H_b, joints)

See :doc:`../guides/mujoco` for more details on MuJoCo integration.

When to Use JAX
---------------

✅ **Good for:**
- Gradient-based optimization
- Computing Jacobians and Hessians
- Processing batches with native batching
- GPU acceleration

❌ **Not ideal for:**
- One-off computations (NumPy is faster)
- Symbolic manipulation (use CasADi)

**See also:** :doc:`../guides/backend_selection`
