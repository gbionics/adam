PyTorch Backend
===============

The PyTorch backend offers GPU acceleration and automatic differentiation. Use it for learning-based control, large-scale optimization, and ML integration.

Key Features
------------

- **GPU Acceleration** – Compute on GPUs
- **Automatic Differentiation** – Compute gradients naturally
- **Native Batching** – Efficiently process multiple configurations
- **Device Flexibility** – Easy CPU/GPU device management
- **ML Integration** – Works seamlessly with PyTorch models

Basic Usage
-----------

.. code-block:: python

    import numpy as np
    import torch
    import adam
    from adam.pytorch import KinDynComputations
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
    w_H_b = torch.tensor(np.eye(4), dtype=torch.float64)
    joints = torch.ones(len(joints_name_list), dtype=torch.float64) * 0.1
    
    # Compute
    M = kinDyn.mass_matrix(w_H_b, joints)
    print(f"Mass matrix shape: {M.shape}")


GPU Support
-----------

Move computations to GPU easily:

.. code-block:: python

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move tensors to GPU
    w_H_b = torch.tensor(np.eye(4), dtype=torch.float64, device=device)
    joints = torch.ones(len(joints_name_list), dtype=torch.float64, device=device) * 0.1
    
    # Compute on GPU
    M = kinDyn.mass_matrix(w_H_b, joints)
    print(f"Computation device: {M.device}")


Automatic Differentiation
--------------------------

Compute gradients naturally:

.. code-block:: python

    # Enable gradient computation
    w_H_b.requires_grad = True
    joints.requires_grad = True
    
    # Forward pass
    M = kinDyn.mass_matrix(w_H_b, joints)
    loss = M.trace()
    
    # Backward pass
    loss.backward()
    
    print(f"Gradient w.r.t. joints:\n{joints.grad}")


Common Operations
-----------------

**Jacobian:**

.. code-block:: python

    J = kinDyn.jacobian('l_sole', w_H_b, joints)
    print(f"Jacobian shape: {J.shape}")

**Forward Kinematics:**

.. code-block:: python

    w_H_f = kinDyn.forward_kinematics('l_sole', w_H_b, joints)
    print(f"End-effector transform:\n{w_H_f}")

**Center of Mass:**

.. code-block:: python

    com = kinDyn.CoM_position(w_H_b, joints)
    J_com = kinDyn.CoM_jacobian(w_H_b, joints)


Learning-Based Control
----------------------

Integration with neural networks:

.. code-block:: python

    import torch.nn as nn
    
    class ControlPolicy(nn.Module):
        def __init__(self, n_dof):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_dof, 64),
                nn.ReLU(),
                nn.Linear(64, n_dof)
            )
        
        def forward(self, joints):
            return self.net(joints)
    
    # Use with adam
    policy = ControlPolicy(len(joints_name_list))
    action = policy(joints)
    
    # Compute acc
    acc = kinDyn.aba(
        w_H_b, joints, base_vel, 
        joints_vel, action
    )

Batch Processing
----------------

Use ``pytorch.KinDynComputations`` to process multiple configurations. 

.. note:: There is a class ``pytorch.KinDynComputationsBatch`` that has the functionality of ``pytorch.KinDynComputations``. 
    It exists to avoid API changes in existing code. New users should prefer ``pytorch.KinDynComputations`` for both single and batched computations.

.. code-block:: python

    from adam.pytorch import KinDynComputations
    
    kinDyn = KinDynComputations(model_path, joints_name_list)
    kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    
    # Create batch (batch dimension is FIRST)
    batch_size = 1024
    w_H_b_batch = torch.tile(torch.eye(4, dtype=torch.float64), (batch_size, 1, 1))
    joints_batch = torch.randn(batch_size, len(joints_name_list), dtype=torch.float64)
    
    # Compute for all configurations at once
    M_batch = kinDyn.mass_matrix(w_H_b_batch, joints_batch)
    print(f"Output shape: {M_batch.shape}")  # (batch_size, 6+n_dof, 6+n_dof)


**Batch Computations:**

.. code-block:: python

    J_batch = kinDyn.jacobian('l_sole', w_H_b_batch, joints_batch)
    # Shape: (batch_size, 6, 6+n)

    w_H_f_batch = kinDyn.forward_kinematics('l_sole', w_H_b_batch, joints_batch)
    # Shape: (batch_size, 4, 4)

    Ag_batch = kinDyn.centroidal_momentum_matrix(w_H_b_batch, joints_batch)
    # Shape: (batch_size, 6, 6+n)

**Batch Gradients:**

.. code-block:: python

    w_H_b_batch.requires_grad = True
    joints_batch.requires_grad = True
    
    # Forward pass
    M_batch = kinDyn.mass_matrix(w_H_b_batch, joints_batch)
    loss = M_batch.mean()
    
    # Backward pass through entire batch
    loss.backward()
    
    print(f"Gradient shape: {joints_batch.grad.shape}")


Batch Use Cases
---------------

**Trajectory Evaluation**

.. code-block:: python

    # Evaluate entire trajectory
    trajectory_configs = torch.randn(trajectory_length, n_dof)
    
    # Compute Jacobians for all configurations
    J_trajectory = kinDyn.jacobian('l_sole', w_H_b_batch, trajectory_configs)

**Simulation Rollouts**

.. code-block:: python

    # Simulate multiple trajectories in parallel
    batch_size = 256
    trajectory_steps = 100
    
    for step in range(trajectory_steps):
        # Compute dynamics for all parallel simulations
        M_batch = kinDyn.mass_matrix(w_H_b_batch, joints_batch)
        # Update joints...


Batch Tips and Tricks
---------------------

**Shape Requirements**

Always remember: batch dimension is **first**

.. code-block:: python

    # ✅ Correct
    w_H_b_batch.shape  # (batch, 4, 4)
    joints_batch.shape  # (batch, n_dof)
    
    # ❌ Wrong
    w_H_b_batch.shape  # (4, 4, batch)
    joints_batch.shape  # (n_dof, batch)

**Consistent dtypes**

.. code-block:: python

    # All tensors should have same dtype
    assert w_H_b_batch.dtype == joints_batch.dtype
    assert w_H_b_batch.dtype == torch.float64  # Recommended

**Memory usage for large batches**

.. code-block:: python

    # For very large batches, use smaller mini-batches
    mini_batch_size = 256  # Adjust based on GPU memory
    
    # Process in chunks
    for i in range(0, total_configs, mini_batch_size):
        batch = configs[i:i+mini_batch_size]
        result = kinDyn_batch.mass_matrix(w_H_b_batch[:len(batch)], batch)


Performance Tips
----------------

**Use float64 for stability (recommended):**

.. code-block:: python

    torch.set_default_dtype(torch.float64)

**Pin memory for faster CPU-GPU transfer:**

.. code-block:: python

    w_H_b = torch.tensor(np.eye(4), dtype=torch.float64, pin_memory=True)

**Use in-place operations carefully:**

.. code-block:: python

    # Be careful with requires_grad when using in-place ops
    joints = joints.clone().detach().requires_grad_(True)


Loading from MuJoCo
-------------------

Load models from MuJoCo and use with PyTorch's autodiff:

.. code-block:: python

    import mujoco
    from robot_descriptions.loaders.mujoco import load_robot_description
    from adam.pytorch import KinDynComputations
    import torch
    
    # Load MuJoCo model
    mj_model = load_robot_description("g1_mj_description")
    
    kinDyn = KinDynComputations.from_mujoco_model(mj_model)
    kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    
    # Use with PyTorch
    w_H_b = torch.eye(4, dtype=torch.float64)
    joints = torch.zeros(kinDyn.NDoF, dtype=torch.float64, requires_grad=True)
    
    M = kinDyn.mass_matrix(w_H_b, joints)
    loss = M.trace()
    loss.backward()  # Gradients in joints.grad

See :doc:`../guides/mujoco` for more on MuJoCo integration.

When to Use PyTorch
-------------------

✅ **Good for:**
- GPU computation
- Learning-based control
- ML pipeline integration
- Large-scale batch
- Gradient-based optimization

❌ **Not ideal for:**
- Simple numeric evaluation (NumPy is easier)
- Symbolic computation (use CasADi instead)

**See also:** :doc:`../guides/backend_selection`
