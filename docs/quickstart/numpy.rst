NumPy Backend
==============

The NumPy backend is the simplest choice for direct computation. Use it for model validation, quick experiments, and debugging.

Basic Usage
-----------

.. code-block:: python

    import numpy as np
    import adam
    from adam.numpy import KinDynComputations
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
    
    # Create KinDynComputations instance
    kinDyn = KinDynComputations(model_path, joints_name_list)
    kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    
    # Define state
    w_H_b = np.eye(4)  # Base at origin
    joints = np.ones(len(joints_name_list)) * 0.1  # All joints at 0.1 rad
    
    # Compute dynamics
    M = kinDyn.mass_matrix(w_H_b, joints)
    print(f"Mass matrix shape: {M.shape}")
    print(f"Mass matrix:\n{M}")
    
    # Forward kinematics
    w_H_f = kinDyn.forward_kinematics('l_sole', w_H_b, joints)
    print(f"Foot transform:\n{w_H_f}")
    
    # Jacobian
    J = kinDyn.jacobian('l_sole', w_H_b, joints)
    print(f"Jacobian shape: {J.shape}")


Common Operations
-----------------

**Centroidal Momentum Matrix**

.. code-block:: python

    Ag = kinDyn.centroidal_momentum_matrix(w_H_b, joints)
    print(f"Centroidal momentum matrix shape: {Ag.shape}")

**Center of Mass**

.. code-block:: python

    com = kinDyn.CoM_position(w_H_b, joints)
    print(f"CoM position: {com}")
    
    J_com = kinDyn.CoM_jacobian(w_H_b, joints)
    print(f"CoM Jacobian shape: {J_com.shape}")

**Bias Forces (Coriolis + Gravity)**

.. code-block:: python

    base_vel = np.zeros(6)
    joints_vel = np.zeros(len(joints_name_list))
    
    h = kinDyn.bias_force(w_H_b, joints, base_vel, joints_vel)
    print(f"Bias force: {h}")

Loading from MuJoCo
-------------------

You can also load models directly from MuJoCo:

.. code-block:: python

    import mujoco
    from robot_descriptions.loaders.mujoco import load_robot_description
    from adam.numpy import KinDynComputations
    
    # Load MuJoCo model
    mj_model = load_robot_description("g1_mj_description")
    
    # Create KinDynComputations from MuJoCo model
    kinDyn = KinDynComputations.from_mujoco_model(mj_model)
    kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    
    # Use as normal
    w_H_b = np.eye(4)
    joints = np.zeros(kinDyn.NDoF)
    M = kinDyn.mass_matrix(w_H_b, joints)

See :doc:`../guides/mujoco` for detailed MuJoCo integration guide.

When to Use NumPy
-----------------

✅ **Good for:**
- Debugging and understanding results
- Code that doesn't need gradients or batching
- Educational examples

❌ **Not ideal for:**
- Computing gradients (use JAX or PyTorch)
- Symbolic formulations (use CasADi)
- Large-scale optimization

**Next steps:** If you need gradients and batching, see :doc:`jax` and :doc:`pytorch`. For optimization, see :doc:`casadi`.
