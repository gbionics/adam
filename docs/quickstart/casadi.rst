CasADi Backend
==============

CasADi excels at symbolic computation and optimization formulation. Use it for optimal control, trajectory optimization, and code generation.

Key Features
------------

- **Symbolic Computation** – Build optimization problems symbolically
- **Both Function Types** – Direct and ``_fun()`` variants
- **Code Generation** – Export to C/Python for embedded systems

Basic Usage
-----------

.. code-block:: python

    import numpy as np
    import casadi as cs
    import adam
    from adam.casadi import KinDynComputations
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
    w_H_b = np.eye(4)
    joints = np.ones(len(joints_name_list)) * 0.1
    
    # Numeric evaluation
    M = kinDyn.mass_matrix(w_H_b, joints)
    print(f"Mass matrix:\n{M}")


Function Variants
-----------------

CasADi provides two ways to call each function:

**Direct method (for immediate evaluation):**

.. code-block:: python

    M = kinDyn.mass_matrix(w_H_b, joints)
    J = kinDyn.jacobian('l_sole', w_H_b, joints)

**Function factory (for reusable compiled functions):**

.. code-block:: python

    M_fun = kinDyn.mass_matrix_fun()
    J_fun = kinDyn.jacobian_fun('l_sole')
    
    # Evaluate multiple times (efficient)
    M1 = M_fun(w_H_b, joints)
    M2 = M_fun(w_H_b, joints + 0.1)


Symbolic Computation
--------------------

Build optimization problems symbolically:

.. code-block:: python

    import casadi as cs
    
    # Create symbolic variables
    w_H_b_sym = cs.SX.sym('H', 4, 4)
    joints_sym = cs.SX.sym('q', len(joints_name_list))
    
    # Symbolic mass matrix
    M_fun = kinDyn.mass_matrix_fun()
    M_sym = M_fun(w_H_b_sym, joints_sym)
    
    # Build symbolic expressions
    M_det = cs.det(M_sym)
    M_trace = cs.trace(M_sym)
    
    print(f"Determinant (symbolic):\n{M_det}")


Optimization Problem
--------------------

Setup a trajectory optimization problem:

.. code-block:: python

    import casadi as cs
    
    # Variables
    q_sym = cs.SX.sym('q', len(joints_name_list))
    q_dot_sym = cs.SX.sym('q_dot', len(joints_name_list))
    
    # Objective: minimize energy
    M_fun = kinDyn.mass_matrix_fun()
    M = M_fun(np.eye(4), q_sym)
    
    kinetic_energy = 0.5 * cs.mtimes(q_dot_sym.T, cs.mtimes(M[:6, :6], q_dot_sym))
    
    # Create optimization problem
    opti = cs.Opti()
    opti.minimize(kinetic_energy)
    
    # Add constraints...
    # Solve...


Inverse Kinematics
------------------

adam provides an IK solver for CasADi:

.. code-block:: python

    from adam.casadi.inverse_kinematics import InverseKinematics, TargetType
    
    # Create IK solver
    ik = InverseKinematics(model_path, joints_name_list)
    
    # Add target
    ik.add_target('l_sole', target_type=TargetType.POSE, as_soft_constraint=True)
    
    # Set desired pose
    desired_pos = np.array([0.3, 0.2, 1.0])
    desired_rot = np.eye(3)
    ik.update_target('l_sole', (desired_pos, desired_rot))
    
    # Solve
    ik.solve()
    w_H_b_sol, q_sol = ik.get_solution()


Tips and Tricks
---------------

**Use DM for numeric results:**

.. code-block:: python

    M_fun = kinDyn.mass_matrix_fun()
    M_result = cs.DM(M_fun(w_H_b, joints))
    print(f"Numeric result: {float(M_result[0, 0])}")

**Mix symbolic and numeric:**

.. code-block:: python

    # Symbolic Jacobian
    J_sym = J_fun(w_H_b_sym, joints_sym)
    
    # Create callable function
    J_callable = cs.Function('J', [w_H_b_sym, joints_sym], [J_sym])
    
    # Evaluate numerically
    J_numeric = J_callable(w_H_b, joints)

**Generate C code:**

.. code-block:: python

    M_fun = kinDyn.mass_matrix_fun()
    M_fun.generate('mass_matrix')  # Generates mass_matrix.c


Loading from MuJoCo
-------------------

Load models from MuJoCo for symbolic optimization:

.. code-block:: python

    import mujoco
    from robot_descriptions.loaders.mujoco import load_robot_description
    from adam.casadi import KinDynComputations
    import casadi as cs
    
    # Load MuJoCo model
    mj_model = load_robot_description("g1_mj_description")
    
    # Create KinDynComputations from MuJoCo model
    kinDyn = KinDynComputations.from_mujoco_model(mj_model)
    kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    
    # Use in optimization
    opti = cs.Opti()
    joints = opti.variable(kinDyn.NDoF)
    
    w_H_b = cs.SX.eye(4)
    M_fun = kinDyn.mass_matrix_fun()
    M = M_fun(w_H_b, joints)
    
    # Add constraints using dynamics
    opti.minimize(cs.trace(M))
    opti.subject_to(cs.fabs(joints) <= 1.0)
    opti.solver('ipopt')
    sol = opti.solve()

See :doc:`../guides/mujoco` for more details on MuJoCo integration.

When to Use CasADi
------------------

✅ **Good for:**
- Optimal control (MPC, trajectory optimization)
- Symbolic formulation
- Code generation for embedded systems
- Building constraint expressions
- Trajectory planning

❌ **Not ideal for:**
- Simple numeric evaluation (NumPy is easier)
- Batching (not supported)

**See also:** :doc:`../guides/backend_selection`
