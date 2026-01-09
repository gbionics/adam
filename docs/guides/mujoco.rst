MuJoCo Integration
==================

adam supports loading robot models directly from `MuJoCo <https://mujoco.org/>`_ ``MjModel`` objects. This enables seamless integration with MuJoCo simulations and models from `robot_descriptions <https://github.com/robot-descriptions/robot_descriptions.py>`_.

Loading MuJoCo Models
---------------------

Use the ``from_mujoco_model()`` class method to create a ``KinDynComputations`` instance:

.. code-block:: python

    import mujoco
    import numpy as np
    from adam import Representations
    from adam.numpy import KinDynComputations

    # Load a MuJoCo model (e.g., from robot_descriptions)
    from robot_descriptions.loaders.mujoco import load_robot_description
    mj_model = load_robot_description("g1_mj_description")

    # Create KinDynComputations directly from MuJoCo model
    kinDyn = KinDynComputations.from_mujoco_model(mj_model)

    # Set velocity representation
    kinDyn.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)

    # Set gravity to match MuJoCo settings
    kinDyn.g = np.concatenate([mj_model.opt.gravity, np.zeros(3)])

Joint Extraction Options
-------------------------

By default, ``from_mujoco_model()`` extracts joint names from the MuJoCo model. You can customize this behavior:

**Using Joint Names (Default)**

.. code-block:: python

    kinDyn = KinDynComputations.from_mujoco_model(mj_model)
    # Uses joint names from mj_model

**Using Actuator Names**

.. code-block:: python

    kinDyn = KinDynComputations.from_mujoco_model(
        mj_model, 
        use_mujoco_actuators=True
    )
    # Uses actuator names instead of joint names

This is useful when your MuJoCo model has actuators defined and you want to work with actuator coordinates.

Working with MuJoCo State
--------------------------

To compute dynamics quantities, you need to extract the robot state from MuJoCo and format it for adam:

.. code-block:: python

    # Create MuJoCo data and set state
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = your_qpos  # Your configuration
    mj_data.qvel[:] = your_qvel  # Your velocities
    mujoco.mj_forward(mj_model, mj_data)

    # Extract base transform from MuJoCo state (for floating-base robots)
    from scipy.spatial.transform import Rotation as R
    
    base_rot = R.from_quat(mj_data.qpos[3:7], scalar_first=True).as_matrix()
    base_pos = mj_data.qpos[0:3]
    w_H_b = np.eye(4)
    w_H_b[:3, :3] = base_rot
    w_H_b[:3, 3] = base_pos

    # Joint positions (excluding free joint)
    # Ensure the serialization between MuJoCo and adam is the same
    joints = mj_data.qpos[7:]

    # Compute dynamics quantities
    M = kinDyn.mass_matrix(w_H_b, joints)
    com_pos = kinDyn.CoM_position(w_H_b, joints)
    J = kinDyn.jacobian('frame_name', w_H_b, joints)

Velocity Representation Differences
------------------------------------

.. warning::
    
    MuJoCo uses a different velocity representation for the floating object.
    See MuJoCo documentation for details: `MuJoCo Floating objects <https://mujoco.readthedocs.io/en/latest/overview.html#floating-objects>`_.

**MuJoCo Free Joint Velocity**

The free joint velocity in MuJoCo is structured as:

.. math::

    \begin{bmatrix} {}^{I}\dot{p}_{B} \\ {}^{B}\omega_{I, B} \end{bmatrix}

where the linear velocity is the **time derivative of the world frame position** and the angular velocity is in the **body frame**.

**adam Mixed Representation**

adam's default mixed representation (``MIXED_REPRESENTATION``) is:

.. math::

    \begin{bmatrix} {}^{I}\dot{p}_{B} \\ {}^{I}\omega_{I, B} \end{bmatrix}

where linear velocity is the **time derivative of the world frame position** and angular velocity is in the **world frame**. 

**Transformation**

To convert MuJoCo velocities to adam's mixed representation:

.. code-block:: python

    # Transform angular velocity from body to world frame
    R = ...  # Rotation matrix
    adam_base_vel = mujoco_base_vel.copy()
    adam_base_vel[3:6] = R @ mujoco_base_vel[3:6]


Backend Support
---------------

All adam backends support MuJoCo model loading. For example:

.. code-block:: python

    from adam.numpy import KinDynComputations
    from adam.jax import KinDynComputations
    from adam.pytorch import KinDynComputations
    from adam.casadi import KinDynComputations
    ...
    kinDyn = KinDynComputations.from_mujoco_model(mj_model)


Example: Neural IK with MuJoCo Model
-------------------------------------

See the `neural_ik.py example <https://github.com/ami-iit/adam/blob/main/examples/neural_ik.py>`_ for a complete demonstration of using adam to train a neural network for inverse kinematics.

Common Pitfalls
---------------

**Joint Ordering**

Ensure that the joint ordering in adam matches your expectations. Use:

.. code-block:: python

    print(kinDyn.rbdalgos.model.joint_names)

to verify the joint names and their order extracted from the MuJoCo model.

**Free Joint Handling**

MuJoCo's free joint (for floating-base robots) is automatically detected and excluded from the actuated joints list. The base transform and base velocity must be provided separately to adam methods.

**Gravity Vector**

Remember to set the gravity vector consistently between MuJoCo and adam:

.. code-block:: python

    # MuJoCo gravity is 3D
    kinDyn.g = np.concatenate([mj_model.opt.gravity, np.zeros(3)])
    # adam expects 6D: [linear_acceleration, angular_acceleration]


Testing with MuJoCo Models
---------------------------

The test suite includes MuJoCo-specific tests in ``tests/test_mujoco.py``.
These tests verify that quantities computed in adam and in mujoco correspond, given the velocity representation differences (see above).
See Also
--------

- :doc:`concepts` - Core concepts including velocity representations
- :doc:`backend_selection` - Choosing the right backend for your use case
- `MuJoCo Documentation <https://mujoco.readthedocs.io/>`_
- `robot_descriptions <https://github.com/robot-descriptions/robot_descriptions.py>`_ - Repository of robot models
