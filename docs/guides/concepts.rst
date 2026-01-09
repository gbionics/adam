Core Concepts
==============

This guide explains the fundamental concepts in adam.

Floating-Base Representation
-----------------------------

adam models robots as a **floating base** articulated system. The state consists of:

- **Base Transform** ``w_H_b``: The 4×4 homogeneous transformation matrix from world frame to base frame
- **Joint Positions** ``joints``: Scalar values for each actuated joint
- **Base Velocity** ``base_vel``: A 6D vector representing the linear and angular velocity of the base
- **Joint Velocities** ``joints_vel``: Scalar velocities for each actuated joint

Robot dynamics
----------------

The robot dynamics are described by the standard equations of motion for floating-base systems:

.. math::

    M(q) \dot{\nu} + C(q, \nu) + g(q) = \tau + J^T(q) f_{ext}

where:

- :math:`M(q)` is the mass matrix
- :math:`C(q, \nu)` are Coriolis and centrifugal effects
- :math:`g(q)` is the gravity term
- :math:`\tau` are the generalized forces/torques
- :math:`J(q)` is the Jacobian matrix of the frame on which external wrenches act
- :math:`f_{ext}` are external wrenches (forces/torques)
- :math:`q` = [w_H_b, joints] is the configuration
- :math:`\nu` = [base_vel, joints_vel] is the configuration velocity
- :math:`\dot{\nu}` is the configuration acceleration


Velocity Representations
------------------------

There are three 6D velocity representations for the floating base (check also useful `iDynTree theory <https://github.com/robotology/idyntree/blob/master/doc/theory.md>`_).
Let :math:`A` denote the world (inertial) frame and :math:`B` denote the base frame. The 6D velocity is stacked as :math:`\begin{bmatrix} v \\ \omega \end{bmatrix}` with linear velocity first.

**Mixed (Default)**

Expressed as ``Representations.MIXED_REPRESENTATION`` (default in adam and iDynTree):

.. math::

    {}^{A[B]}\mathrm{v}_{A,B} = \begin{bmatrix} {}^{A}\dot{o}_{B} \\ {}^{A}\omega_{A,B} \end{bmatrix} = \begin{bmatrix} {}^{A}\dot{o}_{B} \\ (\dot{{}^{A}R_{B}} {}^{A}R_{B}^{\top})^{\vee} \end{bmatrix}

Linear velocity is the time derivative of the base origin position (expressed in world frame :math:`A`), and angular velocity is expressed in the **world frame**.
This hybrid representation is commonly used in humanoid robotics and is the default in adam.


**Left-Trivialized (Body-Fixed)**

Expressed as ``Representations.BODY_FIXED_REPRESENTATION``:

.. math::

    {}^{B}\mathrm{v}_{A,B} = \begin{bmatrix} {}^{B}v_{B} \\ {}^{B}\omega_{A,B} \end{bmatrix} = \begin{bmatrix} {}^{B}R_{A} {}^{A}\dot{o}_{B} \\ ({}^{A}R_{B}^{\top} \dot{{}^{B}R_{A}})^{\vee} \end{bmatrix}

Both linear and angular velocities are expressed in the **base frame** coordinates.
This is the "body-fixed" frame representation.

**Right-Trivialized (Inertial-Fixed)**

Expressed as ``Representations.INERTIAL_FIXED_REPRESENTATION``:

.. math::

    {}^{A}\mathrm{v}_{A,B} = \begin{bmatrix} {}^{A}v_{B} \\ {}^{A}\omega_{A,B} \end{bmatrix} = \begin{bmatrix} {}^{A}\dot{o}_{B} - \dot{{}^{A}R_{B}} {}^{A}R_{B}^{\top} {}^{A}o_{B} \\ (\dot{{}^{A}R_{B}} {}^{A}R_{B}^{\top})^{\vee} \end{bmatrix}

Linear velocity is expressed in the **world frame**, angular velocity in the **world frame**.
This is the "inertial-fixed" frame representation.


**Setting the Representation**

Always set this early when creating a ``KinDynComputations`` object:

.. code-block:: python

    kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    # or
    kinDyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
    # or
    kinDyn.set_frame_velocity_representation(adam.Representations.INERTIAL_FIXED_REPRESENTATION)

The representation affects:

- **Jacobian computations** – Jacobians transform between representations via adjoint matrices
- **Inertia matrix structure** – Mass matrix transforms according to representation
- **Centroidal momentum calculations** – Momentum definitions differ by representation
- **Any velocity-dependent quantities** – Coriolis forces, bias forces, etc.

See the `iDynTree theory documentation <https://github.com/robotology/idyntree/blob/master/doc/theory.md>`_ for mathematical details on 6D velocity representations.

State Format
^^^^^^^^^^^^

Velocities are always stacked as:

.. code-block:: python

    [base_linear_vel, base_angular_vel, joint_velocities]
    # Shape: (3,) + (3,) + (n_dof,) = (6 + n_dof,)


Key Matrices and Quantities
----------------------------

**Mass Matrix** ``M``

Shape: ``(6 + n_dof, 6 + n_dof)``

The inertia matrix relating generalized forces to generalized accelerations:

.. math::

    M(q) \dot{\nu} = \tau - h(q, \nu) - g(q)

where:

- :math:`M` is symmetric and positive definite
- Describes how joint positions affect inertia
- Used in inverse dynamics, control design

**Centroidal Momentum Matrix** ``Ag``

Shape: ``(6, 6 + n_dof)``

Relates generalized velocities to centroidal momentum:

.. math::

    \begin{bmatrix} L \\ h \end{bmatrix} = A_g(q) \nu

where:

- ``L``: Linear momentum
- ``h``: Angular momentum about CoM

**Jacobian** ``J``

Shape: ``(6, 6 + n_dof)``

Relates end-effector velocity to generalized velocities:

.. math::

    v_{ee} = J(q) \nu

Available as:

- ``jacobian(frame_name, ...)`` – frame Jacobian
- ``CoM_jacobian(...)`` – CoM Jacobian
- ``jacobian_dot(...)`` – time derivative

**Forward Kinematics** ``w_H_frame``

Shape: ``(4, 4)``

The transformation from world to any frame:

.. code-block:: python

    w_H_frame = kinDyn.forward_kinematics('frame_name', w_H_b, joints)


Dynamics Algorithms
-------------------

adam uses **Featherstone's Recursive Algorithms** under the hood, which compute all quantities efficiently in O(n) time where n is the number of joints.

**Articulated Body Algorithm (ABA)**

Computes accelerations given forces:

.. math::

    \ddot{q} = \text{ABA}(q, \nu, \tau, f_{ext})

Used in forward dynamics.


**Recursive Newton-Euler Algorithm (RNEA)**

Computes Coriolis, centrifugal, and gravity effects:

.. math::

    h(q, \nu) = C(q,\nu) + g(q) = \text{RNEA}(q, \nu)

**Composite Rigid Body Algorithm (CRBA)**

Computes the mass matrix and centroidal momentum matrix:

.. math::

    M(q), A_g(q) = \text{CRBA}(q)

External Wrenches
-----------------

When computing system acceleration with external forces (e.g., contact forces), pass them as a dictionary:

.. code-block:: python

    external_wrenches = {
        'l_sole': np.array([fx, fy, fz, mx, my, mz]),  # Contact force at left foot
        'r_sole': np.array([fx, fy, fz, mx, my, mz]),  # Contact force at right foot
    }
    
    acc = kinDyn.aba(
        w_H_b, joints, base_vel, joints_vel, tau,
        external_wrenches=external_wrenches
    )

