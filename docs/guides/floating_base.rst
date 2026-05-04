Configurable Floating Base
==========================

By default adam uses the **root link** of the URDF (the link with no parent) as
the floating base. You can change this to any other link at construction time or
at runtime. adam re-roots the kinematic tree internally so that all dynamics
quantities are consistent with the chosen floating base.

Choosing a root link
--------------------

Pass ``root_link`` to the constructor:

.. code-block:: python

    from adam.numpy import KinDynComputations

    kinDyn = KinDynComputations(
        model_path,
        joints_name_list,
        root_link="l_ankle_2",   # any link name in the model
    )
    kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

The same parameter is available for all backends (JAX, CasADi, PyTorch):

.. code-block:: python

    from adam.casadi import KinDynComputations as CasADiKinDyn
    from adam.jax    import KinDynComputations as JAXKinDyn
    from adam.pytorch import KinDynComputations as TorchKinDyn

    kd_cs    = CasADiKinDyn(model_path, joints_name_list, root_link="chest")
    kd_jax   = JAXKinDyn(model_path,    joints_name_list, root_link="chest")
    kd_torch = TorchKinDyn(model_path,  joints_name_list, root_link="chest")

Runtime setter
--------------

You can also change the floating base after construction. This rebuilds the
kinematic tree; for CasADi any cached ``cs.Function`` objects must be
re-requested afterwards.

.. code-block:: python

    kinDyn = KinDynComputations(model_path, joints_name_list)
    # ... compute some quantities with the default root ...

    kinDyn.set_root_link("chest")
    # all subsequent calls use "chest" as the floating base

How it works
------------

When a new root is requested adam finds the path in the kinematic tree from the
new root to the original URDF root and **reverses** every joint along that path.
A reversed joint:

- swaps its parent and child links,
- returns the *inverse* homogeneous transform ``H(q)^{-1}`` in place of the
  original ``H(q)``,
- negates and re-expresses the motion subspace so that the Featherstone
  recursions remain correct.

All joints **not** on the reversal path are unchanged.  The result is
numerically equivalent to iDynTree's ``setFloatingBase`` API.

.. note::

    ``root_link`` must be a **link** name, not a frame name.  In URDF models,
    frames are represented as zero-mass fixed-joint children and are not valid
    floating bases.  Use the parent link of the frame instead.

Joint serialization is independent of the floating base
--------------------------------------------------------

.. important::

    Changing the root link **does not change the joint position / velocity
    vector layout**.  The order of joints in ``joints`` (and the matching
    ``joints_name_list``) is fixed at construction time and stays the same
    regardless of which link is the floating base.

    The user defines the joint ordering once and it never changes — only the
    **base state** part of the inputs (``w_H_b`` and the base velocity)
    reflects the new floating base.

    Concretely, if you construct with::

        kinDyn = KinDynComputations(urdf, ["joint_A", "joint_B"], root_link="l_ankle_2")

    then ``joints[0]`` is always ``joint_A`` and ``joints[1]`` is always
    ``joint_B``, even though the kinematic tree is now rooted at
    ``l_ankle_2``.

.. seealso::

    :doc:`concepts` — velocity representations and the floating-base state.
