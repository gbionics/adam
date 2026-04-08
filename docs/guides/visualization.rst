Visualization
=============

adam provides a lightweight visualization layer based on `viser <https://viser.studio/>`_.
It can render robot models loaded from URDF, MuJoCo, or USD through the same normalized model API.

Installation
------------

Install the visualization dependencies with:

.. code-block:: bash

    pip install adam-robotics[visualization]

If you also want to visualize MuJoCo or USD models, install the corresponding extras as well:

.. code-block:: bash

    pip install adam-robotics[visualization,mujoco]
    pip install adam-robotics[visualization,usd]

Basic Usage
-----------

Create a scene-level ``Visualizer`` and add a robot model to it:

.. code-block:: python

    import numpy as np
    import icub_models
    from adam.numpy import KinDynComputations
    from adam.visualization import Visualizer

    kindyn = KinDynComputations.from_urdf(
        icub_models.get_model_file("iCubGazeboV2_5")
    )

    visualizer = Visualizer(
        world_axes=True,
        ground=True,
        camera_position=(2.5, -2.0, 1.5),
        camera_look_at=(0.0, 0.0, 0.6),
    )

    robot = visualizer.add_model(kindyn, root_name="/icub")

    w_H_b = np.eye(4)
    w_H_b[2, 3] = 0.6
    q = np.zeros(kindyn.NDoF)
    robot.update(w_H_b, q)

``Visualizer`` owns the shared viser scene. ``add_model()`` returns a ``ModelHandle``
for one robot instance in that scene.

Command-Line Viewer
-------------------

If you just want to inspect a model quickly, use the bundled ``adam-model-view`` command:

.. code-block:: bash

    adam-model-view --urdf path/to/robot.urdf
    adam-model-view --mujoco path/to/model.xml
    adam-model-view --usd path/to/robot.usda --robot-prim-path /Robot

The viewer loads the requested model, starts a viser server, adds a default ground plane,
and exposes one joint-slider panel per robot when the model has actuated joints.

Joint Sliders
-------------

You can add a GUI panel with one slider per actuated joint:

.. code-block:: python

    robot.add_joint_sliders(
        folder_name="iCub",
        expand_by_default=False,
    )

Slider limits are taken from the model joint limits when available.

Multiple Robots
---------------

One ``Visualizer`` can host multiple robot models:

.. code-block:: python

    g1 = visualizer.add_model(g1_kindyn, root_name="/g1")
    aliengo = visualizer.add_model(aliengo_kindyn, root_name="/aliengo")

    g1.update(g1_base_transform, g1_joint_positions)
    aliengo.update(aliengo_base_transform, aliengo_joint_positions)

Each model gets its own root in the scene graph and can expose its own slider panel.

Supported Model Sources
-----------------------

The visualizer uses the normalized visuals stored in the adam model, so the same API works for:

- URDF models loaded with ``KinDynComputations.from_urdf(...)``
- MuJoCo models loaded with ``KinDynComputations.from_mujoco_model(mj_model)``
- USD models loaded with ``KinDynComputations.from_usd(...)`` or ``from_usd_stage(...)``

For MuJoCo mesh visuals, adam uses the compiled mesh data already stored in ``MjModel``.
For URDF file meshes, ``trimesh`` is used to load the mesh file before sending it to viser.

Examples
--------

The repository includes ready-to-run examples:

- ``examples/visualization/visualize_urdf.py``
- ``examples/visualization/visualize_mujoco.py``
- ``examples/visualization/visualize_usd.py``
- ``examples/visualization/visualize_multi_robot.py``

See Also
--------

- :doc:`mujoco`
- :doc:`usd`
- :doc:`../modules/visualization`
