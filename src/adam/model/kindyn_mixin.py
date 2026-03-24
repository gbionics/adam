from __future__ import annotations

from typing import Any


class KinDynFactoryMixin:
    """Shared helpers to instantiate KinDyn* classes from different model sources."""

    @property
    def model(self):
        return self.rbdalgos.model

    @classmethod
    def from_urdf(cls: type[KinDynFactoryMixin], urdfstring: Any, *args, **kwargs):
        """Instantiate using a URDF path/string.

        Args:
            urdfstring (str): path/string of a URDF

        Returns:
            KinDynFactoryMixin: An instance of the class initialized with the provided URDF and arguments.
        """
        return cls(urdfstring, *args, **kwargs)

    @classmethod
    def from_mujoco_model(
        cls: type[KinDynFactoryMixin],
        mujoco_model: Any,
        use_mujoco_actuators: bool = False,
        *args,
        **kwargs,
    ):
        """Instantiate using a MuJoCo model.

        Args:
            mujoco_model (MjModel): MuJoCo model instance
            use_mujoco_actuators (bool): If True, use the names of joints under the <actuator> tags in the MuJoCo XML as joint names
            (i.e., set 'joints_name_list' to actuator joint names). This is useful when you want the
            KinDyn* instance to use actuator joint names instead of the default joint names in the xml.
            Default is False.

        Returns:
            KinDynFactoryMixin: An instance of the class initialized with the provided MuJoCo model
        """
        if use_mujoco_actuators:
            # use as joint names the names of joints under the <actuator> tags in the mujoco xml
            actuator_names = [
                mujoco_model.actuator(a).name for a in range(mujoco_model.nu)
            ]
            kwargs.setdefault("joints_name_list", actuator_names)
        return cls(mujoco_model, *args, **kwargs)

    @classmethod
    def from_usd(
        cls: type[KinDynFactoryMixin],
        usd_path: Any,
        robot_prim_path: str | None = None,
        *args,
        **kwargs,
    ):
        """Instantiate using an OpenUSD stage path.

        Args:
            usd_path: USD path/string/pathlib.Path (.usd/.usda/.usdc/.usdz)
            robot_prim_path (str | None): Optional articulation-root prim path to select a specific robot.

        Returns:
            KinDynFactoryMixin: An instance initialized from the provided USD description.
        """
        description = (
            {"usd_path": usd_path, "robot_prim_path": robot_prim_path}
            if robot_prim_path is not None
            else usd_path
        )
        return cls(description, *args, **kwargs)

    @classmethod
    def from_usd_stage(
        cls: type[KinDynFactoryMixin],
        usd_stage: Any,
        robot_prim_path: str | None = None,
        *args,
        **kwargs,
    ):
        """Instantiate using an in-memory pxr.Usd.Stage.

        Args:
            usd_stage: pxr.Usd.Stage instance
            robot_prim_path (str | None): Optional articulation-root prim path to select a specific robot.

        Returns:
            KinDynFactoryMixin: An instance initialized from the provided USD stage.
        """
        description = (
            {"usd_stage": usd_stage, "robot_prim_path": robot_prim_path}
            if robot_prim_path is not None
            else usd_stage
        )
        return cls(description, *args, **kwargs)
