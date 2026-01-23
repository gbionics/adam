from typing import Union

import numpy.typing as npt
import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import Joint, Limits, Pose
from adam.parametric.model.parametric_factories.parametric_link import ParametricLink
import math


class ParametricJoint(Joint):
    """Parametric Joint class"""

    def __init__(
        self,
        joint: urdf_parser_py.urdf.Joint,
        math: SpatialMath,
        parent_link: ParametricLink,
        idx: Union[int, None] = None,
    ) -> None:
        self.math = math
        self.name = joint.name
        self.parent = parent_link.name
        self.parent_parametric = parent_link
        self.child = joint.child
        self.type = joint.joint_type
        # Ensure axis is an ArrayLike for backend math
        self.axis = self._set_axis(joint.axis)
        self.limit = self._set_limits(joint.limit)
        self.idx = idx
        self.joint = joint
        joint_offset = self.parent_parametric.compute_joint_offset(
            joint, self.parent_parametric.link_offset
        )
        self.offset = joint_offset
        self.origin = self.modify(self.parent_parametric.link_offset)

    def _set_axis(self, axis: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            axis (npt.ArrayLike): axis

        Returns:
            npt.ArrayLike: set the axis
        """
        return None if axis is None else self.math.asarray(axis)

    def _set_origin(self, origin: Pose) -> Pose:
        """
        Args:
            origin (Pose): origin

        Returns:
            Pose: set the origin
        """
        return Pose.build(xyz=origin.xyz, rpy=origin.rpy, math=self.math)

    def _set_limits(self, limit: Limits) -> Limits:
        """
        Args:
            limit (Limits): limit

        Returns:
            Limits: set the limits
        """
        joint_lim = math.inf if self.type == "prismatic" else 2 * math.pi
        return Limits(
            lower=-joint_lim if limit is None else limit.lower,
            upper=joint_lim if limit is None else limit.upper,
            effort=math.inf if limit is None else limit.effort,
            velocity=math.inf if limit is None else limit.velocity,
        )

    def modify(self, parent_joint_offset: npt.ArrayLike):
        """
        Args:
            parent_joint_offset (npt.ArrayLike): offset of the parent joint

        Returns:
            npt.ArrayLike: the origin of the joint, parametric with respect to the parent link dimensions
        """

        length = self.parent_parametric.get_principal_length_parametric()
        modified = self.joint.origin

        if modified.xyz[2] < 0:
            modified.xyz[2] = -length + parent_joint_offset - self.offset
        else:
            vo = self.parent_parametric.inertial.origin.xyz[2].array
            modified.xyz[2] = vo + length / 2 - self.offset
        # Return a Pose built with the backend math so xyz/rpy are ArrayLike
        return Pose.build(modified.xyz, modified.rpy, self.math)

    def homogeneous(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint value

        Returns:
            npt.ArrayLike: the homogenous transform of a joint, given q
        """
        if self.type == "fixed":
            xyz = self.origin.xyz
            rpy = self.origin.rpy
            return self.math.H_from_Pos_RPY(xyz, rpy)
        elif self.type in ["revolute", "continuous"]:
            return self.math.H_revolute_joint(
                self.origin.xyz,
                self.origin.rpy,
                self.axis,
                q,
            )
        elif self.type in ["prismatic"]:
            return self.math.H_prismatic_joint(
                self.origin.xyz,
                self.origin.rpy,
                self.axis,
                q,
            )

    def spatial_transform(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint motion

        Returns:
            npt.ArrayLike: spatial transform of the joint given q
        """
        if self.type == "fixed":
            return self.math.X_fixed_joint(self.origin.xyz, self.origin.rpy)
        elif self.type in ["revolute", "continuous"]:
            return self.math.X_revolute_joint(
                self.origin.xyz, self.origin.rpy, self.axis, q
            )
        elif self.type in ["prismatic"]:
            return self.math.X_prismatic_joint(
                self.origin.xyz,
                self.origin.rpy,
                self.axis,
                q,
            )

    def motion_subspace(self) -> npt.ArrayLike:
        """
        Args:
            joint (Joint): Joint

        Returns:
            npt.ArrayLike: motion subspace of the joint
        """
        if self.type == "fixed":
            return self.math.zeros(6, 1)
        elif self.type in ["revolute", "continuous"]:
            axis = self.axis
            z = self.math.zeros(1)
            return self.math.vertcat(z, z, z, axis[0], axis[1], axis[2])
        elif self.type in ["prismatic"]:
            axis = self.axis
            z = self.math.zeros(1)
            return self.math.vertcat(axis[0], axis[1], axis[2], z, z, z)
