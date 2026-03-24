from typing import Any

import numpy.typing as npt

from adam.core.spatial_math import SpatialMath
from adam.model import Inertia, Inertial, Link, Pose
from adam.model.visuals import Visual


class StdLink(Link):
    """Standard Link class"""

    def __init__(
        self,
        link: Any,
        math: SpatialMath,
        *,
        visuals: list[Visual] | None = None,
    ):
        self.math = math
        self.name = link.name
        self.visuals = list(visuals if visuals is not None else link.visuals)
        self.inertial = self._set_inertia(link)
        self.collisions = link.collisions

    def _set_inertia(self, link: Any) -> npt.ArrayLike:
        """
        Args:
            inertia (npt.ArrayLike): inertia

        Returns:
            npt.ArrayLike: set the inertia
        """

        inertial = link.inertial
        if inertial is None:
            return Inertial(
                mass=self.math.asarray(0),
                inertia=Inertia.zero(self.math),
                origin=Pose.zero(self.math),
            )

        inertia = Inertia.build(
            ixx=inertial.inertia.ixx,
            ixy=inertial.inertia.ixy,
            ixz=inertial.inertia.ixz,
            iyy=inertial.inertia.iyy,
            iyz=inertial.inertia.iyz,
            izz=inertial.inertia.izz,
            math=self.math,
        )
        mass = self.math.asarray(inertial.mass)
        pose = Pose.build(inertial.origin.xyz, inertial.origin.rpy, self.math)
        return Inertial(mass=mass, inertia=inertia, origin=pose)

    def spatial_inertia(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: the 6x6 inertia matrix expressed at
                           the origin of the link (with rotation)
        """
        inertia_matrix = self.inertial.inertia.get_matrix()
        mass = self.inertial.mass
        o = self.inertial.origin.xyz
        rpy = self.inertial.origin.rpy
        return self.math.spatial_inertia(inertia_matrix, mass, o, rpy)

    def homogeneous(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: the homogeneous transform of the link
        """
        return self.math.H_from_Pos_RPY(
            self.inertial.origin.xyz,
            self.inertial.origin.rpy,
        )
