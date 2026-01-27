import copy
import math
from enum import Enum

import numpy.typing as npt
import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import Link
from adam.model.abc_factories import Inertia, Inertial, Pose


class Geometry(Enum):
    """The different types of geometries that constitute the URDF"""

    BOX = 1
    CYLINDER = 2
    SPHERE = 3


class Side(Enum):
    """The possible sides of a box geometry"""

    WIDTH = 1
    HEIGHT = 2
    DEPTH = 3


class ParametricLink(Link):
    """Parametric Link class"""

    def __init__(
        self,
        link: urdf_parser_py.urdf.Link,
        math: SpatialMath,
        length_multiplier,
        densities,
    ):
        self.math = math
        self.name = link.name
        self.length_multiplier = length_multiplier
        self.densities = densities
        self.original_visual = copy.deepcopy(link.visual)
        self.visuals = [copy.deepcopy(link.visual)]
        self.geometry_type, self.visual_data = self.get_geometry(self.original_visual)
        original_volume, _ = self.compute_volume(length_multiplier=1.0)
        self.original_density = link.inertial.mass / original_volume
        self.link_offset = self.compute_offset()
        self.volume, self.visual_data_new = self.compute_volume(
            length_multiplier=self.length_multiplier
        )
        self.mass = self.compute_mass()
        # Ensure mass is an ArrayLike for backend math operations
        self.mass = self.math.factory.asarray(self.mass)
        inertia_parametric = self.compute_inertia_parametric()
        origin = self.modify_origin()
        self.inertial = Inertial(
            mass=self.mass, inertia=inertia_parametric, origin=origin
        )
        self.update_visuals()

    def get_principal_length(self):
        """Method computing the principal link length, i.e. the dimension in which the kinematic chain grows"""
        if self.geometry_type == Geometry.CYLINDER:
            xyz_rpy = [
                *self.original_visual.origin.xyz,
                *self.original_visual.origin.rpy,
            ]
            if xyz_rpy[3] < 0.0 or xyz_rpy[4] > 0.0:
                v_l = (
                    2 * self.visual_data.radius
                )  # returning the diameter, since the orientation of the shape is such that the radius is the principal lenght
            else:
                v_l = (
                    self.visual_data.length
                )  # returning the lenght, since the orientation of the shape is such that the radius is the principal lenght
        elif self.geometry_type == Geometry.SPHERE:
            v_l = self.visual_data.radius
        elif self.geometry_type == Geometry.BOX:
            v_l = self.visual_data.size[2]
        else:
            raise ValueError("THE GEOMETRY IS NOT SPECIFIED")
        return v_l

    def get_principal_length_parametric(self):
        """Method computing the principal link length parametric, i.e. the dimension in which the kinematic chain grows"""
        if self.geometry_type == Geometry.CYLINDER:
            xyz_rpy = [
                *self.original_visual.origin.xyz,
                *self.original_visual.origin.rpy,
            ]
            if xyz_rpy[3] < 0.0 or xyz_rpy[4] > 0.0:
                v_l = (
                    2 * self.visual_data_new[1]
                )  # returning the diameter, since the orientation of the shape is such that the radius is the principal lenght
            else:
                v_l = self.visual_data_new[
                    0
                ]  # returning the lenght, since the orientation of the shape is such that the radius is the principal lenght
        elif self.geometry_type == Geometry.SPHERE:
            v_l = self.visual_data_new
        elif self.geometry_type == Geometry.BOX:
            v_l = self.visual_data_new[2]
        else:
            raise ValueError("THE GEOMETRY IS NOT SPECIFIED")
        return v_l

    def compute_offset(self):
        """
        Returns:
            npt.ArrayLike: link offset
        """
        xyz_rpy = [*self.original_visual.origin.xyz, *self.original_visual.origin.rpy]
        v_l = self.get_principal_length()
        v_o = xyz_rpy[2]
        if v_o < 0:
            link_offset = v_l / 2 + v_o
        else:
            link_offset = v_o - v_l / 2
        return link_offset

    def compute_joint_offset(self, joint_i, parent_offset):
        """
        Returns:
            npt.ArrayLike: the child joint offset
        """
        # Taking the principal direction i.e. the length
        v_l = self.get_principal_length()
        j_0 = joint_i.origin.xyz[2]
        if j_0 < 0:
            joint_offset_temp = -(v_l + j_0 - parent_offset)
            joint_offset = joint_offset_temp
        else:
            joint_offset_temp = v_l + parent_offset - j_0
            joint_offset = joint_offset_temp
        return joint_offset

    @staticmethod
    def get_geometry(visual_obj):
        """
        Returns:
            (Geometry, urdf geometry): the geometry of the link and the related urdf object
        """
        if isinstance(visual_obj.geometry, urdf_parser_py.urdf.Box):
            return Geometry.BOX, visual_obj.geometry
        if isinstance(visual_obj.geometry, urdf_parser_py.urdf.Cylinder):
            return Geometry.CYLINDER, visual_obj.geometry
        if isinstance(visual_obj.geometry, urdf_parser_py.urdf.Sphere):
            return Geometry.SPHERE, visual_obj.geometry

        raise NotImplementedError(
            f"The visual type {visual_obj.geometry.__class__} is not supported"
        )

    def compute_volume(self, length_multiplier):
        """
        Returns:
            (npt.ArrayLike, npt.ArrayLike): the volume and the dimension parametric
        """
        volume = 0.0
        visual_data_new = [0.0, 0.0, 0.0]
        """Modifies a link's volume by a given multiplier, in a manner that is logical with the link's geometry"""
        if self.geometry_type == Geometry.BOX:
            visual_data_new[0] = self.visual_data.size[0]  # * self.length_multiplier[0]
            visual_data_new[1] = self.visual_data.size[1]  # * self.length_multiplier[1]
            visual_data_new[2] = self.visual_data.size[2] * length_multiplier
            volume = visual_data_new[0] * visual_data_new[1] * visual_data_new[2]
        elif self.geometry_type == Geometry.CYLINDER:
            visual_data_new[0] = self.visual_data.length * length_multiplier
            visual_data_new[1] = self.visual_data.radius  # * self.length_multiplier[1]
            volume = math.pi * visual_data_new[1] ** 2 * visual_data_new[0]
        elif self.geometry_type == Geometry.SPHERE:
            visual_data_new = self.visual_data.radius * length_multiplier
            volume = 4 * math.pi * visual_data_new**3 / 3
        return volume, visual_data_new

    def compute_mass(self):
        """
        Function that computes the mass starting from the densities, and the link volume
        Returns:
            (npt.ArrayLike): the link mass
        """
        return self.volume * self.densities

    def modify_origin(self):
        """
        Returns:
            (npt.ArrayLike): the link origin parametrized
        """
        origin = self.original_visual.origin
        if self.geometry_type == Geometry.SPHERE:
            "in case of a sphere the origin of the link does not change"
            # convert to Pose to ensure ArrayLike semantics for math operations
            return Pose.build(origin.xyz, origin.rpy, self.math)

        v_o = self.original_visual.origin.xyz[2]
        length = self.get_principal_length_parametric()
        if v_o < 0:
            origin.xyz[2] = self.link_offset - length / 2
        else:
            origin.xyz[2] = length / 2 + self.link_offset
        # convert to Pose to ensure ArrayLike semantics for math operations
        return Pose.build(origin.xyz, origin.rpy, self.math)

    def compute_inertia_parametric(self):
        """
        Returns:
            Inertia Parametric: inertia (ixx, iyy and izz) with the formula that corresponds to the geometry
        Formulas retrieved from https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        """
        xyz_rpy = [*self.original_visual.origin.xyz, *self.original_visual.origin.rpy]
        # mass = self.mass.array if hasattr(self.mass, "array") else self.mass
        mass = self.mass.array
        if self.geometry_type == Geometry.BOX:
            coeff_x = (self.visual_data_new[1] ** 2 + self.visual_data_new[2] ** 2) / 12
            coeff_y = (self.visual_data_new[0] ** 2 + self.visual_data_new[2] ** 2) / 12
            coeff_z = (self.visual_data_new[0] ** 2 + self.visual_data_new[1] ** 2) / 12
            ixx = mass * coeff_x
            iyy = mass * coeff_y
            izz = mass * coeff_z
        elif self.geometry_type == Geometry.CYLINDER:
            coeff_xy = (
                3 * (self.visual_data_new[1] ** 2) + self.visual_data_new[0] ** 2
            ) / 12
            coeff_zz = self.visual_data_new[1] ** 2 / 2
            ixx = mass * coeff_xy
            iyy = mass * coeff_xy
            izz = mass * coeff_zz
            # Handle orientation swaps without item assignment
            if xyz_rpy[3] > 0 and xyz_rpy[4] == 0.0 and xyz_rpy[5] == 0.0:
                iyy, izz = izz, ixx
            elif xyz_rpy[4] > 0.0:
                ixx, izz = izz, iyy
        elif self.geometry_type == Geometry.SPHERE:
            coeff_s = 2 * self.visual_data_new**2 / 5
            ixx = mass * coeff_s
            iyy, izz = ixx, ixx
        else:
            # Default to zeros if geometry is unknown
            ixx, iyy, izz = 0.0, 0.0, 0.0
        iyz, ixz, ixy = 0.0, 0.0, 0.0
        # make the components ArrayLike
        ixx = self.math.asarray(ixx).array
        iyy = self.math.asarray(iyy).array
        izz = self.math.asarray(izz).array
        ixy = self.math.asarray(ixy).array
        ixz = self.math.asarray(ixz).array
        iyz = self.math.asarray(iyz).array

        return Inertia.build(
            ixx=ixx, ixy=ixy, ixz=ixz, iyy=iyy, iyz=iyz, izz=izz, math=self.math
        )

    def spatial_inertia(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: the 6x6 inertia matrix expressed at the
                           origin of the link (with rotation)
        """
        inertia_matrix = self.inertial.inertia.get_matrix()
        mass = self.mass
        xyz = self.inertial.origin.xyz
        rpy = self.inertial.origin.rpy
        return self.math.spatial_inertia_with_parameters(inertia_matrix, mass, xyz, rpy)

    def homogeneous(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: the homogeneous transform of the link
        """
        xyz = self.inertial.origin.xyz
        rpy = self.inertial.origin.rpy
        return self.math.H_from_Pos_RPY(
            xyz,
            rpy,
        )

    def update_visuals(self):
        if self.geometry_type == Geometry.BOX:
            self.visuals[0].geometry.size = self.visual_data_new
            self.visuals[0].origin.xyz[2] = self.inertial.origin.xyz[2]
        elif self.geometry_type == Geometry.CYLINDER:
            self.visuals[0].geometry.length = self.visual_data_new[0]
            self.visuals[0].origin.xyz[2] = self.inertial.origin.xyz[2]
        elif self.geometry_type == Geometry.SPHERE:
            self.visuals[0].geometry.radius = self.visual_data_new
