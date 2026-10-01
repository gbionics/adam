# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import warnings

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from adam.core.constants import Representations
from adam.core.rbd_algorithms import RBDAlgorithms
from adam.jax.jax_like import SpatialMath
from adam.model import Model
from adam.parametric.model import ParametricLink, URDFParametricModelFactory


class KinDynComputationsParametric:
    """This is a small class that retrieves robot quantities using Jax for Floating Base systems.
    This is parametric w.r.t the link length and densities.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        links_name_list: list,
        root_link: str = None,
        gravity: np.array = jnp.array([0, 0, -9.80665, 0, 0, 0]),
    ) -> None:
        """
        Args:
            urdfstring (str): either path or string of the urdf
            joints_name_list (list): list of the actuated joints
            links_name_list (list): list of parametric links
            root_link (str, optional): Deprecated. The root link is automatically chosen as the link with no parent in the URDF. Defaults to None.
        """
        self.math = SpatialMath()
        self.links_name_list = links_name_list
        self.g = gravity
        self.urdfstring = urdfstring
        self.joints_name_list = joints_name_list
        self.representation = Representations.MIXED_REPRESENTATION  # Default
        if root_link is not None:
            warnings.warn(
                "The root_link argument is not used. The root link is automatically chosen as the link with no parent in the URDF",
                DeprecationWarning,
                stacklevel=2,
            )

    def set_frame_velocity_representation(
        self, representation: Representations
    ) -> None:
        """Sets the representation of the velocity of the frames

        Args:
            representation (Representations): The representation of the velocity
        """
        self.representation = representation

    def mass_matrix(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        length_multiplier: jnp.array,
        densities: jnp.array,
    ):
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links

        Returns:
            M (jnp.array): Mass Matrix
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        [M, _] = self.rbdalgos.crba(base_transform, joint_positions)
        return M.array

    def centroidal_momentum_matrix(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        length_multiplier: jnp.array,
        densities: jnp.array,
    ):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links

        Returns:
            Jcc (jnp.array): Centroidal Momentum matrix
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        [_, Jcm] = self.rbdalgos.crba(base_transform, joint_positions)
        return Jcm.array

    def relative_jacobian(
        self,
        frame: str,
        joint_positions: jnp.array,
        length_multiplier: jnp.array,
        densities: jnp.array,
    ):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (jnp.array): The joints position
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links

        Returns:
            J (jnp.array): The Jacobian between the root and the frame
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.relative_jacobian(frame, joint_positions).array

    def jacobian_dot(
        self,
        frame: str,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        joint_velocities: jnp.array,
        length_multiplier: jnp.array,
        densities: jnp.array,
    ) -> jnp.array:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            base_velocity (jnp.array): The base velocity
            joint_velocities (jnp.array): The joint velocities
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links

        Returns:
            Jdot (jnp.array): The Jacobian derivative relative to the frame
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.jacobian_dot(
            frame, base_transform, joint_positions, base_velocity, joint_velocities
        ).array

    def forward_kinematics(
        self,
        frame: str,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        length_multiplier: jnp.array,
        densities: jnp.array,
    ):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links

        Returns:
            T_fk (jnp.array): The fk represented as Homogenous transformation matrix
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.forward_kinematics(
            frame, base_transform, joint_positions
        ).array

    def forward_kinematics_fun(
        self, frame, length_multiplier: jnp.array, densities: jnp.array
    ):
        return lambda T, joint_positions: self.forward_kinematics(
            frame, T, joint_positions
        )

    def jacobian(
        self,
        frame: str,
        base_transform,
        joint_positions,
        length_multiplier: jnp.array,
        densities: jnp.array,
    ):
        """Returns the Jacobian relative to the specified frame

        Args

            frame (str): The frame to which the jacobian will be computed
            base_transform (jnp.array): The homogenous transform from base to world frame
            s (jnp.array): The joints position
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links

        Returns:
            J_tot (jnp.array): The Jacobian relative to the frame
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.jacobian(frame, base_transform, joint_positions).array

    def bias_force(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        s_dot: jnp.array,
        length_multiplier: jnp.array,
        densities: jnp.array,
    ) -> jnp.array:
        """Returns the bias force of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            base_velocity (jnp.array): The base velocity
            s_dot (jnp.array): The joints velocity
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links


        Returns:
            h (jnp.array): the bias force
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.rnea(
            base_transform, joint_positions, base_velocity, s_dot, self.g
        ).array

    def coriolis_term(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        s_dot: jnp.array,
        length_multiplier: jnp.array,
        densities: jnp.array,
    ) -> jnp.array:
        """Returns the coriolis term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            base_velocity (jnp.array): The base velocity
            s_dot (jnp.array): The joints velocity
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links

        Returns:
            C (jnp.array): the Coriolis term
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity,
            s_dot,
            np.zeros(6),
        ).array

    def gravity_term(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        length_multiplier: jnp.array,
        densities: jnp.array,
    ) -> jnp.array:
        """Returns the gravity term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links

        Returns:
            G (jnp.array): the gravity term
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            np.zeros(6),
            np.zeros(self.NDoF),
            self.g,
        ).array

    def CoM_position(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        length_multiplier: jnp.array,
        densities: jnp.array,
    ) -> jnp.array:
        """Returns the CoM position

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links

        Returns:
            com (jnp.array): The CoM position
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.CoM_position(base_transform, joint_positions).array

    def get_total_mass(
        self, length_multiplier: jnp.array, densities: jnp.array
    ) -> float:
        """Returns the total mass of the robot

        Args:
            length_multiplier (jnp.array): The length multiplier of the parametrized links
            densities (jnp.array): The densities of the parametrized links

        Returns:
            mass: The total mass
        """
        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.get_total_mass()

    def get_original_densities(self) -> list[float]:
        """Returns the original densities of the parametric links

        Returns:
            densities: The original densities of the parametric links
        """
        densities = []
        model = self.rbdalgos.model
        for name in self.links_name_list:
            link = model.links[name]
            assert isinstance(link, ParametricLink)
            densities.append(link.original_density)
        return densities
