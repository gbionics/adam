# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import warnings

import numpy as np

from adam.core.constants import Representations
from adam.core.rbd_algorithms import RBDAlgorithms
from adam.model import Model, build_model_factory
from adam.model.kindyn_mixin import KinDynFactoryMixin
from adam.numpy.numpy_like import SpatialMath
from adam.core.array_api_math import spec_from_reference


class KinDynComputations(KinDynFactoryMixin):
    """This is a small class that retrieves robot quantities using NumPy for Floating Base systems."""

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list = None,
        dtype: np.dtype = np.float64,
        root_link: str = None,
        gravity: np.array = np.array([0, 0, -9.80665, 0, 0, 0], dtype=np.float32),
    ) -> None:
        """
        Args:
            urdfstring (str): path/string of a URDF or a MuJoCo MjModel
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): Deprecated. The root link is automatically chosen as the link with no parent in the URDF. Defaults to None.
        """
        ref = np.array(0.0, dtype=dtype)
        math = SpatialMath(spec_from_reference(ref))
        factory = build_model_factory(description=urdfstring, math=math)
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = model.NDoF
        self.g = np.asarray(gravity, dtype=dtype)
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
        self.rbdalgos.set_frame_velocity_representation(representation)

    def mass_matrix(
        self, base_transform: np.ndarray, joint_positions: np.ndarray
    ) -> np.ndarray:
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            M (np.ndarray): Mass Matrix
        """
        [M, _] = self.rbdalgos.crba(base_transform, joint_positions)
        return M.array

    def centroidal_momentum_matrix(
        self, base_transform: np.ndarray, s: np.ndarray
    ) -> np.ndarray:
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joint positions

        Returns:
            Jcc (np.ndarray): Centroidal Momentum matrix
        """
        [_, Jcm] = self.rbdalgos.crba(base_transform, s)
        return Jcm.array

    def forward_kinematics(
        self, frame: str, base_transform: np.ndarray, joint_positions: np.ndarray
    ) -> np.ndarray:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            H (np.ndarray): The fk represented as Homogenous transformation matrix
        """
        return self.rbdalgos.forward_kinematics(
            frame, base_transform, joint_positions
        ).array.squeeze()

    def jacobian(
        self, frame: str, base_transform: np.ndarray, joint_positions: np.ndarray
    ) -> np.ndarray:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            J_tot (np.ndarray): The Jacobian relative to the frame
        """
        return self.rbdalgos.jacobian(
            frame, base_transform, joint_positions
        ).array.squeeze()

    def relative_jacobian(self, frame: str, joint_positions: np.ndarray) -> np.ndarray:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (np.ndarray): The joints position

        Returns:
            J (np.ndarray): The Jacobian between the root and the frame
        """
        return self.rbdalgos.relative_jacobian(frame, joint_positions).array

    def jacobian_dot(
        self,
        frame: str,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        base_velocity: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> np.ndarray:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            base_velocity (np.ndarray): The base velocity
            joint_velocities (np.ndarray): The joint velocities

        Returns:
            Jdot (np.ndarray): The Jacobian derivative relative to the frame
        """
        return self.rbdalgos.jacobian_dot(
            frame, base_transform, joint_positions, base_velocity, joint_velocities
        ).array.squeeze()

    def CoM_position(
        self, base_transform: np.ndarray, joint_positions: np.ndarray
    ) -> np.ndarray:
        """Returns the CoM position

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            CoM (np.ndarray): The CoM position
        """
        return self.rbdalgos.CoM_position(
            base_transform, joint_positions
        ).array.squeeze()

    def CoM_jacobian(
        self, base_transform: np.ndarray, joint_positions: np.ndarray
    ) -> np.ndarray:
        """Returns the CoM Jacobian

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            Jcom (np.ndarray): The CoM Jacobian
        """
        return self.rbdalgos.CoM_jacobian(
            base_transform, joint_positions
        ).array.squeeze()

    def bias_force(
        self,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        base_velocity: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> np.ndarray:
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            base_velocity (np.ndarray): The base velocity
            joint_velocities (np.ndarray): The joint velocities

        Returns:
            h (np.ndarray): the bias force
        """
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            self.g,
        ).array.squeeze()

    def coriolis_term(
        self,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        base_velocity: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> np.ndarray:
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            base_velocity (np.ndarray): The base velocity
            joint_velocities (np.ndarray): The joint velocities

        Returns:
            C (np.ndarray): the Coriolis term
        """
        # set in the bias force computation the gravity term to zero
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            np.zeros(6),
        ).array.squeeze()

    def gravity_term(
        self, base_transform: np.ndarray, joint_positions: np.ndarray
    ) -> np.ndarray:
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            G (np.ndarray): the gravity term
        """
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            np.zeros(6),
            np.zeros(self.NDoF),
            self.g,
        ).array.squeeze()

    def get_total_mass(self) -> float:
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.rbdalgos.get_total_mass()

    def aba(
        self,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        base_velocity: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray,
        external_wrenches: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Featherstone Articulated-Body Algorithm (floating base, O(n)).

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            base_velocity (np.ndarray): The base velocity
            joint_velocities (np.ndarray): The joint velocities
            joint_torques (np.ndarray): The joint torques
            external_wrenches (dict[str, np.ndarray], optional): External wrenches applied to the robot. Defaults to None.

        Returns:
            np.ndarray: The base acceleration and the joint accelerations
        """

        return self.rbdalgos.aba(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            joint_torques,
            self.g,
            external_wrenches,
        ).array.squeeze()

    def link_poses(
        self, base_transform: np.ndarray, joint_positions: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Return root-to-link transforms for the whole model.

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
        Returns:
            dict[str, np.ndarray]: A dictionary mapping link names to their poses as homogenous transformation matrices
        """
        return self.rbdalgos.link_poses(base_transform, joint_positions)
