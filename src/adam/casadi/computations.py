# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import warnings

import casadi as cs
import numpy as np

from adam.casadi.casadi_like import SpatialMath
from adam.core import RBDAlgorithms
from adam.core.constants import Representations
from adam.model import Model, build_model_factory
from adam.model.kindyn_mixin import KinDynFactoryMixin


class KinDynComputations(KinDynFactoryMixin):
    """Class that retrieves robot quantities using CasADi for Floating Base systems."""

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list = None,
        root_link: str = None,
        gravity: np.array = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0]),
        f_opts: dict = dict(jit=False, jit_options=dict(flags="-Ofast"), cse=True),
    ) -> None:
        """
        Args:
            urdfstring (str): path/string of a URDF or a MuJoCo MjModel.
                NOTE: The parameter name `urdfstring` is deprecated and will be renamed to `model` in a future release.
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): Deprecated. The root link is automatically chosen as the link with no parent in the URDF. Defaults to None.
        """
        math = SpatialMath()
        factory = build_model_factory(description=urdfstring, math=math)
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        self.g = gravity
        self.f_opts = f_opts
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

    def mass_matrix_fun(self) -> cs.Function:
        """Returns the Mass Matrix functions computed the CRBA

        Returns:
            M (casADi function): Mass Matrix
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        M, _ = self.rbdalgos.crba(base_transform, joint_positions)
        return cs.Function(
            "M", [base_transform, joint_positions], [M.array], self.f_opts
        )

    def centroidal_momentum_matrix_fun(self) -> cs.Function:
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Returns:
            Jcc (casADi function): Centroidal Momentum matrix
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        _, Jcm = self.rbdalgos.crba(base_transform, joint_positions)
        return cs.Function(
            "Jcm", [base_transform, joint_positions], [Jcm.array], self.f_opts
        )

    def forward_kinematics_fun(self, frame: str) -> cs.Function:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed

        Returns:
            H (casADi function): The fk represented as Homogenous transformation matrix
        """
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_transform = cs.SX.sym("H", 4, 4)
        H = self.rbdalgos.forward_kinematics(frame, base_transform, joint_positions)
        return cs.Function(
            "H", [base_transform, joint_positions], [H.array], self.f_opts
        )

    def jacobian_fun(self, frame: str) -> cs.Function:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_tot (casADi function): The Jacobian relative to the frame
        """
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_transform = cs.SX.sym("H", 4, 4)
        J_tot = self.rbdalgos.jacobian(frame, base_transform, joint_positions)
        return cs.Function(
            "J_tot", [base_transform, joint_positions], [J_tot.array], self.f_opts
        )

    def relative_jacobian_fun(self, frame: str) -> cs.Function:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain

        Returns:
            J (casADi function): The Jacobian between the root and the frame
        """
        joint_positions = cs.SX.sym("s", self.NDoF)
        J = self.rbdalgos.relative_jacobian(frame, joint_positions)
        return cs.Function("J", [joint_positions], [J.array], self.f_opts)

    def jacobian_dot_fun(self, frame: str) -> cs.Function:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_dot (casADi function): The Jacobian derivative relative to the frame
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_velocity = cs.SX.sym("v_b", 6)
        joint_velocities = cs.SX.sym("s_dot", self.NDoF)
        J_dot = self.rbdalgos.jacobian_dot(
            frame, base_transform, joint_positions, base_velocity, joint_velocities
        )
        return cs.Function(
            "J_dot",
            [base_transform, joint_positions, base_velocity, joint_velocities],
            [J_dot.array],
            self.f_opts,
        )

    def CoM_position_fun(self) -> cs.Function:
        """Returns the CoM position

        Returns:
            CoM (casADi function): The CoM position
        """
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_transform = cs.SX.sym("H", 4, 4)
        com_pos = self.rbdalgos.CoM_position(base_transform, joint_positions)
        return cs.Function(
            "CoM_pos", [base_transform, joint_positions], [com_pos.array], self.f_opts
        )

    def CoM_jacobian_fun(self) -> cs.Function:
        """Returns the CoM Jacobian

        Returns:
            J_com (casADi function): The CoM Jacobian
        """
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_transform = cs.SX.sym("H", 4, 4)
        J_com = self.rbdalgos.CoM_jacobian(base_transform, joint_positions)
        return cs.Function(
            "J_com", [base_transform, joint_positions], [J_com.array], self.f_opts
        )

    def bias_force_fun(self) -> cs.Function:
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            h (casADi function): the bias force
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_velocity = cs.SX.sym("v_b", 6)
        joint_velocities = cs.SX.sym("s_dot", self.NDoF)
        h = self.rbdalgos.rnea(
            base_transform, joint_positions, base_velocity, joint_velocities, self.g
        )
        return cs.Function(
            "h",
            [base_transform, joint_positions, base_velocity, joint_velocities],
            [h.array],
            self.f_opts,
        )

    def coriolis_term_fun(self) -> cs.Function:
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            C (casADi function): the Coriolis term
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_velocity = cs.SX.sym("v_b", 6)
        joint_velocities = cs.SX.sym("s_dot", self.NDoF)
        # set in the bias force computation the gravity term to zero
        C = self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            np.zeros(6),
        )
        return cs.Function(
            "C",
            [base_transform, joint_positions, base_velocity, joint_velocities],
            [C.array],
            self.f_opts,
        )

    def gravity_term_fun(self) -> cs.Function:
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            G (casADi function): the gravity term
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        # set in the bias force computation the velocity to zero
        G = self.rbdalgos.rnea(
            base_transform, joint_positions, np.zeros(6), np.zeros(self.NDoF), self.g
        )
        return cs.Function(
            "G", [base_transform, joint_positions], [G.array], self.f_opts
        )

    def get_total_mass(self) -> float:
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.rbdalgos.get_total_mass()

    def mass_matrix(self, base_transform: cs.SX, joint_positions: cs.SX):
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (cs.SX): The homogenous transform from base to world frame
            joint_positions (cs.SX): The joints position

        Returns:
            M (cs.SX): Mass Matrix
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.mass_matrix_fun()"
            )

        M, _ = self.rbdalgos.crba(base_transform, joint_positions)
        return M.array

    def centroidal_momentum_matrix(self, base_transform: cs.SX, joint_positions: cs.SX):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (cs.SX): The homogenous transform from base to world frame
            joint_positions (cs.SX): The joints position

        Returns:
            Jcc (cs.SX): Centroidal Momentum matrix
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.centroidal_momentum_matrix_fun()"
            )

        _, Jcm = self.rbdalgos.crba(base_transform, joint_positions)
        return Jcm.array

    def relative_jacobian(self, frame: str, joint_positions: cs.SX):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (cs.SX): The joints position

        Returns:
            J (cs.SX): The Jacobian between the root and the frame
        """
        if isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.relative_jacobian_fun()"
            )

        return self.rbdalgos.relative_jacobian(frame, joint_positions).array

    def jacobian_dot(
        self,
        frame: str,
        base_transform: cs.SX,
        joint_positions: cs.SX,
        base_velocity: cs.SX,
        joint_velocities: cs.SX,
    ) -> cs.SX:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (cs.SX): The homogenous transform from base to world frame
            joint_positions (cs.SX): The joints position
            base_velocity (cs.SX): The base velocity
            joint_velocities (cs.SX): The joint velocities

        Returns:
            Jdot (cs.SX): The Jacobian derivative relative to the frame
        """
        if (
            isinstance(base_transform, cs.MX)
            and isinstance(joint_positions, cs.MX)
            and isinstance(base_velocity, cs.MX)
            and isinstance(joint_velocities, cs.MX)
        ):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.jacobian_dot_fun()"
            )
        return self.rbdalgos.jacobian_dot(
            frame, base_transform, joint_positions, base_velocity, joint_velocities
        ).array

    def forward_kinematics(
        self,
        frame: str,
        base_transform: cs.SX,
        joint_positions: cs.SX,
    ):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (cs.SX): The homogenous transform from base to world frame
            joint_positions (cs.SX): The joints position

        Returns:
            H (cs.SX): The fk represented as Homogenous transformation matrix
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.forward_kinematics_fun()"
            )

        return self.rbdalgos.forward_kinematics(
            frame, base_transform, joint_positions
        ).array

    def link_poses(
        self,
        base_transform: cs.SX,
        joint_positions: cs.SX,
    ) -> dict[str, cs.SX]:
        """Return root-to-link transforms for the whole model.

        Args:
            base_transform (cs.SX): Homogenous transform from base to world
            joint_positions (cs.SX): The joints position

        Returns:
            dict[str, cs.SX]: Link poses as homogenous transformation matrices
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use SX inputs for link_poses()."
            )

        return {
            name: transform.array
            for name, transform in self.rbdalgos.link_poses(
                base_transform,
                joint_positions,
            ).items()
        }

    def jacobian(self, frame: str, base_transform, joint_positions):
        """Returns the Jacobian relative to the specified frame

        Args:
            base_transform (cs.SX): The homogenous transform from base to world frame
            s (cs.SX): The joints position
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_tot (cs.SX): The Jacobian relative to the frame
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.jacobian_fun()"
            )

        return self.rbdalgos.jacobian(frame, base_transform, joint_positions).array

    def bias_force(
        self,
        base_transform: cs.SX,
        joint_positions: cs.SX,
        base_velocity: cs.SX,
        joint_velocities: cs.SX,
    ) -> cs.SX:
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (cs.SX): The homogenous transform from base to world frame
            joint_positions (cs.SX): The joints position
            base_velocity (cs.SX): The base velocity
            joint_velocities (cs.SX): The joints velocity

        Returns:
            h (cs.SX): the bias force
        """
        if (
            isinstance(base_transform, cs.MX)
            and isinstance(joint_positions, cs.MX)
            and isinstance(base_velocity, cs.MX)
            and isinstance(joint_velocities, cs.MX)
        ):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.bias_force_fun()"
            )

        return self.rbdalgos.rnea(
            base_transform, joint_positions, base_velocity, joint_velocities, self.g
        ).array

    def coriolis_term(
        self,
        base_transform: cs.SX,
        joint_positions: cs.SX,
        base_velocity: cs.SX,
        joint_velocities: cs.SX,
    ) -> cs.SX:
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (cs.SX): The homogenous transform from base to world frame
            joint_positions (cs.SX): The joints position
            base_velocity (cs.SX): The base velocity
            joint_velocities (cs.SX): The joints velocity

        Returns:
            C (cs.SX): the Coriolis term
        """
        if (
            isinstance(base_transform, cs.MX)
            and isinstance(joint_positions, cs.MX)
            and isinstance(base_velocity, cs.MX)
            and isinstance(joint_velocities, cs.MX)
        ):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.coriolis_term_fun()"
            )

        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            np.zeros(6),
        ).array

    def gravity_term(self, base_transform: cs.SX, joint_positions: cs.SX) -> cs.SX:
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (cs.SX): The homogenous transform from base to world frame
            joint_positions (cs.SX): The joints position

        Returns:
            G (cs.SX): the gravity term
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.gravity_term_fun()"
            )

        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            np.zeros(6),
            np.zeros(self.NDoF),
            self.g,
        ).array

    def aba(
        self,
        base_transform: cs.SX,
        joint_positions: cs.SX,
        base_velocity: cs.SX,
        joint_velocities: cs.SX,
        joint_torques: cs.SX,
        external_wrenches: dict[str, cs.SX] | None = None,
    ) -> cs.SX:
        """Featherstone Articulated-Body Algorithm (floating base, O(n)).

        Args:
            base_transform (cs.SX): The homogenous transform from base to world frame
            joint_positions (cs.SX): The joints position
            base_velocity (cs.SX): The base velocity
            joint_velocities (cs.SX): The joint velocities
            joint_torques (cs.SX): The joint torques
            external_wrenches (dict[str, cs.SX], optional): External wrenches applied to the robot. Defaults to None.

        Returns:
            cs.SX: The base acceleration and the joint accelerations
        """
        if (
            isinstance(base_transform, cs.MX)
            and isinstance(joint_positions, cs.MX)
            and isinstance(base_velocity, cs.MX)
            and isinstance(joint_velocities, cs.MX)
            and isinstance(joint_torques, cs.MX)
        ):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.aba_fun()"
            )

        return self.rbdalgos.aba(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            joint_torques,
            self.g,
            external_wrenches,
        ).array

    def aba_fun(self) -> cs.Function:
        """Returns the Articulated Body Algorithm function for forward dynamics

        Returns:
            qdd (casADi function): The joint accelerations and base acceleration
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_velocity = cs.SX.sym("v_b", 6)
        joint_velocities = cs.SX.sym("s_dot", self.NDoF)
        joint_torques = cs.SX.sym("tau", self.NDoF)

        qdd = self.rbdalgos.aba(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            joint_torques,
            self.g,
            None,  # external_wrenches not supported in symbolic form
        )
        return cs.Function(
            "qdd",
            [
                base_transform,
                joint_positions,
                base_velocity,
                joint_velocities,
                joint_torques,
            ],
            [qdd.array],
            self.f_opts,
        )

    def CoM_position(self, base_transform: cs.SX, joint_positions: cs.SX) -> cs.SX:
        """Returns the CoM position

        Args:
            base_transform (cs.SX): The homogenous transform from base to world frame
            joint_positions (cs.SX): The joints position

        Returns:
            CoM (cs.SX): The CoM position
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.CoM_position_fun()"
            )

        return self.rbdalgos.CoM_position(base_transform, joint_positions).array

    def CoM_jacobian(self, base_transform: cs.SX, joint_positions: cs.SX) -> cs.SX:
        """Returns the CoM Jacobian

        Args:
            base_transform (cs.SX): The homogenous transform from base to world frame
            joint_positions (cs.SX): The joints position

        Returns:
            J_com (cs.SX): The CoM Jacobian
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.CoM_jacobian_fun()"
            )

        return self.rbdalgos.CoM_jacobian(base_transform, joint_positions).array
