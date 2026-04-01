from __future__ import annotations
import dataclasses
from enum import Enum, auto
from typing import Any
import casadi as cs
import numpy as np
from adam.casadi import KinDynComputations
from liecasadi import SO3, SE3


class TargetType(Enum):
    """Type of IK target supported by the solver."""

    POSITION = auto()
    ROTATION = auto()
    POSE = auto()


class FramesConstraint(Enum):
    """Type of constraint for the frames in the IK problem."""

    BALL = auto()  # Ball constraint (position only)
    FIXED = auto()  # Fixed constraint (position + rotation)


@dataclasses.dataclass
class Target:
    """Dataclass to store target information."""

    target_type: TargetType
    frame: str
    weight: float = 1.0
    as_soft_constraint: bool = True
    forward_kinematics_function: cs.Function = None
    param_pos: cs.SX = None
    param_rot: cs.SX = None


class InverseKinematics:
    def __init__(
        self,
        urdf_path: str,
        joints_list: list[str],
        joint_limits_active: bool = True,
        solver_settings: dict[str, Any] = None,
        convert_revolute_to_spherical: bool = False,
    ):
        """Initialize the InverseKinematics solver.

        Args:
            urdf_path (str): Path to the URDF file.
            joints_list (list[str]): List of joint names.
            joint_limits_active (bool, optional): If True, enforces joint limits. Defaults to True.
            solver_settings (dict[str, ], optional): Settings for the solver. Defaults to None.
        """
        self.kd = KinDynComputations(
            urdf_path,
            joints_list,
            convert_revolute_to_spherical=convert_revolute_to_spherical,
        )
        self.ndof = self.kd.NDoF
        self.joints_list = self.kd.rbdalgos.model.actuated_joints

        # Opti variables --------------------------------------------------
        self.opti = cs.Opti()
        self.base_pos = self.opti.variable(3)  # translation (x,y,z)
        self.base_quat = self.opti.variable(4)  # unit quaternion (x,y,z,w)
        self.joint_pos = self.opti.variable(self.ndof)
        self.base_homogeneous = SE3.from_position_quaternion(
            self.base_pos, self.base_quat
        ).as_matrix()
        self.opti.set_initial(self.base_pos, [0, 0, 0])  # default to origin
        self.opti.set_initial(self.base_quat, [0, 0, 0, 1])  # identity quaternion
        self.opti.subject_to(cs.sumsqr(self.base_quat) == 1)  # enforce unit quaternion
        self.opti.set_initial(self.joint_pos, np.zeros(self.ndof))  # default to zero

        self.targets: dict[str, Target] = {}
        self.cost_terms: list[cs.MX] = []
        self._problem_built = False
        self._cached_sol = None
        solver_settings = (
            {
                "ipopt": {"print_level": 0},
            }
            if solver_settings is None
            else solver_settings
        )
        self.joint_limits_active = joint_limits_active
        self.set_default_joint_limit_constraints()

        self.opti.solver("ipopt", solver_settings)

    def get_opti_variables(self) -> dict[str, cs.SX]:
        """Get the optimization variables of the IK solver.

        Returns:
            dict[str, cs.SX]: Dictionary containing the optimization variables.

            - "base_pos": Position of the base (3D vector).
            - "base_quat": Quaternion representing the base orientation (4D vector with x, y, z, w).
            - "joint_pos": Joint variables (Vector with length equal to the number of non-fixed joints).
            - "base_homogeneous": Homogeneous transformation matrix of the base (4x4 matrix, from position and quaternion).

        """
        return {
            "base_pos": self.base_pos,
            "base_quat": self.base_quat,
            "joint_pos": self.joint_pos,
            "base_homogeneous": self.base_homogeneous,
        }

    def set_solver(self, solver_name: str, solver_settings: dict[str, Any]):
        """Set the solver for the optimization problem.

        Args:
            solver_name (str): The name of the solver to use.
            solver_settings (dict[str, Any]): The settings for the solver.
        """
        self.opti.solver(solver_name, solver_settings)
        self._cached_sol = None

    def get_opti(self) -> cs.Opti:
        """Get the CasADi Opti object for this IK solver. This can be used to add custom constraints or objectives.

        Returns:
            cs.Opti: The CasADi Opti object.
        """
        return self.opti

    def get_kin_dyn_computations(self) -> KinDynComputations:
        """Get the KinDynComputations object used by this IK solver.

        Returns:
            KinDynComputations: The KinDynComputations object.
        """
        return self.kd

    def set_default_joint_limit_constraints(self):
        """Set joint limit constraints if active."""
        if self.joint_limits_active:
            # Add joint limits as constraints
            for i, joint_name in enumerate(self.joints_list):
                joint_limit = self.kd.rbdalgos.model.joints[joint_name].limit
                if joint_limit is not None:
                    self.opti.subject_to(
                        self.opti.bounded(
                            joint_limit.lower, self.joint_pos[i], joint_limit.upper
                        )
                    )

    def set_joint_limits(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        """Set custom joint limits for the optimization problem.

        Args:
            lower_bounds (np.ndarray): Lower bounds for each joint.
            upper_bounds (np.ndarray): Upper bounds for each joint.
        """
        self._ensure_graph_modifiable()
        if len(lower_bounds) != self.ndof or len(upper_bounds) != self.ndof:
            raise ValueError("Bounds must match the number of degrees of freedom")
        for i in range(self.ndof):
            self.opti.subject_to(
                self.opti.bounded(lower_bounds[i], self.joint_pos[i], upper_bounds[i])
            )

    def set_joint_limit(self, joint_name: str, lower_limit: float, upper_limit: float):
        """Set custom joint limits for a specific joint for the optimization problem.

        Args:
            joint_name (str): The name of the joint to set limits for.
            lower_limit (float): The lower limit for the joint.
            upper_limit (float): The upper limit for the joint.
        """
        self._ensure_graph_modifiable()
        if joint_name not in self.joints_list:
            raise ValueError(f"Joint '{joint_name}' not found in joints list")
        idx = self.joints_list.index(joint_name)
        self.opti.subject_to(
            self.opti.bounded(lower_limit, self.joint_pos[idx], upper_limit)
        )

    def add_target_position(
        self,
        frame: str,
        *,
        as_soft_constraint: bool = True,
        weight: float = 1.0,
    ):
        """Add a target position for the IK solver.

        Args:
            frame (str): The name of the frame to target.
            as_soft_constraint (bool): If False, adds the target as a hard constraint. If True, adds it as a soft constraint (cost term).
            weight (float): The weight of the target, ignored if `as_soft_constraint` is `False`.

        Raises:
            ValueError: If the target frame already exists.
        """
        self._ensure_graph_modifiable()
        if frame in self.targets:
            raise ValueError(f"Target '{frame}' already exists")

        p_des = self.opti.parameter(3)
        self.opti.set_value(p_des, np.zeros(3))  # default to origin
        p_fk = self.kd.forward_kinematics_fun(frame)(
            self.base_transform(), self.joint_pos
        )[:3, 3]

        if as_soft_constraint:
            self.cost_terms.append(weight * cs.sumsqr(p_fk - p_des))
        else:
            self.opti.subject_to(p_fk == p_des)

        self.targets[frame] = Target(
            target_type=TargetType.POSITION,
            frame=frame,
            weight=weight,
            as_soft_constraint=as_soft_constraint,
            param_pos=p_des,
            forward_kinematics_function=self.kd.forward_kinematics_fun(frame),
        )

    def add_frames_constraint(
        self,
        parent_frame: str,
        child_frame: str,
        constraint_type: FramesConstraint,
        as_soft_constraint: bool = True,
        weight: float = 1e5,
    ):
        """Add a constraint between two frames.

        Args:
            parent_frame (str): The name of the parent frame.
            child_frame (str): The name of the child frame.
            constraint_type (FramesConstraint): Type of constraint to apply.
            as_soft_constraint (bool): If False, adds the constraint as a hard constraint. If True, adds it as a soft constraint (cost term).
            weight (float): Weight for the constraint in the cost function, ignored if `as_soft_constraint` is `False`.
        """
        self._ensure_graph_modifiable()
        # check that frames are different
        if parent_frame == child_frame:
            raise ValueError("Parent and child frames must be different")
        H_parent = self.kd.forward_kinematics_fun(parent_frame)(
            self.base_transform(), self.joint_pos
        )
        H_child = self.kd.forward_kinematics_fun(child_frame)(
            self.base_transform(), self.joint_pos
        )
        p_parent, p_child = H_parent[:3, 3], H_child[:3, 3]
        R_parent, R_child = H_parent[:3, :3], H_child[:3, :3]
        if constraint_type is FramesConstraint.BALL:
            # Ball constraint: child frame must be positioned relative to parent frame
            if as_soft_constraint:
                slack = self.opti.variable(1)
                self.opti.subject_to(cs.sumsqr(p_child - p_parent) <= slack)
                self.opti.subject_to(
                    self.opti.bounded(0, slack, 1e-3)
                )  # small slack to allow for numerical stability
                self.cost_terms.append(cs.sumsqr(slack) * weight)
            else:
                self.opti.subject_to(p_child == p_parent)

        elif constraint_type is FramesConstraint.FIXED:
            # Fixed constraint: child frame must be aligned and positioned with parent frame
            if as_soft_constraint:
                slack = self.opti.variable(1)
                # Positional error: squared distance between frame origins
                pos_err_sq = cs.sumsqr(p_child - p_parent)
                # Chordal rotation cost: 3 - trace(R_parent^T @ R_child)
                rot_err_sq = 3 - cs.trace(R_parent.T @ R_child)
                # Total fixed-frame error combines position and rotation terms
                total_err = pos_err_sq + rot_err_sq
                self.opti.subject_to(total_err <= slack)
                self.opti.subject_to(self.opti.bounded(0, slack, 1e-3))
                self.cost_terms.append(cs.sumsqr(slack) * weight)
            else:
                self.opti.subject_to(p_child == p_parent)
                self.opti.subject_to(R_child == R_parent)
        else:
            raise ValueError(f"Unsupported constraint type: {constraint_type.name}")

    def add_ball_constraint(
        self,
        parent_frame: str,
        child_frame: str,
        as_soft_constraint: bool = True,
        weight: float = 1e5,
    ):
        """Add a ball constraint between two frames.

        Args:
            parent_frame (str): The name of the parent frame.
            child_frame (str): The name of the child frame.
            as_soft_constraint (bool): If False, adds the constraint as a hard constraint. If True, adds it as a soft constraint (cost term).
            weight (float): Weight for the constraint in the cost function, ignored if `as_soft_constraint` is `False`
        """
        self.add_frames_constraint(
            parent_frame, child_frame, FramesConstraint.BALL, as_soft_constraint, weight
        )

    def add_fixed_constraint(
        self,
        parent_frame: str,
        child_frame: str,
        as_soft_constraint: bool = True,
        weight: float = 1e5,
    ):
        """Add a fixed constraint between two frames.

        Args:
            parent_frame (str): The name of the parent frame.
            child_frame (str): The name of the child frame.
            as_soft_constraint (bool): If False, adds the constraint as a hard constraint. If True, adds it as a soft constraint (cost term).
            weight (float): Weight for the constraint in the cost function, ignored if `as_soft_constraint` is `False`
        """
        self.add_frames_constraint(
            parent_frame,
            child_frame,
            FramesConstraint.FIXED,
            as_soft_constraint,
            weight,
        )

    def add_min_distance_constraint(self, frames_list: list[str], distance: float):
        """Add a distance constraint between frames.

        Args:
            frames_list (list[str]): List of frame names to apply the distance constraint.
            distance (float): Minimum distance between consecutive frames.
        """
        self._ensure_graph_modifiable()
        if len(frames_list) < 2:
            raise ValueError("At least two frames are required for distance constraint")
        for i in range(len(frames_list) - 1):
            print(
                f"Adding distance constraint between {frames_list[i]} and {frames_list[i + 1]}"
            )
            frame_i, frame_j = frames_list[i], frames_list[i + 1]
            p_i = self.kd.forward_kinematics_fun(frame_i)(
                self.base_transform(), self.joint_pos
            )[:3, 3]
            p_j = self.kd.forward_kinematics_fun(frame_j)(
                self.base_transform(), self.joint_pos
            )[:3, 3]
            dist_sq = cs.sumsqr(p_i - p_j)
            self.opti.subject_to(dist_sq >= distance**2)

    def add_target_orientation(
        self, frame: str, *, as_soft_constraint: bool = True, weight: float = 1.0
    ):
        """Add an orientation target for a frame.

        Args:
            frame (str): The name of the frame to target.
            as_soft_constraint (bool): If False, adds the target as a hard constraint. If True, adds it as a soft constraint (cost term).
            weight (float): Weight for the target in the cost function, ignored if `as_soft_constraint` is `False`.

        Raises:
            ValueError: If the target frame already exists.
        """
        self._ensure_graph_modifiable()
        if frame in self.targets:
            raise ValueError(f"Target '{frame}' already exists")

        R_des = self.opti.parameter(3, 3)
        self.opti.set_value(R_des, np.eye(3))  # default to identity rotation
        H_fk = self.kd.forward_kinematics_fun(frame)(
            self.base_transform(), self.joint_pos
        )
        R_fk = H_fk[:3, :3]
        # Chordal rotation cost: 3 - trace(R_des^T @ R_fk)
        # Equivalent to ||R_des^T @ R_fk - I||_F^2 / 2
        rot_err_sq = 3 - cs.trace(R_des.T @ R_fk)

        if as_soft_constraint:
            self.cost_terms.append(weight * rot_err_sq)
        else:
            self.opti.subject_to(rot_err_sq == 0)

        self.targets[frame] = Target(
            target_type=TargetType.ROTATION,
            frame=frame,
            weight=weight,
            as_soft_constraint=as_soft_constraint,
            param_rot=R_des,
            forward_kinematics_function=self.kd.forward_kinematics_fun(frame),
        )

    def add_target_pose(
        self, frame: str, *, as_soft_constraint: bool = True, weight: float = 1.0
    ):
        """Add a pose target for a frame.

        Args:
            frame (str): The name of the frame to target.
            as_soft_constraint (bool): If False, adds the target as a hard constraint. If True, adds it as a soft constraint (cost term).
            weight (float): Weight for the target in the cost function, ignored if `as_soft_constraint` is `False`.
        Raises:
            ValueError: If the target frame already exists.
        """
        self._ensure_graph_modifiable()
        if frame in self.targets:
            raise ValueError(f"Target '{frame}' already exists")

        p_des = self.opti.parameter(3)
        R_des = self.opti.parameter(3, 3)
        self.opti.set_value(p_des, np.zeros(3))  # default to origin
        self.opti.set_value(R_des, np.eye(3))  # default to identity rotation
        H_fk = self.kd.forward_kinematics_fun(frame)(
            self.base_transform(), self.joint_pos
        )
        p_fk, R_fk = H_fk[:3, 3], H_fk[:3, :3]

        pos_err_sq = cs.sumsqr(p_fk - p_des)
        # Chordal rotation cost: 3 - trace(R_des^T @ R_fk)
        # Equivalent to ||R_des^T @ R_fk - I||_F^2 / 2
        rot_err_sq = 3 - cs.trace(R_des.T @ R_fk)

        if as_soft_constraint:
            self.cost_terms.append(weight * (pos_err_sq + rot_err_sq))
        else:
            self.opti.subject_to(p_fk == p_des)
            self.opti.subject_to(rot_err_sq == 0)

        self.targets[frame] = Target(
            target_type=TargetType.POSE,
            frame=frame,
            weight=weight,
            as_soft_constraint=as_soft_constraint,
            param_pos=p_des,
            param_rot=R_des,
            forward_kinematics_function=self.kd.forward_kinematics_fun(frame),
        )

    def add_target(
        self,
        frame: str,
        target_type: TargetType,
        *,
        as_soft_constraint: bool = True,
        weight: float = 1.0,
    ):
        """Add a target for a frame.

        Args:
            frame (str): The name of the frame to target.
            target_type (TargetType): The type of target (position, rotation, or pose).
            as_soft_constraint (bool): If False, adds the target as a hard constraint. If True, adds it as a soft constraint (cost term).
            weight (float): Weight for the target in the cost function, ignored if `as_soft_constraint` is `False`.
        """
        if target_type is TargetType.POSITION:
            self.add_target_position(
                frame, weight=weight, as_soft_constraint=as_soft_constraint
            )
        elif target_type is TargetType.ROTATION:
            self.add_target_orientation(
                frame, weight=weight, as_soft_constraint=as_soft_constraint
            )
        elif target_type is TargetType.POSE:
            self.add_target_pose(
                frame, weight=weight, as_soft_constraint=as_soft_constraint
            )
        else:
            raise ValueError("Unsupported target type")

    def update_target_position(self, frame: str, position: np.ndarray):
        """Update the target position for a frame.

        Args:
            frame (str): The name of the frame to update.
            position (np.ndarray): The new target position.
        """
        self._check_target_type(frame, TargetType.POSITION)
        self.opti.set_value(self.targets[frame].param_pos, position)

    def update_target_orientation(self, frame: str, rotation: np.ndarray):
        """Update the target orientation for a frame.

        Args:
            frame (str): The name of the frame to update.
            rotation (np.ndarray): The new target rotation.
        """
        self._check_target_type(frame, TargetType.ROTATION)
        self.opti.set_value(self.targets[frame].param_rot, rotation)

    def update_target_pose(
        self, frame: str, position: np.ndarray, rotation: np.ndarray
    ):
        """Update the target pose for a frame.

        Args:
            frame (str): The name of the frame to update.
            position (np.ndarray): The new target position.
            rotation (np.ndarray): The new target rotation.
        """
        self._check_target_type(frame, TargetType.POSE)
        self.opti.set_value(self.targets[frame].param_pos, position)
        self.opti.set_value(self.targets[frame].param_rot, rotation)

    def update_target(
        self, frame: str, target: np.ndarray | tuple[np.ndarray, np.ndarray]
    ):
        """Update the target for a frame.

        Args:
            frame (str): The name of the frame to update.
            target (Union[np.ndarray, tuple[np.ndarray, np.ndarray]]): The new target position, rotation, or pose.

        Raises:
            ValueError: If the target is of an invalid type.
            ValueError: If the frame is not a pose target.
            RuntimeError: If the optimization problem has already been built.
        """
        if frame not in self.targets:
            raise ValueError(f"Unknown target '{frame}'")
        ttype = self.targets[frame].target_type
        if ttype is TargetType.POSITION:
            self.update_target_position(frame, target)  # type: ignore[arg-type]
        elif ttype is TargetType.ROTATION:
            self.update_target_orientation(frame, target)  # type: ignore[arg-type]
        elif ttype is TargetType.POSE:
            if not (isinstance(target, (list, tuple)) and len(target) == 2):
                raise ValueError("Pose update expects (position, rotation)")
            self.update_target_pose(frame, *target)  # type: ignore[arg-type]
        else:
            raise RuntimeError("Unknown target type")

    def set_initial_guess(self, base_transform: np.ndarray, joint_values: np.ndarray):
        """Set the initial guess for the optimization problem.

        Args:
            base_transform (np.ndarray): 4x4 transformation matrix for the floating base.
            joint_values (np.ndarray): Initial joint values for the robot.
        """
        pos, rot = base_transform[:3, 3], base_transform[:3, :3]
        quat = SO3.from_matrix(rot).as_quat().coeffs()  # (x,y,z,w)
        self.opti.set_initial(self.base_pos, pos)
        self.opti.set_initial(self.base_quat, quat)
        self.opti.set_initial(self.joint_pos, joint_values)

    def solve(self):
        """Solve the optimization problem."""
        if not self._problem_built:
            self._finalize_problem()
        if self._cached_sol is not None:
            # Warm‑start
            self.opti.set_initial(self.base_pos, self._cached_sol.value(self.base_pos))
            self.opti.set_initial(
                self.base_quat, self._cached_sol.value(self.base_quat)
            )
            self.opti.set_initial(
                self.joint_pos, self._cached_sol.value(self.joint_pos)
            )
        self._cached_sol = self.opti.solve()

    def get_solution(self):
        """Get the solution of the optimization problem.

        Raises:
            RuntimeError: If the optimization problem has not been solved yet.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The 4x4 transformation matrix and joint values.
        """
        if self._cached_sol is None:
            raise RuntimeError("No solution yet - call solve() first")
        pos = self._cached_sol.value(self.base_pos)
        quat = self._cached_sol.value(self.base_quat)
        joints = self._cached_sol.value(self.joint_pos)
        H = SE3.from_position_quaternion(pos, quat).as_matrix()  # 4x4 numpy
        return np.array(H), np.array(joints)

    def base_transform(self):
        """Symbolic 4x4 transform of the floating base (SX)."""
        return SE3.from_position_quaternion(self.base_pos, self.base_quat).as_matrix()

    def _ensure_graph_modifiable(self):
        if self._problem_built:
            raise RuntimeError(
                "Cannot add new targets after the problem has been assembled (first solve())."
            )

    def _finalize_problem(self):
        """Finalize the optimization problem.

        Raises:
            RuntimeError: If the optimization problem has already been built.
        """
        if self.cost_terms:
            self.opti.minimize(cs.sum(cs.vertcat(*self.cost_terms)))
        else:
            raise RuntimeError(
                "No targets added to the problem. Please add at least one target."
            )
        self._problem_built = True

    def _check_target_type(self, frame: str, expected: TargetType):
        """Check the target type for a given frame.

        Args:
            frame (str): The name of the frame to check.
            expected (TargetType): The expected target type.

        Raises:
            ValueError: If the target type does not match the expected type.
        """
        if self.targets[frame].target_type is not expected:
            raise ValueError(f"Target '{frame}' is not of type {expected.name}")

    def add_cost(self, cost: cs.MX):
        """Add a custom cost term to the optimization problem.

        This method appends the provided cost term to the `cost_terms` list.
        During the `_finalize_problem` step, all cost terms in this list are
        combined using `cs.sum(cs.vertcat(*self.cost_terms))` and minimized
        as part of the objective function.

        Note: This method ensures that the optimization graph is still modifiable
        before adding the cost term. Once the problem is finalized (after the
        first call to `solve()`), no new cost terms can be added.
        Args:
            cost (cs.MX): The cost term to add.
        """
        self._ensure_graph_modifiable()
        self.cost_terms.append(cost)
