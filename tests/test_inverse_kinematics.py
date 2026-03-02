import numpy as np
import pytest
from conftest import RobotCfg, State
from liecasadi import SO3

from adam.casadi import KinDynComputations
from adam.casadi.inverse_kinematics import InverseKinematics

# Root link used to constrain the floating base
ROOT_LINK = "root_link"

# End-effector frames used as IK targets — exist in both iCubGenova04 and StickBot
IK_TARGET_FRAMES = ["l_hand", "r_hand", "l_ankle_2", "r_ankle_2"]

TOL = 1e-4  # metres / radians  (warm start)


def _fk_poses(
    kd: KinDynComputations, base_transform: np.ndarray, joint_pos: np.ndarray
) -> dict:
    """Return {frame: (p, R)} for the root link and all end-effector frames."""
    poses = {}
    for frame in [ROOT_LINK] + IK_TARGET_FRAMES:
        H = np.array(kd.forward_kinematics_fun(frame)(base_transform, joint_pos))
        poses[frame] = (H[:3, 3], H[:3, :3])
    return poses


def _build_ik_position(robot_cfg: RobotCfg) -> InverseKinematics:
    """IK with position-only targets for end-effectors (+ full pose on root link)."""
    ik = InverseKinematics(
        robot_cfg.model_path,
        robot_cfg.joints_name_list,
        joint_limits_active=True,
        solver_settings={"ipopt": {"print_level": 0, "tol": 1e-15, "max_iter": 4000}},
    )
    ik.add_target_pose(ROOT_LINK, as_soft_constraint=True, weight=1e8)
    for frame in IK_TARGET_FRAMES:
        ik.add_target_position(frame, as_soft_constraint=True, weight=1e8)
    return ik


def _build_ik_orientation(robot_cfg: RobotCfg) -> InverseKinematics:
    """IK with orientation-only targets for end-effectors (+ full pose on root link)."""
    ik = InverseKinematics(
        robot_cfg.model_path,
        robot_cfg.joints_name_list,
        joint_limits_active=True,
        solver_settings={"ipopt": {"print_level": 0, "tol": 1e-15, "max_iter": 4000}},
    )
    ik.add_target_pose(ROOT_LINK, as_soft_constraint=True, weight=1e8)
    for frame in IK_TARGET_FRAMES:
        ik.add_target_orientation(frame, as_soft_constraint=True, weight=1e8)
    return ik


def _build_ik_pose(robot_cfg: RobotCfg) -> InverseKinematics:
    """IK with full pose targets for end-effectors (+ full pose on root link)."""
    ik = InverseKinematics(
        robot_cfg.model_path,
        robot_cfg.joints_name_list,
        joint_limits_active=True,
        solver_settings={"ipopt": {"print_level": 0, "tol": 1e-15, "max_iter": 4000}},
    )
    ik.add_target_pose(ROOT_LINK, as_soft_constraint=True, weight=1e8)
    for frame in IK_TARGET_FRAMES:
        ik.add_target_pose(frame, as_soft_constraint=True, weight=1e8)
    return ik


@pytest.fixture(scope="module")
def ik_reference(
    tests_setup,
) -> tuple[KinDynComputations, RobotCfg, np.ndarray, np.ndarray]:
    """Expose kd, robot_cfg, H_ref and q_ref (clamped to joint limits) from tests_setup."""
    robot_cfg, state = tests_setup
    kd = KinDynComputations(robot_cfg.model_path, robot_cfg.joints_name_list)

    lower = np.array(
        [kd.rbdalgos.model.joints[j].limit.lower for j in robot_cfg.joints_name_list]
    )
    upper = np.array(
        [kd.rbdalgos.model.joints[j].limit.upper for j in robot_cfg.joints_name_list]
    )
    q_ref = np.clip(state.joints_pos, lower, upper)
    H_ref = state.H  # base pose from tests_setup

    return kd, robot_cfg, H_ref, q_ref


def test_ik_position(ik_reference):
    """Position-only targets: IK should match FK positions within tolerance."""
    kd, robot_cfg, H_ref, q_ref = ik_reference
    poses = _fk_poses(kd, H_ref, q_ref)

    ik = _build_ik_position(robot_cfg)
    ik.update_target(ROOT_LINK, poses[ROOT_LINK])
    for frame in IK_TARGET_FRAMES:
        ik.update_target_position(frame, poses[frame][0])

    noise = np.random.default_rng(0).normal(scale=0.01, size=q_ref.shape)
    ik.set_initial_guess(H_ref, q_ref + noise)
    ik.solve()
    H_sol, q_sol = ik.get_solution()

    sol_poses = _fk_poses(kd, H_sol, q_sol)
    for frame in IK_TARGET_FRAMES:
        p_des = poses[frame][0]
        p_sol = sol_poses[frame][0]
        pos_err = np.linalg.norm(p_sol - p_des)
        assert pos_err < TOL, f"[{frame}] position error too large: {pos_err:.6f} m"


def test_ik_orientation(ik_reference):
    """Orientation-only targets: IK should match FK orientations within tolerance."""
    kd, robot_cfg, H_ref, q_ref = ik_reference
    poses = _fk_poses(kd, H_ref, q_ref)

    ik = _build_ik_orientation(robot_cfg)
    ik.update_target(ROOT_LINK, poses[ROOT_LINK])
    for frame in IK_TARGET_FRAMES:
        ik.update_target_orientation(frame, poses[frame][1])

    noise = np.random.default_rng(0).normal(scale=0.01, size=q_ref.shape)
    ik.set_initial_guess(H_ref, q_ref + noise)
    ik.solve()
    H_sol, q_sol = ik.get_solution()

    sol_poses = _fk_poses(kd, H_sol, q_sol)
    for frame in IK_TARGET_FRAMES:
        R_des = poses[frame][1]
        R_sol = sol_poses[frame][1]
        R_rel = SO3.from_matrix(R_des).inverse() * SO3.from_matrix(R_sol)
        rot_err = float(np.linalg.norm(np.array(R_rel.log().vec)))
        assert rot_err < TOL, f"[{frame}] rotation error too large: {rot_err:.6f} rad"


def test_ik_pose(ik_reference):
    """Full pose targets: IK should match FK poses and joint angles within tolerance."""
    kd, robot_cfg, H_ref, q_ref = ik_reference
    poses = _fk_poses(kd, H_ref, q_ref)

    ik = _build_ik_pose(robot_cfg)
    for frame, (p, R) in poses.items():
        ik.update_target(frame, (p, R))

    noise = np.random.default_rng(0).normal(scale=0.01, size=q_ref.shape)
    ik.set_initial_guess(H_ref, q_ref + noise)
    ik.solve()
    H_sol, q_sol = ik.get_solution()

    sol_poses = _fk_poses(kd, H_sol, q_sol)
    for frame in IK_TARGET_FRAMES:
        p_des, R_des = poses[frame]
        p_sol, R_sol = sol_poses[frame]

        pos_err = np.linalg.norm(p_sol - p_des)
        R_rel = SO3.from_matrix(R_des).inverse() * SO3.from_matrix(R_sol)
        rot_err = float(np.linalg.norm(np.array(R_rel.log().vec)))

        assert pos_err < TOL, f"[{frame}] position error too large: {pos_err:.6f} m"
        assert rot_err < TOL, f"[{frame}] rotation error too large: {rot_err:.6f} rad"

    joint_err = np.linalg.norm(q_sol - q_ref)
    assert joint_err < TOL, f"joint error too large: {joint_err:.6f} rad"
