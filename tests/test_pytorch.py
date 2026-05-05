import numpy as np
import pytest
import torch
from conftest import RobotCfg, State, to_numpy

from adam.pytorch import KinDynComputations


@pytest.fixture(scope="module")
def setup_test(tests_setup, device) -> KinDynComputations | RobotCfg | State:
    robot_cfg, state = tests_setup
    if robot_cfg.root_link is not None:
        pytest.skip("root link parametrization tested in numpy and casadi only")
    adam_kin_dyn = KinDynComputations(
        robot_cfg.model_path,
        robot_cfg.joints_name_list,
        device=device,
        dtype=torch.float64,
    )
    adam_kin_dyn.set_frame_velocity_representation(robot_cfg.velocity_representation)
    # convert state quantities to torch tensors
    state.H = torch.as_tensor(state.H, device=device)
    state.joints_pos = torch.as_tensor(state.joints_pos, device=device)
    state.base_vel = torch.as_tensor(state.base_vel, device=device)
    state.joints_vel = torch.as_tensor(state.joints_vel, device=device)
    state.gravity = torch.as_tensor(state.gravity, device=device)

    return adam_kin_dyn, robot_cfg, state


def test_mass_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_mass_matrix = robot_cfg.idyn_function_values.mass_matrix
    adam_mass_matrix = adam_kin_dyn.mass_matrix(state.H, state.joints_pos)
    assert to_numpy(adam_mass_matrix) - idyn_mass_matrix == pytest.approx(0.0, abs=1e-5)


def test_CMM(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_cmm = robot_cfg.idyn_function_values.centroidal_momentum_matrix
    adam_cmm = adam_kin_dyn.centroidal_momentum_matrix(state.H, state.joints_pos)
    assert to_numpy(adam_cmm) - idyn_cmm == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_com = robot_cfg.idyn_function_values.CoM_position
    adam_com = adam_kin_dyn.CoM_position(state.H, state.joints_pos)
    assert to_numpy(adam_com) - idyn_com == pytest.approx(0.0, abs=1e-5)


def test_CoM_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_com_jacobian = robot_cfg.idyn_function_values.CoM_jacobian
    adam_com_jacobian = adam_kin_dyn.CoM_jacobian(state.H, state.joints_pos)
    assert to_numpy(adam_com_jacobian) - idyn_com_jacobian == pytest.approx(
        0.0, abs=1e-5
    )


def test_total_mass(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_total_mass = robot_cfg.idyn_function_values.total_mass
    assert to_numpy(adam_kin_dyn.get_total_mass()) - idyn_total_mass == pytest.approx(
        0.0, abs=1e-5
    )


def test_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian
    adam_jacobian = adam_kin_dyn.jacobian("l_sole", state.H, state.joints_pos)
    assert to_numpy(adam_jacobian) - idyn_jacobian == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian_non_actuated
    adam_jacobian = adam_kin_dyn.jacobian("head", state.H, state.joints_pos)
    assert to_numpy(adam_jacobian) - idyn_jacobian == pytest.approx(0.0, abs=1e-5)


def test_jacobian_dot(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian_dot_nu = robot_cfg.idyn_function_values.jacobian_dot_nu
    adam_jacobian_dot_nu = adam_kin_dyn.jacobian_dot(
        "l_sole", state.H, state.joints_pos, state.base_vel, state.joints_vel
    ) @ torch.concatenate((state.base_vel, state.joints_vel), axis=0)
    assert idyn_jacobian_dot_nu - to_numpy(adam_jacobian_dot_nu) == pytest.approx(
        0.0, abs=1e-5
    )


def test_relative_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.relative_jacobian
    adam_jacobian = adam_kin_dyn.relative_jacobian("l_sole", state.joints_pos)
    assert idyn_jacobian - to_numpy(adam_jacobian) == pytest.approx(0.0, abs=1e-5)


def test_fk(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics
    adam_H = adam_kin_dyn.forward_kinematics("l_sole", state.H, state.joints_pos)
    assert idyn_H - to_numpy(adam_H) == pytest.approx(0.0, abs=1e-5)


def test_link_poses(setup_test):
    adam_kin_dyn, _robot_cfg, state = setup_test
    link_poses = adam_kin_dyn.link_poses(state.H, state.joints_pos)
    link_name = next(iter(link_poses))
    adam_H = adam_kin_dyn.forward_kinematics(link_name, state.H, state.joints_pos)

    assert to_numpy(link_poses[link_name]) - to_numpy(adam_H) == pytest.approx(
        0.0,
        abs=1e-5,
    )


def test_fk_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics_non_actuated
    adam_H = adam_kin_dyn.forward_kinematics("head", state.H, state.joints_pos)
    assert idyn_H - to_numpy(adam_H) == pytest.approx(0.0, abs=1e-5)


def test_bias_force(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_h = robot_cfg.idyn_function_values.bias_force
    adam_h = adam_kin_dyn.bias_force(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )
    assert idyn_h - to_numpy(adam_h) == pytest.approx(0.0, abs=1e-4)


def test_coriolis_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_coriolis = robot_cfg.idyn_function_values.coriolis_term
    adam_coriolis = adam_kin_dyn.coriolis_term(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )
    assert idyn_coriolis - to_numpy(adam_coriolis) == pytest.approx(0.0, abs=1e-4)


def test_gravity_term(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_gravity = robot_cfg.idyn_function_values.gravity_term
    adam_gravity = adam_kin_dyn.gravity_term(state.H, state.joints_pos)
    assert idyn_gravity - to_numpy(adam_gravity) == pytest.approx(0.0, abs=1e-4)


def test_aba(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    torques = (
        torch.randn(
            len(state.joints_pos),
            dtype=state.joints_pos.dtype,
            device=state.joints_pos.device,
        )
        * 10
    )
    H = state.H
    joints_pos = state.joints_pos
    base_vel = state.base_vel
    joints_vel = state.joints_vel

    wrenches = {
        "l_sole": torch.randn(
            6, dtype=state.joints_pos.dtype, device=state.joints_pos.device
        )
        * 10,
        "torso_1": torch.randn(
            6, dtype=state.joints_pos.dtype, device=state.joints_pos.device
        )
        * 10,
        "head": torch.randn(
            6, dtype=state.joints_pos.dtype, device=state.joints_pos.device
        )
        * 10,
    }

    adam_qdd = adam_kin_dyn.aba(
        base_transform=H,
        joint_positions=joints_pos,
        base_velocity=base_vel,
        joint_velocities=joints_vel,
        joint_torques=torques,
        external_wrenches=wrenches,
    )

    M = adam_kin_dyn.mass_matrix(H, joints_pos)
    h = adam_kin_dyn.bias_force(H, joints_pos, base_vel, joints_vel)

    generalized_external_wrenches = torch.zeros(
        6 + len(joints_pos), dtype=H.dtype, device=H.device
    )
    for frame, wrench in wrenches.items():
        J = adam_kin_dyn.jacobian(frame, H, joints_pos)
        generalized_external_wrenches += J.T @ wrench

    base_wrench = torch.zeros(6, dtype=H.dtype, device=H.device)
    full_tau = torch.concatenate([base_wrench, torques])
    residual = M @ adam_qdd + h - full_tau - generalized_external_wrenches

    assert to_numpy(residual) == pytest.approx(0.0, abs=1e-4)
