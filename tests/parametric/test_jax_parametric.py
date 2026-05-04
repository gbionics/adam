import numpy as np
import pytest
from conftest import RobotCfg, State

from adam.parametric.jax import KinDynComputationsParametric


@pytest.fixture(scope="module")
def setup_test(tests_setup) -> KinDynComputationsParametric | RobotCfg | State:
    robot_cfg, state = tests_setup
    # skip the tests if the model is not the StickBot
    if robot_cfg.robot_name != "StickBot":
        pytest.skip("Skipping the test because the model is not StickBot")
    if robot_cfg.root_link is not None:
        pytest.skip("root link parametrization tested in numpy and casadi only")
    link_name_list = ["chest"]
    adam_kin_dyn = KinDynComputationsParametric(
        robot_cfg.model_path, robot_cfg.joints_name_list, link_name_list
    )
    # This is the original density value associated to the chest link, computed as mass/volume
    original_density = [628.0724496264945]
    original_length = np.ones(len(link_name_list))
    adam_kin_dyn.set_frame_velocity_representation(robot_cfg.velocity_representation)
    return adam_kin_dyn, robot_cfg, state, original_density, original_length


def test_mass_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_mass_matrix = robot_cfg.idyn_function_values.mass_matrix
    adam_mass_matrix = adam_kin_dyn.mass_matrix(
        state.H, state.joints_pos, original_length, original_density
    )
    assert adam_mass_matrix - idyn_mass_matrix == pytest.approx(0.0, abs=1e-5)


def test_CMM(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_cmm = robot_cfg.idyn_function_values.centroidal_momentum_matrix
    adam_cmm = adam_kin_dyn.centroidal_momentum_matrix(
        state.H, state.joints_pos, original_length, original_density
    )
    assert adam_cmm - idyn_cmm == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_com = robot_cfg.idyn_function_values.CoM_position
    adam_com = adam_kin_dyn.CoM_position(
        state.H, state.joints_pos, original_length, original_density
    )
    assert adam_com - idyn_com == pytest.approx(0.0, abs=1e-5)


def test_total_mass(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_total_mass = robot_cfg.idyn_function_values.total_mass
    assert adam_kin_dyn.get_total_mass(
        original_length, original_density
    ) - idyn_total_mass == pytest.approx(0.0, abs=1e-5)


def test_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian
    adam_jacobian = adam_kin_dyn.jacobian(
        "l_sole", state.H, state.joints_pos, original_length, original_density
    )
    assert adam_jacobian - idyn_jacobian == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian_non_actuated
    adam_jacobian = adam_kin_dyn.jacobian(
        "head", state.H, state.joints_pos, original_length, original_density
    )
    assert adam_jacobian - idyn_jacobian == pytest.approx(0.0, abs=1e-5)


def test_jacobian_dot(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_jacobian_dot_nu = robot_cfg.idyn_function_values.jacobian_dot_nu
    adam_jacobian_dot_nu = adam_kin_dyn.jacobian_dot(
        "l_sole",
        state.H,
        state.joints_pos,
        state.base_vel,
        state.joints_vel,
        original_length,
        original_density,
    ) @ np.concatenate((state.base_vel, state.joints_vel))
    assert idyn_jacobian_dot_nu - adam_jacobian_dot_nu == pytest.approx(0.0, abs=1e-5)


def test_relative_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.relative_jacobian
    adam_jacobian = adam_kin_dyn.relative_jacobian(
        "l_sole", state.joints_pos, original_length, original_density
    )
    assert idyn_jacobian - adam_jacobian == pytest.approx(0.0, abs=1e-5)


def test_fk(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics
    adam_H = adam_kin_dyn.forward_kinematics(
        "l_sole", state.H, state.joints_pos, original_length, original_density
    )
    assert idyn_H - adam_H == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics_non_actuated
    adam_H = adam_kin_dyn.forward_kinematics(
        "head", state.H, state.joints_pos, original_length, original_density
    )
    assert idyn_H - adam_H == pytest.approx(0.0, abs=1e-5)


def test_bias_force(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_bias_force = robot_cfg.idyn_function_values.bias_force
    adam_bias_force = adam_kin_dyn.bias_force(
        state.H,
        state.joints_pos,
        state.base_vel,
        state.joints_vel,
        original_length,
        original_density,
    )
    assert idyn_bias_force - adam_bias_force == pytest.approx(0.0, abs=1e-4)


def test_coriolis_term(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_coriolis_gravity = robot_cfg.idyn_function_values.coriolis_term
    adam_coriolis_gravity = adam_kin_dyn.coriolis_term(
        state.H,
        state.joints_pos,
        state.base_vel,
        state.joints_vel,
        original_length,
        original_density,
    )
    assert idyn_coriolis_gravity - adam_coriolis_gravity == pytest.approx(0.0, abs=1e-4)


def test_gravity_term(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_gravity = robot_cfg.idyn_function_values.gravity_term
    adam_gravity = adam_kin_dyn.gravity_term(
        state.H, state.joints_pos, original_length, original_density
    )
    assert np.allclose(idyn_gravity, adam_gravity, atol=1e-4)
