import casadi as cs
import numpy as np
import pytest
from conftest import RobotCfg, State, compute_idyntree_values

from adam.model import Model
from adam.model.conversions.idyntree import to_idyntree_model
from adam.numpy.numpy_like import SpatialMath
from adam.parametric.casadi import KinDynComputationsParametric
from adam.parametric.model.parametric_factories.parametric_model import (
    URDFParametricModelFactory,
)


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
    adam_kin_dyn.set_frame_velocity_representation(robot_cfg.velocity_representation)
    original_density = [628.0724496264945]
    original_length = np.ones(len(link_name_list))
    factory = URDFParametricModelFactory(
        path=robot_cfg.model_path,
        math=SpatialMath(),
        links_name_list=link_name_list,
        length_multiplier=original_length,
        densities=original_density,
    )
    model = Model.build(factory=factory, joints_name_list=robot_cfg.joints_name_list)

    robot_cfg.kin_dyn.loadRobotModel(to_idyntree_model(model))
    robot_cfg.idyn_function_values = compute_idyntree_values(robot_cfg.kin_dyn, state)
    return adam_kin_dyn, robot_cfg, state, original_density, original_length


def test_mass_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_mass_matrix = robot_cfg.idyn_function_values.mass_matrix
    adam_mass_matrix = cs.DM(
        adam_kin_dyn.mass_matrix_fun()(
            state.H, state.joints_pos, original_length, original_density
        )
    )
    assert adam_mass_matrix - idyn_mass_matrix == pytest.approx(0.0, abs=1e-5)


def test_CMM(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_cmm = robot_cfg.idyn_function_values.centroidal_momentum_matrix
    adam_cmm = cs.DM(
        adam_kin_dyn.centroidal_momentum_matrix_fun()(
            state.H, state.joints_pos, original_length, original_density
        )
    )
    assert adam_cmm - idyn_cmm == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_com = robot_cfg.idyn_function_values.CoM_position
    adam_com = cs.DM(
        adam_kin_dyn.CoM_position_fun()(
            state.H, state.joints_pos, original_length, original_density
        )
    )
    assert adam_com - idyn_com == pytest.approx(0.0, abs=1e-5)


def test_total_mass(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_total_mass = robot_cfg.idyn_function_values.total_mass
    adam_total_mass = cs.DM(
        adam_kin_dyn.get_total_mass()(original_length, original_density)
    )
    assert adam_total_mass - idyn_total_mass == pytest.approx(0.0, abs=1e-5)


def test_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian
    adam_jacobian = cs.DM(
        adam_kin_dyn.jacobian_fun("l_sole")(
            state.H, state.joints_pos, original_length, original_density
        )
    )
    assert adam_jacobian - idyn_jacobian == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian_non_actuated
    adam_jacobian = cs.DM(
        adam_kin_dyn.jacobian_fun("head")(
            state.H, state.joints_pos, original_length, original_density
        )
    )
    assert adam_jacobian - idyn_jacobian == pytest.approx(0.0, abs=1e-5)


def test_jacobian_dot(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_jacobian_dot_nu = robot_cfg.idyn_function_values.jacobian_dot_nu
    adam_jacobian_dot_nu = cs.DM(
        adam_kin_dyn.jacobian_dot_fun("l_sole")(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
            original_length,
            original_density,
        )
    ) @ np.concatenate((state.base_vel, state.joints_vel))
    assert idyn_jacobian_dot_nu - adam_jacobian_dot_nu == pytest.approx(0.0, abs=1e-5)


def test_relative_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.relative_jacobian
    adam_jacobian = cs.DM(
        adam_kin_dyn.relative_jacobian_fun("l_sole")(
            state.joints_pos, original_length, original_density
        )
    )
    assert idyn_jacobian - adam_jacobian == pytest.approx(0.0, abs=1e-5)


def test_fk(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics
    adam_H = cs.DM(
        adam_kin_dyn.forward_kinematics_fun("l_sole")(
            state.H, state.joints_pos, original_length, original_density
        )
    )
    assert idyn_H - adam_H == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics_non_actuated
    adam_H = cs.DM(
        adam_kin_dyn.forward_kinematics_fun("head")(
            state.H, state.joints_pos, original_length, original_density
        )
    )
    assert idyn_H - adam_H == pytest.approx(0.0, abs=1e-5)


def test_bias_force(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_h = robot_cfg.idyn_function_values.bias_force
    adam_h = cs.DM(
        adam_kin_dyn.bias_force_fun()(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
            original_length,
            original_density,
        )
    )
    assert idyn_h - adam_h == pytest.approx(0.0, abs=1e-5)


def test_coriolis_term(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_coriolis = robot_cfg.idyn_function_values.coriolis_term
    adam_coriolis = cs.DM(
        adam_kin_dyn.coriolis_term_fun()(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
            original_length,
            original_density,
        )
    )
    assert idyn_coriolis - adam_coriolis == pytest.approx(0.0, abs=1e-5)


def test_gravity_term(setup_test):
    adam_kin_dyn, robot_cfg, state, original_density, original_length = setup_test
    idyn_gravity = robot_cfg.idyn_function_values.gravity_term
    adam_gravity = cs.DM(
        adam_kin_dyn.gravity_term_fun()(
            state.H, state.joints_pos, original_length, original_density
        )
    )
    assert idyn_gravity - adam_gravity == pytest.approx(0.0, abs=1e-5)
