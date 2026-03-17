import numpy as np
import pytest
from conftest import RobotCfg, State

import idyntree.bindings as idyntree
import urdf_usd_converter
from adam.model import Model, build_model_factory
from adam.numpy import KinDynComputations
from adam.numpy.numpy_like import SpatialMath


@pytest.fixture(scope="module")
def setup_test(tests_setup, tmp_path_factory) -> KinDynComputations | RobotCfg | State:
    robot_cfg, state = tests_setup

    out_dir = tmp_path_factory.mktemp("newton_usd")

    # Use the Newton urdf-usd-converter to convert URDF → USD.
    converter = urdf_usd_converter.Converter()
    asset = converter.convert(
        str(robot_cfg.model_path),
        str(out_dir / robot_cfg.robot_name),
    )

    # Let ADAM auto-discover the articulation root in the Newton-generated USD.
    adam_kin_dyn = KinDynComputations.from_usd(
        asset.path,
        joints_name_list=robot_cfg.joints_name_list,
    )
    adam_kin_dyn.set_frame_velocity_representation(robot_cfg.velocity_representation)

    return adam_kin_dyn, robot_cfg, state


def test_mass_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_mass_matrix = robot_cfg.idyn_function_values.mass_matrix
    adam_mass_matrix = adam_kin_dyn.mass_matrix(state.H, state.joints_pos)
    assert adam_mass_matrix - idyn_mass_matrix == pytest.approx(0.0, abs=2e-2)


def test_CMM(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_cmm = robot_cfg.idyn_function_values.centroidal_momentum_matrix
    adam_cmm = adam_kin_dyn.centroidal_momentum_matrix(state.H, state.joints_pos)
    assert adam_cmm - idyn_cmm == pytest.approx(0.0, abs=2e-2)


def test_CoM_pos(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_com = robot_cfg.idyn_function_values.CoM_position
    adam_com = adam_kin_dyn.CoM_position(state.H, state.joints_pos)
    assert adam_com - idyn_com == pytest.approx(0.0, abs=1e-5)


def test_CoM_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_com_jacobian = robot_cfg.idyn_function_values.CoM_jacobian
    adam_com_jacobian = adam_kin_dyn.CoM_jacobian(state.H, state.joints_pos)
    assert adam_com_jacobian - idyn_com_jacobian == pytest.approx(0.0, abs=1e-5)


def test_total_mass(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_total_mass = robot_cfg.idyn_function_values.total_mass
    assert adam_kin_dyn.get_total_mass() - idyn_total_mass == pytest.approx(
        0.0, abs=1e-5
    )


def test_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian
    adam_jacobian = adam_kin_dyn.jacobian("l_sole", state.H, state.joints_pos)
    assert adam_jacobian - idyn_jacobian == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian_non_actuated
    adam_jacobian = adam_kin_dyn.jacobian("head", state.H, state.joints_pos)
    assert adam_jacobian - idyn_jacobian == pytest.approx(0.0, abs=1e-5)


def test_jacobian_dot(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian_dot_nu = robot_cfg.idyn_function_values.jacobian_dot_nu
    adam_jacobian_dot_nu = adam_kin_dyn.jacobian_dot(
        "l_sole", state.H, state.joints_pos, state.base_vel, state.joints_vel
    ) @ np.concatenate((state.base_vel, state.joints_vel))
    assert idyn_jacobian_dot_nu - adam_jacobian_dot_nu == pytest.approx(0.0, abs=1e-5)


def test_relative_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.relative_jacobian
    adam_jacobian = adam_kin_dyn.relative_jacobian("l_sole", state.joints_pos)
    assert idyn_jacobian - adam_jacobian == pytest.approx(0.0, abs=1e-5)


def test_fk(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics
    adam_H = adam_kin_dyn.forward_kinematics("l_sole", state.H, state.joints_pos)
    assert idyn_H - adam_H == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics_non_actuated
    adam_H = adam_kin_dyn.forward_kinematics("head", state.H, state.joints_pos)
    assert idyn_H - adam_H == pytest.approx(0.0, abs=1e-5)


def test_bias_force(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_h = robot_cfg.idyn_function_values.bias_force
    adam_h = adam_kin_dyn.bias_force(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )
    assert idyn_h - adam_h == pytest.approx(0.0, abs=2.5e-1)


def test_coriolis_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_coriolis = robot_cfg.idyn_function_values.coriolis_term
    adam_coriolis = adam_kin_dyn.coriolis_term(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )
    assert idyn_coriolis - adam_coriolis == pytest.approx(0.0, abs=2.5e-1)


def test_gravity_term(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_gravity = robot_cfg.idyn_function_values.gravity_term
    adam_gravity = adam_kin_dyn.gravity_term(state.H, state.joints_pos)
    assert idyn_gravity - adam_gravity == pytest.approx(0.0, abs=1e-4)


def test_aba(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    torques = np.random.randn(len(state.joints_pos)) * 10
    H = state.H
    joints_pos = state.joints_pos
    base_vel = state.base_vel
    joints_vel = state.joints_vel

    wrenches = {
        "l_sole": np.random.randn(6) * 10,
        "torso_1": np.random.randn(6) * 10,
        "head": np.random.randn(6) * 10,
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

    generalized_external_wrenches = np.zeros(6 + len(joints_pos))
    for frame, wrench in wrenches.items():
        J = adam_kin_dyn.jacobian(frame, H, joints_pos)
        generalized_external_wrenches += J.T @ wrench

    base_wrench = np.zeros(6)
    full_tau = np.concatenate([base_wrench, torques])
    residual = M @ adam_qdd + h - full_tau - generalized_external_wrenches

    assert residual == pytest.approx(0.0, abs=1e-4)


@pytest.mark.xfail(
    reason=(
        "urdf-usd-converter currently preserves rotated inertial tensors using the "
        "opposite URDF inertial-axis convention from adam/idyntree for this minimal case"
    ),
    strict=False,
)
def test_newton_usd_preserves_rotated_inertial_frame(tmp_path):
    urdf = """
    <robot name="rotated_inertia">
      <link name="base">
        <inertial>
          <origin xyz="0.1 -0.2 0.3" rpy="0.4 -0.3 0.2"/>
          <mass value="3.5"/>
          <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.6"/>
        </inertial>
      </link>
      <link name="tip"/>
      <joint name="base_to_tip" type="fixed">
        <parent link="base"/>
        <child link="tip"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
      </joint>
    </robot>
    """.strip()

    urdf_path = tmp_path / "rotated_inertia.urdf"
    urdf_path.write_text(urdf)

    converter = urdf_usd_converter.Converter()
    asset = converter.convert(str(urdf_path), str(tmp_path / "rotated_inertia"))

    math = SpatialMath()
    model_urdf = Model.build(
        factory=build_model_factory(description=str(urdf_path), math=math),
        joints_name_list=[],
    )
    model_usd = Model.build(
        factory=build_model_factory(description=asset.path, math=math),
        joints_name_list=[],
    )

    idyntree_loader = idyntree.ModelLoader()
    assert idyntree_loader.loadModelFromFile(str(urdf_path))
    idyntree_model = idyntree_loader.model()
    idyntree_base = idyntree_model.getLink(idyntree_model.getLinkIndex("base"))

    original_link = model_urdf.links["base"]
    usd_link = model_usd.links["base"]

    np.testing.assert_allclose(
        original_link.inertial.origin.xyz.array,
        idyntree_base.getInertia().getCenterOfMass().toNumPy(),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        original_link.spatial_inertia().array,
        idyntree_base.getInertia().asMatrix().toNumPy(),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        usd_link.inertial.origin.xyz.array,
        original_link.inertial.origin.xyz.array,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        usd_link.spatial_inertia().array,
        original_link.spatial_inertia().array,
        atol=1e-6,
        rtol=1e-6,
    )
