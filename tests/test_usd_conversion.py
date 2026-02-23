import numpy as np
import pytest
from conftest import RobotCfg, State

pytest.importorskip("pxr")

from adam.model import Model, build_model_factory
from adam.numpy.computations import KinDynComputations
from adam.numpy.numpy_like import SpatialMath


@pytest.fixture(scope="module")
def setup_test(
    tests_setup, tmp_path_factory
) -> tuple[KinDynComputations, KinDynComputations, RobotCfg, State]:
    robot_cfg, state = tests_setup

    out_dir = tmp_path_factory.mktemp(f"usd_{robot_cfg.robot_name}")
    usd_path = out_dir / f"{robot_cfg.robot_name}.usda"

    model_factory = build_model_factory(
        description=robot_cfg.model_path,
        math=SpatialMath(),
    )
    model = Model.build(
        factory=model_factory,
        joints_name_list=robot_cfg.joints_name_list,
    )
    out = model.to_usd(usd_path, robot_prim_path="/Robot")

    kd_urdf = KinDynComputations.from_urdf(
        robot_cfg.model_path,
        joints_name_list=robot_cfg.joints_name_list,
    )
    kd_usd = KinDynComputations.from_usd(
        out,
        robot_prim_path="/Robot",
        joints_name_list=robot_cfg.joints_name_list,
    )

    kd_urdf.set_frame_velocity_representation(robot_cfg.velocity_representation)
    kd_usd.set_frame_velocity_representation(robot_cfg.velocity_representation)

    return kd_urdf, kd_usd, robot_cfg, state


def test_model_to_usd(setup_test):
    kd_urdf, kd_usd, _robot_cfg, state = setup_test

    assert kd_urdf.NDoF == kd_usd.NDoF
    assert set(kd_urdf.rbdalgos.model.links) == set(kd_usd.rbdalgos.model.links)

    H_base = state.H
    q = state.joints_pos

    links = list(kd_urdf.rbdalgos.model.links.keys())
    for frame in links[: min(5, len(links))]:
        fk_urdf = kd_urdf.forward_kinematics(frame, H_base, q)
        fk_usd = kd_usd.forward_kinematics(frame, H_base, q)
        np.testing.assert_allclose(fk_usd, fk_urdf, atol=1e-5, rtol=1e-5)

    tip = links[-1]
    jac_urdf = kd_urdf.jacobian(tip, H_base, q)
    jac_usd = kd_usd.jacobian(tip, H_base, q)
    np.testing.assert_allclose(jac_usd, jac_urdf, atol=1e-5, rtol=1e-5)

    com_urdf = kd_urdf.CoM_position(H_base, q)
    com_usd = kd_usd.CoM_position(H_base, q)
    np.testing.assert_allclose(com_usd, com_urdf, atol=1e-5, rtol=1e-5)
