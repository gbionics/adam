"""Tests for child-Xform frame support in the USDModelFactory."""

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from adam.numpy.computations import KinDynComputations


def _build_two_link_stage_with_xform_frames(tmp_path):
    """Create a minimal USD stage with rigid bodies and child Xform frames."""
    usd_path = str(tmp_path / "test_frames.usda")
    stage = Usd.Stage.CreateNew(usd_path)

    # Root Xform with ArticulationRoot
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())

    # --- base_link (rigid body) ---
    base = UsdGeom.Xform.Define(stage, "/Robot/base_link")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(base.GetPrim())
    mass_api.GetMassAttr().Set(1.0)
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.01, 0.01, 0.01))

    # --- child_link (rigid body at offset) ---
    child = UsdGeom.Xform.Define(stage, "/Robot/child_link")
    UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
    mass_api2 = UsdPhysics.MassAPI.Apply(child.GetPrim())
    mass_api2.GetMassAttr().Set(1.0)
    mass_api2.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.01, 0.01, 0.01))

    # --- Revolute joint connecting base_link -> child_link ---
    rev_joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/joint1")
    rev_joint.GetBody0Rel().SetTargets(["/Robot/base_link"])
    rev_joint.GetBody1Rel().SetTargets(["/Robot/child_link"])
    rev_joint.GetAxisAttr().Set("Y")
    rev_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.5))
    rev_joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    rev_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rev_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # --- child Xform frame "tip" under child_link (not a rigid body) ---
    tip = UsdGeom.Xform.Define(stage, "/Robot/child_link/tip")
    tip.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.5))
    tip.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # --- child Xform frame "sensor_frame" under base_link ---
    sensor = UsdGeom.Xform.Define(stage, "/Robot/base_link/sensor_frame")
    sensor.AddTranslateOp().Set(Gf.Vec3f(0.1, 0.2, 0.3))
    sensor.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    stage.Save()
    return usd_path


@pytest.fixture(scope="module")
def kindyn(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("usd_frames")
    usd_path = _build_two_link_stage_with_xform_frames(tmp_path)
    kd = KinDynComputations.from_usd(usd_path, joints_name_list=["joint1"])
    return kd


def test_xform_frames_appear_as_frames(kindyn):
    """Child Xform prims should appear as frames in the model."""
    frame_names = set(kindyn.rbdalgos.model.frames.keys())
    assert "tip" in frame_names
    assert "sensor_frame" in frame_names


def test_xform_frame_fk_identity_base(kindyn):
    """FK to child Xform frames at zero joint config."""
    H = np.eye(4)
    q = np.zeros(kindyn.NDoF)

    # 'tip' at (0,0,0.5) on child_link, which is at (0,0,0.5) from base
    # Total: (0, 0, 1.0)
    fk_tip = kindyn.forward_kinematics("tip", H, q)
    np.testing.assert_allclose(fk_tip[:3, 3], [0.0, 0.0, 1.0], atol=1e-10)

    # 'sensor_frame' at (0.1, 0.2, 0.3) on base_link at origin
    fk_sensor = kindyn.forward_kinematics("sensor_frame", H, q)
    np.testing.assert_allclose(fk_sensor[:3, 3], [0.1, 0.2, 0.3], atol=1e-10)


def test_xform_frame_fk_moves_with_joint(kindyn):
    """FK to child Xform frame changes when parent joint moves."""
    H = np.eye(4)
    q_90 = np.array([np.pi / 2])

    fk_tip = kindyn.forward_kinematics("tip", H, q_90)
    # At 90 deg around Y: (0,0,0.5) link + (0,0,0.5) frame → X direction
    np.testing.assert_allclose(fk_tip[:3, 3], [0.5, 0.0, 0.5], atol=1e-6)

    # sensor_frame on base_link is unaffected by joint1
    fk_sensor = kindyn.forward_kinematics("sensor_frame", H, q_90)
    np.testing.assert_allclose(fk_sensor[:3, 3], [0.1, 0.2, 0.3], atol=1e-10)


def test_xform_frame_not_in_links(kindyn):
    """Child Xform frames should not appear as links."""
    link_names = set(kindyn.rbdalgos.model.links.keys())
    assert "tip" not in link_names
    assert "sensor_frame" not in link_names
