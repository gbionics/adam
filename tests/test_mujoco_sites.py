"""Tests for MuJoCo site support in the MujocoModelFactory."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from adam.numpy.computations import KinDynComputations

# A minimal MuJoCo model with a two-link arm and sites attached to the bodies.
_MJCF_WITH_SITES = """\
<mujoco model="site_test">
  <worldbody>
    <body name="base" pos="0 0 0">
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <joint name="free_joint" type="free"/>
      <geom type="sphere" size="0.05"/>
      <site name="base_site" pos="0.1 0.2 0.3" quat="1 0 0 0"/>
      <body name="link1" pos="0 0 0.5">
        <inertial pos="0 0 0.25" mass="1" diaginertia="0.01 0.01 0.001"/>
        <joint name="joint1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.5"/>
        <site name="tip" pos="0 0 0.5" quat="1 0 0 0"/>
        <site name="mid" pos="0 0 0.25"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def kindyn():
    model = mujoco.MjModel.from_xml_string(_MJCF_WITH_SITES)
    kd = KinDynComputations(model, joints_name_list=["joint1"])
    return kd


def test_sites_appear_as_frames(kindyn):
    """Sites should be discovered as frames in the model."""
    frame_names = set(kindyn.rbdalgos.model.frames.keys())
    assert "base_site" in frame_names
    assert "tip" in frame_names
    assert "mid" in frame_names


def test_site_fk_identity_base(kindyn):
    """FK to a site at zero joint config with identity base transform."""
    H = np.eye(4)
    q = np.zeros(kindyn.NDoF)

    # 'tip' is at pos=(0,0,0.5) on 'link1' which is at pos=(0,0,0.5) from 'base'
    # So global position at zero config: (0, 0, 0.5 + 0.5) = (0, 0, 1.0)
    fk_tip = kindyn.forward_kinematics("tip", H, q)
    np.testing.assert_allclose(fk_tip[:3, 3], [0.0, 0.0, 1.0], atol=1e-10)

    # 'mid' is at pos=(0,0,0.25) on 'link1'
    # Global: (0, 0, 0.5 + 0.25) = (0, 0, 0.75)
    fk_mid = kindyn.forward_kinematics("mid", H, q)
    np.testing.assert_allclose(fk_mid[:3, 3], [0.0, 0.0, 0.75], atol=1e-10)

    # 'base_site' is at pos=(0.1, 0.2, 0.3) on 'base' which is at origin
    fk_base_site = kindyn.forward_kinematics("base_site", H, q)
    np.testing.assert_allclose(fk_base_site[:3, 3], [0.1, 0.2, 0.3], atol=1e-10)


def test_site_fk_moves_with_joint(kindyn):
    """FK to a site should change when the parent joint moves."""
    H = np.eye(4)
    q_zero = np.zeros(kindyn.NDoF)
    q_90 = np.array([np.pi / 2])  # 90 deg around Y axis

    fk_zero = kindyn.forward_kinematics("tip", H, q_zero)
    fk_90 = kindyn.forward_kinematics("tip", H, q_90)

    # At zero: tip at (0, 0, 1.0)
    # At 90 deg around Y: the 0.5 link + 0.5 site along Z rotates to X
    # Expected: (0.5, 0, 0.5)
    np.testing.assert_allclose(fk_zero[:3, 3], [0.0, 0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(fk_90[:3, 3], [0.5, 0.0, 0.5], atol=1e-6)


def test_site_does_not_appear_as_link(kindyn):
    """Sites should not appear in links (they are frames, not links)."""
    link_names = set(kindyn.rbdalgos.model.links.keys())
    assert "tip" not in link_names
    assert "mid" not in link_names
    assert "base_site" not in link_names
