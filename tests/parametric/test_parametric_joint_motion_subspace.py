"""Test parametric joint motion subspace dimensions."""

import pytest
import numpy as np

from adam.parametric.model.parametric_factories.parametric_joint import ParametricJoint
from adam.numpy.numpy_like import SpatialMath
from adam.model import Pose


class SimpleLink:
    """Minimal link object for testing."""

    def __init__(self):
        self.name = "test_link"
        self.link_offset = 0.0
        self.inertial = type("obj", (object,), {"origin": Pose.zero(SpatialMath())})()

    def compute_joint_offset(self, joint, parent_offset):
        return 0.0

    def get_principal_length_parametric(self):
        return 1.0


class SimpleJoint:
    """Minimal joint object for testing."""

    def __init__(self, joint_type, axis):
        self.name = f"test_{joint_type}"
        self.parent = "link1"
        self.child = "link2"
        self.joint_type = joint_type
        self.axis = axis
        self.origin = type("obj", (object,), {"xyz": [0, 0, 0], "rpy": [0, 0, 0]})()
        self.limit = None


@pytest.fixture
def spatial_math():
    """Create a SpatialMath instance."""
    return SpatialMath()


@pytest.fixture
def simple_link(spatial_math):
    """Create a SimpleLink instance."""
    return SimpleLink()


def test_parametric_prismatic_joint_motion_subspace_dimension(spatial_math, simple_link):
    """Test that parametric prismatic joint motion subspace has correct dimension (6x1)."""
    joint = SimpleJoint("prismatic", [1, 0, 0])
    parametric_joint = ParametricJoint(joint, spatial_math, simple_link)
    motion_sub = parametric_joint.motion_subspace()

    # Check shape is (6, 1)
    assert motion_sub.shape == (6, 1), f"Expected shape (6, 1), got {motion_sub.shape}"

    # Check total size is 6
    assert motion_sub.array.size == 6, f"Expected 6 elements, got {motion_sub.array.size}"


def test_parametric_prismatic_joint_motion_subspace_values(spatial_math, simple_link):
    """Test that parametric prismatic joint motion subspace has correct values."""
    # Test X-axis prismatic joint
    joint = SimpleJoint("prismatic", [1, 0, 0])
    parametric_joint = ParametricJoint(joint, spatial_math, simple_link)
    motion_sub = parametric_joint.motion_subspace()

    expected = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    np.testing.assert_array_almost_equal(motion_sub.array, expected)

    # Test Y-axis prismatic joint
    joint = SimpleJoint("prismatic", [0, 1, 0])
    parametric_joint = ParametricJoint(joint, spatial_math, simple_link)
    motion_sub = parametric_joint.motion_subspace()

    expected = np.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    np.testing.assert_array_almost_equal(motion_sub.array, expected)

    # Test Z-axis prismatic joint
    joint = SimpleJoint("prismatic", [0, 0, 1])
    parametric_joint = ParametricJoint(joint, spatial_math, simple_link)
    motion_sub = parametric_joint.motion_subspace()

    expected = np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    np.testing.assert_array_almost_equal(motion_sub.array, expected)


def test_parametric_revolute_joint_motion_subspace_values(spatial_math, simple_link):
    """Test that parametric revolute joint motion subspace has correct values."""
    # Test X-axis revolute joint
    joint = SimpleJoint("revolute", [1, 0, 0])
    parametric_joint = ParametricJoint(joint, spatial_math, simple_link)
    motion_sub = parametric_joint.motion_subspace()

    expected = np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]])
    np.testing.assert_array_almost_equal(motion_sub.array, expected)
