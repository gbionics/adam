"""Test joint motion subspace dimensions."""

import pytest
import numpy as np

from adam.model.std_factories.std_joint import StdJoint
from adam.numpy.numpy_like import SpatialMath


class SimpleJoint:
    """Minimal joint object for testing."""

    def __init__(self, joint_type, axis):
        self.name = f"test_{joint_type}"
        self.parent = "link1"
        self.child = "link2"
        self.joint_type = joint_type
        self.axis = axis
        self.origin = None
        self.limit = None


@pytest.fixture
def spatial_math():
    """Create a SpatialMath instance."""
    return SpatialMath()


def test_prismatic_joint_motion_subspace_dimension(spatial_math):
    """Test that prismatic joint motion subspace has correct dimension (6x1)."""
    joint = SimpleJoint("prismatic", [1, 0, 0])
    std_joint = StdJoint(joint, spatial_math)
    motion_sub = std_joint.motion_subspace()

    # Check shape is (6, 1)
    assert motion_sub.shape == (6, 1), f"Expected shape (6, 1), got {motion_sub.shape}"

    # Check total size is 6
    assert motion_sub.array.size == 6, f"Expected 6 elements, got {motion_sub.array.size}"


def test_prismatic_joint_motion_subspace_values(spatial_math):
    """Test that prismatic joint motion subspace has correct values."""
    # Test X-axis prismatic joint
    joint = SimpleJoint("prismatic", [1, 0, 0])
    std_joint = StdJoint(joint, spatial_math)
    motion_sub = std_joint.motion_subspace()

    expected = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    np.testing.assert_array_almost_equal(motion_sub.array, expected)

    # Test Y-axis prismatic joint
    joint = SimpleJoint("prismatic", [0, 1, 0])
    std_joint = StdJoint(joint, spatial_math)
    motion_sub = std_joint.motion_subspace()

    expected = np.array([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
    np.testing.assert_array_almost_equal(motion_sub.array, expected)

    # Test Z-axis prismatic joint
    joint = SimpleJoint("prismatic", [0, 0, 1])
    std_joint = StdJoint(joint, spatial_math)
    motion_sub = std_joint.motion_subspace()

    expected = np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    np.testing.assert_array_almost_equal(motion_sub.array, expected)


def test_revolute_joint_motion_subspace_dimension(spatial_math):
    """Test that revolute joint motion subspace has correct dimension (6x1)."""
    joint = SimpleJoint("revolute", [1, 0, 0])
    std_joint = StdJoint(joint, spatial_math)
    motion_sub = std_joint.motion_subspace()

    # Check shape is (6, 1)
    assert motion_sub.shape == (6, 1), f"Expected shape (6, 1), got {motion_sub.shape}"

    # Check total size is 6
    assert motion_sub.array.size == 6, f"Expected 6 elements, got {motion_sub.array.size}"


def test_revolute_joint_motion_subspace_values(spatial_math):
    """Test that revolute joint motion subspace has correct values."""
    # Test X-axis revolute joint
    joint = SimpleJoint("revolute", [1, 0, 0])
    std_joint = StdJoint(joint, spatial_math)
    motion_sub = std_joint.motion_subspace()

    expected = np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]])
    np.testing.assert_array_almost_equal(motion_sub.array, expected)

    # Test Y-axis revolute joint
    joint = SimpleJoint("revolute", [0, 1, 0])
    std_joint = StdJoint(joint, spatial_math)
    motion_sub = std_joint.motion_subspace()

    expected = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [0.0]])
    np.testing.assert_array_almost_equal(motion_sub.array, expected)

    # Test Z-axis revolute joint
    joint = SimpleJoint("revolute", [0, 0, 1])
    std_joint = StdJoint(joint, spatial_math)
    motion_sub = std_joint.motion_subspace()

    expected = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
    np.testing.assert_array_almost_equal(motion_sub.array, expected)


def test_fixed_joint_motion_subspace(spatial_math):
    """Test that fixed joint motion subspace is all zeros."""
    joint = SimpleJoint("fixed", None)
    std_joint = StdJoint(joint, spatial_math)
    motion_sub = std_joint.motion_subspace()

    # Check shape is (6, 1)
    assert motion_sub.shape == (6, 1), f"Expected shape (6, 1), got {motion_sub.shape}"

    # Check all zeros
    expected = np.zeros((6, 1))
    np.testing.assert_array_almost_equal(motion_sub.array, expected)
