"""Direct tests of the joint transforms in ``adam.core.spatial_math``.

Every batched result is compared with (a) stacking independent unbatched
evaluations and (b) an independent NumPy reference written in this file.
"""

import numpy as np
import pytest

try:  # keep float64 for JAX, as the other test modules do
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

from adam.core.array_api_math import spec_from_reference

Q_SHAPES = [(), (1,), (5,), (2, 3)]
XYZ = np.array([0.1, -0.2, 0.3])
RPY = np.array([0.3, -0.5, 0.7])
AXIS = np.array([0.0, 0.6, 0.8])


def _backend(name):
    if name == "numpy":
        from adam.numpy.numpy_like import SpatialMath

        ref = np.array(0.0, dtype=np.float64)
        return SpatialMath(spec_from_reference(ref)), lambda x: np.asarray(x)
    if name == "jax":
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        from adam.jax.jax_like import SpatialMath

        ref = jnp.array(0.0, dtype=jnp.float64)
        return SpatialMath(spec_from_reference(ref)), lambda x: np.asarray(x)
    if name == "torch":
        import torch
        from adam.pytorch.torch_like import SpatialMath

        ref = torch.tensor(0.0, dtype=torch.float64)
        return (
            SpatialMath(spec_from_reference(ref)),
            lambda x: x.detach().cpu().numpy(),
        )
    raise ValueError(name)


@pytest.fixture(params=["numpy", "jax", "torch"])
def backend(request):
    pytest.importorskip(request.param)
    return _backend(request.param)


def _rot_rpy(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rot_axis(axis, q):
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def _ref_H(kind, q):
    H = np.eye(4)
    H[:3, :3] = _rot_rpy(RPY)
    H[:3, 3] = XYZ
    if kind == "revolute":
        H[:3, :3] = H[:3, :3] @ _rot_axis(AXIS, q)
    elif kind == "prismatic":
        H[:3, 3] = XYZ + q * AXIS
    return H


def _ref_X(kind, q):
    T = _ref_H(kind, q)
    R = T[:3, :3].T
    p = -R @ T[:3, 3]
    S = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[:3, 3:] = S @ R
    X[3:, 3:] = R
    return X


def _call(math, name, kind, q_np):
    xyz, rpy, axis = (math.asarray(a) for a in (XYZ, RPY, AXIS))
    if kind == "fixed":
        if name == "H":
            return math.H_from_Pos_RPY(xyz, rpy)
        return math.X_fixed_joint(xyz, rpy)
    q = math.asarray(q_np)
    fn = getattr(math, f"{name}_{kind}_joint")
    return fn(xyz, rpy, axis, q)


@pytest.mark.parametrize("name", ["H", "X"])
@pytest.mark.parametrize("kind", ["revolute", "prismatic", "fixed"])
@pytest.mark.parametrize("q_shape", Q_SHAPES, ids=str)
def test_joint_transform_batching(backend, name, kind, q_shape):
    math, to_np = backend
    rng = np.random.default_rng(0)
    q = rng.uniform(-2, 2, size=q_shape)
    out = to_np(_call(math, name, kind, q).array)
    matrix = (4, 4) if name == "H" else (6, 6)

    if kind == "fixed":
        # q is irrelevant: fixed joints must stay unbatched
        assert out.shape == matrix
        ref = (_ref_H if name == "H" else _ref_X)(kind, 0.0)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
        return

    assert out.shape == q_shape + matrix
    assert out.dtype == np.float64

    ref_fn = _ref_H if name == "H" else _ref_X
    for idx in np.ndindex(*q_shape):
        # independent unbatched evaluation through the library
        single = to_np(_call(math, name, kind, np.asarray(q[idx])).array)
        assert single.shape == matrix
        np.testing.assert_allclose(out[idx], single, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            out[idx], ref_fn(kind, q[idx]), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_joint_transform_does_not_mutate_static_inputs(backend, kind):
    math, to_np = backend
    xyz, rpy, axis = (math.asarray(a) for a in (XYZ, RPY, AXIS))
    q = math.asarray(np.linspace(-1, 1, 6).reshape(2, 3))
    fn = getattr(math, f"H_{kind}_joint")
    fn(xyz, rpy, axis, q)
    np.testing.assert_array_equal(to_np(xyz.array), XYZ)
    np.testing.assert_array_equal(to_np(rpy.array), RPY)
    np.testing.assert_array_equal(to_np(axis.array), AXIS)


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_joint_transform_static_inputs_are_not_tiled(backend, kind, monkeypatch):
    math, _ = backend
    calls = []
    monkeypatch.setattr(math, "tile", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(math.factory, "tile", lambda *a, **k: calls.append(1))
    _call(math, "H", kind, np.zeros((7,)))
    assert not calls


@pytest.mark.parametrize("name", ["H", "X"])
@pytest.mark.parametrize("kind", ["revolute", "prismatic", "fixed"])
def test_joint_transform_casadi_unbatched(name, kind):
    cs = pytest.importorskip("casadi")
    from adam.casadi.casadi_like import SpatialMath

    math = SpatialMath()
    q = 0.37
    got = np.array(cs.DM(_call(math, name, kind, q).array))
    ref = (_ref_H if name == "H" else _ref_X)(kind, q)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_prismatic_translation_broadcasts_q_over_axis(backend):
    """A batch of size 3 must not be mistaken for a 3-vector."""
    math, to_np = backend
    q = np.array([0.1, 0.2, 0.3])
    out = to_np(_call(math, "H", "prismatic", q).array)
    expected = XYZ + q[:, None] * AXIS
    np.testing.assert_allclose(out[:, :3, 3], expected, rtol=1e-12, atol=1e-12)
