"""Shape, value and dynamics checks of the public NumPy/JAX/PyTorch APIs under batching.

Batched outputs are compared with independent unbatched evaluations, and ABA is
checked against the library dynamics convention
``M @ qdd + h = [0; tau] + sum_i J_i^T @ wrench_i``.
"""

import numpy as np
import pytest
from batching_utils import (
    FRAME,
    GRAVITY,
    JOINTS,
    N,
    OPS,
    REPRESENTATIONS,
    STATE_KEYS,
    URDF,
    Adapter,
    make_state,
    sample,
)

BATCH_SHAPES = [(1,), (4,), (2, 3)]
ALL_SHAPES = [()] + BATCH_SHAPES
TOL = dict(rtol=1e-9, atol=1e-9)


@pytest.fixture(params=["numpy", "jax", "torch"])
def adapter(request):
    pytest.importorskip(request.param)
    return Adapter(request.param)


@pytest.mark.parametrize("rep", REPRESENTATIONS, ids=lambda r: r.name)
@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("batch", ALL_SHAPES, ids=str)
def test_shape_and_values_match_unbatched(adapter, rep, op, batch):
    adapter.kd.set_frame_velocity_representation(rep)
    state = make_state(np.random.default_rng(1), batch)
    args = adapter.inputs(state)
    snapshot = [adapter.to_np(a).copy() for a in args]

    out = OPS[op].fn(adapter.kd, *args)
    assert tuple(out.shape) == batch + OPS[op].shape
    assert out.dtype == adapter.dtype
    got = adapter.to_np(out)
    assert np.isfinite(got).all()

    for idx in np.ndindex(*batch):
        single = adapter.to_np(adapter.call(op, sample(state, idx)))
        assert single.shape == OPS[op].shape
        np.testing.assert_allclose(got[idx], single, **TOL)

    for arg, before in zip(args, snapshot):
        np.testing.assert_array_equal(adapter.to_np(arg), before)


@pytest.mark.parametrize("rep", REPRESENTATIONS, ids=lambda r: r.name)
@pytest.mark.parametrize("batch", ALL_SHAPES, ids=str)
def test_mass_matrix_is_symmetric(adapter, rep, batch):
    adapter.kd.set_frame_velocity_representation(rep)
    state = make_state(np.random.default_rng(2), batch)
    M = adapter.to_np(adapter.call("mass", state))
    np.testing.assert_allclose(M, np.swapaxes(M, -1, -2), **TOL)


@pytest.mark.parametrize("root", [None, "l3"], ids=["root", "rerooted"])
@pytest.mark.parametrize("with_wrench", [False, True], ids=["no_wrench", "wrench"])
@pytest.mark.parametrize("rep", REPRESENTATIONS, ids=lambda r: r.name)
@pytest.mark.parametrize("batch", ALL_SHAPES, ids=str)
def test_aba_dynamics_residual(adapter, rep, batch, with_wrench, root):
    if root is not None:
        adapter.kd.set_root_link(root)
    adapter.kd.set_frame_velocity_representation(rep)
    rng = np.random.default_rng(3)
    state = make_state(rng, batch)
    # "tool" is a frame, the others are tree nodes
    wrenches = {f: rng.normal(size=(6,)) * 5 for f in ("l4", "l1", "tool")}
    if not with_wrench:
        wrenches = {}

    H, q, v, qd, tau = args = adapter.inputs(state)
    external = {f: adapter.to(w) for f, w in wrenches.items()}
    qdd = adapter.to_np(adapter.kd.aba(*args, external_wrenches=external or None))
    assert qdd.shape == batch + (6 + N,)

    M = adapter.to_np(adapter.kd.mass_matrix(H, q))
    h = adapter.to_np(adapter.kd.bias_force(H, q, v, qd))
    rhs = np.concatenate([np.zeros(batch + (6,)), state["tau"]], axis=-1)
    for frame, w in wrenches.items():
        J = adapter.to_np(adapter.kd.jacobian(frame, H, q))
        rhs = rhs + np.einsum("...ji,j->...i", J, w)
    residual = np.einsum("...ij,...j->...i", M, qdd) + h - rhs
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


@pytest.mark.parametrize("batch", [(1,), (3,)], ids=str)
def test_aba_batched_external_wrench_matches_unbatched(adapter, batch):
    rng = np.random.default_rng(4)
    state = make_state(rng, batch)
    wrench = rng.normal(size=batch + (6,)) * 5

    def aba(state, wrench):
        external = {FRAME: adapter.to(wrench)}
        return adapter.to_np(
            adapter.kd.aba(*adapter.inputs(state), external_wrenches=external)
        )

    out = aba(state, wrench)
    for idx in np.ndindex(*batch):
        np.testing.assert_allclose(
            out[idx], aba(sample(state, idx), wrench[idx]), **TOL
        )


@pytest.mark.parametrize("batch", [(), (16,)], ids=str)
def test_static_caches_are_not_mutated(adapter, batch):
    algos = adapter.kd.rbdalgos
    caches = [
        *algos._spatial_inertias,
        *algos._motion_subspaces,
        algos._root_spatial_transform,
        algos._root_motion_subspace,
    ]
    before = [adapter.to_np(c.array).copy() for c in caches]
    args = adapter.inputs(make_state(np.random.default_rng(1), batch))
    for rep in REPRESENTATIONS:
        adapter.kd.set_frame_velocity_representation(rep)
        adapter.kd.mass_matrix(args[0], args[1])
        adapter.kd.aba(*args)
        adapter.kd.aba(*args, external_wrenches={FRAME: adapter.to(np.ones(6))})
    for cache, expected in zip(caches, before):
        np.testing.assert_array_equal(adapter.to_np(cache.array), expected)


@pytest.mark.parametrize("batch", [(), (1,), (16,), (2, 3)], ids=str)
def test_dynamics_neither_tile_nor_invert(adapter, monkeypatch, batch):
    """Static data is broadcast rather than tiled; one-DoF ABA pivots use a reciprocal."""
    math = adapter.kd.rbdalgos.math

    def forbid(name):
        def fail(*args, **kwargs):
            raise AssertionError(f"{name} was called")

        return fail

    for owner in (math, math.factory):
        monkeypatch.setattr(owner, "tile", forbid("tile"))
    monkeypatch.setattr(math, "inv", forbid("inv"))

    H, q, v, qd, tau = adapter.inputs(make_state(np.random.default_rng(0), batch))
    adapter.kd.mass_matrix(H, q)
    adapter.kd.centroidal_momentum_matrix(H, q)
    adapter.kd.aba(H, q, v, qd, tau)
    adapter.kd.aba(H, q, v, qd, tau, external_wrenches={FRAME: adapter.to(np.ones(6))})


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("batch", [(), (1,), (2, 3)], ids=str)
def test_float32_dtype_is_preserved(backend, op, batch):
    pytest.importorskip(backend)
    ad = Adapter(backend, dtype="float32")
    state = make_state(np.random.default_rng(5), batch)
    out = ad.call(op, state)
    assert out.dtype == ad.dtype
    assert tuple(out.shape) == batch + OPS[op].shape
    _assert_close_to_float64(ad.to_np(out), op, state)


def _assert_close_to_float64(got, op, state):
    ref = Adapter("numpy")
    expected = ref.to_np(ref.call(op, state))
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4 * scale, err_msg=op)


def _usable_torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        (torch.ones(2, device="cuda") * 2).cpu()
    except RuntimeError as exc:  # e.g. GPU architecture unsupported by this build
        pytest.skip(f"CUDA present but unusable: {exc}")
    return torch


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("batch", [(), (1,), (4,)], ids=str)
def test_torch_cuda_keeps_device_and_dtype(op, batch):
    torch = _usable_torch_cuda()
    from adam.pytorch import KinDynComputations

    kd = KinDynComputations(
        URDF,
        JOINTS,
        device=torch.device("cuda"),
        dtype=torch.float32,
        gravity=torch.as_tensor(GRAVITY),
    )
    state = make_state(np.random.default_rng(6), batch)
    args = [
        torch.as_tensor(state[k], dtype=torch.float32, device="cuda")
        for k in STATE_KEYS
    ]
    out = OPS[op].fn(kd, *args)
    assert out.device.type == "cuda" and out.dtype == torch.float32
    assert tuple(out.shape) == batch + OPS[op].shape
    _assert_close_to_float64(out.cpu().numpy(), op, state)


@pytest.mark.parametrize("op", list(OPS))
def test_jax_gpu_keeps_device(op):
    jax = pytest.importorskip("jax")
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("no JAX GPU backend")
    ad = Adapter("jax")
    args = [
        jax.device_put(a, gpu)
        for a in ad.inputs(make_state(np.random.default_rng(7), (4,)))
    ]
    out = OPS[op].fn(ad.kd, *args)
    assert out.devices() == {gpu}


@pytest.mark.parametrize("n_dof", [1, 2, 3, 4])
def test_casadi_matches_numpy_for_any_dof_count(n_dof):
    """CasADi is 2-D only, so a two-DoF model hits its two-column concatenation corner case."""
    cs = pytest.importorskip("casadi")
    from adam.casadi import KinDynComputations as CasadiKD
    from adam.numpy import KinDynComputations as NumpyKD

    joints = JOINTS[:n_dof]
    ck = CasadiKD(URDF, joints, gravity=GRAVITY)
    nk = NumpyKD(URDF, joints, gravity=GRAVITY)
    rng = np.random.default_rng(n_dof)
    s = make_state(rng, ())
    H, v = s["H"], s["v"]
    q, qd, tau = (rng.uniform(-1, 1, n_dof) for _ in range(3))
    checks = {
        "mass": (ck.mass_matrix_fun()(H, q), nk.mass_matrix(H, q)),
        "cmm": (
            ck.centroidal_momentum_matrix_fun()(H, q),
            nk.centroidal_momentum_matrix(H, q),
        ),
        "com_jacobian": (ck.CoM_jacobian_fun()(H, q), nk.CoM_jacobian(H, q)),
        "jacobian": (ck.jacobian_fun(FRAME)(H, q), nk.jacobian(FRAME, H, q)),
        "relative_jacobian": (
            ck.relative_jacobian_fun(FRAME)(q),
            nk.relative_jacobian(FRAME, q),
        ),
        "jacobian_dot": (
            ck.jacobian_dot_fun(FRAME)(H, q, v, qd),
            nk.jacobian_dot(FRAME, H, q, v, qd),
        ),
        "bias": (ck.bias_force_fun()(H, q, v, qd), nk.bias_force(H, q, v, qd)),
        "aba": (ck.aba_fun()(H, q, v, qd, tau), nk.aba(H, q, v, qd, tau)),
    }
    for name, (got, ref) in checks.items():
        got = np.array(cs.DM(got)).reshape(ref.shape)
        np.testing.assert_allclose(got, ref, err_msg=name, **TOL)


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("batch", [(1,), (3,)], ids=str)
def test_parametric_eager_keeps_batch_dimension(backend, batch):
    """Eager shape/value check only; parametric JIT/autodiff batching is not covered."""
    pytest.importorskip(backend)
    ad = Adapter(backend)  # for the array conversions
    if backend == "jax":
        from adam.parametric.jax import KinDynComputationsParametric
    else:
        from adam.parametric.pytorch import KinDynComputationsParametric
    kd = KinDynComputationsParametric(URDF, JOINTS, ["l2"])
    lm, dens = ad.to([1.3]), ad.to([900.0])
    ops = {
        "mass": ((6 + N, 6 + N), lambda H, q, v, qd: kd.mass_matrix(H, q, lm, dens)),
        "bias": ((6 + N,), lambda H, q, v, qd: kd.bias_force(H, q, v, qd, lm, dens)),
        "coriolis": (
            (6 + N,),
            lambda H, q, v, qd: kd.coriolis_term(H, q, v, qd, lm, dens),
        ),
        "gravity": ((6 + N,), lambda H, q, v, qd: kd.gravity_term(H, q, lm, dens)),
        "com": ((3,), lambda H, q, v, qd: kd.CoM_position(H, q, lm, dens)),
    }
    state = make_state(np.random.default_rng(8), batch)

    def run(fn, state):
        return ad.to_np(fn(*ad.inputs(state)[:4]))

    for name, (shape, fn) in ops.items():
        out = run(fn, state)
        assert out.shape == batch + shape, name
        for idx in np.ndindex(*batch):
            np.testing.assert_allclose(
                out[idx], run(fn, sample(state, idx)), err_msg=name, **TOL
            )
