"""Autodiff and compilation checks for batched ABA and mass matrix (JAX, PyTorch)."""

import numpy as np
import pytest
from batching_utils import N, OPS, REPRESENTATIONS, Adapter, make_state, sample

BATCHES = [(1,), (4,)]
TOL = dict(rtol=1e-9, atol=1e-9)
W_MASS = np.random.default_rng(11).normal(size=(6 + N, 6 + N))
W_ABA = np.random.default_rng(12).normal(size=(6 + N,))


def _adapter(backend, rep):
    pytest.importorskip(backend)
    ad = Adapter(backend)
    ad.kd.set_frame_velocity_representation(rep)
    return ad


# ----------------------------------------------------------------------- JAX


@pytest.mark.parametrize("rep", REPRESENTATIONS, ids=lambda r: r.name)
@pytest.mark.parametrize(
    "op", ["mass", "cmm", "bias", "aba", "fk", "jacobian", "com", "com_jacobian"]
)
@pytest.mark.parametrize("batch", [(), (1,), (4,), (2, 3)], ids=str)
def test_jax_jit_matches_eager(rep, op, batch):
    ad = _adapter("jax", rep)
    import jax

    args = ad.inputs(make_state(np.random.default_rng(6), batch))
    fn = OPS[op].fn
    eager = fn(ad.kd, *args)
    jitted = jax.jit(lambda *a: fn(ad.kd, *a))(*args)
    assert eager.shape == jitted.shape == batch + OPS[op].shape
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), **TOL)


def _jax_losses(ad, state):
    H, v = ad.to(state["H"]), ad.to(state["v"])

    def mass_loss(q):
        return (ad.kd.mass_matrix(H, q) * W_MASS).sum()

    def aba_loss(q, qd, tau):
        return (ad.kd.aba(H, q, v, qd, tau) * W_ABA).sum()

    return mass_loss, aba_loss


@pytest.mark.parametrize("rep", REPRESENTATIONS, ids=lambda r: r.name)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
def test_jax_grad_mass_and_aba(rep, batch):
    ad = _adapter("jax", rep)
    import jax

    state = make_state(np.random.default_rng(7), batch)
    mass_loss, aba_loss = _jax_losses(ad, state)
    x = [ad.to(state[k]) for k in ("q", "qd", "tau")]

    val_m, g_m = jax.jit(jax.value_and_grad(mass_loss))(x[0])
    val_a, g_a = jax.jit(jax.value_and_grad(aba_loss, argnums=(0, 1, 2)))(*x)
    np.testing.assert_allclose(g_m, jax.grad(mass_loss)(x[0]), **TOL)
    assert np.isfinite(float(val_m)) and np.isfinite(float(val_a))
    for g, xi in zip((g_m, *g_a), (x[0], *x)):
        assert g.shape == xi.shape and g.dtype == xi.dtype
        assert np.isfinite(np.asarray(g)).all()

    # central finite differences along a random direction (float64)
    rng = np.random.default_rng(8)
    eps = 1e-6
    d = [rng.normal(size=state[k].shape) for k in ("q", "qd", "tau")]

    def shifted(loss, s, n_args):
        keys = ("q", "qd", "tau")[:n_args]
        return float(loss(*(ad.to(state[k] + s * di) for k, di in zip(keys, d))))

    for loss, grads in ((mass_loss, (g_m,)), (aba_loss, g_a)):
        fd = (shifted(loss, eps, len(grads)) - shifted(loss, -eps, len(grads))) / (
            2 * eps
        )
        analytic = sum(float(np.sum(np.asarray(g) * di)) for g, di in zip(grads, d))
        np.testing.assert_allclose(fd, analytic, rtol=1e-5, atol=1e-6)

    # the gradient of the batch sum equals the per-sample gradients
    for b in range(batch[0]):
        s = sample(state, (b,))
        mass_loss_b, aba_loss_b = _jax_losses(ad, s)
        xb = [ad.to(s[k]) for k in ("q", "qd", "tau")]
        np.testing.assert_allclose(
            jax.grad(mass_loss_b)(xb[0]), np.asarray(g_m)[b], **TOL
        )
        for g_single, g_batch in zip(jax.grad(aba_loss_b, argnums=(0, 1, 2))(*xb), g_a):
            np.testing.assert_allclose(g_single, np.asarray(g_batch)[b], **TOL)


# ------------------------------------------------------------------- PyTorch


def _torch_leaves(ad, state):
    return [ad.to(state[k]).requires_grad_(True) for k in ("q", "qd", "tau")]


def _torch_backward(ad, state):
    """Backpropagates the weighted mass-matrix and ABA losses; returns (q, qd, tau)."""
    H, v = ad.to(state["H"]), ad.to(state["v"])
    q, qd, tau = leaves = _torch_leaves(ad, state)
    M = ad.kd.mass_matrix(H, q)
    qdd = ad.kd.aba(H, q, v, qd, tau)
    assert M.requires_grad and qdd.requires_grad
    (M * ad.to(W_MASS)).sum().backward()
    (qdd * ad.to(W_ABA)).sum().backward()
    return leaves


@pytest.mark.parametrize("rep", REPRESENTATIONS, ids=lambda r: r.name)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
def test_torch_backward_mass_and_aba(rep, batch):
    ad = _adapter("torch", rep)
    import torch

    state = make_state(np.random.default_rng(9), batch)
    leaves = _torch_backward(ad, state)
    for x in leaves:
        assert x.grad.shape == x.shape and x.grad.dtype == x.dtype
        assert torch.isfinite(x.grad).all()

    for b in range(batch[0]):
        for batched, single in zip(leaves, _torch_backward(ad, sample(state, (b,)))):
            np.testing.assert_allclose(
                batched.grad[b].numpy(), single.grad.numpy(), **TOL
            )


@pytest.mark.parametrize("rep", REPRESENTATIONS, ids=lambda r: r.name)
@pytest.mark.parametrize("batch", BATCHES, ids=str)
def test_torch_gradcheck(rep, batch):
    ad = _adapter("torch", rep)
    import torch

    state = make_state(np.random.default_rng(10), batch)
    H, v = ad.to(state["H"]), ad.to(state["v"])
    q, qd, tau = _torch_leaves(ad, state)
    tol = dict(eps=1e-6, atol=1e-5, rtol=1e-4)

    assert torch.autograd.gradcheck(lambda q_: ad.kd.mass_matrix(H, q_), (q,), **tol)
    assert torch.autograd.gradcheck(
        lambda q_, qd_, tau_: ad.kd.aba(H, q_, v, qd_, tau_), (q, qd, tau), **tol
    )
