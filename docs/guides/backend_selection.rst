Choosing a Backend
===================

adam supports multiple backends, each optimized for different use cases. This guide helps you choose the right one for your application.

.. Quick Comparison
.. ----------------

.. +-------------+--------+--------+--------+--------+
.. | Feature     | NumPy  | JAX    | CasADi | PyTorch|
.. +=============+========+========+========+========+
.. | **Speed**   | Good   | Excellent | Good   | Excellent |
.. | **Symbolic**| ❌    | ❌     | ✅    | ❌    |
.. | **Autodiff**| ❌    | ✅     | ✅    | ✅    |
.. | **GPU**     | ❌    | ✅     | ❌    | ✅    |
.. | **Batch**   | Manual | Native | Manual | Native |
.. | **Learning Curve** | Easy | Medium | Medium | Easy |
.. +-------------+--------+--------+--------+--------+

.. Detailed Breakdown
.. ------------------

NumPy
^^^^^

**Best for:** Quick prototyping, validation, simplicity

- Direct numerical computation
- No compilation or setup overhead
- Easiest to debug and understand
- Good for initial model verification

**When to use:**

- Validating robot models
- Quick experiments
- Educational purposes

**Example:**

.. code-block:: python

    from adam.numpy import KinDynComputations
    kinDyn = KinDynComputations(model_path, joints_list)
    M = kinDyn.mass_matrix(w_H_b, joints)


JAX
^^^

**Best for:** Research, optimization, vectorized computations

- JIT compilation for speed
- Automatic differentiation (grad, jacobian, hessian)
- Native batch support

**When to use:**

- Gradient-based optimization
- Computing derivatives and Jacobians
- Processing batches of configurations
- Control design with auto-differentiation

**Gotchas:**

- First call with jit is slow (compilation)
- Can't use Python branching on traced values

**Example:**

.. code-block:: python

    from adam.jax import KinDynComputations
    from jax import jit, grad
    
    kinDyn = KinDynComputations(model_path, joints_list)
    
    @jit
    def compute_and_grad(w_H_b, joints):
        M = kinDyn.mass_matrix(w_H_b, joints)
        grad_fn = grad(lambda q: kinDyn.mass_matrix(w_H_b, q).sum())
        return M, grad_fn(joints)


CasADi
^^^^^^

**Best for:** Optimal control, trajectory optimization, symbolic formulation

- Symbolic computation for optimization
- Function generation for code generation
- Both numeric and symbolic evaluation
- Constraint handling

**When to use:**

- Nonlinear model predictive control (NMPC)
- Trajectory optimization
- Building optimization problems
- When you need symbolic expressions
- Code generation for embedded systems

**Example:**

.. code-block:: python

    import casadi as cs
    from adam.casadi import KinDynComputations
    
    kinDyn = KinDynComputations(model_path, joints_list)
    
    # Numeric evaluation
    M_fun = kinDyn.mass_matrix_fun()
    M = M_fun(w_H_b, joints)
    
    # Symbolic computation
    w_H_b_sym = cs.SX.sym('H', 4, 4)
    joints_sym = cs.SX.sym('q', n_dof)
    M_sym = M_fun(w_H_b_sym, joints_sym)


PyTorch
^^^^^^^

**Best for:** Machine learning, GPU acceleration, batched computation

- GPU usage out of the box
- Native batch support
- Integration with neural networks
- Automatic differentiation

**When to use:**

- Learning-based control
- Large-scale batch processing
- GPU-accelerated robotics
- Integration with ML frameworks

**Gotchas:**

- GPU memory can be limiting
- Requires careful dtype/device handling

**Example:**

.. code-block:: python

    import torch
    from adam.pytorch import KinDynComputations
    
    kinDyn = KinDynComputations(model_path, joints_list)
    
    # Batched computation
    batch_size = 1024
    w_H_b_batch = torch.randn(batch_size, 4, 4)
    joints_batch = torch.randn(batch_size, n_dof)
    
    M_batch = kinDyn.mass_matrix(w_H_b_batch, joints_batch)  # Shape: (batch, 6+n, 6+n)
