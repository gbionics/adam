"""Shared robot, inputs and backend adapter for the batching tests."""

from typing import Callable, NamedTuple

import numpy as np

try:  # must precede `import adam`: KinDynComputations builds a default gravity at import
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

from adam import Representations

# Tiny tree with revolute, prismatic and fixed joints; "tool" is massless, so adam
# treats it as a frame, and l2 has box geometry so the parametric models can rescale it.
URDF = """<?xml version="1.0"?>
<robot name="rpf_chain">
  <link name="base">
    <inertial>
      <origin xyz="0.01 0.02 0.03" rpy="0 0 0"/>
      <mass value="4.0"/>
      <inertia ixx="0.20" ixy="0.01" ixz="0.02" iyy="0.25" iyz="0.015" izz="0.30"/>
    </inertial>
  </link>
  <link name="l1">
    <inertial>
      <origin xyz="0.0 0.05 0.1" rpy="0.1 0.2 0.3"/>
      <mass value="1.5"/>
      <inertia ixx="0.05" ixy="0.002" ixz="0.001" iyy="0.06" iyz="0.003" izz="0.04"/>
    </inertial>
  </link>
  <link name="l2">
    <inertial>
      <origin xyz="0.02 0.0 0.05" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.03" ixy="0.001" ixz="0.0" iyy="0.035" iyz="0.002" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0.0 0.0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.08 0.1"/>
      </geometry>
    </visual>
  </link>
  <link name="l3">
    <inertial>
      <origin xyz="0.0 0.0 0.04" rpy="0 0.2 0"/>
      <mass value="0.8"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.001" iyy="0.025" iyz="0.0" izz="0.015"/>
    </inertial>
  </link>
  <link name="l4">
    <inertial>
      <origin xyz="0.03 0.01 0.0" rpy="0 0 0.4"/>
      <mass value="0.6"/>
      <inertia ixx="0.01" ixy="0.0005" ixz="0.0" iyy="0.012" iyz="0.0" izz="0.008"/>
    </inertial>
  </link>
  <link name="l5">
    <inertial>
      <origin xyz="0.0 0.0 0.02" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.008" ixy="0.0" ixz="0.0" iyy="0.009" iyz="0.0" izz="0.006"/>
    </inertial>
  </link>
  <link name="tool"/>

  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="l1"/>
    <origin xyz="0.1 0.2 -0.1" rpy="0.3 -0.2 0.5"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <joint name="j2" type="prismatic">
    <parent link="l1"/>
    <child link="l2"/>
    <origin xyz="0.0 0.1 0.2" rpy="-0.4 0.1 0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="10"/>
  </joint>
  <joint name="jf" type="fixed">
    <parent link="l2"/>
    <child link="l3"/>
    <origin xyz="0.05 -0.1 0.15" rpy="0.2 0.4 -0.3"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="l3"/>
    <child link="l4"/>
    <origin xyz="0.0 0.0 0.1" rpy="0.0 0.0 0.0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <joint name="j4" type="prismatic">
    <parent link="base"/>
    <child link="l5"/>
    <origin xyz="-0.1 0.0 0.1" rpy="0.1 0.2 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="10"/>
  </joint>
  <joint name="jt" type="fixed">
    <parent link="l4"/>
    <child link="tool"/>
    <origin xyz="0.02 -0.03 0.05" rpy="0.3 -0.2 0.1"/>
  </joint>
</robot>
"""
JOINTS = ["j1", "j2", "j3", "j4"]
N = len(JOINTS)
# The default gravity of the NumPy/PyTorch wrappers is float32-rounded; use an exact vector.
GRAVITY = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0])
FRAME = "l4"
REPRESENTATIONS = [
    Representations.MIXED_REPRESENTATION,
    Representations.BODY_FIXED_REPRESENTATION,
    Representations.INERTIAL_FIXED_REPRESENTATION,
]
STATE_KEYS = ("H", "q", "v", "qd", "tau")


class Op(NamedTuple):
    shape: tuple  # unbatched output shape
    fn: Callable  # (kd, H, q, v, qd, tau) -> output


OPS = {
    "fk": Op((4, 4), lambda kd, H, q, v, qd, tau: kd.forward_kinematics(FRAME, H, q)),
    "jacobian": Op((6, 6 + N), lambda kd, H, q, v, qd, tau: kd.jacobian(FRAME, H, q)),
    "jacobian_dot": Op(
        (6, 6 + N), lambda kd, H, q, v, qd, tau: kd.jacobian_dot(FRAME, H, q, v, qd)
    ),
    "com": Op((3,), lambda kd, H, q, v, qd, tau: kd.CoM_position(H, q)),
    "com_jacobian": Op((3, 6 + N), lambda kd, H, q, v, qd, tau: kd.CoM_jacobian(H, q)),
    "mass": Op((6 + N, 6 + N), lambda kd, H, q, v, qd, tau: kd.mass_matrix(H, q)),
    "cmm": Op(
        (6, 6 + N), lambda kd, H, q, v, qd, tau: kd.centroidal_momentum_matrix(H, q)
    ),
    "bias": Op((6 + N,), lambda kd, H, q, v, qd, tau: kd.bias_force(H, q, v, qd)),
    "coriolis": Op(
        (6 + N,), lambda kd, H, q, v, qd, tau: kd.coriolis_term(H, q, v, qd)
    ),
    "gravity": Op((6 + N,), lambda kd, H, q, v, qd, tau: kd.gravity_term(H, q)),
    "aba": Op((6 + N,), lambda kd, H, q, v, qd, tau: kd.aba(H, q, v, qd, tau)),
}


class Adapter:
    """Builds a backend's KinDynComputations and converts arrays to and from it."""

    def __init__(self, backend: str, dtype: str = "float64"):
        self.backend = backend
        if backend == "numpy":
            from adam.numpy import KinDynComputations

            self.xp, self.dtype = np, np.float64
            self.kd = KinDynComputations(URDF, JOINTS, gravity=GRAVITY)
        elif backend == "jax":
            import jax.numpy as jnp
            from adam.jax import KinDynComputations

            self.xp, self.dtype = jnp, getattr(jnp, dtype)
            self.kd = KinDynComputations(
                URDF, JOINTS, dtype=self.dtype, gravity=jnp.asarray(GRAVITY)
            )
        else:
            import torch
            from adam.pytorch import KinDynComputations

            self.xp, self.dtype = torch, getattr(torch, dtype)
            self.kd = KinDynComputations(
                URDF,
                JOINTS,
                device=torch.device("cpu"),
                dtype=self.dtype,
                gravity=torch.as_tensor(GRAVITY, dtype=torch.float64),
            )

    def to(self, x):
        return self.xp.asarray(np.array(x), dtype=self.dtype)

    def to_np(self, x) -> np.ndarray:
        if self.backend == "torch":
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def inputs(self, state: dict) -> list:
        return [self.to(state[k]) for k in STATE_KEYS]

    def call(self, op: str, state: dict):
        return OPS[op].fn(self.kd, *self.inputs(state))


def make_state(rng: np.random.Generator, batch: tuple) -> dict:
    """Random base pose (uniform rotation), joint positions/velocities and torques."""
    quat = rng.normal(size=batch + (4,))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    R = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(batch + (3, 3))
    H = np.broadcast_to(np.eye(4), batch + (4, 4)).copy()
    H[..., :3, :3] = R
    H[..., :3, 3] = rng.uniform(-1, 1, size=batch + (3,))
    return {
        "H": H,
        "q": rng.uniform(-1.5, 1.5, size=batch + (N,)),
        "v": rng.uniform(-1, 1, size=batch + (6,)),
        "qd": rng.uniform(-1, 1, size=batch + (N,)),
        "tau": rng.uniform(-1, 1, size=batch + (N,)),
    }


def sample(state: dict, idx: tuple) -> dict:
    return {k: v[idx] for k, v in state.items()}
