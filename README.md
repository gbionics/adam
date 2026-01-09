# 🤖 adam

[![adam](https://github.com/ami-iit/adam/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ami-iit/adam/actions/workflows/tests.yml)
[![](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://github.com/ami-iit/adam/blob/main/LICENSE)

**Automatic Differentiation for rigid-body-dynamics Algorithms**

**adam** computes rigid-body dynamics for floating-base robots. Built on Featherstone's algorithms and available across multiple backends:

- 🔥 **JAX** – compile, vectorize, and differentiate with XLA
- 🎯 **CasADi** – symbolic computation for optimization and control
- 🔦 **PyTorch** – GPU acceleration and batched operations
- 🐍 **NumPy** – simple numerical evaluation

All backends share the same interface and produce numerically consistent results, letting you pick the tool that fits your use case.

## 📦 Installation

### With pip

```bash
# JAX backend
pip install adam-robotics[jax]

# CasADi backend
pip install adam-robotics[casadi]

# PyTorch backend
pip install adam-robotics[pytorch]

# All backends
pip install adam-robotics[all]
```

### With conda

```bash
# CasADi backend
conda create -n adamenv -c conda-forge adam-robotics-casadi

# JAX backend (Linux/macOS only)
conda create -n adamenv -c conda-forge adam-robotics-jax

# PyTorch backend (Linux/macOS only)
conda create -n adamenv -c conda-forge adam-robotics-pytorch

# All backends (Linux/macOS only)
conda create -n adamenv -c conda-forge adam-robotics-all
```

### From source

```bash
git clone https://github.com/ami-iit/adam.git
cd adam
pip install .[jax]  # or [casadi], [pytorch], [all]
```

## 🚀 Quick Start

Load a robot model and compute dynamics quantities:

### JAX

> [!NOTE]
> Check the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)

```python
import adam
from adam.jax import KinDynComputations
import icub_models
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

# if you want to icub-models https://github.com/robotology/icub-models to retrieve the urdf
model_path = icub_models.get_model_file("iCubGazeboV2_5")
# The joint list
joints_name_list = [
    'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
    'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
    'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
    'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
    'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
]

kinDyn = KinDynComputations(model_path, joints_name_list)
# Set velocity representation (3 options available):
# 1. MIXED_REPRESENTATION (default) - time derivative of base origin position (expressed in world frame) + world-frame angular velocity
kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
# 2. BODY_FIXED_REPRESENTATION - both linear & angular velocity in body frame
# kinDyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
# 3. INERTIAL_FIXED_REPRESENTATION - world-frame linear & angular velocity
# kinDyn.set_frame_velocity_representation(adam.Representations.INERTIAL_FIXED_REPRESENTATION)

w_H_b = np.eye(4)
joints = np.ones(len(joints_name_list))
M = kinDyn.mass_matrix(w_H_b, joints)
print(M)
w_H_f = kinDyn.forward_kinematics('frame_name', w_H_b, joints)

# JAX functions can also be jitted!
# For example:

def frame_forward_kinematics(w_H_b, joints):
    # This is needed since str is not a valid JAX type
    return kinDyn.forward_kinematics('frame_name', w_H_b, joints)

jitted_frame_fk = jit(frame_forward_kinematics)
w_H_f = jitted_frame_fk(w_H_b, joints)

# JAX natively supports batching
joints_batch = jnp.tile(joints, (1024, 1))
w_H_b_batch = jnp.tile(w_H_b, (1024, 1, 1))
w_H_f_batch = kinDyn.forward_kinematics('frame_name', w_H_b_batch, joints_batch)
```

> [!NOTE]
> The first call of the jitted function can be slow, since JAX needs to compile the function. Then it will be faster!

### CasADi

```python
import casadi as cs
import adam
from adam.casadi import KinDynComputations
import icub_models
import numpy as np

# if you want to icub-models https://github.com/robotology/icub-models to retrieve the urdf
model_path = icub_models.get_model_file("iCubGazeboV2_5")
# The joint list
joints_name_list = [
    'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
    'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
    'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
    'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
    'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
]

kinDyn = KinDynComputations(model_path, joints_name_list)
# Set velocity representation (3 options available):
# 1. MIXED_REPRESENTATION (default) - time derivative of position + world-frame angular velocity
kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
# 2. BODY_FIXED_REPRESENTATION - both linear & angular velocity in body frame
# kinDyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
# 3. INERTIAL_FIXED_REPRESENTATION - world-frame linear & angular velocity
# kinDyn.set_frame_velocity_representation(adam.Representations.INERTIAL_FIXED_REPRESENTATION)

w_H_b = np.eye(4)
joints = np.ones(len(joints_name_list))
M = kinDyn.mass_matrix_fun()
print(M(w_H_b, joints))

# If you want to use the symbolic version
w_H_b = cs.SX.eye(4)
joints = cs.SX.sym('joints', len(joints_name_list))
M = kinDyn.mass_matrix_fun()
print(M(w_H_b, joints))

# This is usable also with casadi.MX
w_H_b = cs.MX.eye(4)
joints = cs.MX.sym('joints', len(joints_name_list))
M = kinDyn.mass_matrix_fun()
print(M(w_H_b, joints))
```

### PyTorch

```python
import adam
from adam.pytorch import KinDynComputations
import icub_models
import numpy as np

# if you want to icub-models https://github.com/robotology/icub-models to retrieve the urdf
model_path = icub_models.get_model_file("iCubGazeboV2_5")
# The joint list
joints_name_list = [
    'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
    'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
    'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
    'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
    'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
]

kinDyn = KinDynComputations(model_path, joints_name_list)
# choose the representation you want to use the body fixed representation
kinDyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
# or, if you want to use the mixed representation (that is the default)
kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
w_H_b = np.eye(4)
joints = np.ones(len(joints_name_list))
M = kinDyn.mass_matrix(w_H_b, joints)
print(M)
```

### PyTorch Batched

Use `pytorch.KinDynComputations` to process also multiple configurations.

> [!NOTE]
> There is a class `pytorch.KinDynComputationsBatch` that has the functionality of `pytorch.KinDynComputations`. It exists to avoid API changes in existing code. New users should prefer `pytorch.KinDynComputations` for both single and batched computations.


```python
import adam
from adam.pytorch import KinDynComputations
import icub_models

# if you want to icub-models
model_path = icub_models.get_model_file("iCubGazeboV2_5")
# The joint list
joints_name_list = [
    'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
    'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
    'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
    'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
    'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
]

kinDyn = KinDynComputations(model_path, joints_name_list)
# choose the representation you want to use the body fixed representation
kinDyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
# or, if you want to use the mixed representation (that is the default)
kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
w_H_b = np.eye(4)
joints = np.ones(len(joints_name_list))

num_samples = 1024
w_H_b_batch = torch.tensor(np.tile(w_H_b, (num_samples, 1, 1)), dtype=torch.float32)
joints_batch = torch.tensor(np.tile(joints, (num_samples, 1)), dtype=torch.float32)

M = kinDyn.mass_matrix(w_H_b_batch, joints_batch)
w_H_f = kinDyn.forward_kinematics('frame_name', w_H_b_batch, joints_batch)
```

### MuJoCo

adam supports loading models directly from [MuJoCo](https://mujoco.org/) `MjModel` objects. This is useful when working with MuJoCo simulations or models from [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py).

```python
import mujoco
import numpy as np
from adam import Representations
from adam.numpy import KinDynComputations

# Load a MuJoCo model (e.g., from robot_descriptions)
from robot_descriptions.loaders.mujoco import load_robot_description
mj_model = load_robot_description("g1_mj_description")

# Create KinDynComputations directly from MuJoCo model
kinDyn = KinDynComputations.from_mujoco_model(mj_model)

# Set velocity representation (default is mixed)
kinDyn.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)

# Set gravity to match MuJoCo settings
kinDyn.g = np.concatenate([mj_model.opt.gravity, np.zeros(3)])

# Create MuJoCo data and set state
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = your_qpos  # Your configuration
mj_data.qvel[:] = your_qvel  # Your velocities
mujoco.mj_forward(mj_model, mj_data)

# Extract base transform from MuJoCo state (for floating-base robots)
from scipy.spatial.transform import Rotation as R
base_rot = R.from_quat(mj_data.qpos[3:7], scalar_first=True).as_matrix()
base_pos = mj_data.qpos[0:3]
w_H_b = np.eye(4)
w_H_b[:3, :3] = base_rot
w_H_b[:3, 3] = base_pos

# Joint positions (excluding free joint).
# Be sure the serialization between mujoco and adam is the same
joints = mj_data.qpos[7:]

# Compute dynamics quantities
M = kinDyn.mass_matrix(w_H_b, joints)
com_pos = kinDyn.CoM_position(w_H_b, joints)
J = kinDyn.jacobian('frame_name', w_H_b, joints)
```

> [!NOTE]
> When using `from_mujoco_model`, adam automatically extracts the joint names from the MuJoCo model. You can also specify `use_mujoco_actuators=True` to use actuator names instead of joint names.

> [!WARNING]
> MuJoCo uses a different velocity representation for the floating base. The free joint velocity in MuJoCo is `[I \dot{p}_B, B \omega_B]`, while mixed representation uses `[I \dot{p}_B, I \omega_B]`. Make sure to handle this transformation when comparing with MuJoCo computations.

### Inverse Kinematics

```python
import casadi as cs
import numpy as np
import adam
from adam.casadi import KinDynComputations
from adam.casadi.inverse_kinematics import InverseKinematics, TargetType

# Load model
model_path = ...
joints_name_list = [...]

# Create IK solver
ik = InverseKinematics(model_path, joints_name_list)
ik.add_target("l_sole", target_type=TargetType.POSE, as_soft_constraint=True, weight=1.0)

# Update target and solve
desired_position = np.array([0.3, 0.2, 1.0])
desired_orientation = np.eye(3)
ik.update_target("l_sole", (desired_position, desired_orientation))
ik.solve()

# Get solution
w_H_b_sol, q_sol = ik.get_solution()
print("Base pose:\n", w_H_b_sol)
print("Joint values:\n", q_sol)
```

## 📚 Features

- **Kinematics**: Forward kinematics, Jacobians (frame and base)
- **Dynamics**: Mass matrix, Coriolis/centrifugal forces and gravity, Articulated body algorithm
- **Centroidal**: Centroidal momentum matrix and derivatives
- **Differentiation**: Get gradients, Jacobians, and Hessians automatically
- **Symbolic**: Build computation graphs with CasADi for optimization
- **Batched**: Process multiple configurations in parallel with PyTorch

## 📖 Documentation

See the [full documentation](https://adam-robotics.readthedocs.io/en/latest/) for detailed API reference, more examples, and theory.

## 🧪 Testing

Run tests to verify installation:

```bash
pip install .[test]  # Install test dependencies
pytest tests/
```

See `tests/` folder for comprehensive examples across all backends.

## 🤝 Contributing

Found a bug or have a feature idea? Open an [issue](https://github.com/ami-iit/adam/issues) or submit a [pull request](https://github.com/ami-iit/adam/pulls)! 🚀

> [!WARNING]
> This is a project under active development. API may change.

## 📄 License

BSD 3-Clause License – see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Built on [Roy Featherstone's Rigid Body Dynamics Algorithms](https://link.springer.com/book/10.1007/978-1-4899-7560-7) and references like [Traversaro's A Unified View of the Equations of Motion](https://traversaro.github.io/traversaro-phd-thesis/traversaro-phd-thesis.pdf).
