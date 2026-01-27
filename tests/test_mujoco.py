import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from robot_descriptions.loaders.mujoco import load_robot_description

from adam import Representations
from adam.numpy.computations import KinDynComputations

DESCRIPTION_NAMES = ["g1_mj_description", "aliengo_mj_description"]


def _load_model(description: str) -> mujoco.MjModel:
    print(f"Loading robot description '{description}'")
    try:
        model = load_robot_description(description)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Cannot load {description}: {exc}")
    if model is None:  # pragma: no cover - loader returns None on missing path
        pytest.skip(f"{description} not available (loader returned None)")
    return model


def _random_configuration(
    model: mujoco.MjModel, rng: np.random.Generator
) -> np.ndarray:
    q = rng.normal(scale=1.0, size=model.nq)
    # Normalize the free joint quaternion (indices 3:7)
    quat = q[3:7]
    quat = quat / np.linalg.norm(quat)
    q[3:7] = quat
    return q


def _random_velocity(model: mujoco.MjModel, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(scale=1.0, size=model.nv)


def _set_state(
    model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, qvel: np.ndarray
):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)


def _base_velocity_transform(
    base_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MuJoCo encodes a free-joint qvel as [I v_B, B ω_B] (I v_B = I \\dot{p}_B).
    ADAM mixed vel representation uses [I v_B, I ω_B]. Given R_IB = base_rotation,
      I ω_B = R_IB @ (B ω_B).
    Adj maps qvel_MJ -> nu_ADAM for the base; Adj_inv reverses it.
    """
    Adj = np.block(
        [
            [np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), base_rotation],
        ]
    )
    Adj_inv = np.block(
        [
            [np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), base_rotation.T],
        ]
    )
    return Adj, Adj_inv


def _joint_permutation(
    model: mujoco.MjModel, kd: KinDynComputations
) -> tuple[np.ndarray, list[str], list[str]]:
    mj_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        for j_id in range(model.njnt)
        if model.jnt_type[j_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    adam_joint_names = kd.rbdalgos.model.actuated_joints
    perm = [mj_joint_names.index(name) for name in adam_joint_names]
    P = np.eye(len(mj_joint_names))[:, perm]
    # TODO: the P in this test is an identity matrix. The actuated joints and the MuJoCo joints are the same and in the same order. The cases in which this is not true should be handled by the user.
    return P, mj_joint_names, adam_joint_names


def _compose_transforms(
    base_rotation: np.ndarray, permutation: np.ndarray, n_act: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # MuJoCo qvel layout (free joint first) is [I v_B, B ω_B, joint_vel_XML]:
    #   - linear velocity of the base origin: I v_B = I \dot{p}_B
    #   - angular velocity of the base expressed in the body frame B
    #   - joint rates in the XML joint order
    # ADAM mixed-velocity qdot is [I v_B, I ω_B, joint_vel_ADAM]:
    #   - linear matches MuJoCo: I v_B = I \dot{p}_B
    #   - angular expressed in I (I ω_B = R_IB B ω_B)
    #   - joint rates reordered to ADAM's actuated_joints ordering
    # Let R_IB = base_rotation. Then:
    #   I ω_B = R_IB @ (B ω_B)
    #   qdot_ADAM = [I v_B; I ω_B; P^T qdot_XML] = S_inv @ qvel_MJ
    # where
    #   C_inv = blkdiag(I3, R_IB^T),  S_inv = [[C_inv, 0],[0, P]]
    # For generalized forces (dual map):
    #   tau_MJ = S_inv.T @ tau_ADAM, i.e. base torques use R_IB and joints use P.
    # Remember: the permutation matrix here is an identity.
    Adj, Adj_inv = _base_velocity_transform(base_rotation)
    S_inv = np.block(
        [
            [Adj_inv, np.zeros((6, n_act))],
            [np.zeros((n_act, 6)), permutation],
        ]
    )
    S_inv_force = np.block(
        [
            [Adj_inv.T, np.zeros((6, n_act))],
            [np.zeros((n_act, 6)), permutation.T],
        ]
    )
    return Adj, S_inv, S_inv_force


def _system_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    masses = model.body_mass[:, None]
    return np.sum(masses * data.xipos, axis=0) / np.sum(masses)


def _system_com_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    total_mass = np.sum(model.body_mass)
    jac_total = np.zeros((3, model.nv))
    for body_id in range(1, model.nbody):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
        jac_total += model.body_mass[body_id] * jacp
    return jac_total / total_mass


def _bias_forces(
    model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, gravity: np.ndarray
) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    original_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = gravity
    mujoco.mj_forward(model, data)
    model.opt.gravity[:] = original_gravity
    return data.qfrc_bias.copy()


@pytest.fixture(scope="module", params=DESCRIPTION_NAMES)
def mujoco_setup(request):
    model = _load_model(request.param)

    rng = np.random.default_rng(42)
    qpos = _random_configuration(model, rng)
    qvel = _random_velocity(model, rng)

    data = mujoco.MjData(model)
    _set_state(model, data, qpos, qvel)

    kd = KinDynComputations.from_mujoco_model(model)
    kd.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)
    kd.g = np.concatenate([model.opt.gravity, np.zeros(3)])

    base_rot = R.from_quat(
        data.qpos[3:7], scalar_first=True
    ).as_matrix()  # free joint quaternion is at indices 3:7
    base_pos = data.qpos[:3]

    P, mj_joint_names, adam_joint_names = _joint_permutation(model, kd)
    Adj, S_inv, S_inv_force = _compose_transforms(base_rot, P, kd.rbdalgos.NDoF)
    base_transform = np.eye(4)
    base_transform[:3, :3] = base_rot
    base_transform[:3, 3] = base_pos

    q_joints = P.T @ qpos[7:]  # reorder joints to ADAM ordering
    base_vel = Adj @ qvel[:6]
    qd = P.T @ qvel[6:]
    return dict(
        model=model,
        data=data,
        kd=kd,
        base_transform=base_transform,
        base_vel=base_vel,
        q_joints=q_joints,
        qd=qd,
        qpos=qpos,
        qvel=qvel,
        joint_permutation=P,
        base_transform_matrix=Adj,
        velocity_transform=S_inv,
        force_transform=S_inv_force,
        mj_joint_names=mj_joint_names,
        adam_joint_names=adam_joint_names,
    )


def test_forward_kinematics(mujoco_setup):
    model = mujoco_setup["model"]
    data = mujoco_setup["data"]
    kd = mujoco_setup["kd"]
    base_transform = mujoco_setup["base_transform"]
    q_joints = mujoco_setup["q_joints"]

    for link_name in kd.rbdalgos.model.links:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        if body_id < 0:
            pytest.skip(f"Missing body {link_name} in mujoco model")
        T_adam = kd.forward_kinematics(link_name, base_transform, q_joints)
        R_body = data.xmat[body_id].reshape(3, 3)
        p = data.xpos[body_id]
        T_mj = np.eye(4)
        T_mj[:3, :3] = R_body
        T_mj[:3, 3] = p
        np.testing.assert_allclose(T_adam, T_mj, atol=1e-4, rtol=1e-4)


def test_jacobian(mujoco_setup):
    model = mujoco_setup["model"]
    data = mujoco_setup["data"]
    kd = mujoco_setup["kd"]
    base_transform = mujoco_setup["base_transform"]
    q_joints = mujoco_setup["q_joints"]
    n_act = kd.rbdalgos.NDoF
    C = mujoco_setup["base_transform_matrix"]
    P = mujoco_setup["joint_permutation"]

    for link_name in kd.rbdalgos.model.links:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        if body_id < 0:
            pytest.skip(f"{DESCRIPTION_NAME} missing body {link_name} in mujoco model")
        J_adam = kd.jacobian(link_name, base_transform, q_joints)
        J_adam_j = J_adam[:, -n_act:]
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        J_mj = np.vstack([jacp, jacr])
        J_mj_joint = J_mj[:, 6:]
        J_mj_joint_reordered = J_mj_joint @ P
        J_adam_base = J_adam[:, :6] @ C
        np.testing.assert_allclose(J_mj[:, :6], J_adam_base, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(J_mj_joint_reordered, J_adam_j, atol=1e-4, rtol=1e-4)


def test_mass_matrix(mujoco_setup):
    model = mujoco_setup["model"]
    data = mujoco_setup["data"]
    kd = mujoco_setup["kd"]
    base_transform = mujoco_setup["base_transform"]
    q_joints = mujoco_setup["q_joints"]
    S_inv = mujoco_setup["velocity_transform"]
    M_mj = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M_mj, data.qM)
    M_adam = kd.mass_matrix(base_transform, q_joints)
    # Remove MuJoCo joint armature (rotor inertia) from the full mass matrix.
    M_mj_no_arm = M_mj.copy()
    armature_joint = model.dof_armature[6:]
    M_mj_no_arm[6:, 6:] -= np.diag(armature_joint)
    # Change coordinates: qdot_adam = S qdot_mj, so M_adam = S^{-T} M_mj S^{-1}
    M_expected = S_inv.T @ M_mj_no_arm @ S_inv
    np.testing.assert_allclose(M_expected, M_adam, atol=2e-2, rtol=1e-3)


def test_total_mass(mujoco_setup):
    model = mujoco_setup["model"]
    kd = mujoco_setup["kd"]
    np.testing.assert_allclose(
        kd.get_total_mass(), np.sum(model.body_mass), rtol=1e-6, atol=1e-6
    )


def test_com_position(mujoco_setup):
    model = mujoco_setup["model"]
    data = mujoco_setup["data"]
    kd = mujoco_setup["kd"]
    base_transform = mujoco_setup["base_transform"]
    q_joints = mujoco_setup["q_joints"]
    com_mj = _system_com(model, data)
    com_adam = kd.CoM_position(base_transform, q_joints)
    np.testing.assert_allclose(com_adam, com_mj, atol=1e-4, rtol=1e-4)


def test_com_jacobian(mujoco_setup):
    model = mujoco_setup["model"]
    data = mujoco_setup["data"]
    kd = mujoco_setup["kd"]
    base_transform = mujoco_setup["base_transform"]
    q_joints = mujoco_setup["q_joints"]
    Adj_inv = mujoco_setup["velocity_transform"]

    jac_mj = _system_com_jacobian(model, data)
    jac_mj_expected = jac_mj @ Adj_inv
    jac_adam = kd.CoM_jacobian(base_transform, q_joints)
    np.testing.assert_allclose(jac_adam, jac_mj_expected, atol=1e-4, rtol=1e-4)


def test_bias_force(mujoco_setup):
    kd = mujoco_setup["kd"]
    base_transform = mujoco_setup["base_transform"]
    q_joints = mujoco_setup["q_joints"]
    base_vel = mujoco_setup["base_vel"]
    qd = mujoco_setup["qd"]
    bias_mj = mujoco_setup["data"].qfrc_bias.copy()
    tau_expected = mujoco_setup["force_transform"] @ bias_mj
    tau_adam = kd.bias_force(base_transform, q_joints, base_vel, qd)
    np.testing.assert_allclose(tau_adam, tau_expected, atol=1e-3, rtol=1e-4)


def test_gravity_term(mujoco_setup):
    model = mujoco_setup["model"]
    kd = mujoco_setup["kd"]
    base_transform = mujoco_setup["base_transform"]
    q_joints = mujoco_setup["q_joints"]
    qpos = mujoco_setup["qpos"]
    zeros = np.zeros_like(mujoco_setup["qvel"])
    bias_mj = _bias_forces(model, qpos, zeros, model.opt.gravity.copy())
    tau_expected = mujoco_setup["force_transform"] @ bias_mj
    tau_adam = kd.gravity_term(base_transform, q_joints)
    np.testing.assert_allclose(tau_adam, tau_expected, atol=1e-3, rtol=1e-4)


def test_coriolis_term(mujoco_setup):
    model = mujoco_setup["model"]
    kd = mujoco_setup["kd"]
    base_transform = mujoco_setup["base_transform"]
    q_joints = mujoco_setup["q_joints"]
    base_vel = mujoco_setup["base_vel"]
    qd = mujoco_setup["qd"]
    qpos = mujoco_setup["qpos"]
    qvel = mujoco_setup["qvel"]
    zero_gravity = np.zeros(3)
    bias_mj = _bias_forces(model, qpos, qvel, zero_gravity)
    tau_expected = mujoco_setup["force_transform"] @ bias_mj
    original_g = kd.g
    kd.g = np.concatenate([zero_gravity, np.zeros(3)])
    tau_adam = kd.coriolis_term(base_transform, q_joints, base_vel, qd)
    kd.g = original_g
    np.testing.assert_allclose(tau_adam, tau_expected, atol=1e-3, rtol=1e-4)
