import dataclasses
import logging
import os
from itertools import product

import icub_models
import idyntree.bindings as idyntree
import numpy as np
import pytest
import requests

from adam import Representations
from scipy.spatial.transform import Rotation


@dataclasses.dataclass
class State:
    H: np.ndarray
    joints_pos: np.ndarray
    base_vel: np.ndarray
    joints_vel: np.ndarray
    gravity: np.ndarray


@dataclasses.dataclass
class IDynFunctionValues:
    mass_matrix: np.ndarray
    centroidal_momentum_matrix: np.ndarray
    CoM_position: np.ndarray
    CoM_jacobian: np.ndarray
    total_mass: float
    jacobian: np.ndarray
    jacobian_non_actuated: np.ndarray
    jacobian_dot_nu: np.ndarray
    relative_jacobian: np.ndarray
    forward_kinematics: np.ndarray
    forward_kinematics_non_actuated: np.ndarray
    bias_force: np.ndarray
    coriolis_term: np.ndarray
    gravity_term: np.ndarray


@dataclasses.dataclass
class RobotCfg:
    robot_name: str
    velocity_representation: Representations
    model_path: str
    joints_name_list: list
    n_dof: int
    kin_dyn: idyntree.KinDynComputations
    idyn_function_values: IDynFunctionValues


VELOCITY_REPRESENTATIONS = [
    Representations.MIXED_REPRESENTATION,
    Representations.BODY_FIXED_REPRESENTATION,
    Representations.INERTIAL_FIXED_REPRESENTATION,
]

ROBOTS = [
    "iCubGenova04",
    "StickBot",
]


def get_robot_model_path(robot_name: str) -> str:
    if robot_name == "StickBot":
        model_path = "stickbot.urdf"
        if not os.path.exists(model_path):
            url = "https://raw.githubusercontent.com/icub-tech-iit/ergocub-gazebo-simulations/2a9f2d3849610c5731f925ced54b56a3333d8b0b/models/stickBot/model.urdf"
            response = requests.get(url)
            with open(model_path, "wb") as file:
                file.write(response.content)
    else:
        model_path = str(icub_models.get_model_file(robot_name))
    return model_path


TEST_CONFIGURATIONS = list(product(VELOCITY_REPRESENTATIONS, ROBOTS))


@pytest.fixture(scope="module", params=TEST_CONFIGURATIONS, ids=str)
def tests_setup(request) -> RobotCfg | State:

    velocity_representation, robot_name = request.param

    np.random.seed(42)

    model_path = get_robot_model_path(robot_name)

    joints_name_list = [
        "torso_pitch",
        "torso_roll",
        "torso_yaw",
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_shoulder_yaw",
        "l_elbow",
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_shoulder_yaw",
        "r_elbow",
        "l_hip_pitch",
        "l_hip_roll",
        "l_hip_yaw",
        "l_knee",
        "l_ankle_pitch",
        "l_ankle_roll",
        "r_hip_pitch",
        "r_hip_roll",
        "r_hip_yaw",
        "r_knee",
        "r_ankle_pitch",
        "r_ankle_roll",
    ]

    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Showing the robot tree.")

    robot_iDyn = idyntree.ModelLoader()
    robot_iDyn.loadReducedModelFromFile(model_path, joints_name_list)

    kin_dyn = idyntree.KinDynComputations()
    kin_dyn.loadRobotModel(robot_iDyn.model())

    if velocity_representation == Representations.BODY_FIXED_REPRESENTATION:
        idyn_representation = idyntree.BODY_FIXED_REPRESENTATION
    elif velocity_representation == Representations.MIXED_REPRESENTATION:
        idyn_representation = idyntree.MIXED_REPRESENTATION
    elif velocity_representation == Representations.INERTIAL_FIXED_REPRESENTATION:
        idyn_representation = idyntree.INERTIAL_FIXED_REPRESENTATION
    else:
        raise ValueError(f"Unknown velocity representation: {velocity_representation}")
    kin_dyn.setFrameVelocityRepresentation(idyn_representation)

    n_dof = len(joints_name_list)
    # base quantities
    xyz = (np.random.rand(3) - 0.5) * 5
    rpy = (np.random.rand(3) - 0.5) * 5
    base_vel = (np.random.rand(6) - 0.5) * 5
    # joints quantitites
    joints_val = (np.random.rand(n_dof) - 0.5) * 5
    joints_dot_val = (np.random.rand(n_dof) - 0.5) * 5

    g = np.array([0, 0, -9.80665])
    # H_b = SpatialMath().H_from_Pos_RPY(xyz, rpy).array
    R_b = Rotation.from_euler("xyz", rpy).as_matrix()
    H_b = np.eye(4)
    H_b[:3, :3] = R_b
    H_b[:3, 3] = xyz

    state = State(
        H=H_b,
        joints_pos=joints_val,
        base_vel=base_vel,
        joints_vel=joints_dot_val,
        gravity=g,
    )

    idyn_function_values = compute_idyntree_values(kin_dyn, state)

    robot_cfg = RobotCfg(
        robot_name=robot_name,
        velocity_representation=velocity_representation,
        model_path=model_path,
        joints_name_list=joints_name_list,
        n_dof=n_dof,
        kin_dyn=kin_dyn,
        idyn_function_values=idyn_function_values,
    )

    yield robot_cfg, state


def compute_idyntree_values(
    kin_dyn: idyntree.KinDynComputations, state: State
) -> IDynFunctionValues:
    kin_dyn.setRobotState(
        state.H, state.joints_pos, state.base_vel, state.joints_vel, state.gravity
    )

    # mass matrix
    idyn_mass_matrix = idyntree.MatrixDynSize()
    kin_dyn.getFreeFloatingMassMatrix(idyn_mass_matrix)
    idyn_mass_matrix = idyn_mass_matrix.toNumPy()

    # centroidal momentum matrix
    idyn_cmm = idyntree.MatrixDynSize()
    kin_dyn.getCentroidalTotalMomentumJacobian(idyn_cmm)
    idyn_cmm = idyn_cmm.toNumPy()

    # CoM position
    idyn_com = kin_dyn.getCenterOfMassPosition().toNumPy()

    # Com jacobian
    idyn_com_jacobian = idyntree.MatrixDynSize(3, kin_dyn.model().getNrOfDOFs() + 6)
    kin_dyn.getCenterOfMassJacobian(idyn_com_jacobian)
    idyn_com_jacobian = idyn_com_jacobian.toNumPy()

    # total mass
    total_mass = kin_dyn.model().getTotalMass()

    # jacobian
    idyn_jacobian = idyntree.MatrixDynSize(6, kin_dyn.model().getNrOfDOFs() + 6)
    kin_dyn.getFrameFreeFloatingJacobian("l_sole", idyn_jacobian)
    idyn_jacobian = idyn_jacobian.toNumPy()

    # jacobian_non_actuated
    idyn_jacobian_non_actuated = idyntree.MatrixDynSize(
        6, kin_dyn.model().getNrOfDOFs() + 6
    )
    kin_dyn.getFrameFreeFloatingJacobian("head", idyn_jacobian_non_actuated)
    idyn_jacobian_non_actuated = idyn_jacobian_non_actuated.toNumPy()

    # jacobian_dot_nu
    idyn_jacobian_dot_nu = kin_dyn.getFrameBiasAcc("l_sole").toNumPy()

    # relative_jacobian
    # set the base pose to the identity to get the relative jacobian
    kin_dyn.setRobotState(
        np.eye(4),
        state.joints_pos,
        state.base_vel,
        state.joints_vel,
        state.gravity,
    )
    idyn_relative_jacobian = idyntree.MatrixDynSize(
        6, kin_dyn.model().getNrOfDOFs() + 6
    )
    kin_dyn.getFrameFreeFloatingJacobian("l_sole", idyn_relative_jacobian)
    idyn_relative_jacobian = idyn_relative_jacobian.toNumPy()[:, 6:]

    # forward_kinematics
    # set back the state to the original one
    kin_dyn.setRobotState(
        state.H, state.joints_pos, state.base_vel, state.joints_vel, state.gravity
    )
    idyn_forward_kinematics = (
        kin_dyn.getWorldTransform("l_sole").asHomogeneousTransform().toNumPy()
    )

    # forward_kinematics_non_actuated
    idyn_forward_kinematics_non_actuated = (
        kin_dyn.getWorldTransform("head").asHomogeneousTransform().toNumPy()
    )

    # bias_force
    idyn_bias_force = idyntree.FreeFloatingGeneralizedTorques(kin_dyn.model())
    assert kin_dyn.generalizedBiasForces(idyn_bias_force)
    idyn_bias_force = np.concatenate(
        (
            idyn_bias_force.baseWrench().toNumPy(),
            idyn_bias_force.jointTorques().toNumPy(),
        )
    )

    # coriolis_term
    # set gravity to zero to get only the coriolis term
    kin_dyn.setRobotState(
        state.H, state.joints_pos, state.base_vel, state.joints_vel, np.zeros(3)
    )
    idyn_coriolis_term = idyntree.FreeFloatingGeneralizedTorques(kin_dyn.model())
    assert kin_dyn.generalizedBiasForces(idyn_coriolis_term)
    idyn_coriolis_term = np.concatenate(
        (
            idyn_coriolis_term.baseWrench().toNumPy(),
            idyn_coriolis_term.jointTorques().toNumPy(),
        )
    )

    # gravity_term
    # set gravity to the actual value and velocities to zero to get only the gravity term
    kin_dyn.setRobotState(
        state.H,
        state.joints_pos,
        np.zeros(6),
        np.zeros(kin_dyn.model().getNrOfDOFs()),
        state.gravity,
    )
    idyn_gravity_term = idyntree.FreeFloatingGeneralizedTorques(kin_dyn.model())
    assert kin_dyn.generalizedBiasForces(idyn_gravity_term)
    idyn_gravity_term = np.concatenate(
        (
            idyn_gravity_term.baseWrench().toNumPy(),
            idyn_gravity_term.jointTorques().toNumPy(),
        )
    )

    return IDynFunctionValues(
        mass_matrix=idyn_mass_matrix,
        centroidal_momentum_matrix=idyn_cmm,
        CoM_position=idyn_com,
        CoM_jacobian=idyn_com_jacobian,
        total_mass=total_mass,
        jacobian=idyn_jacobian,
        jacobian_non_actuated=idyn_jacobian_non_actuated,
        jacobian_dot_nu=idyn_jacobian_dot_nu,
        relative_jacobian=idyn_relative_jacobian,
        forward_kinematics=idyn_forward_kinematics,
        forward_kinematics_non_actuated=idyn_forward_kinematics_non_actuated,
        bias_force=idyn_bias_force,
        coriolis_term=idyn_coriolis_term,
        gravity_term=idyn_gravity_term,
    )


@pytest.fixture(scope="session")
def device():
    import torch

    """Pick CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(x):
    """Convert a torch tensor to a numpy array, handling gradients if present."""
    if x.device.type == "cuda":
        x = x.cpu()
    return x.detach().numpy()
