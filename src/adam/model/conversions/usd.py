from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from adam.core.spatial_math import ArrayLike
from adam.model.model import Model
from adam.model.std_factories.std_model import URDFModelFactory
from adam.numpy.numpy_like import SpatialMath


def _to_numpy(x: Any) -> np.ndarray:
    val = x.array if isinstance(x, ArrayLike) else x
    return np.asarray(val, dtype=float)


def _to_float(x: Any) -> float:
    return float(_to_numpy(x).reshape(-1)[0])


def _axis_to_usd_token(axis: np.ndarray, tol: float = 1e-5) -> str:
    a = np.asarray(axis, dtype=float).reshape(-1)
    if a.size != 3:
        raise ValueError(f"Joint axis must be 3D, got shape {a.shape}.")

    n = np.linalg.norm(a)
    if n == 0.0:
        raise ValueError(
            "Joint axis has zero norm and cannot be converted to USD axis token."
        )
    a = a / n

    idx = int(np.argmax(np.abs(a)))
    principal = float(a[idx])
    residue = np.delete(np.abs(a), idx)
    if abs(abs(principal) - 1.0) > tol or np.any(residue > tol):
        # Keep a valid USD Physics axis token and store the exact axis in
        # a custom attribute ("adam:axis") handled by the ADAM USD loader.
        return "X"

    token = ["X", "Y", "Z"][idx]
    return token if principal >= 0.0 else f"-{token}"


def _write_joint_axis(joint_prim: Any, axis: np.ndarray, Sdf: Any, Gf: Any) -> None:
    joint_prim.CreateAxisAttr(_axis_to_usd_token(axis))
    axis_n = axis / np.linalg.norm(axis)
    joint_prim.GetPrim().CreateAttribute("adam:axis", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(float(axis_n[0]), float(axis_n[1]), float(axis_n[2]))
    )


def _inertia_to_principal_axes(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # I is a symmetric rotational inertia tensor about the CoM, expressed in
    # the body/link frame of the USD rigid body that will receive principalAxes.
    I_sym = 0.5 * (I + I.T)
    eigvals, eigvecs = np.linalg.eigh(I_sym)

    # Ensure right-handed principal basis for quaternion conversion.
    if np.linalg.det(eigvecs) < 0.0:
        eigvecs[:, 0] *= -1.0

    # USD stores principalAxes as the rotation from the body frame in which I
    # is expressed to the principal-inertia frame, i.e. the transpose/inverse
    # of the matrix whose columns are the principal axes expressed in the body
    # frame.
    quat_wxyz = R.from_matrix(eigvecs.T).as_quat(scalar_first=True)
    return eigvals, quat_wxyz


def model_to_usd_stage(
    model: Model,
    *,
    stage: Any | None = None,
    robot_prim_path: str = "/Robot",
) -> Any:
    """Convert an ADAM model to an OpenUSD stage containing one articulation robot."""
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'pxr' (OpenUSD) package is required for USD conversion."
        ) from exc

    stage = Usd.Stage.CreateInMemory() if stage is None else stage
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)

    robot = UsdGeom.Xform.Define(stage, robot_prim_path)
    robot_prim = robot.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
    stage.SetDefaultPrim(robot_prim)

    # Author assetInfo.name per USD asset structure best practices.
    robot_prim.SetAssetInfoByKey("name", model.name)

    link_paths: dict[str, str] = {}
    all_links = dict(model.links)
    all_links.update(model.frames)

    for link in all_links.values():
        link_path = f"{robot_prim_path}/{link.name}"
        link_paths[link.name] = link_path

        link_xf = UsdGeom.Xform.Define(stage, link_path)
        link_prim = link_xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link_prim)

        mass_api = UsdPhysics.MassAPI.Apply(link_prim)
        mass_api.CreateMassAttr(_to_float(link.inertial.mass))

        inertia = link.inertial.inertia
        I_inertial = np.array(
            [
                [
                    _to_float(inertia.ixx),
                    _to_float(inertia.ixy),
                    _to_float(inertia.ixz),
                ],
                [
                    _to_float(inertia.ixy),
                    _to_float(inertia.iyy),
                    _to_float(inertia.iyz),
                ],
                [
                    _to_float(inertia.ixz),
                    _to_float(inertia.iyz),
                    _to_float(inertia.izz),
                ],
            ],
            dtype=float,
        )
        # ADAM stores the rotational inertia in the inertial frame described by
        # link.inertial.origin, while USD principalAxes is defined from the link
        # frame. Rotate the tensor into the link frame before diagonalizing it.
        inertial_rpy = _to_numpy(link.inertial.origin.rpy).reshape(-1)
        R_link_from_inertial = R.from_euler("xyz", inertial_rpy).as_matrix()
        I_link = R_link_from_inertial @ I_inertial @ R_link_from_inertial.T

        principal_vals, principal_quat = _inertia_to_principal_axes(I_link)

        com_xyz = _to_numpy(link.inertial.origin.xyz).reshape(-1)
        mass_api.CreateCenterOfMassAttr(
            Gf.Vec3f(float(com_xyz[0]), float(com_xyz[1]), float(com_xyz[2]))
        )
        mass_api.CreateDiagonalInertiaAttr(
            Gf.Vec3f(
                float(principal_vals[0]),
                float(principal_vals[1]),
                float(principal_vals[2]),
            )
        )
        mass_api.CreatePrincipalAxesAttr(
            Gf.Quatf(
                float(principal_quat[0]),
                float(principal_quat[1]),
                float(principal_quat[2]),
                float(principal_quat[3]),
            )
        )

    joints_scope_path = f"{robot_prim_path}/__joints__"
    UsdGeom.Scope.Define(stage, joints_scope_path)

    for joint in model.joints.values():
        if joint.parent not in link_paths or joint.child not in link_paths:
            continue

        joint_path = f"{joints_scope_path}/{joint.name}"
        if joint.type in {"revolute", "continuous"}:
            joint_prim = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            axis = _to_numpy(joint.axis).reshape(-1)
            _write_joint_axis(joint_prim, axis, Sdf, Gf)
            if (
                joint.limit is not None
                and np.isfinite(_to_float(joint.limit.lower))
                and np.isfinite(_to_float(joint.limit.upper))
                and joint.type == "revolute"
            ):
                joint_prim.CreateLowerLimitAttr(
                    np.degrees(_to_float(joint.limit.lower))
                )
                joint_prim.CreateUpperLimitAttr(
                    np.degrees(_to_float(joint.limit.upper))
                )
        elif joint.type == "prismatic":
            joint_prim = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            axis = _to_numpy(joint.axis).reshape(-1)
            _write_joint_axis(joint_prim, axis, Sdf, Gf)
            if (
                joint.limit is not None
                and np.isfinite(_to_float(joint.limit.lower))
                and np.isfinite(_to_float(joint.limit.upper))
            ):
                # Prismatic limits are linear (metres) — no angle conversion.
                joint_prim.CreateLowerLimitAttr(_to_float(joint.limit.lower))
                joint_prim.CreateUpperLimitAttr(_to_float(joint.limit.upper))
        elif joint.type == "fixed":
            joint_prim = UsdPhysics.FixedJoint.Define(stage, joint_path)
        else:
            raise ValueError(f"Unsupported joint type for USD conversion: {joint.type}")

        joint_prim.CreateBody0Rel().SetTargets([link_paths[joint.parent]])
        joint_prim.CreateBody1Rel().SetTargets([link_paths[joint.child]])

        xyz = _to_numpy(joint.origin.xyz).reshape(-1)
        rpy = _to_numpy(joint.origin.rpy).reshape(-1)
        quat_wxyz = R.from_euler("xyz", rpy).as_quat(scalar_first=True)

        joint_prim.CreateLocalPos0Attr(
            Gf.Vec3f(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        )
        joint_prim.CreateLocalRot0Attr(
            Gf.Quatf(
                float(quat_wxyz[0]),
                float(quat_wxyz[1]),
                float(quat_wxyz[2]),
                float(quat_wxyz[3]),
            )
        )

        # Keep child-side local frame identity to match ADAM joint semantics.
        joint_prim.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint_prim.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    return stage


def model_to_usd(
    model: Model,
    usd_output_path: str | pathlib.Path,
    *,
    robot_prim_path: str = "/Robot",
) -> pathlib.Path:
    """Convert an ADAM model to a USD file containing one articulation robot."""
    usd_output_path = pathlib.Path(usd_output_path)

    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'pxr' (OpenUSD) package is required for USD conversion."
        ) from exc

    if usd_output_path.exists():
        usd_output_path.unlink()

    stage = Usd.Stage.CreateNew(str(usd_output_path))
    model_to_usd_stage(model, stage=stage, robot_prim_path=robot_prim_path)
    stage.GetRootLayer().Save()
    return usd_output_path


def urdf_to_usd(
    urdf_path: str | pathlib.Path,
    usd_output_path: str | pathlib.Path,
    *,
    joints_name_list: list[str] | None = None,
    robot_prim_path: str = "/Robot",
) -> pathlib.Path:

    math = SpatialMath()
    urdf_factory = URDFModelFactory(path=urdf_path, math=math)

    if joints_name_list is None:
        joints_name_list = [
            j.name for j in urdf_factory.get_joints() if j.type != "fixed"
        ]

    model = Model.build(factory=urdf_factory, joints_name_list=joints_name_list)

    return model_to_usd(
        model=model,
        usd_output_path=usd_output_path,
        robot_prim_path=robot_prim_path,
    )
