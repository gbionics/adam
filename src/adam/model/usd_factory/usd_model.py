from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from adam.core.spatial_math import SpatialMath
from adam.model.abc_factories import Limits, ModelFactory
from adam.model.std_factories.std_joint import StdJoint
from adam.model.std_factories.std_link import StdLink

if TYPE_CHECKING:
    from pxr import Usd


@dataclass
class USDOrigin:
    xyz: np.ndarray
    rpy: np.ndarray


@dataclass
class USDInertia:
    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float


@dataclass
class USDInertial:
    mass: float
    inertia: USDInertia
    origin: USDOrigin


@dataclass
class USDLink:
    name: str
    inertial: USDInertial
    visuals: list
    collisions: list


@dataclass
class USDJoint:
    name: str
    parent: str
    child: str
    joint_type: str
    axis: Optional[np.ndarray]
    origin: USDOrigin
    limit: Optional[Limits]


def _quat_to_xyzw(q: Any) -> np.ndarray:
    imag = q.GetImaginary()
    return np.array([imag[0], imag[1], imag[2], q.GetReal()], dtype=float)


def _quat_to_rpy(q: Any) -> np.ndarray:
    quat_xyzw = _quat_to_xyzw(q)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Gimbal lock detected.*",
            category=UserWarning,
        )
        return R.from_quat(quat_xyzw).as_euler("xyz")


def _vec3_to_np(v: Any) -> np.ndarray:
    return np.array([v[0], v[1], v[2]], dtype=float)


def _is_identity_quat(q: Any) -> bool:
    quat_xyzw = _quat_to_xyzw(q)
    return np.allclose(quat_xyzw, np.array([0.0, 0.0, 0.0, 1.0])) or np.allclose(
        quat_xyzw, np.array([0.0, 0.0, 0.0, -1.0])
    )


class USDModelFactory(ModelFactory):
    """Factory that builds a robot model from an OpenUSD stage/path."""

    def __init__(
        self,
        usd_source: str | pathlib.Path | "Usd.Stage",
        math: SpatialMath,
        robot_prim_path: str | None = None,
    ):
        self.math = math
        self.Usd, self.UsdGeom, self.UsdPhysics, self.Gf = self._import_openusd()
        self.stage = self._load_stage(usd_source)
        self.robot_prim = self._resolve_robot_prim(robot_prim_path)
        self.name = self.robot_prim.GetName() or "usd_robot"

        self._links, self._path_to_link_name = self._build_links()
        self._joints = self._build_joints()
        self._child_map = self._build_child_map()

    def _import_openusd(self):
        try:
            from pxr import Gf, Usd, UsdGeom, UsdPhysics
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'pxr' (OpenUSD) package is required to load USD models."
            ) from exc
        return Usd, UsdGeom, UsdPhysics, Gf

    def _load_stage(self, usd_source: str | pathlib.Path | "Usd.Stage"):
        if isinstance(usd_source, self.Usd.Stage):
            return usd_source

        if isinstance(usd_source, pathlib.Path):
            path = usd_source
        elif isinstance(usd_source, str):
            path = pathlib.Path(usd_source)
        else:
            raise ValueError(
                f"Unsupported USD source type: {type(usd_source).__name__}. "
                "Expected a path/string or pxr.Usd.Stage."
            )

        stage = self.Usd.Stage.Open(str(path))
        if stage is None:
            raise ValueError(f"Unable to open USD stage from path: {path}")
        return stage

    def _find_articulation_roots(self) -> list[Any]:
        roots = []
        for prim in self.stage.TraverseAll():
            if prim.HasAPI(self.UsdPhysics.ArticulationRootAPI):
                roots.append(prim)
        return roots

    def _resolve_robot_prim(self, robot_prim_path: str | None):
        if robot_prim_path is not None:
            prim = self.stage.GetPrimAtPath(robot_prim_path)
            if not prim.IsValid():
                raise ValueError(
                    f"robot_prim_path '{robot_prim_path}' does not exist in the USD stage."
                )
            if not prim.HasAPI(self.UsdPhysics.ArticulationRootAPI):
                raise ValueError(
                    f"Prim '{robot_prim_path}' is not a UsdPhysics articulation root."
                )
            return prim

        roots = self._find_articulation_roots()
        if len(roots) == 0:
            raise ValueError(
                "No UsdPhysics articulation root found in USD stage. "
                "Provide 'robot_prim_path' in the description payload if needed."
            )
        if len(roots) > 1:
            paths = [str(p.GetPath()) for p in roots]
            raise ValueError(
                "Multiple articulation roots found in USD stage. "
                "Select the robot explicitly with description={'usd_path': ..., 'robot_prim_path': ...}. "
                f"Candidates: {paths}"
            )
        return roots[0]

    def _axis_token_to_vector(self, axis_token: Any) -> Optional[np.ndarray]:
        if axis_token is None:
            return None
        token = str(axis_token).strip().upper()
        if token in {"X", "+X"}:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        if token in {"Y", "+Y"}:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        if token in {"Z", "+Z"}:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        if token == "-X":
            return np.array([-1.0, 0.0, 0.0], dtype=float)
        if token == "-Y":
            return np.array([0.0, -1.0, 0.0], dtype=float)
        if token == "-Z":
            return np.array([0.0, 0.0, -1.0], dtype=float)
        raise ValueError(f"Unsupported USD joint axis token: {axis_token}")

    def _mass_api_to_inertial(self, prim: Any) -> USDInertial:
        mass_api = self.UsdPhysics.MassAPI.Get(self.stage, prim.GetPath())

        mass_attr = mass_api.GetMassAttr()
        diagonal_inertia_attr = mass_api.GetDiagonalInertiaAttr()
        center_of_mass_attr = mass_api.GetCenterOfMassAttr()
        principal_axes_attr = mass_api.GetPrincipalAxesAttr()

        mass = (
            float(mass_attr.Get())
            if mass_attr.IsValid() and mass_attr.HasAuthoredValueOpinion()
            else 0.0
        )
        diagonal_inertia = (
            _vec3_to_np(diagonal_inertia_attr.Get())
            if diagonal_inertia_attr.IsValid()
            and diagonal_inertia_attr.HasAuthoredValueOpinion()
            else np.zeros(3)
        )
        center_of_mass = (
            _vec3_to_np(center_of_mass_attr.Get())
            if center_of_mass_attr.IsValid()
            and center_of_mass_attr.HasAuthoredValueOpinion()
            else np.zeros(3)
        )
        principal_axes = (
            principal_axes_attr.Get()
            if principal_axes_attr.IsValid()
            and principal_axes_attr.HasAuthoredValueOpinion()
            else self.Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        )

        return USDInertial(
            mass=mass,
            inertia=USDInertia(
                ixx=float(diagonal_inertia[0]),
                ixy=0.0,
                ixz=0.0,
                iyy=float(diagonal_inertia[1]),
                iyz=0.0,
                izz=float(diagonal_inertia[2]),
            ),
            origin=USDOrigin(
                xyz=center_of_mass,
                rpy=_quat_to_rpy(principal_axes),
            ),
        )

    def _build_links(self) -> tuple[list[StdLink], dict[str, str]]:
        links: list[StdLink] = []
        path_to_link_name: dict[str, str] = {}
        names_seen: set[str] = set()
        robot_path = self.robot_prim.GetPath()

        for prim in self.stage.TraverseAll():
            if not prim.GetPath().HasPrefix(robot_path):
                continue
            if not prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
                continue

            name = prim.GetName()
            if name in names_seen:
                raise ValueError(
                    f"Duplicate rigid-body name '{name}' under robot root '{robot_path}'. "
                    "Use unique prim names inside the robot subtree."
                )
            names_seen.add(name)

            link = USDLink(
                name=name,
                inertial=self._mass_api_to_inertial(prim),
                visuals=[],
                collisions=[],
            )
            links.append(StdLink(link, self.math))
            path_to_link_name[str(prim.GetPath())] = name

        if not links:
            raise ValueError(
                f"No rigid bodies found under robot root '{robot_path}' in USD stage."
            )

        return links, path_to_link_name

    def _joint_schema_and_type_from_prim(self, prim: Any) -> tuple[Any, str]:
        if prim.IsA(self.UsdPhysics.RevoluteJoint):
            return self.UsdPhysics.RevoluteJoint(prim), "revolute"
        if prim.IsA(self.UsdPhysics.PrismaticJoint):
            return self.UsdPhysics.PrismaticJoint(prim), "prismatic"
        if prim.IsA(self.UsdPhysics.FixedJoint):
            return self.UsdPhysics.FixedJoint(prim), "fixed"
        raise ValueError(
            f"Unsupported USD joint type '{prim.GetTypeName()}' at {prim.GetPath()}."
        )

    def _joint_limits(self, joint_schema: Any, joint_type: str) -> Optional[Limits]:
        if joint_type == "fixed":
            return None

        lower_attr = joint_schema.GetLowerLimitAttr()
        upper_attr = joint_schema.GetUpperLimitAttr()
        has_limits = (
            lower_attr.IsValid()
            and upper_attr.IsValid()
            and (
                lower_attr.HasAuthoredValueOpinion()
                or upper_attr.HasAuthoredValueOpinion()
            )
        )
        if not has_limits:
            return None

        lower = float(lower_attr.Get())
        upper = float(upper_attr.Get())
        return Limits(lower=lower, upper=upper, effort=None, velocity=None)

    def _local_pose_to_origin(self, joint: Any) -> USDOrigin:
        pos0 = joint.GetLocalPos0Attr().Get()
        rot0 = joint.GetLocalRot0Attr().Get()
        pos1 = joint.GetLocalPos1Attr().Get()
        rot1 = joint.GetLocalRot1Attr().Get()

        if pos0 is None:
            pos0 = self.Gf.Vec3f(0.0, 0.0, 0.0)
        if rot0 is None:
            rot0 = self.Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        if pos1 is None:
            pos1 = self.Gf.Vec3f(0.0, 0.0, 0.0)
        if rot1 is None:
            rot1 = self.Gf.Quatf(1.0, 0.0, 0.0, 0.0)

        # Current implementation assumes the child-side local joint frame is identity.
        # This is true for common robot exports and keeps a direct URDF-style mapping.
        if not np.allclose(_vec3_to_np(pos1), np.zeros(3)) or not _is_identity_quat(
            rot1
        ):
            raise ValueError(
                "USD joint localPos1/localRot1 must be identity for this loader. "
                f"Found non-identity at joint {joint.GetPath()}."
            )

        return USDOrigin(xyz=_vec3_to_np(pos0), rpy=_quat_to_rpy(rot0))

    def _build_joint_from_prim(self, prim: Any) -> Optional[StdJoint]:
        joint = self.UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()

        if not body1_targets:
            return None

        child_path = str(body1_targets[0])
        parent_path = str(body0_targets[0]) if body0_targets else None

        child_name = self._path_to_link_name.get(child_path)
        if child_name is None:
            return None

        # If the parent is world/outside articulation, keep child as tree root by skipping this joint.
        if parent_path is None:
            return None
        parent_name = self._path_to_link_name.get(parent_path)
        if parent_name is None:
            return None

        joint_schema, joint_type = self._joint_schema_and_type_from_prim(prim)
        if joint_type == "fixed":
            axis = None
        else:
            axis_attr = prim.GetAttribute("adam:axis")
            if axis_attr.IsValid() and axis_attr.HasAuthoredValueOpinion():
                axis = _vec3_to_np(axis_attr.Get())
            else:
                axis = self._axis_token_to_vector(joint_schema.GetAxisAttr().Get())

        usd_joint = USDJoint(
            name=prim.GetName(),
            parent=parent_name,
            child=child_name,
            joint_type=joint_type,
            axis=axis,
            origin=self._local_pose_to_origin(joint),
            limit=self._joint_limits(joint_schema, joint_type),
        )
        return StdJoint(usd_joint, self.math)

    def _build_joints(self) -> list[StdJoint]:
        joints: list[StdJoint] = []
        robot_path = self.robot_prim.GetPath()

        for prim in self.stage.TraverseAll():
            if not prim.GetPath().HasPrefix(robot_path):
                continue
            if not prim.IsA(self.UsdPhysics.Joint):
                continue
            built = self._build_joint_from_prim(prim)
            if built is not None:
                joints.append(built)

        return joints

    def _build_child_map(self) -> dict[str, list[str]]:
        child_map: dict[str, list[str]] = {}
        for joint in self._joints:
            child_map.setdefault(joint.parent, []).append(joint.child)
        return child_map

    def build_joint(self, joint) -> StdJoint:  # pragma: no cover - required by ABC
        raise NotImplementedError("USDModelFactory does not build joints externally")

    def build_link(self, link) -> StdLink:  # pragma: no cover - required by ABC
        raise NotImplementedError("USDModelFactory does not build links externally")

    def get_joints(self) -> list[StdJoint]:
        return self._joints

    def _has_non_fixed_joint(self, link_name: str) -> bool:
        return any(j.child == link_name and j.type != "fixed" for j in self._joints)

    def get_links(self) -> list[StdLink]:
        return [
            link
            for link in self._links
            if (
                float(link.inertial.mass.array) != 0.0
                or link.name in self._child_map.keys()
                or self._has_non_fixed_joint(link.name)
            )
        ]

    def get_frames(self) -> list[StdLink]:
        return [
            link
            for link in self._links
            if float(link.inertial.mass.array) == 0.0
            and link.name not in self._child_map.keys()
            and not self._has_non_fixed_joint(link.name)
        ]
