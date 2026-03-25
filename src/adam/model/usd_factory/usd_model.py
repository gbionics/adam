from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from adam.core.spatial_math import SpatialMath
from adam.model.abc_factories import Limits, ModelFactory, Pose
from adam.model.std_factories.std_joint import StdJoint
from adam.model.std_factories.std_link import StdLink
from adam.model.visuals import (
    BoxVisualGeometry,
    CylinderVisualGeometry,
    EmbeddedMeshVisualGeometry,
    SphereVisualGeometry,
    Visual,
    VisualMaterial,
    capsule_mesh_geometry,
)

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
    visuals: list[Visual]
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


def _quat_to_wxyz(q: Any) -> np.ndarray:
    imag = q.GetImaginary()
    return np.array([q.GetReal(), imag[0], imag[1], imag[2]], dtype=float)


def _rotation_from_usd_quat(q: Any) -> R:
    return R.from_quat(_quat_to_wxyz(q), scalar_first=True)


def _vec3_to_np(v: Any) -> np.ndarray:
    return np.array([v[0], v[1], v[2]], dtype=float)


def _is_identity_quat(q: Any) -> bool:
    quat_wxyz = _quat_to_wxyz(q)
    return np.allclose(quat_wxyz, np.array([1.0, 0.0, 0.0, 0.0])) or np.allclose(
        quat_wxyz, np.array([-1.0, 0.0, 0.0, 0.0])
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
        self._xform_cache = self.UsdGeom.XformCache()

        self._links, self._path_to_link_name = self._build_links()
        self._joints = self._build_joints()
        self._child_map = self._build_child_map()
        self._frame_links, self._frame_joints = self._build_frame_xforms()

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

        # USD stores principalAxes as the rotation from the link frame to the
        # principal-inertia frame (R_link_to_principal).  Rotate the diagonal
        # inertia tensor back into the link frame directly, avoiding any
        # Euler-angle conversion and the associated gimbal-lock singularity.
        #
        #   I_link = R^T @ diag(Ixx, Iyy, Izz) @ R
        #
        # where R = R.from_quat(principal_axes) maps link → principal frame.
        R_principal = _rotation_from_usd_quat(principal_axes)
        R_mat = R_principal.as_matrix()
        I_diag = np.diag(diagonal_inertia)
        I_link = R_mat.T @ I_diag @ R_mat

        return USDInertial(
            mass=mass,
            inertia=USDInertia(
                ixx=float(I_link[0, 0]),
                ixy=float(I_link[0, 1]),
                ixz=float(I_link[0, 2]),
                iyy=float(I_link[1, 1]),
                iyz=float(I_link[1, 2]),
                izz=float(I_link[2, 2]),
            ),
            origin=USDOrigin(xyz=center_of_mass, rpy=np.zeros(3)),
        )

    @staticmethod
    def _attr_value(attr: Any, default: Any = None) -> Any:
        if attr is None or not attr.IsValid() or not attr.HasAuthoredValueOpinion():
            return default
        value = attr.Get()
        return default if value is None else value

    @staticmethod
    def _vec_array_to_np(values: Any) -> np.ndarray:
        if values is None:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(
            [[value[0], value[1], value[2]] for value in values],
            dtype=float,
        )

    @staticmethod
    def _triangulate_face_indices(
        face_vertex_counts: np.ndarray,
        face_vertex_indices: np.ndarray,
    ) -> np.ndarray:
        triangles: list[list[int]] = []
        cursor = 0
        for count in face_vertex_counts:
            count = int(count)
            face = face_vertex_indices[cursor : cursor + count]
            cursor += count
            if count < 3:
                continue
            for index in range(1, count - 1):
                triangles.append([int(face[0]), int(face[index]), int(face[index + 1])])
        return np.asarray(triangles, dtype=np.uint32)

    @staticmethod
    def _rotation_from_z_to(axis: np.ndarray) -> np.ndarray:
        axis = np.asarray(axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 1e-12:
            return np.eye(3, dtype=float)
        axis = axis / axis_norm
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        if np.allclose(axis, z_axis):
            return np.eye(3, dtype=float)
        if np.allclose(axis, -z_axis):
            return R.from_rotvec(
                np.pi * np.array([1.0, 0.0, 0.0], dtype=float)
            ).as_matrix()
        rotvec = np.cross(z_axis, axis)
        angle = np.arccos(np.clip(np.dot(z_axis, axis), -1.0, 1.0))
        rotvec = rotvec / np.linalg.norm(rotvec) * angle
        return R.from_rotvec(rotvec).as_matrix()

    def _decompose_transform(
        self,
        transform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        linear = transform[:3, :3]
        translation = transform[:3, 3].copy()
        u, _, vt = np.linalg.svd(linear)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        scale = np.diag(rotation.T @ linear)
        return translation, rotation, scale

    def _pose_from_transform(self, transform: np.ndarray) -> Pose:
        translation, rotation, _scale = self._decompose_transform(transform)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Gimbal lock detected.*",
                category=UserWarning,
            )
            rpy = R.from_matrix(rotation).as_euler("xyz")
        return Pose.build(translation, rpy, self.math)

    def _visual_material(self, prim: Any) -> VisualMaterial | None:
        color_attr = prim.GetAttribute("primvars:displayColor")
        opacity_attr = prim.GetAttribute("primvars:displayOpacity")
        colors = self._attr_value(color_attr)
        opacities = self._attr_value(opacity_attr)
        if colors is None:
            return None

        color = np.asarray([colors[0][0], colors[0][1], colors[0][2]], dtype=float)
        opacity = (
            float(opacities[0]) if opacities is not None and len(opacities) > 0 else 1.0
        )
        return VisualMaterial(
            rgba=(float(color[0]), float(color[1]), float(color[2]), opacity)
        )

    def _is_effectively_visible(self, prim: Any) -> bool:
        imageable = self.UsdGeom.Imageable(prim)
        if not imageable:
            return True
        return imageable.ComputeVisibility() != self.UsdGeom.Tokens.invisible

    def _supported_visual_prim(self, prim: Any) -> bool:
        return (
            prim.IsA(self.UsdGeom.Mesh)
            or prim.IsA(self.UsdGeom.Cube)
            or prim.IsA(self.UsdGeom.Sphere)
            or prim.IsA(self.UsdGeom.Cylinder)
            or prim.IsA(self.UsdGeom.Capsule)
        )

    def _should_include_visual_prim(self, link_path: Any, prim: Any) -> bool:
        prim_path = prim.GetPath()
        if prim_path == link_path or not prim_path.HasPrefix(link_path):
            return False

        if not self._is_effectively_visible(prim):
            return False

        purpose = self._attr_value(prim.GetAttribute("purpose"))
        if purpose in {"guide", "proxy"}:
            return False

        for rigid_path in self._rigid_body_paths:
            if (
                rigid_path != link_path
                and rigid_path.HasPrefix(link_path)
                and prim_path.HasPrefix(rigid_path)
            ):
                return False

        return self._supported_visual_prim(prim)

    def _axis_token_spec(self, axis_token: Any) -> tuple[np.ndarray, int, str]:
        if axis_token == self.UsdGeom.Tokens.x:
            return np.array([1.0, 0.0, 0.0], dtype=float), 0, "x"
        if axis_token == self.UsdGeom.Tokens.y:
            return np.array([0.0, 1.0, 0.0], dtype=float), 1, "y"
        return np.array([0.0, 0.0, 1.0], dtype=float), 2, "z"

    def _relative_transform(self, link_prim: Any, prim: Any) -> np.ndarray:
        world_link = np.asarray(
            self._xform_cache.GetLocalToWorldTransform(link_prim),
            dtype=float,
        ).T
        world_prim = np.asarray(
            self._xform_cache.GetLocalToWorldTransform(prim),
            dtype=float,
        ).T
        return np.linalg.inv(world_link) @ world_prim

    def _visual_from_geom_prim(self, link_prim: Any, prim: Any) -> Visual | None:
        relative_transform = self._relative_transform(link_prim, prim)
        origin = self._pose_from_transform(relative_transform)
        translation, rotation, scale = self._decompose_transform(relative_transform)
        material = self._visual_material(prim)

        if prim.IsA(self.UsdGeom.Cube):
            cube = self.UsdGeom.Cube(prim)
            size = float(cube.GetSizeAttr().Get())
            geometry = BoxVisualGeometry(
                size=tuple(float(size * abs(axis_scale)) for axis_scale in scale)
            )
        elif prim.IsA(self.UsdGeom.Sphere):
            sphere = self.UsdGeom.Sphere(prim)
            radius = float(sphere.GetRadiusAttr().Get())
            geometry = SphereVisualGeometry(
                radius=float(radius * np.max(np.abs(scale)))
            )
        elif prim.IsA(self.UsdGeom.Cylinder):
            cylinder = self.UsdGeom.Cylinder(prim)
            radius = float(cylinder.GetRadiusAttr().Get())
            height = float(cylinder.GetHeightAttr().Get())
            axis_token = self._attr_value(cylinder.GetAxisAttr(), self.UsdGeom.Tokens.z)
            axis_vector, length_axis, _axis_name = self._axis_token_spec(axis_token)
            axis_rotation = self._rotation_from_z_to(axis_vector)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Gimbal lock detected.*",
                    category=UserWarning,
                )
                origin = Pose.build(
                    translation,
                    R.from_matrix(rotation @ axis_rotation).as_euler("xyz"),
                    self.math,
                )
            radial_axes = [idx for idx in range(3) if idx != length_axis]
            radial_scale = float(np.max(np.abs(scale[radial_axes])))
            geometry = CylinderVisualGeometry(
                radius=float(radius * radial_scale),
                length=float(height * abs(scale[length_axis])),
            )
        elif prim.IsA(self.UsdGeom.Capsule):
            capsule = self.UsdGeom.Capsule(prim)
            radius = float(capsule.GetRadiusAttr().Get())
            height = float(capsule.GetHeightAttr().Get())
            axis_token = self._attr_value(capsule.GetAxisAttr(), self.UsdGeom.Tokens.z)
            _axis_vector, length_axis, axis_name = self._axis_token_spec(axis_token)
            radial_axes = [idx for idx in range(3) if idx != length_axis]
            radial_scale = float(np.max(np.abs(scale[radial_axes])))
            geometry = capsule_mesh_geometry(
                radius=float(radius * radial_scale),
                cylindrical_length=float(height * abs(scale[length_axis])),
                axis=axis_name,
            )
        elif prim.IsA(self.UsdGeom.Mesh):
            mesh = self.UsdGeom.Mesh(prim)
            points = self._vec_array_to_np(mesh.GetPointsAttr().Get())
            face_vertex_counts = np.asarray(
                mesh.GetFaceVertexCountsAttr().Get(),
                dtype=int,
            )
            face_vertex_indices = np.asarray(
                mesh.GetFaceVertexIndicesAttr().Get(),
                dtype=int,
            )
            faces = self._triangulate_face_indices(
                face_vertex_counts,
                face_vertex_indices,
            )
            if len(points) == 0 or len(faces) == 0:
                return None
            geometry = EmbeddedMeshVisualGeometry(
                vertices=np.asarray(points * scale.reshape(1, 3), dtype=np.float32),
                faces=np.asarray(faces, dtype=np.uint32),
            )
        else:
            return None

        return Visual(
            origin=origin,
            geometry=geometry,
            material=material,
            name=prim.GetName() or None,
        )

    def _link_visuals(self, link_prim: Any) -> list[Visual]:
        link_path = link_prim.GetPath()
        visuals: list[Visual] = []
        for prim in self.Usd.PrimRange(link_prim, self.Usd.TraverseInstanceProxies()):
            if not self._should_include_visual_prim(link_path, prim):
                continue
            visual = self._visual_from_geom_prim(link_prim, prim)
            if visual is not None:
                visuals.append(visual)
        return visuals

    def _joint_connected_rigid_body_paths(
        self,
        seed_path: str,
        candidate_paths: set[str],
    ) -> set[str]:
        connected_paths = {seed_path}
        changed = True
        while changed:
            changed = False
            for prim in self.stage.TraverseAll():
                if not prim.IsA(self.UsdPhysics.Joint):
                    continue
                joint = self.UsdPhysics.Joint(prim)
                body_paths = set()
                body0_targets = joint.GetBody0Rel().GetTargets()
                body1_targets = joint.GetBody1Rel().GetTargets()
                if body0_targets:
                    body0_path = str(body0_targets[0])
                    if body0_path in candidate_paths:
                        body_paths.add(body0_path)
                if body1_targets:
                    body1_path = str(body1_targets[0])
                    if body1_path in candidate_paths:
                        body_paths.add(body1_path)
                if body_paths and body_paths & connected_paths:
                    new_paths = body_paths - connected_paths
                    if new_paths:
                        connected_paths.update(new_paths)
                        changed = True
        return connected_paths

    def _discover_rigid_body_prims(self) -> list[Any]:
        robot_path = self.robot_prim.GetPath()
        if not self.robot_prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
            selected_paths = {
                str(prim.GetPath())
                for prim in self.stage.TraverseAll()
                if prim.HasAPI(self.UsdPhysics.RigidBodyAPI)
                and prim.GetPath().HasPrefix(robot_path)
            }
        else:
            scope_path = robot_path.GetParentPath()
            candidate_paths = {
                str(prim.GetPath())
                for prim in self.stage.TraverseAll()
                if prim.HasAPI(self.UsdPhysics.RigidBodyAPI)
                and (
                    prim.GetPath() == robot_path or prim.GetPath().HasPrefix(scope_path)
                )
            }
            selected_paths = self._joint_connected_rigid_body_paths(
                seed_path=str(robot_path),
                candidate_paths=candidate_paths,
            )

        return [
            prim
            for prim in self.stage.TraverseAll()
            if prim.HasAPI(self.UsdPhysics.RigidBodyAPI)
            and str(prim.GetPath()) in selected_paths
        ]

    def _build_links(self) -> tuple[list[StdLink], dict[str, str]]:
        links: list[StdLink] = []
        path_to_link_name: dict[str, str] = {}
        names_seen: set[str] = set()
        rigid_body_prims = self._discover_rigid_body_prims()
        robot_path = self.robot_prim.GetPath()

        for prim in rigid_body_prims:
            name = prim.GetName()
            if name in names_seen:
                raise ValueError(
                    f"Duplicate rigid-body name '{name}' under robot root '{robot_path}'. "
                    "Use unique prim names inside the robot subtree."
                )
            names_seen.add(name)
            path_to_link_name[str(prim.GetPath())] = name

        if not rigid_body_prims:
            raise ValueError(
                f"No rigid bodies found for robot root '{robot_path}' in USD stage."
            )

        self._rigid_body_paths = tuple(prim.GetPath() for prim in rigid_body_prims)

        for prim in rigid_body_prims:
            link = USDLink(
                name=prim.GetName(),
                inertial=self._mass_api_to_inertial(prim),
                visuals=self._link_visuals(prim),
                collisions=[],
            )
            links.append(StdLink(link, self.math, visuals=link.visuals))

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

        # USD stores revolute limits in degrees; ADAM uses radians internally.
        if joint_type == "revolute":
            lower = np.radians(lower)
            upper = np.radians(upper)

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

        # Compose the parent-side and child-side local frames into a
        # single URDF-style joint origin:  T_origin = T_parent * T_child^{-1}
        # When localPos1/localRot1 is identity this reduces to the parent frame.
        p0 = _vec3_to_np(pos0)
        p1 = _vec3_to_np(pos1)
        R0 = _rotation_from_usd_quat(rot0)
        R1 = _rotation_from_usd_quat(rot1)

        # Combined rotation: R0 * R1^{-1}
        R_combined = R0 * R1.inv()
        # Combined translation: p0 - R_combined * p1
        p_combined = p0 - R_combined.apply(p1)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Gimbal lock detected.*",
                category=UserWarning,
            )
            rpy_combined = R_combined.as_euler("xyz")

        return USDOrigin(xyz=p_combined, rpy=rpy_combined)

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
                # ADAM-exported USD stores the exact axis in the body frame.
                axis = _vec3_to_np(axis_attr.Get())
            else:
                # The USD axis token is defined in the joint anchor frame.
                # Transform it to the child body frame via R_localRot1,
                # since ADAM (like URDF) expects the axis in the child frame.
                # localRot1 encodes the rotation FROM the joint/anchor frame
                # TO the child body frame.
                axis_in_anchor = self._axis_token_to_vector(
                    joint_schema.GetAxisAttr().Get()
                )
                rot1 = joint.GetLocalRot1Attr().Get()
                if rot1 is not None and not _is_identity_quat(rot1):
                    axis = _rotation_from_usd_quat(rot1).apply(axis_in_anchor)
                else:
                    axis = axis_in_anchor

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
        # Search the entire stage for joints rather than only under the
        # articulation root.  Some converters (e.g. Newton urdf-usd-converter)
        # place joints in a sibling scope (e.g. /Robot/Physics/) rather than
        # under the articulation root (e.g. /Robot/Geometry/root_link).
        # _build_joint_from_prim already filters by checking that body targets
        # belong to our discovered rigid bodies, so a stage-wide scan is safe.
        joints: list[StdJoint] = []

        for prim in self.stage.TraverseAll():
            if not prim.IsA(self.UsdPhysics.Joint):
                continue
            built = self._build_joint_from_prim(prim)
            if built is not None:
                joints.append(built)

        return joints

    def _build_frame_xforms(self) -> tuple[list[StdLink], list[StdJoint]]:
        """Discover child Xform prims of rigid bodies and expose them as frames."""
        import warnings as _warn

        frame_links: list[StdLink] = []
        frame_joints: list[StdJoint] = []
        existing_names = set(self._path_to_link_name.values())

        for parent_path, parent_name in list(self._path_to_link_name.items()):
            parent_prim = self.stage.GetPrimAtPath(parent_path)
            for child_prim in parent_prim.GetChildren():
                if child_prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
                    continue
                if child_prim.IsA(self.UsdPhysics.Joint):
                    continue
                if not child_prim.IsA(self.UsdGeom.Xformable):
                    continue

                child_name = child_prim.GetName()
                if child_name in existing_names:
                    continue

                xformable = self.UsdGeom.Xformable(child_prim)
                local_transform = xformable.GetLocalTransformation()
                translate = local_transform.ExtractTranslation()
                rot_quat = local_transform.ExtractRotationQuat()
                q_xyzw = _quat_to_xyzw(rot_quat)
                R_frame = R.from_quat(q_xyzw)
                with _warn.catch_warnings():
                    _warn.filterwarnings(
                        "ignore",
                        message="Gimbal lock detected.*",
                        category=UserWarning,
                    )
                    rpy = R_frame.as_euler("xyz")

                zero_inertial = USDInertial(
                    mass=0.0,
                    inertia=USDInertia(
                        ixx=0.0, ixy=0.0, ixz=0.0, iyy=0.0, iyz=0.0, izz=0.0
                    ),
                    origin=USDOrigin(xyz=np.zeros(3), rpy=np.zeros(3)),
                )
                frame_link = USDLink(
                    name=child_name,
                    inertial=zero_inertial,
                    visuals=[],
                    collisions=[],
                )
                frame_links.append(StdLink(frame_link, self.math))

                frame_joint = USDJoint(
                    name=f"{parent_name}_to_{child_name}_fixed",
                    parent=parent_name,
                    child=child_name,
                    joint_type="fixed",
                    axis=None,
                    origin=USDOrigin(
                        xyz=np.array(
                            [translate[0], translate[1], translate[2]], dtype=float
                        ),
                        rpy=rpy,
                    ),
                    limit=None,
                )
                frame_joints.append(StdJoint(frame_joint, self.math))
                existing_names.add(child_name)

        return frame_links, frame_joints

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
        return self._joints + self._frame_joints

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
        body_frames = [
            link
            for link in self._links
            if float(link.inertial.mass.array) == 0.0
            and link.name not in self._child_map.keys()
            and not self._has_non_fixed_joint(link.name)
        ]
        return body_frames + self._frame_links
