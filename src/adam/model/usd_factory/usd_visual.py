from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from adam.core.spatial_math import SpatialMath
from adam.model.abc_factories import Pose
from adam.model.visuals import (
    BoxVisualGeometry,
    CylinderVisualGeometry,
    EmbeddedMeshVisualGeometry,
    SphereVisualGeometry,
    Visual,
    VisualMaterial,
    capsule_mesh_geometry,
)


class USDVisualBuilder:
    def __init__(
        self,
        *,
        Usd: Any,
        UsdGeom: Any,
        xform_cache: Any,
        math: SpatialMath,
        rigid_body_paths: tuple[Any, ...],
    ):
        self.Usd = Usd
        self.UsdGeom = UsdGeom
        self._xform_cache = xform_cache
        self.math = math
        self._rigid_body_paths = rigid_body_paths

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

    @staticmethod
    def _decompose_transform(
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

    def link_visuals(self, link_prim: Any) -> list[Visual]:
        link_path = link_prim.GetPath()
        visuals: list[Visual] = []
        for prim in self.Usd.PrimRange(link_prim, self.Usd.TraverseInstanceProxies()):
            if not self._should_include_visual_prim(link_path, prim):
                continue
            visual = self._visual_from_geom_prim(link_prim, prim)
            if visual is not None:
                visuals.append(visual)
        return visuals
