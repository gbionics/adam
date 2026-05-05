from __future__ import annotations

from typing import Any, Callable

import numpy as np

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


class MujocoVisualBuilder:
    def __init__(
        self,
        *,
        mujoco: Any,
        mj_model: Any,
        math: SpatialMath,
        normalize_quaternion: Callable[[np.ndarray], np.ndarray],
        quat_to_rpy: Callable[[np.ndarray], np.ndarray],
    ):
        self.mujoco = mujoco
        self.mj_model = mj_model
        self.math = math
        self._normalize_quaternion = normalize_quaternion
        self._quat_to_rpy = quat_to_rpy

    def _geom_name(self, geom_id: int) -> str:
        name = self.mujoco.mj_id2name(
            self.mj_model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        return name if name is not None else f"geom_{geom_id}"

    def _pose_from_pos_quat(self, pos: np.ndarray, quat: np.ndarray) -> Pose:
        return Pose.build(
            np.asarray(pos, dtype=float),
            self._quat_to_rpy(quat),
            self.math,
        )

    def _compiled_mesh_geometry(self, mesh_id: int) -> EmbeddedMeshVisualGeometry:
        vert_start = int(self.mj_model.mesh_vertadr[mesh_id])
        vert_count = int(self.mj_model.mesh_vertnum[mesh_id])
        face_start = int(self.mj_model.mesh_faceadr[mesh_id])
        face_count = int(self.mj_model.mesh_facenum[mesh_id])
        return EmbeddedMeshVisualGeometry(
            vertices=np.asarray(
                self.mj_model.mesh_vert[vert_start : vert_start + vert_count],
                dtype=np.float32,
            ).copy(),
            faces=np.asarray(
                self.mj_model.mesh_face[face_start : face_start + face_count],
                dtype=np.uint32,
            ).copy(),
        )

    def _geom_visual(self, geom_id: int) -> Visual | None:
        geom_type = int(self.mj_model.geom_type[geom_id])
        geom_size = np.asarray(self.mj_model.geom_size[geom_id], dtype=float)
        geom_pos = np.asarray(self.mj_model.geom_pos[geom_id], dtype=float)
        geom_quat = self._normalize_quaternion(
            np.asarray(self.mj_model.geom_quat[geom_id], dtype=float)
        )
        material = VisualMaterial(
            rgba=tuple(float(value) for value in self.mj_model.geom_rgba[geom_id])
        )

        if geom_type == self.mujoco.mjtGeom.mjGEOM_BOX:
            geometry = BoxVisualGeometry(
                size=tuple(float(2.0 * v) for v in geom_size[:3])
            )
            origin = self._pose_from_pos_quat(geom_pos, geom_quat)
        elif geom_type == self.mujoco.mjtGeom.mjGEOM_SPHERE:
            geometry = SphereVisualGeometry(radius=float(geom_size[0]))
            origin = self._pose_from_pos_quat(geom_pos, geom_quat)
        elif geom_type == self.mujoco.mjtGeom.mjGEOM_CYLINDER:
            geometry = CylinderVisualGeometry(
                radius=float(geom_size[0]),
                length=float(2.0 * geom_size[1]),
            )
            origin = self._pose_from_pos_quat(geom_pos, geom_quat)
        elif geom_type == self.mujoco.mjtGeom.mjGEOM_CAPSULE:
            geometry = capsule_mesh_geometry(
                radius=float(geom_size[0]),
                cylindrical_length=float(2.0 * geom_size[1]),
            )
            origin = self._pose_from_pos_quat(geom_pos, geom_quat)
        elif geom_type == self.mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(self.mj_model.geom_dataid[geom_id])
            geometry = self._compiled_mesh_geometry(mesh_id)
            origin = self._pose_from_pos_quat(geom_pos, geom_quat)
        else:
            return None

        return Visual(
            name=self._geom_name(geom_id),
            origin=origin,
            geometry=geometry,
            material=material,
        )

    def _geom_signature(self, geom_id: int) -> tuple:
        return (
            int(self.mj_model.geom_type[geom_id]),
            int(self.mj_model.geom_dataid[geom_id]),
            tuple(float(v) for v in self.mj_model.geom_size[geom_id]),
            tuple(float(v) for v in self.mj_model.geom_pos[geom_id]),
            tuple(float(v) for v in self.mj_model.geom_quat[geom_id]),
        )

    def link_visuals(self, body_id: int) -> list[Visual]:
        visuals: list[Visual] = []
        geom_ids = [
            geom_id
            for geom_id, geom_body_id in enumerate(self.mj_model.geom_bodyid)
            if int(geom_body_id) == body_id
        ]
        non_contact_signatures = {
            self._geom_signature(geom_id)
            for geom_id in geom_ids
            if int(self.mj_model.geom_contype[geom_id]) == 0
            and int(self.mj_model.geom_conaffinity[geom_id]) == 0
        }

        for geom_id in geom_ids:
            signature = self._geom_signature(geom_id)
            if (
                int(self.mj_model.geom_contype[geom_id]) != 0
                or int(self.mj_model.geom_conaffinity[geom_id]) != 0
            ) and signature in non_contact_signatures:
                continue
            visual = self._geom_visual(geom_id)
            if visual is not None:
                visuals.append(visual)
        return visuals
