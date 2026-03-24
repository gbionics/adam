from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import warnings

import numpy as np
from scipy.spatial.transform import Rotation as R

from adam.core.spatial_math import SpatialMath
from adam.model.abc_factories import Limits, ModelFactory
from adam.model.visuals import (
    BoxVisualGeometry,
    CylinderVisualGeometry,
    EmbeddedMeshVisualGeometry,
    SphereVisualGeometry,
    Visual,
    VisualMaterial,
)
from adam.model.std_factories.std_joint import StdJoint
from adam.model.std_factories.std_link import StdLink

# Type checking only - doesn't execute at runtime
if TYPE_CHECKING:
    import mujoco


@dataclass
class MujocoOrigin:
    xyz: np.ndarray
    rpy: np.ndarray


@dataclass
class MujocoInertia:
    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float


@dataclass
class MujocoInertial:
    mass: float
    inertia: MujocoInertia
    origin: MujocoOrigin


@dataclass
class MujocoLink:
    name: str
    inertial: MujocoInertial
    visuals: list[Visual]
    collisions: list


@dataclass
class MujocoJoint:
    name: str
    parent: str
    child: str
    joint_type: str
    axis: Optional[np.ndarray]
    origin: MujocoOrigin
    limit: Optional[Limits]


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _rotate_vector(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector using quaternion [w, x, y, z]."""
    rot = R.from_quat(quat, scalar_first=True).as_matrix()
    return rot @ vec


def _quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Gimbal lock detected.*",
            category=UserWarning,
        )
        return R.from_quat(quat, scalar_first=True).as_euler("xyz")


class MujocoModelFactory(ModelFactory):
    """Factory that builds a model starting from a mujoco.MjModel."""

    def __init__(self, mj_model: "mujoco.MjModel", math: SpatialMath):
        self.math = math
        self.mujoco = self._import_mujoco()
        self.mj_model = self._model_exists(mj_model)
        fallback_name = "mujoco_model"
        self.name = getattr(self.mj_model, "name", None) or fallback_name

        self._links = self._build_links()
        self._child_map = self._build_child_map()
        self._joints = self._build_joints()
        self._site_links, self._site_joints = self._build_sites()

    def _import_mujoco(self):
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise ImportError(
                "The 'MuJoCo' package is required to load MuJoCo models."
            ) from exc
        return mujoco

    def _model_exists(self, mj_model):
        if isinstance(mj_model, self.mujoco.MjModel):
            return mj_model

        raise ValueError(
            f"Expected a MuJoCo MjModel object, but got {type(mj_model).__name__}."
        )

    def _body_name(self, body_id: int) -> str:
        name = self.mujoco.mj_id2name(
            self.mj_model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        return name if name is not None else f"body_{body_id}"

    def _joint_name(self, joint_id: int) -> str:
        name = self.mujoco.mj_id2name(
            self.mj_model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        return name if name is not None else f"joint_{joint_id}"

    def _link_inertial(self, body_id: int) -> MujocoInertial:
        mass = float(self.mj_model.body_mass[body_id])
        inertia_diagonal = self.mj_model.body_inertia[body_id]
        inertia = MujocoInertia(
            ixx=float(inertia_diagonal[0]),
            ixy=0.0,
            ixz=0.0,
            iyy=float(inertia_diagonal[1]),
            iyz=0.0,
            izz=float(inertia_diagonal[2]),
        )

        ipos = np.array(self.mj_model.body_ipos[body_id], dtype=float)
        iquat = _normalize_quaternion(
            np.array(self.mj_model.body_iquat[body_id], dtype=float)
        )
        origin = MujocoOrigin(
            xyz=ipos,
            rpy=R.from_quat(iquat, scalar_first=True).as_euler("xyz"),
        )
        return MujocoInertial(mass=mass, inertia=inertia, origin=origin)

    def _geom_name(self, geom_id: int) -> str:
        name = self.mujoco.mj_id2name(
            self.mj_model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        return name if name is not None else f"geom_{geom_id}"

    def _pose_from_pos_quat(
        self, pos: np.ndarray, quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(pos, dtype=float), _quat_to_rpy(quat)

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
        geom_quat = _normalize_quaternion(
            np.asarray(self.mj_model.geom_quat[geom_id], dtype=float)
        )
        material = VisualMaterial(
            rgba=tuple(float(value) for value in self.mj_model.geom_rgba[geom_id])
        )

        if geom_type == self.mujoco.mjtGeom.mjGEOM_BOX:
            geometry = BoxVisualGeometry(size=tuple(float(2.0 * v) for v in geom_size[:3]))
            xyz, rpy = self._pose_from_pos_quat(geom_pos, geom_quat)
        elif geom_type == self.mujoco.mjtGeom.mjGEOM_SPHERE:
            geometry = SphereVisualGeometry(radius=float(geom_size[0]))
            xyz, rpy = self._pose_from_pos_quat(geom_pos, geom_quat)
        elif geom_type in (
            self.mujoco.mjtGeom.mjGEOM_CYLINDER,
            self.mujoco.mjtGeom.mjGEOM_CAPSULE,
        ):
            geometry = CylinderVisualGeometry(
                radius=float(geom_size[0]),
                length=float(2.0 * geom_size[1]),
            )
            xyz, rpy = self._pose_from_pos_quat(geom_pos, geom_quat)
        elif geom_type == self.mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(self.mj_model.geom_dataid[geom_id])
            xyz, rpy = self._pose_from_pos_quat(geom_pos, geom_quat)
            geometry = self._compiled_mesh_geometry(mesh_id)
        else:
            return None

        return Visual(
            name=self._geom_name(geom_id),
            origin=MujocoOrigin(xyz=xyz, rpy=rpy),
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

    def _link_visuals(self, body_id: int) -> list[Visual]:
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

    def _build_links(self) -> list[StdLink]:
        links: list[StdLink] = []
        for body_id in range(1, self.mj_model.nbody):
            link = MujocoLink(
                name=self._body_name(body_id),
                inertial=self._link_inertial(body_id),
                visuals=self._link_visuals(body_id),
                collisions=[],
            )
            links.append(StdLink(link, self.math, visuals=link.visuals))
        return links

    def _build_child_map(self) -> dict[str, list[str]]:
        child_map: dict[str, list[str]] = {}
        for body_id in range(1, self.mj_model.nbody):
            parent_id = int(self.mj_model.body_parentid[body_id])
            parent_name = self._body_name(parent_id) if parent_id > 0 else None
            if parent_name is None:
                continue
            child_map.setdefault(parent_name, []).append(self._body_name(body_id))
        return child_map

    def _joint_origin(self, body_id: int, joint_id: Optional[int]) -> MujocoOrigin:
        body_pos = np.array(self.mj_model.body_pos[body_id], dtype=float)
        body_quat = _normalize_quaternion(
            np.array(self.mj_model.body_quat[body_id], dtype=float)
        )
        xyz = body_pos
        if joint_id is not None:
            j_pos = np.array(self.mj_model.jnt_pos[joint_id], dtype=float)
            if np.linalg.norm(j_pos) > 0.0:
                xyz = xyz + _rotate_vector(body_quat, j_pos)
        rpy = R.from_quat(body_quat, scalar_first=True).as_euler("xyz")
        return MujocoOrigin(xyz=xyz, rpy=rpy)

    def _build_limits(self, joint_id: int, joint_type: str) -> Optional[Limits]:
        if joint_type == "fixed":
            return None
        limited = bool(self.mj_model.jnt_limited[joint_id])
        if not limited:
            return None
        lower, upper = self.mj_model.jnt_range[joint_id]
        return Limits(lower=lower, upper=upper, effort=None, velocity=None)

    def _joint_type(self, mj_type: int) -> str:
        if mj_type == self.mujoco.mjtJoint.mjJNT_HINGE:
            return "revolute"
        if mj_type == self.mujoco.mjtJoint.mjJNT_SLIDE:
            return "prismatic"
        return "unsupported"

    def _build_joint(
        self,
        body_id: int,
        joint_id: Optional[int],
        parent_name: str,
        joint_type: str,
    ) -> StdJoint:
        child_name = self._body_name(body_id)
        name = (
            self._joint_name(joint_id)
            if joint_id is not None
            else f"{parent_name}_to_{child_name}_fixed"
        )
        axis = (
            np.array(self.mj_model.jnt_axis[joint_id], dtype=float)
            if joint_type != "fixed" and joint_id is not None
            else None
        )
        origin = self._joint_origin(body_id, joint_id)
        limit = (
            self._build_limits(joint_id, joint_type) if joint_id is not None else None
        )
        joint = MujocoJoint(
            name=name,
            parent=parent_name,
            child=child_name,
            joint_type=joint_type,
            axis=axis,
            origin=origin,
            limit=limit,
        )
        return StdJoint(joint, self.math)

    def _build_joints(self) -> list[StdJoint]:
        joints: list[StdJoint] = []
        for body_id in range(1, self.mj_model.nbody):
            parent_id = int(self.mj_model.body_parentid[body_id])
            if parent_id < 1:
                continue
            parent_name = self._body_name(parent_id)
            joint_start = int(self.mj_model.body_jntadr[body_id])
            joint_num = int(self.mj_model.body_jntnum[body_id])

            if joint_num == 0:
                joints.append(
                    self._build_joint(
                        body_id=body_id,
                        joint_id=None,
                        parent_name=parent_name,
                        joint_type="fixed",
                    )
                )
                continue

            for joint_id in range(joint_start, joint_start + joint_num):
                joint_type = self._joint_type(int(self.mj_model.jnt_type[joint_id]))
                if joint_type == "unsupported":
                    # Skip free/ball joints; base pose is provided externally.
                    continue
                joints.append(
                    self._build_joint(
                        body_id=body_id,
                        joint_id=joint_id,
                        parent_name=parent_name,
                        joint_type=joint_type,
                    )
                )
        return joints

    def _site_name(self, site_id: int) -> str:
        name = self.mujoco.mj_id2name(
            self.mj_model, self.mujoco.mjtObj.mjOBJ_SITE, site_id
        )
        return name if name is not None else f"site_{site_id}"

    def _build_sites(self) -> tuple[list[StdLink], list[StdJoint]]:
        """Build zero-mass links and fixed joints for MuJoCo sites."""
        site_links: list[StdLink] = []
        site_joints: list[StdJoint] = []
        existing_names = {link.name for link in self._links}

        for site_id in range(self.mj_model.nsite):
            site_name = self._site_name(site_id)
            if site_name in existing_names:
                continue

            body_id = int(self.mj_model.site_bodyid[site_id])
            if body_id < 1:
                # Site attached to the world body; skip.
                continue
            parent_name = self._body_name(body_id)

            site_pos = np.array(self.mj_model.site_pos[site_id], dtype=float)
            site_quat = _normalize_quaternion(
                np.array(self.mj_model.site_quat[site_id], dtype=float)
            )
            rpy = R.from_quat(site_quat, scalar_first=True).as_euler("xyz")

            zero_inertia = MujocoInertia(
                ixx=0.0, ixy=0.0, ixz=0.0, iyy=0.0, iyz=0.0, izz=0.0
            )
            zero_origin = MujocoOrigin(xyz=np.zeros(3), rpy=np.zeros(3))
            site_link = MujocoLink(
                name=site_name,
                inertial=MujocoInertial(
                    mass=0.0, inertia=zero_inertia, origin=zero_origin
                ),
                visuals=[],
                collisions=[],
            )
            site_links.append(StdLink(site_link, self.math))

            site_joint = MujocoJoint(
                name=f"{parent_name}_to_{site_name}_fixed",
                parent=parent_name,
                child=site_name,
                joint_type="fixed",
                axis=None,
                origin=MujocoOrigin(xyz=site_pos, rpy=rpy),
                limit=None,
            )
            site_joints.append(StdJoint(site_joint, self.math))
            existing_names.add(site_name)

        return site_links, site_joints

    def build_joint(self, joint) -> StdJoint:  # pragma: no cover - required by ABC
        raise NotImplementedError("MujocoModelFactory does not build joints externally")

    def build_link(self, link) -> StdLink:  # pragma: no cover - required by ABC
        raise NotImplementedError("MujocoModelFactory does not build links externally")

    def get_joints(self) -> list[StdJoint]:
        return self._joints + self._site_joints

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
        return body_frames + self._site_links
