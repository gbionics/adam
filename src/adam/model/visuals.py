from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import urdf_parser_py.urdf
from scipy.spatial.transform import Rotation

from adam.model.abc_factories import Pose


@dataclasses.dataclass(frozen=True, slots=True)
class VisualMaterial:
    rgba: tuple[float, float, float, float] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class BoxVisualGeometry:
    size: tuple[float, float, float]


@dataclasses.dataclass(frozen=True, slots=True)
class CylinderVisualGeometry:
    radius: float
    length: float


@dataclasses.dataclass(frozen=True, slots=True)
class SphereVisualGeometry:
    radius: float


@dataclasses.dataclass(frozen=True, slots=True)
class MeshVisualGeometry:
    filename: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclasses.dataclass(frozen=True, slots=True)
class EmbeddedMeshVisualGeometry:
    vertices: np.ndarray
    faces: np.ndarray


VisualGeometry = (
    BoxVisualGeometry
    | CylinderVisualGeometry
    | SphereVisualGeometry
    | MeshVisualGeometry
    | EmbeddedMeshVisualGeometry
)


@dataclasses.dataclass(frozen=True, slots=True)
class Visual:
    """Normalized visual attached to a link, independent of the source format."""

    origin: Pose
    geometry: VisualGeometry
    material: VisualMaterial | None = None
    name: str | None = None


def _rotation_from_z_axis(axis: str) -> np.ndarray:
    axis = axis.lower()
    if axis == "z":
        return np.eye(3, dtype=np.float32)
    if axis == "x":
        return Rotation.from_euler("y", 90, degrees=True).as_matrix().astype(np.float32)
    if axis == "y":
        return (
            Rotation.from_euler("x", -90, degrees=True).as_matrix().astype(np.float32)
        )
    raise ValueError(f"Unsupported capsule axis {axis!r}. Expected one of x, y, z.")


def capsule_mesh_geometry(
    radius: float,
    cylindrical_length: float,
    *,
    axis: str = "z",
    radial_segments: int = 24,
    hemisphere_segments: int = 12,
) -> EmbeddedMeshVisualGeometry:
    """Build a procedural capsule mesh aligned with the requested axis."""
    radius = float(radius)
    cylindrical_length = max(float(cylindrical_length), 0.0)
    if radius <= 0.0:
        raise ValueError(f"`radius` must be positive. Got {radius}.")
    if radial_segments < 3:
        raise ValueError("`radial_segments` must be at least 3.")
    if hemisphere_segments < 2:
        raise ValueError("`hemisphere_segments` must be at least 2.")

    cylinder_half_length = 0.5 * cylindrical_length
    angles = np.linspace(0.0, 2.0 * np.pi, radial_segments, endpoint=False)
    circle_xy = np.stack((np.cos(angles), np.sin(angles)), axis=1)

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    def append_ring(z_value: float, ring_radius: float) -> np.ndarray:
        start_idx = len(vertices)
        for x_value, y_value in circle_xy:
            vertices.append(
                [ring_radius * x_value, ring_radius * y_value, float(z_value)]
            )
        return np.arange(start_idx, start_idx + radial_segments, dtype=np.uint32)

    bottom_pole = len(vertices)
    vertices.append([0.0, 0.0, -(cylinder_half_length + radius)])

    rings: list[np.ndarray] = []
    lower_phis = np.linspace(-0.5 * np.pi, 0.0, hemisphere_segments + 1)[1:]
    for phi in lower_phis:
        rings.append(
            append_ring(
                -cylinder_half_length + radius * np.sin(phi),
                radius * np.cos(phi),
            )
        )

    upper_phi_values = np.linspace(0.0, 0.5 * np.pi, hemisphere_segments + 1)
    if cylinder_half_length <= 1e-12:
        upper_phi_values = upper_phi_values[1:-1]
    else:
        upper_phi_values = upper_phi_values[:-1]

    for phi in upper_phi_values:
        rings.append(
            append_ring(
                cylinder_half_length + radius * np.sin(phi),
                radius * np.cos(phi),
            )
        )

    top_pole = len(vertices)
    vertices.append([0.0, 0.0, cylinder_half_length + radius])

    first_ring = rings[0]
    last_ring = rings[-1]
    for ring_index in range(radial_segments):
        next_index = (ring_index + 1) % radial_segments
        faces.append(
            [bottom_pole, int(first_ring[next_index]), int(first_ring[ring_index])]
        )

    for lower_ring, upper_ring in zip(rings, rings[1:]):
        for ring_index in range(radial_segments):
            next_index = (ring_index + 1) % radial_segments
            faces.append(
                [
                    int(lower_ring[ring_index]),
                    int(upper_ring[ring_index]),
                    int(upper_ring[next_index]),
                ]
            )
            faces.append(
                [
                    int(lower_ring[ring_index]),
                    int(upper_ring[next_index]),
                    int(lower_ring[next_index]),
                ]
            )

    for ring_index in range(radial_segments):
        next_index = (ring_index + 1) % radial_segments
        faces.append([top_pole, int(last_ring[ring_index]), int(last_ring[next_index])])

    vertices_array = np.asarray(vertices, dtype=np.float32)
    if axis.lower() != "z":
        vertices_array = vertices_array @ _rotation_from_z_axis(axis).T

    return EmbeddedMeshVisualGeometry(
        vertices=vertices_array,
        faces=np.asarray(faces, dtype=np.uint32),
    )


def _resolve_mesh_filename(
    filename: str, resource_roots: tuple[pathlib.Path, ...]
) -> str:
    cleaned = filename[len("file://") :] if filename.startswith("file://") else filename
    candidate = pathlib.Path(cleaned)

    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())

    if cleaned.startswith("package://"):
        package_path = cleaned[len("package://") :]
        package_name, _, relative = package_path.partition("/")
        if package_name and relative:
            for root in resource_roots:
                direct = (
                    root / relative
                    if root.name == package_name
                    else root / package_name / relative
                )
                if direct.exists():
                    return str(direct.resolve())
        return filename

    for root in resource_roots:
        resolved = root / cleaned
        if resolved.exists():
            return str(resolved.resolve())

    if candidate.exists():
        return str(candidate.resolve())

    return filename


def normalize_urdf_visual(
    visual: urdf_parser_py.urdf.Visual,
    *,
    math,
    resource_roots: tuple[pathlib.Path, ...] = (),
) -> Visual:
    origin = (
        Pose.zero(math)
        if visual.origin is None
        else Pose.build(visual.origin.xyz, visual.origin.rpy, math)
    )

    geometry = visual.geometry
    if isinstance(geometry, urdf_parser_py.urdf.Box):
        normalized_geometry = BoxVisualGeometry(
            size=tuple(float(value) for value in geometry.size)
        )
    elif isinstance(geometry, urdf_parser_py.urdf.Cylinder):
        normalized_geometry = CylinderVisualGeometry(
            radius=float(geometry.radius),
            length=float(geometry.length),
        )
    elif isinstance(geometry, urdf_parser_py.urdf.Sphere):
        normalized_geometry = SphereVisualGeometry(radius=float(geometry.radius))
    elif isinstance(geometry, urdf_parser_py.urdf.Mesh):
        normalized_geometry = MeshVisualGeometry(
            filename=_resolve_mesh_filename(geometry.filename, resource_roots),
            scale=(
                tuple(float(value) for value in geometry.scale)
                if geometry.scale is not None
                else (1.0, 1.0, 1.0)
            ),
        )
    else:
        raise NotImplementedError(
            f"The visual type {geometry.__class__.__name__} is not supported"
        )

    material = None
    if visual.material is not None and visual.material.color is not None:
        material = VisualMaterial(
            rgba=tuple(float(value) for value in visual.material.color.rgba)
        )

    return Visual(
        name=getattr(visual, "name", None),
        origin=origin,
        geometry=normalized_geometry,
        material=material,
    )
