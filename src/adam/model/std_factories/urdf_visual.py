from __future__ import annotations

import pathlib

import urdf_parser_py.urdf

from adam.model.abc_factories import Pose
from adam.model.visuals import (
    BoxVisualGeometry,
    CylinderVisualGeometry,
    MeshVisualGeometry,
    SphereVisualGeometry,
    Visual,
    VisualMaterial,
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
