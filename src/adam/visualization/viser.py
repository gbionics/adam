from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import trimesh
except ImportError:  # pragma: no cover - optional dependency
    trimesh = None

try:
    import viser
except ImportError:  # pragma: no cover - optional dependency
    viser = None

from adam.model.abc_factories import Pose
from adam.model.visuals import (
    BoxVisualGeometry,
    CylinderVisualGeometry,
    EmbeddedMeshVisualGeometry,
    MeshVisualGeometry,
    SphereVisualGeometry,
    Visual,
)

_IDENTITY_WXYZ = R.identity().as_quat(scalar_first=True)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "array"):
        value = value.array
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "full"):
        value = value.full()
    return np.asarray(value, dtype=float)


def _transform_to_viser(transform: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = _to_numpy(transform)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {matrix.shape!r}.")
    return (
        R.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True),
        matrix[:3, 3].copy(),
    )


def _pose_to_matrix(origin: Pose) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R.from_euler("xyz", _to_numpy(origin.rpy)).as_matrix()
    transform[:3, 3] = _to_numpy(origin.xyz)
    return transform


def _normalize_rgba(
    rgba: tuple[float, float, float, float] | None,
    default_rgb: tuple[int, int, int],
) -> tuple[tuple[int, int, int], float | None]:
    if rgba is None:
        return default_rgb, None

    rgba_array = np.asarray(rgba, dtype=float)
    if np.max(rgba_array[:3]) <= 1.0:
        rgb = tuple(
            int(np.clip(round(channel * 255.0), 0, 255)) for channel in rgba_array[:3]
        )
    else:
        rgb = tuple(int(np.clip(round(channel), 0, 255)) for channel in rgba_array[:3])

    alpha = float(rgba_array[3])
    return rgb, None if alpha >= 0.999 else alpha


def _extract_model(kindyn: Any):
    if hasattr(kindyn, "model"):
        return kindyn.model
    if hasattr(kindyn, "rbdalgos") and hasattr(kindyn.rbdalgos, "model"):
        return kindyn.rbdalgos.model
    raise TypeError(
        "Visualizer expects a KinDynComputations-like object exposing `model` "
        "and `link_poses()`."
    )


def _normalize_node_name(root_name: str) -> str:
    if not root_name.startswith("/"):
        root_name = f"/{root_name}"
    return root_name.rstrip("/")


def _scene_path(name: str, *, root_name: str | None = None) -> str:
    if name.startswith("/"):
        return _normalize_node_name(name)
    if root_name is None:
        return _normalize_node_name(name)
    return f"{_normalize_node_name(root_name)}/{name.lstrip('/')}"


class ModelHandle:
    """Scene node bundle for one robot instance attached to a Visualizer."""

    def __init__(
        self,
        visualizer: Visualizer,
        kindyn: Any,
        *,
        root_name: str,
        base_transform: Any | None = None,
        joint_positions: Any | None = None,
        show_frames: bool = False,
        axes_length: float = 0.1,
        axes_radius: float = 0.01,
        origin_radius: float | None = None,
        default_color: tuple[int, int, int] = (160, 160, 160),
    ) -> None:
        self.visualizer = visualizer
        self.kindyn = kindyn
        self.model = _extract_model(kindyn)
        self.root_name = _normalize_node_name(root_name)
        self.show_frames = show_frames
        self.axes_length = axes_length
        self.axes_radius = axes_radius
        self.origin_radius = origin_radius
        self.default_color = default_color
        self._rendered = False
        self._current_base_transform = np.eye(4, dtype=float)
        self._current_joint_positions = np.zeros(self.model.NDoF, dtype=float)
        self._frame_handles: dict[str, Any] = {}
        self._visual_handles: dict[tuple[str, int], Any] = {}
        self._visual_local_transforms: dict[tuple[str, int], np.ndarray] = {}
        self._joint_slider_handles: dict[str, Any] = {}
        self._joint_slider_defaults = np.zeros(self.model.NDoF, dtype=float)
        self._joint_slider_sync_active = False
        self._tree_link_names = tuple(node.name for node in self.model.tree)
        self._visuals_by_link = {
            node.name: list(node.link.visuals)
            for node in self.model.tree
            if node.link.visuals
        }

        self.render()
        self.update(
            np.eye(4, dtype=float) if base_transform is None else base_transform,
            (
                np.zeros(self.model.NDoF, dtype=float)
                if joint_positions is None
                else joint_positions
            ),
        )

    def scene_path(self, name: str) -> str:
        return _scene_path(name, root_name=self.root_name)

    def render(self) -> None:
        if self._rendered:
            return

        if self.show_frames:
            for node in self.model.tree:
                self._frame_handles[node.name] = self.visualizer.scene.add_frame(
                    self._link_frame_name(node.name),
                    show_axes=True,
                    axes_length=self.axes_length,
                    axes_radius=self.axes_radius,
                    origin_radius=self.origin_radius,
                )

        for link_name, visuals in self._visuals_by_link.items():
            for visual_index, visual in enumerate(visuals):
                self._visual_local_transforms[(link_name, visual_index)] = (
                    _pose_to_matrix(visual.origin)
                )
                self._visual_handles[(link_name, visual_index)] = self._add_visual_node(
                    link_name,
                    visual,
                    visual_index,
                )

        self._rendered = True

    def update(self, base_transform: Any, joint_positions: Any) -> None:
        base_matrix = _to_numpy(base_transform)
        if base_matrix.shape != (4, 4):
            raise ValueError(
                f"`base_transform` must have shape (4, 4). Got {base_matrix.shape!r}."
            )

        joint_vector = _to_numpy(joint_positions)
        if joint_vector.shape != (self.model.NDoF,):
            raise ValueError(
                f"`joint_positions` must have shape ({self.model.NDoF},). "
                f"Got {joint_vector.shape!r}."
            )

        self._current_base_transform = base_matrix.copy()
        self._current_joint_positions = joint_vector.copy()

        link_transforms = self._compute_link_poses(base_matrix, joint_vector)
        link_names = (
            self._tree_link_names if self.show_frames else self._visuals_by_link
        )

        for link_name in link_names:
            link_transform = link_transforms[link_name]
            if self.show_frames:
                frame_handle = self._frame_handles[link_name]
                frame_handle.wxyz, frame_handle.position = _transform_to_viser(
                    link_transform
                )

            for visual_index, visual in enumerate(
                self._visuals_by_link.get(link_name, ())
            ):
                visual_transform = (
                    link_transform
                    @ self._visual_local_transforms[(link_name, visual_index)]
                )
                handle = self._visual_handles[(link_name, visual_index)]
                handle.wxyz, handle.position = _transform_to_viser(visual_transform)

    def _compute_link_poses(
        self, base_transform: np.ndarray, joint_positions: np.ndarray
    ) -> dict[str, np.ndarray]:
        return {
            name: _to_numpy(transform)
            for name, transform in self.kindyn.link_poses(
                base_transform, joint_positions
            ).items()
        }

    def add_joint_sliders(
        self,
        *,
        folder_name: str = "Joints",
        expand_by_default: bool = False,
        step: float | None = None,
        initial_joint_positions: Any | None = None,
    ) -> dict[str, Any]:
        if self.model.NDoF == 0:
            return {}
        if self._joint_slider_handles:
            return self._joint_slider_handles

        if initial_joint_positions is not None:
            self.update(self._current_base_transform, initial_joint_positions)

        self._joint_slider_defaults = self._current_joint_positions.copy()
        with self.visualizer.gui.add_folder(
            folder_name, expand_by_default=expand_by_default
        ):
            reset_button = self.visualizer.gui.add_button("Reset Joints")
            for joint_name in self.model.actuated_joints:
                joint_index = self.model.joints[joint_name].idx
                lower, upper = self._joint_slider_limits(joint_name)
                initial_value = float(
                    np.clip(self._current_joint_positions[joint_index], lower, upper)
                )
                self._joint_slider_defaults[joint_index] = initial_value
                slider = self.visualizer.gui.add_slider(
                    joint_name,
                    min=lower,
                    max=upper,
                    step=self._joint_slider_step(lower, upper, step),
                    initial_value=initial_value,
                )

                @slider.on_update
                def _on_slider_update(event, *, joint_index=joint_index):
                    if self._joint_slider_sync_active:
                        return
                    self._current_joint_positions[joint_index] = float(
                        event.target.value
                    )
                    self.update(
                        self._current_base_transform, self._current_joint_positions
                    )

                self._joint_slider_handles[joint_name] = slider

            @reset_button.on_click
            def _on_reset(_event):
                self._joint_slider_sync_active = True
                try:
                    for joint_index, joint_name in enumerate(
                        self.model.actuated_joints
                    ):
                        self._joint_slider_handles[joint_name].value = float(
                            self._joint_slider_defaults[joint_index]
                        )
                finally:
                    self._joint_slider_sync_active = False
                self.update(self._current_base_transform, self._joint_slider_defaults)

        return self._joint_slider_handles

    def _joint_slider_limits(self, joint_name: str) -> tuple[float, float]:
        joint = self.model.joints[joint_name]
        if joint.limit is not None:
            lower = joint.limit.lower
            upper = joint.limit.upper
            if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                return lower, upper
        if joint.type == "revolute":
            return -float(np.pi), float(np.pi)
        if joint.type == "prismatic":
            return -1.0, 1.0
        return -1.0, 1.0

    @staticmethod
    def _joint_slider_step(
        lower: float,
        upper: float,
        step: float | None,
    ) -> float:
        if step is not None and step > 0.0:
            return step
        span = upper - lower
        if span <= 0.0:
            return 0.01
        return max(span / 200.0, 1e-3)

    def _link_frame_name(self, link_name: str) -> str:
        return f"{self.root_name}/frames/{link_name}"

    def _link_visual_root_name(self, link_name: str) -> str:
        return f"{self.root_name}/{link_name}"

    def _visual_name(self, link_name: str, visual_index: int, name: str | None) -> str:
        # Use generic suffix if no name provided, otherwise use the actual name
        # (or append index if the name appears multiple times in this link)
        if name is None:
            suffix = f"visual_{visual_index}"
        else:
            # Count how many visuals in this link share the same name.
            # If the name is unique, use it as-is; if duplicates exist,
            # append the index to disambiguate (e.g. "arm_0", "arm_1").
            duplicate_count = sum(
                1
                for visual in self._visuals_by_link.get(link_name, ())
                if visual.name == name
            )
            suffix = name if duplicate_count == 1 else f"{name}_{visual_index}"
        return f"{self._link_visual_root_name(link_name)}/{suffix}"

    def _add_visual_node(self, link_name: str, visual: Visual, visual_index: int):
        rgba = visual.material.rgba if visual.material is not None else None
        color, opacity = _normalize_rgba(rgba, self.default_color)
        name = self._visual_name(link_name, visual_index, visual.name)

        if isinstance(visual.geometry, BoxVisualGeometry):
            return self.visualizer.scene.add_box(
                name,
                color=color,
                dimensions=np.asarray(visual.geometry.size, dtype=float),
                opacity=opacity,
                wxyz=_IDENTITY_WXYZ.copy(),
                position=np.zeros(3, dtype=float),
            )

        if isinstance(visual.geometry, CylinderVisualGeometry):
            return self.visualizer.scene.add_cylinder(
                name,
                radius=float(visual.geometry.radius),
                height=float(visual.geometry.length),
                color=color,
                opacity=opacity,
                wxyz=_IDENTITY_WXYZ.copy(),
                position=np.zeros(3, dtype=float),
            )

        if isinstance(visual.geometry, SphereVisualGeometry):
            return self.visualizer.scene.add_icosphere(
                name,
                radius=float(visual.geometry.radius),
                color=color,
                opacity=opacity,
                wxyz=_IDENTITY_WXYZ.copy(),
                position=np.zeros(3, dtype=float),
            )

        if isinstance(visual.geometry, EmbeddedMeshVisualGeometry):
            return self.visualizer.scene.add_mesh_simple(
                name,
                vertices=np.asarray(visual.geometry.vertices, dtype=np.float32),
                faces=np.asarray(visual.geometry.faces, dtype=np.uint32),
                color=color,
                opacity=opacity,
                wxyz=_IDENTITY_WXYZ.copy(),
                position=np.zeros(3, dtype=float),
            )

        if isinstance(visual.geometry, MeshVisualGeometry):
            mesh_path = pathlib.Path(visual.geometry.filename)
            if not mesh_path.is_absolute() or not mesh_path.exists():
                raise FileNotFoundError(
                    "Mesh paths must be resolved by the model factory before visualization. "
                    f"Got: {visual.geometry.filename}"
                )
            if trimesh is None:  # pragma: no cover - exercised via monkeypatch
                raise ImportError(
                    "The optional dependency 'trimesh' is required for visualization. "
                    "Install it with `pip install adam-robotics[visualization]`."
                )
            mesh = trimesh.load(mesh_path, force="mesh")
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            scale = np.asarray(visual.geometry.scale, dtype=np.float32)
            if scale.shape == ():
                vertices = vertices * float(scale)
            else:
                vertices = vertices * scale.reshape(1, 3)
            return self.visualizer.scene.add_mesh_simple(
                name,
                vertices=vertices,
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                color=color,
                opacity=opacity,
                wxyz=_IDENTITY_WXYZ.copy(),
                position=np.zeros(3, dtype=float),
            )

        raise NotImplementedError(
            f"Unsupported visual geometry: {visual.geometry.__class__.__name__}"
        )


class Visualizer:
    """Scene-level viser wrapper that can host one or more robot models."""

    def __init__(
        self,
        kindyn: Any | None = None,
        *,
        server: Any | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        up_direction: str = "+z",
        world_axes: bool = False,
        ground: bool = False,
        ground_name: str = "/ground",
        ground_width: float = 160.0,
        ground_height: float = 160.0,
        ground_plane: str = "xy",
        ground_cell_size: float = 0.25,
        ground_section_size: float = 0.5,
        default_lights: bool | None = None,
        default_lights_cast_shadow: bool = True,
        camera_position: tuple[float, float, float] | None = None,
        camera_look_at: tuple[float, float, float] | None = None,
        root_name: str = "/adam",
        show_frames: bool = False,
        axes_length: float = 0.1,
        axes_radius: float = 0.01,
        origin_radius: float | None = None,
        default_color: tuple[int, int, int] = (160, 160, 160),
    ) -> None:
        self._owns_server = server is None
        if server is None:
            if viser is None:  # pragma: no cover - exercised via monkeypatch
                raise ImportError(
                    "The optional dependency 'viser' is required for visualization. "
                    "Install it with `pip install adam-robotics[visualization]`."
                )
            self.server = viser.ViserServer(host=host, port=port)
        else:
            self.server = server
        self._models: dict[str, ModelHandle] = {}
        self._default_model: ModelHandle | None = None

        self._configure_scene(
            up_direction=up_direction,
            world_axes=world_axes,
            ground=ground,
            ground_name=ground_name,
            ground_width=ground_width,
            ground_height=ground_height,
            ground_plane=ground_plane,
            ground_cell_size=ground_cell_size,
            ground_section_size=ground_section_size,
            default_lights=default_lights,
            default_lights_cast_shadow=default_lights_cast_shadow,
            camera_position=camera_position,
            camera_look_at=camera_look_at,
        )

        if kindyn is not None:
            self._default_model = self.add_model(
                kindyn,
                root_name=root_name,
                show_frames=show_frames,
                axes_length=axes_length,
                axes_radius=axes_radius,
                origin_radius=origin_radius,
                default_color=default_color,
            )

    @property
    def scene(self) -> Any:
        scene = getattr(self.server, "scene", None)
        if scene is None:
            raise AttributeError(
                "The configured viser server does not expose a scene API."
            )
        return scene

    @property
    def gui(self) -> Any:
        gui = getattr(self.server, "gui", None)
        if gui is None:
            raise AttributeError(
                "The configured viser server does not expose a GUI API."
            )
        return gui

    def scene_path(self, name: str, *, root_name: str | None = None) -> str:
        return _scene_path(name, root_name=root_name)

    def add_scene_node(
        self,
        method_name: str,
        name: str,
        *,
        root_name: str | None = None,
        **kwargs,
    ) -> Any:
        method = getattr(self.scene, method_name, None)
        if method is None:
            raise AttributeError(
                f"The configured viser scene does not expose '{method_name}()'."
            )
        return method(self.scene_path(name, root_name=root_name), **kwargs)

    def add_model(
        self,
        kindyn: Any,
        *,
        name: str | None = None,
        root_name: str | None = None,
        show_frames: bool = False,
        axes_length: float = 0.1,
        axes_radius: float = 0.01,
        origin_radius: float | None = None,
        default_color: tuple[int, int, int] = (160, 160, 160),
        base_transform: Any | None = None,
        joint_positions: Any | None = None,
    ) -> ModelHandle:
        if root_name is None:
            if name is None:
                model = _extract_model(kindyn)
                name = getattr(model, "name", None) or f"model_{len(self._models)}"
            root_name = f"/{name}"
        normalized_root_name = _normalize_node_name(root_name)
        if normalized_root_name in self._models:
            raise ValueError(
                f"A model is already registered at root '{normalized_root_name}'."
            )

        handle = ModelHandle(
            self,
            kindyn,
            root_name=normalized_root_name,
            base_transform=base_transform,
            joint_positions=joint_positions,
            show_frames=show_frames,
            axes_length=axes_length,
            axes_radius=axes_radius,
            origin_radius=origin_radius,
            default_color=default_color,
        )
        self._models[normalized_root_name] = handle

        if self._default_model is None:
            self._default_model = handle

        return handle

    @property
    def models(self) -> dict[str, ModelHandle]:
        return dict(self._models)

    def update(self, base_transform: Any, joint_positions: Any) -> None:
        if self._default_model is None:
            raise RuntimeError(
                "This Visualizer owns only a scene. Add a robot with `add_model()` "
                "and update the returned handle."
            )
        self._default_model.update(base_transform, joint_positions)

    def add_joint_sliders(self, *args, **kwargs):
        if self._default_model is None:
            raise RuntimeError(
                "This Visualizer owns only a scene. Add a robot with `add_model()` "
                "and add sliders on the returned handle."
            )
        return self._default_model.add_joint_sliders(*args, **kwargs)

    def close(self) -> None:
        if self._owns_server and hasattr(self.server, "stop"):
            self.server.stop()

    def _configure_scene(
        self,
        *,
        up_direction: str,
        world_axes: bool,
        ground: bool,
        ground_name: str,
        ground_width: float,
        ground_height: float,
        ground_plane: str,
        ground_cell_size: float,
        ground_section_size: float,
        default_lights: bool | None,
        default_lights_cast_shadow: bool,
        camera_position: tuple[float, float, float] | None,
        camera_look_at: tuple[float, float, float] | None,
    ) -> None:
        scene = getattr(self.server, "scene", None)
        if scene is None:
            return

        if hasattr(scene, "set_up_direction"):
            scene.set_up_direction(up_direction)

        if default_lights is not None and hasattr(scene, "configure_default_lights"):
            scene.configure_default_lights(
                enabled=default_lights,
                cast_shadow=default_lights_cast_shadow,
            )

        if world_axes and hasattr(scene, "world_axes"):
            scene.world_axes.visible = True

        if ground and hasattr(scene, "add_grid"):
            scene.add_grid(
                ground_name,
                width=ground_width,
                height=ground_height,
                plane=ground_plane,
                cell_size=ground_cell_size,
                section_size=ground_section_size,
                cell_color=(214, 217, 223),
                cell_thickness=0.9,
                section_color=(214, 217, 223),
                section_thickness=0.9,
                plane_color=(248, 249, 251),
                plane_opacity=1.0,
                shadow_opacity=0.22,
            )

        initial_camera = getattr(self.server, "initial_camera", None)
        if initial_camera is not None:
            if camera_position is not None:
                initial_camera.position = camera_position
            if camera_look_at is not None:
                initial_camera.look_at = camera_look_at
