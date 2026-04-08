from __future__ import annotations

import pathlib
from dataclasses import dataclass

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from adam.model.abc_factories import Pose
from adam.model.visuals import EmbeddedMeshVisualGeometry
from adam.numpy import KinDynComputations
from adam.visualization import Visualizer
import adam.visualization.viser as visualization_viser


def _build_kindyn(description, joints_name_list=None) -> KinDynComputations:
    kwargs = {}
    if joints_name_list is not None:
        kwargs["joints_name_list"] = joints_name_list

    if isinstance(description, mujoco.MjModel):
        return KinDynComputations.from_mujoco_model(description, **kwargs)

    if isinstance(description, pathlib.Path):
        if description.suffix.lower() in {".xml", ".mjcf"}:
            return KinDynComputations.from_mujoco_model(
                mujoco.MjModel.from_xml_path(str(description)),
                **kwargs,
            )
        description = str(description)

    if isinstance(description, str):
        stripped = description.lstrip()
        if stripped.startswith("<") and "<robot" in stripped[:2048].lower():
            return KinDynComputations.from_urdf(description, **kwargs)
        if stripped.startswith("<") and "<mujoco" in stripped[:2048].lower():
            return KinDynComputations.from_mujoco_model(
                mujoco.MjModel.from_xml_string(description), **kwargs
            )

        suffix = pathlib.Path(description).suffix.lower()
        if suffix == ".urdf":
            return KinDynComputations.from_urdf(description, **kwargs)
        if suffix in {".xml", ".mjcf"}:
            return KinDynComputations.from_mujoco_model(
                mujoco.MjModel.from_xml_path(description), **kwargs
            )

    raise TypeError(f"Unsupported test model description: {type(description)!r}")


def _as_numpy(value) -> np.ndarray:
    while hasattr(value, "array"):
        value = value.array
    array = np.asarray(value)
    if array.dtype == object:
        array = np.vectorize(
            lambda item: item.array if hasattr(item, "array") else item,
            otypes=[float],
        )(array)
    return np.asarray(array, dtype=float)


def _new_usd_test_stage():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    return stage, Gf, UsdGeom, UsdPhysics


def _add_usd_rigid_body(stage, usd_geom, usd_physics, gf, path: str):
    body = usd_geom.Xform.Define(stage, path)
    prim = body.GetPrim()
    usd_physics.RigidBodyAPI.Apply(prim)
    mass_api = usd_physics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(1.0)
    mass_api.CreateCenterOfMassAttr(gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreateDiagonalInertiaAttr(gf.Vec3f(1.0, 1.0, 1.0))
    mass_api.CreatePrincipalAxesAttr(gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return prim


@dataclass
class FakeHandle:
    name: str
    kwargs: dict
    position: np.ndarray | None = None
    wxyz: np.ndarray | None = None


@dataclass
class FakeWorldAxes:
    visible: bool = False


@dataclass
class FakeInitialCamera:
    position: tuple[float, float, float] | None = None
    look_at: tuple[float, float, float] | None = None


@dataclass
class FakeGuiEvent:
    target: object
    client_id: None = None
    client: None = None


class FakeSliderHandle:
    def __init__(self, label: str, kwargs: dict) -> None:
        self.label = label
        self.kwargs = kwargs.copy()
        self._value = kwargs["initial_value"]
        self._callbacks = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        event = FakeGuiEvent(target=self)
        for callback in list(self._callbacks):
            callback(event)

    def on_update(self, func):
        self._callbacks.append(func)
        return func


class FakeButtonHandle:
    def __init__(self, label: str) -> None:
        self.label = label
        self._callbacks = []

    def on_click(self, func):
        self._callbacks.append(func)
        return func

    def click(self) -> None:
        event = FakeGuiEvent(target=self)
        for callback in list(self._callbacks):
            callback(event)


class FakeFolderHandle:
    def __init__(self, gui: "FakeGui", label: str, kwargs: dict) -> None:
        self.gui = gui
        self.label = label
        self.kwargs = kwargs.copy()

    def __enter__(self):
        self.gui._folder_stack.append(self.label)
        return self

    def __exit__(self, *args) -> None:
        del args
        self.gui._folder_stack.pop()


class FakeGui:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self._folder_stack: list[str] = []

    def add_folder(self, label: str, **kwargs) -> FakeFolderHandle:
        self.calls.append(("add_folder", label, kwargs.copy()))
        return FakeFolderHandle(self, label, kwargs)

    def add_slider(self, label: str, **kwargs) -> FakeSliderHandle:
        self.calls.append(("add_slider", label, kwargs.copy()))
        return FakeSliderHandle(label, kwargs)

    def add_button(self, label: str, **kwargs) -> FakeButtonHandle:
        self.calls.append(("add_button", label, kwargs.copy()))
        return FakeButtonHandle(label)


class FakeScene:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.world_axes = FakeWorldAxes()

    def set_up_direction(self, direction: str) -> None:
        self.calls.append(("set_up_direction", direction, {}))

    def configure_default_lights(
        self, *, enabled: bool = True, cast_shadow: bool = True
    ) -> None:
        self.calls.append(
            (
                "configure_default_lights",
                "",
                {"enabled": enabled, "cast_shadow": cast_shadow},
            )
        )

    def _record(self, method: str, name: str, kwargs: dict) -> FakeHandle:
        handle = FakeHandle(name=name, kwargs=kwargs.copy())
        self.calls.append((method, name, kwargs.copy()))
        return handle

    def add_frame(self, name: str, **kwargs) -> FakeHandle:
        return self._record("add_frame", name, kwargs)

    def add_box(self, name: str, **kwargs) -> FakeHandle:
        return self._record("add_box", name, kwargs)

    def add_cylinder(self, name: str, **kwargs) -> FakeHandle:
        return self._record("add_cylinder", name, kwargs)

    def add_icosphere(self, name: str, **kwargs) -> FakeHandle:
        return self._record("add_icosphere", name, kwargs)

    def add_mesh_simple(self, name: str, **kwargs) -> FakeHandle:
        return self._record("add_mesh_simple", name, kwargs)

    def add_grid(self, name: str, **kwargs) -> FakeHandle:
        return self._record("add_grid", name, kwargs)


class FakeServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self.scene = FakeScene()
        self.gui = FakeGui()
        self.initial_camera = FakeInitialCamera()
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def get_port(self) -> int:
        return self.port


@pytest.fixture
def fake_viser(monkeypatch):
    fake_module = type("FakeViserModule", (), {"ViserServer": FakeServer})
    monkeypatch.setattr(visualization_viser, "viser", fake_module)
    return fake_module


def test_visualizer_builds_scene_and_updates_link_frames(fake_viser):
    urdf = """
    <robot name="simple_robot">
      <link name="base">
        <visual>
          <origin xyz="0 0 0.1" rpy="0 0 0"/>
          <geometry><box size="0.2 0.3 0.4"/></geometry>
        </visual>
      </link>
      <link name="link1">
        <visual>
          <geometry><cylinder radius="0.05" length="0.3"/></geometry>
        </visual>
      </link>
      <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
      </joint>
    </robot>
    """
    kindyn = _build_kindyn(urdf, joints_name_list=["joint1"])
    assert isinstance(kindyn.model.links["base"].visuals[0].origin, Pose)
    visualizer = Visualizer(
        world_axes=True,
        ground=True,
        camera_position=(2.0, -2.0, 1.0),
        camera_look_at=(0.0, 0.0, 0.5),
        default_lights=True,
    )
    model_handle = visualizer.add_model(kindyn, root_name="/robot", show_frames=True)

    frame_calls = [
        call for call in visualizer.server.scene.calls if call[0] == "add_frame"
    ]
    assert [call[1] for call in frame_calls] == [
        "/robot/frames/base",
        "/robot/frames/link1",
    ]
    visual_calls = [
        call
        for call in visualizer.server.scene.calls
        if call[0] in {"add_box", "add_cylinder", "add_icosphere", "add_mesh_simple"}
    ]
    assert [call[1] for call in visual_calls] == [
        "/robot/base/visual_0",
        "/robot/link1/visual_0",
    ]
    assert visualizer.server.scene.world_axes.visible is True
    assert visualizer.server.initial_camera.position == (2.0, -2.0, 1.0)
    assert visualizer.server.initial_camera.look_at == (0.0, 0.0, 0.5)
    assert ("add_grid", "/ground") in [
        (method, name) for method, name, _ in visualizer.server.scene.calls
    ]
    ground_call = next(
        call
        for call in visualizer.server.scene.calls
        if call[0] == "add_grid" and call[1] == "/ground"
    )
    assert ground_call[2]["plane_color"] == (248, 249, 251)
    assert ground_call[2]["cell_color"] == (214, 217, 223)
    assert ground_call[2]["section_color"] == (214, 217, 223)
    assert ground_call[2]["section_thickness"] == pytest.approx(0.9)
    assert ("configure_default_lights", "") in [
        (method, name) for method, name, _ in visualizer.server.scene.calls
    ]

    model_handle.update(
        np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        np.array([np.pi / 2]),
    )

    base_handle = model_handle._frame_handles["base"]
    link_handle = model_handle._frame_handles["link1"]
    base_visual_handle = model_handle._visual_handles[("base", 0)]
    link_visual_handle = model_handle._visual_handles[("link1", 0)]

    assert base_handle.position == pytest.approx([1.0, 2.0, 3.0])
    assert link_handle.position == pytest.approx([1.0, 2.0, 4.0])
    assert base_visual_handle.position == pytest.approx([1.0, 2.0, 3.1])
    assert link_visual_handle.position == pytest.approx([1.0, 2.0, 4.0])
    assert link_handle.wxyz == pytest.approx(
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        abs=1e-6,
    )


def test_link_poses_matches_per_link_forward_kinematics():
    urdf = """
    <robot name="simple_robot">
      <link name="base"/>
      <link name="link1"/>
      <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
      </joint>
    </robot>
    """
    kindyn = _build_kindyn(urdf, joints_name_list=["joint1"])
    base_transform = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    joint_positions = np.array([np.pi / 4])

    link_poses = kindyn.link_poses(base_transform, joint_positions)

    assert tuple(link_poses) == ("base", "link1")
    for link_name in link_poses:
        assert _as_numpy(link_poses[link_name]) == pytest.approx(
            _as_numpy(
                kindyn.forward_kinematics(link_name, base_transform, joint_positions)
            )
        )


def test_visualizer_rejects_wrong_joint_vector_length(fake_viser):
    urdf = """
    <robot name="simple_robot">
      <link name="base"/>
      <link name="link1"/>
      <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <axis xyz="0 0 1"/>
      </joint>
    </robot>
    """
    kindyn = _build_kindyn(urdf, joints_name_list=["joint1"])
    model_handle = Visualizer().add_model(kindyn)

    with pytest.raises(ValueError, match="joint_positions"):
        model_handle.update(np.eye(4), np.zeros(2))


def test_visualizer_uses_bulk_link_poses_when_available(fake_viser):
    urdf = """
    <robot name="simple_robot">
      <link name="base"/>
      <link name="link1">
        <visual>
          <geometry><box size="0.1 0.1 0.1"/></geometry>
        </visual>
      </link>
      <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
      </joint>
    </robot>
    """
    base_kindyn = _build_kindyn(urdf, joints_name_list=["joint1"])

    class ProxyKindyn:
        def __init__(self, kindyn):
            self.model = kindyn.model
            self._kindyn = kindyn
            self.link_poses_calls = 0

        def link_poses(self, base_transform, joint_positions):
            self.link_poses_calls += 1
            return self._kindyn.link_poses(base_transform, joint_positions)

        def forward_kinematics(self, *_args, **_kwargs):
            raise AssertionError("Visualizer should use link_poses() when available.")

    kindyn = ProxyKindyn(base_kindyn)
    model_handle = Visualizer().add_model(kindyn, root_name="/robot", show_frames=True)

    model_handle.update(np.eye(4), np.array([0.25]))

    assert kindyn.link_poses_calls == 2


def test_visualizer_joint_sliders_follow_joint_limits_and_update_pose(fake_viser):
    urdf = """
    <robot name="slider_robot">
      <link name="base"/>
      <link name="link1">
        <visual>
          <geometry><box size="0.1 0.1 0.1"/></geometry>
        </visual>
      </link>
      <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.0" upper="1.0" effort="10.0" velocity="10.0"/>
      </joint>
    </robot>
    """
    kindyn = _build_kindyn(urdf, joints_name_list=["joint1"])
    model_handle = Visualizer().add_model(kindyn, show_frames=True)

    sliders = model_handle.add_joint_sliders(folder_name="Robot Joints")
    slider = sliders["joint1"]

    slider_calls = [
        call for call in model_handle.visualizer.gui.calls if call[0] == "add_slider"
    ]
    assert len(slider_calls) == 1
    _, _, slider_kwargs = slider_calls[0]
    assert slider_kwargs["min"] == pytest.approx(-1.0)
    assert slider_kwargs["max"] == pytest.approx(1.0)
    assert slider_kwargs["initial_value"] == pytest.approx(0.0)

    slider.value = 0.5

    assert model_handle._current_joint_positions == pytest.approx([0.5])
    assert model_handle._frame_handles["link1"].wxyz == pytest.approx(
        [np.cos(0.25), 0.0, 0.0, np.sin(0.25)],
        abs=1e-6,
    )


def test_multiple_visualizers_can_share_one_server(fake_viser):
    urdf = """
    <robot name="shared_scene_robot">
      <link name="base">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="1.0"/>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
        <visual>
          <geometry><box size="0.2 0.2 0.2"/></geometry>
        </visual>
      </link>
    </robot>
    """
    kindyn_a = _build_kindyn(urdf)
    kindyn_b = _build_kindyn(urdf)
    visualizer = Visualizer()
    visualizer.add_model(kindyn_a, root_name="/robot_a")
    visualizer.add_model(kindyn_b, root_name="/robot_b")

    recorded_names = [name for method, name, _ in visualizer.server.scene.calls]
    assert "/robot_a/base/visual_0" in recorded_names
    assert "/robot_b/base/visual_0" in recorded_names


def test_visualizer_exposes_scene_gui_and_scene_helpers(fake_viser):
    visualizer = Visualizer()

    assert visualizer.scene is visualizer.server.scene
    assert visualizer.gui is visualizer.server.gui
    assert visualizer.scene_path("debug/box") == "/debug/box"
    assert visualizer.scene_path("tool", root_name="/extras") == "/extras/tool"
    assert (
        visualizer.scene_path("/world/target", root_name="/ignored") == "/world/target"
    )

    visualizer.add_scene_node(
        "add_box",
        "debug/box",
        dimensions=np.array([0.1, 0.2, 0.3]),
        color=(255, 0, 0),
    )
    visualizer.add_scene_node(
        "add_frame",
        "tool",
        root_name="/extras",
        show_axes=True,
    )

    recorded = [(method, name) for method, name, _ in visualizer.server.scene.calls]
    assert ("add_box", "/debug/box") in recorded
    assert ("add_frame", "/extras/tool") in recorded


def test_model_handle_exposes_namespaced_scene_path(fake_viser):
    urdf = """
    <robot name="helper_robot">
      <link name="base">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="1.0"/>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>
    </robot>
    """
    kindyn = _build_kindyn(urdf)
    visualizer = Visualizer()
    model_handle = visualizer.add_model(kindyn, root_name="/robot")

    assert model_handle.scene_path("markers/box") == "/robot/markers/box"
    assert model_handle.scene_path("/world/target") == "/world/target"

    visualizer.add_scene_node(
        "add_frame",
        "target",
        root_name=model_handle.root_name,
        show_axes=False,
    )
    visualizer.add_scene_node(
        "add_box",
        "markers/box",
        root_name=model_handle.root_name,
        dimensions=np.array([0.1, 0.1, 0.1]),
        color=(0, 255, 0),
    )

    recorded = [(method, name) for method, name, _ in visualizer.server.scene.calls]
    assert ("add_frame", "/robot/target") in recorded
    assert ("add_box", "/robot/markers/box") in recorded


def test_visualizer_resolves_package_meshes_from_model_roots(
    fake_viser, monkeypatch, tmp_path: pathlib.Path
):
    package_root = tmp_path / "share" / "robot_pkg"
    mesh_path = package_root / "meshes" / "part.obj"
    model_path = package_root / "robots" / "demo" / "model.urdf"
    mesh_path.parent.mkdir(parents=True)
    model_path.parent.mkdir(parents=True)
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
            ]
        )
    )
    model_path.write_text("""
        <robot name="mesh_robot">
          <link name="base">
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="1.0"/>
              <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
            </inertial>
            <visual>
              <geometry>
                <mesh filename="package://robot_pkg/meshes/part.obj" scale="1 2 3"/>
              </geometry>
              <material name="gray">
                <color rgba="0.5 0.5 0.5 0.4"/>
              </material>
            </visual>
          </link>
        </robot>
        """)

    class FakeMesh:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

    monkeypatch.setattr(
        visualization_viser,
        "viser",
        type("FakeViserModule", (), {"ViserServer": FakeServer}),
    )
    monkeypatch.setattr(
        visualization_viser,
        "trimesh",
        type(
            "FakeTrimeshModule",
            (),
            {"load": staticmethod(lambda path, force="mesh": FakeMesh())},
        ),
    )

    kindyn = _build_kindyn(str(model_path))
    visual_record = kindyn.model.links["base"].visuals[0]
    assert visual_record.geometry.filename == str(mesh_path.resolve())

    visualizer = Visualizer()
    visualizer.add_model(kindyn)

    mesh_calls = [
        call for call in visualizer.server.scene.calls if call[0] == "add_mesh_simple"
    ]
    assert len(mesh_calls) == 1
    _, _, kwargs = mesh_calls[0]
    assert "scale" not in kwargs
    assert kwargs["vertices"] == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        )
    )
    assert kwargs["opacity"] == pytest.approx(0.4)


def test_mujoco_factory_populates_visuals_and_visualizer_renders(fake_viser):
    xml = """
    <mujoco model="test">
      <worldbody>
        <body name="base">
          <geom name="gbox" type="box" size="0.1 0.2 0.3" rgba="1 0 0 1"/>
          <body name="link1" pos="0 0 0.5">
            <joint name="joint1" type="hinge" axis="0 0 1"/>
            <geom name="gsphere" type="sphere" size="0.15" rgba="0 1 0 1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    mj_model = mujoco.MjModel.from_xml_string(xml)
    kindyn = _build_kindyn(mj_model, joints_name_list=["joint1"])

    assert len(kindyn.model.links["base"].visuals) == 1
    assert len(kindyn.model.links["link1"].visuals) == 1
    assert isinstance(kindyn.model.links["base"].visuals[0].origin, Pose)
    assert isinstance(kindyn.model.links["link1"].visuals[0].origin, Pose)

    visualizer = Visualizer()
    visualizer.add_model(kindyn, root_name="/adam")
    recorded = [(method, name) for method, name, _ in visualizer.server.scene.calls]
    assert ("add_box", "/adam/base/gbox") in recorded
    assert ("add_icosphere", "/adam/link1/gsphere") in recorded


def test_usd_factory_populates_visuals_and_visualizer_renders(fake_viser):
    stage, Gf, UsdGeom, UsdPhysics = _new_usd_test_stage()

    robot = UsdGeom.Xform.Define(stage, "/Robot")
    robot_prim = robot.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
    stage.SetDefaultPrim(robot_prim)

    base = UsdGeom.Xform.Define(stage, "/Robot/base")
    base_prim = base.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base_prim)
    mass_api = UsdPhysics.MassAPI.Apply(base_prim)
    mass_api.CreateMassAttr(1.0)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    cube = UsdGeom.Cube.Define(stage, "/Robot/base/box_visual")
    cube.CreateSizeAttr(0.4)
    cube.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.2))

    mesh = UsdGeom.Mesh.Define(stage, "/Robot/base/mesh_visual")
    mesh.AddTranslateOp().Set(Gf.Vec3f(0.3, 0.0, 0.0))
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
            Gf.Vec3f(0.0, 0.0, 1.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3])

    kindyn = KinDynComputations.from_usd_stage(stage, joints_name_list=[])
    visuals = kindyn.model.links["base"].visuals

    assert len(visuals) == 2

    box_visual = next(visual for visual in visuals if visual.name == "box_visual")
    mesh_visual = next(visual for visual in visuals if visual.name == "mesh_visual")

    assert isinstance(box_visual.origin, Pose)
    assert isinstance(mesh_visual.origin, Pose)
    assert _as_numpy(box_visual.origin.xyz) == pytest.approx([0.0, 0.0, 0.2])
    assert box_visual.geometry.size == pytest.approx((0.4, 0.4, 0.4))
    assert isinstance(mesh_visual.geometry, EmbeddedMeshVisualGeometry)
    assert mesh_visual.geometry.vertices == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    visualizer = Visualizer()
    visualizer.add_model(kindyn, root_name="/usd")
    recorded = [(method, name) for method, name, _ in visualizer.server.scene.calls]
    assert ("add_box", "/usd/base/box_visual") in recorded
    assert ("add_mesh_simple", "/usd/base/mesh_visual") in recorded


def test_visualizer_renders_compiled_mujoco_meshes_without_trimesh(
    fake_viser, monkeypatch, tmp_path: pathlib.Path
):
    mesh_dir = tmp_path / "assets"
    mesh_path = mesh_dir / "part.obj"
    mjcf_path = tmp_path / "robot.xml"
    mesh_dir.mkdir(parents=True)
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
            ]
        )
    )
    mjcf_path.write_text("""
        <mujoco model="mesh_robot">
          <compiler meshdir="assets"/>
          <asset>
            <mesh name="part_mesh" file="part.obj"/>
          </asset>
          <worldbody>
            <body name="base">
              <geom name="gmesh" type="mesh" mesh="part_mesh" rgba="0.2 0.2 0.2 1"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    monkeypatch.setattr(visualization_viser, "viser", fake_viser)
    monkeypatch.setattr(
        visualization_viser,
        "trimesh",
        type(
            "ForbiddenTrimesh",
            (),
            {
                "load": staticmethod(
                    lambda *args, **kwargs: (_ for _ in ()).throw(
                        AssertionError(
                            "MuJoCo compiled meshes should not import trimesh."
                        )
                    )
                )
            },
        ),
    )

    mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    kindyn = _build_kindyn(mj_model)
    visual_record = kindyn.model.links["base"].visuals[0]
    assert isinstance(visual_record.geometry, EmbeddedMeshVisualGeometry)
    mesh_id = int(mj_model.geom_dataid[0])
    v0 = int(mj_model.mesh_vertadr[mesh_id])
    nv = int(mj_model.mesh_vertnum[mesh_id])
    f0 = int(mj_model.mesh_faceadr[mesh_id])
    nf = int(mj_model.mesh_facenum[mesh_id])
    expected_vertices = np.asarray(mj_model.mesh_vert[v0 : v0 + nv], dtype=np.float32)
    expected_faces = np.asarray(mj_model.mesh_face[f0 : f0 + nf], dtype=np.uint32)
    assert visual_record.geometry.vertices == pytest.approx(expected_vertices)
    assert np.array_equal(visual_record.geometry.faces, expected_faces)

    visualizer = Visualizer()
    visualizer.add_model(kindyn)

    mesh_calls = [
        call for call in visualizer.server.scene.calls if call[0] == "add_mesh_simple"
    ]
    assert len(mesh_calls) == 1
    _, _, kwargs = mesh_calls[0]
    assert kwargs["vertices"] == pytest.approx(expected_vertices)
    assert np.array_equal(kwargs["faces"], expected_faces)


def test_mujoco_factory_uses_compiled_mesh_pose_and_skips_collisions(
    tmp_path: pathlib.Path,
):
    mesh_dir = tmp_path / "assets"
    mesh_path = mesh_dir / "part.obj"
    mjcf_path = tmp_path / "robot.xml"
    mesh_dir.mkdir(parents=True)
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
            ]
        )
    )
    mjcf_path.write_text("""
        <mujoco model="mesh_robot">
          <compiler meshdir="assets"/>
          <asset>
            <mesh name="part_mesh" file="part.obj"/>
          </asset>
          <worldbody>
            <body name="base">
              <geom type="mesh" mesh="part_mesh" contype="0" conaffinity="0"/>
              <geom type="mesh" mesh="part_mesh"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    kindyn = _build_kindyn(mj_model)

    visuals = kindyn.model.links["base"].visuals
    assert len(visuals) == 1

    visual = visuals[0]
    assert isinstance(visual.origin, Pose)
    geom_pos = np.asarray(mj_model.geom_pos[0], dtype=float)
    geom_quat = np.asarray(mj_model.geom_quat[0], dtype=float)
    visual_xyz = np.asarray(
        (
            visual.origin.xyz.array
            if hasattr(visual.origin.xyz, "array")
            else visual.origin.xyz
        ),
        dtype=float,
    )
    visual_rpy = np.asarray(
        (
            visual.origin.rpy.array
            if hasattr(visual.origin.rpy, "array")
            else visual.origin.rpy
        ),
        dtype=float,
    )

    expected_rotation = R.from_quat(geom_quat, scalar_first=True)
    assert isinstance(visual.geometry, EmbeddedMeshVisualGeometry)
    assert visual_xyz == pytest.approx(geom_pos)
    assert R.from_euler("xyz", visual_rpy).as_matrix() == pytest.approx(
        expected_rotation.as_matrix()
    )


def test_mujoco_capsule_visuals_render_as_mesh(fake_viser):
    xml = """
    <mujoco model="capsule_robot">
      <worldbody>
        <body name="base">
          <geom name="capsule" type="capsule" size="0.1 0.2" rgba="0.3 0.4 0.5 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mj_model = mujoco.MjModel.from_xml_string(xml)
    kindyn = _build_kindyn(mj_model)

    visual = kindyn.model.links["base"].visuals[0]
    assert isinstance(visual.origin, Pose)
    assert isinstance(visual.geometry, EmbeddedMeshVisualGeometry)
    assert visual.geometry.vertices.shape[0] > 0
    assert visual.geometry.faces.shape[0] > 0

    visualizer = Visualizer()
    visualizer.add_model(kindyn, root_name="/capsule")

    recorded = [(method, name) for method, name, _ in visualizer.server.scene.calls]
    assert ("add_mesh_simple", "/capsule/base/capsule") in recorded
    assert ("add_cylinder", "/capsule/base/capsule") not in recorded


def test_usd_factory_respects_robot_prim_path_for_root_rigid_body_selection():
    stage, Gf, UsdGeom, UsdPhysics = _new_usd_test_stage()

    robot_a = UsdGeom.Xform.Define(stage, "/RobotA")
    robot_a_prim = robot_a.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot_a_prim)
    UsdPhysics.RigidBodyAPI.Apply(robot_a_prim)
    mass_api = UsdPhysics.MassAPI.Apply(robot_a_prim)
    mass_api.CreateMassAttr(1.0)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    UsdGeom.Cube.Define(stage, "/RobotA/box").CreateSizeAttr(0.2)

    robot_b = UsdGeom.Xform.Define(stage, "/RobotB")
    robot_b_prim = robot_b.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot_b_prim)
    UsdPhysics.RigidBodyAPI.Apply(robot_b_prim)
    mass_api = UsdPhysics.MassAPI.Apply(robot_b_prim)
    mass_api.CreateMassAttr(1.0)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    UsdGeom.Cube.Define(stage, "/RobotB/box").CreateSizeAttr(0.2)

    kindyn = KinDynComputations.from_usd_stage(
        stage, robot_prim_path="/RobotA", joints_name_list=[]
    )

    assert set(kindyn.model.links) == {"RobotA"}


def test_usd_factory_skips_inherited_invisible_visuals():
    stage, Gf, UsdGeom, UsdPhysics = _new_usd_test_stage()

    robot = UsdGeom.Xform.Define(stage, "/Robot")
    robot_prim = robot.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
    stage.SetDefaultPrim(robot_prim)

    _add_usd_rigid_body(stage, UsdGeom, UsdPhysics, Gf, "/Robot/base")
    visible_box = UsdGeom.Cube.Define(stage, "/Robot/base/visible_box")
    visible_box.CreateSizeAttr(0.2)

    hidden_scope = UsdGeom.Xform.Define(stage, "/Robot/base/hidden_scope")
    UsdGeom.Imageable(hidden_scope.GetPrim()).CreateVisibilityAttr(
        UsdGeom.Tokens.invisible
    )
    hidden_box = UsdGeom.Cube.Define(stage, "/Robot/base/hidden_scope/hidden_box")
    hidden_box.CreateSizeAttr(0.2)

    kindyn = KinDynComputations.from_usd_stage(stage, joints_name_list=[])

    assert [visual.name for visual in kindyn.model.links["base"].visuals] == [
        "visible_box"
    ]


def test_usd_factory_imports_capsule_visuals_and_visualizer_renders_mesh(fake_viser):
    stage, Gf, UsdGeom, UsdPhysics = _new_usd_test_stage()

    robot = UsdGeom.Xform.Define(stage, "/Robot")
    robot_prim = robot.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
    stage.SetDefaultPrim(robot_prim)

    _add_usd_rigid_body(stage, UsdGeom, UsdPhysics, Gf, "/Robot/base")
    capsule = UsdGeom.Capsule.Define(stage, "/Robot/base/capsule_visual")
    capsule.CreateRadiusAttr(0.1)
    capsule.CreateHeightAttr(0.4)
    capsule.CreateAxisAttr(UsdGeom.Tokens.y)
    capsule.AddTranslateOp().Set(Gf.Vec3f(0.2, 0.0, 0.3))

    kindyn = KinDynComputations.from_usd_stage(stage, joints_name_list=[])
    visual = kindyn.model.links["base"].visuals[0]

    assert isinstance(visual.origin, Pose)
    assert isinstance(visual.geometry, EmbeddedMeshVisualGeometry)
    assert np.asarray(visual.origin.xyz.array, dtype=float) == pytest.approx(
        [0.2, 0.0, 0.3]
    )
    assert visual.geometry.vertices.shape[0] > 0
    assert visual.geometry.faces.shape[0] > 0

    visualizer = Visualizer()
    visualizer.add_model(kindyn, root_name="/usd_capsule")

    recorded = [(method, name) for method, name, _ in visualizer.server.scene.calls]
    assert ("add_mesh_simple", "/usd_capsule/base/capsule_visual") in recorded


def test_usd_factory_imports_instance_proxy_mesh_visuals(fake_viser):
    stage, Gf, UsdGeom, UsdPhysics = _new_usd_test_stage()

    robot = UsdGeom.Xform.Define(stage, "/Robot")
    robot_prim = robot.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
    stage.SetDefaultPrim(robot_prim)

    _add_usd_rigid_body(stage, UsdGeom, UsdPhysics, Gf, "/Robot/base")

    shared_visual = UsdGeom.Xform.Define(stage, "/visuals/shared_base")
    shared_visual.AddTranslateOp().Set(Gf.Vec3f(0.1, 0.0, 0.2))
    shared_mesh = UsdGeom.Mesh.Define(stage, "/visuals/shared_base/mesh")
    shared_mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    shared_mesh.CreateFaceVertexCountsAttr([3])
    shared_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    visuals_prim = UsdGeom.Xform.Define(stage, "/Robot/base/visuals").GetPrim()
    visuals_prim.GetReferences().AddInternalReference("/visuals/shared_base")
    visuals_prim.SetInstanceable(True)

    kindyn = KinDynComputations.from_usd_stage(stage, joints_name_list=[])
    visual = kindyn.model.links["base"].visuals[0]

    assert isinstance(visual.origin, Pose)
    assert isinstance(visual.geometry, EmbeddedMeshVisualGeometry)
    assert np.asarray(visual.origin.xyz.array, dtype=float) == pytest.approx(
        [0.1, 0.0, 0.2]
    )

    visualizer = Visualizer()
    visualizer.add_model(kindyn, root_name="/usd_instance")

    recorded = [(method, name) for method, name, _ in visualizer.server.scene.calls]
    assert ("add_mesh_simple", "/usd_instance/base/mesh") in recorded


def test_visualizer_keeps_duplicate_visual_names_unique(fake_viser):
    stage, Gf, UsdGeom, UsdPhysics = _new_usd_test_stage()

    robot = UsdGeom.Xform.Define(stage, "/Robot")
    robot_prim = robot.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)
    stage.SetDefaultPrim(robot_prim)

    _add_usd_rigid_body(stage, UsdGeom, UsdPhysics, Gf, "/Robot/base")

    for group_name, x_offset in [("group_a", 0.0), ("group_b", 0.2)]:
        group = UsdGeom.Xform.Define(stage, f"/Robot/base/{group_name}")
        group.AddTranslateOp().Set(Gf.Vec3f(x_offset, 0.0, 0.0))
        mesh = UsdGeom.Mesh.Define(stage, f"/Robot/base/{group_name}/mesh")
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(0.1, 0.0, 0.0),
                Gf.Vec3f(0.0, 0.1, 0.0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    kindyn = KinDynComputations.from_usd_stage(stage, joints_name_list=[])

    visualizer = Visualizer()
    visualizer.add_model(kindyn, root_name="/usd_dups")

    mesh_names = [
        name
        for method, name, _ in visualizer.server.scene.calls
        if method == "add_mesh_simple"
    ]
    assert len(mesh_names) == 2
    assert len(set(mesh_names)) == 2
    assert mesh_names == ["/usd_dups/base/mesh_0", "/usd_dups/base/mesh_1"]


def test_visualizer_raises_clear_error_without_viser(monkeypatch):
    monkeypatch.setattr(visualization_viser, "viser", None)

    urdf = """
    <robot name="simple_robot">
      <link name="base">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="1.0"/>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>
    </robot>
    """
    kindyn = _build_kindyn(urdf)

    with pytest.raises(ImportError, match="adam-robotics\\[visualization\\]"):
        Visualizer(kindyn)
