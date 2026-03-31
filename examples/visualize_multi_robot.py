"""
Load iCub, StickBot, iCub-from-USD, Unitree G1, and Aliengo into one shared viser scene.

Usage
-----
    PYTHONPATH=src python3 examples/visualize_multi_robot.py

Dependencies
------------
pip install icub-models mujoco robot_descriptions requests viser trimesh usd-core urdf-usd-converter
"""

from __future__ import annotations

import argparse
import pathlib
import tempfile
import time

import numpy as np

try:
    import icub_models
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "This example needs 'icub-models'. Install with: pip install icub-models"
    ) from exc

try:
    import mujoco
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "This example needs 'mujoco'. Install with: pip install mujoco"
    ) from exc

try:
    from robot_descriptions import aliengo_mj_description, g1_mj_description
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "This example needs 'robot_descriptions'. "
        "Install with: pip install robot_descriptions"
    ) from exc

from adam.numpy import KinDynComputations
from adam.visualization import Visualizer
from _example_utils import (
    DEFAULT_STICKBOT_URDF,
    convert_urdf_to_usd,
    resolve_stickbot_path,
)

DEFAULT_ICUB_MODEL = "iCubGazeboV2_5"
DEFAULT_ICUB_HEIGHT = 0.6
DEFAULT_STICKBOT_HEIGHT = 0.6
DEFAULT_STICKBOT_USD_HEIGHT = 0.6
DEFAULT_G1_HEIGHT = 0.793
DEFAULT_ALIENGO_HEIGHT = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize iCub, StickBot, iCub loaded from USD, "
            "G1, and Aliengo in one shared viser scene"
        )
    )
    parser.add_argument(
        "--icub-model",
        default=DEFAULT_ICUB_MODEL,
        help=(
            "iCub model name exposed by icub_models " f"(default: {DEFAULT_ICUB_MODEL})"
        ),
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface for the viser server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viser server",
    )
    parser.add_argument(
        "--show-frames",
        action="store_true",
        help="Render link frames in addition to visuals",
    )
    parser.add_argument(
        "--stickbot-urdf",
        default=str(DEFAULT_STICKBOT_URDF),
        help=("Path to the StickBot URDF " f"(default: {DEFAULT_STICKBOT_URDF})"),
    )
    return parser.parse_args()


def _base_transform(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4)
    transform[0, 3] = x
    transform[1, 3] = y
    transform[2, 3] = z
    return transform


def _build_mujoco_kindyn(mjcf_path: str | pathlib.Path) -> KinDynComputations:
    resolved_path = pathlib.Path(mjcf_path).expanduser().resolve()
    mj_model = mujoco.MjModel.from_xml_path(str(resolved_path))
    return KinDynComputations.from_mujoco_model(mj_model)


def main() -> None:
    args = parse_args()

    icub_path = icub_models.get_model_file(args.icub_model)
    stickbot_path = resolve_stickbot_path(args.stickbot_urdf)

    with tempfile.TemporaryDirectory(prefix="adam_multi_robot_usd_") as usd_dir:
        icub_usd_path = convert_urdf_to_usd(icub_path, usd_dir)
        robot_configs = [
            dict(
                label="iCub",
                loader=KinDynComputations.from_urdf,
                description=icub_path,
                root_name="/icub",
                base_transform=_base_transform(-3.2, 0.0, DEFAULT_ICUB_HEIGHT),
                slider_folder="iCub",
            ),
            dict(
                label="StickBot",
                loader=KinDynComputations.from_urdf,
                description=str(stickbot_path),
                root_name="/stickbot",
                base_transform=_base_transform(-1.6, 0.0, DEFAULT_STICKBOT_HEIGHT),
                slider_folder="StickBot",
            ),
            dict(
                label="iCub USD",
                loader=KinDynComputations.from_usd,
                description=str(icub_usd_path),
                root_name="/icub_usd",
                base_transform=_base_transform(0.0, 0.0, DEFAULT_STICKBOT_USD_HEIGHT),
                slider_folder="iCub USD",
            ),
            dict(
                label="G1",
                loader=_build_mujoco_kindyn,
                description=g1_mj_description.MJCF_PATH,
                root_name="/g1",
                base_transform=_base_transform(1.6, 0.0, DEFAULT_G1_HEIGHT),
                slider_folder="G1",
            ),
            dict(
                label="Aliengo",
                loader=_build_mujoco_kindyn,
                description=aliengo_mj_description.MJCF_PATH,
                root_name="/aliengo",
                base_transform=_base_transform(3.2, 0.0, DEFAULT_ALIENGO_HEIGHT),
                slider_folder="Aliengo",
            ),
        ]

        visualizer = Visualizer(
            host=args.host,
            port=args.port,
            world_axes=True,
            ground=True,
            ground_width=160.0,
            ground_height=160.0,
            ground_plane="xy",
            camera_position=(7.5, -7.0, 2.8),
            camera_look_at=(0.0, 0.0, 0.75),
        )

        for config in robot_configs:
            kindyn = config["loader"](config["description"])
            model_handle = visualizer.add_model(
                kindyn,
                root_name=config["root_name"],
                show_frames=args.show_frames,
            )

            joint_positions = np.zeros(kindyn.NDoF)
            model_handle.update(config["base_transform"], joint_positions)
            if kindyn.NDoF > 0:
                model_handle.add_joint_sliders(
                    folder_name=config["slider_folder"],
                    expand_by_default=False,
                )

        actual_port = (
            visualizer.server.get_port()
            if hasattr(visualizer.server, "get_port")
            else args.port
        )
        print("Loaded multi-robot scene.")
        print(f"iCub URDF: {icub_path}")
        print(f"StickBot URDF: {stickbot_path}")
        print(f"iCub USD: {icub_usd_path}")
        print(f"G1 MJCF: {g1_mj_description.MJCF_PATH}")
        print(f"Aliengo MJCF: {aliengo_mj_description.MJCF_PATH}")
        print(f"Viser server running at http://{args.host}:{actual_port}")
        print(
            "Joint sliders are available in the iCub, StickBot, "
            "iCub USD, G1, and Aliengo folders."
        )
        print("Press Ctrl+C to exit.")

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            visualizer.close()


if __name__ == "__main__":  # pragma: no cover - example entry point
    main()
