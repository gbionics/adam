"""
Load a USD robot model with ADAM and visualize it with viser.

Usage
-----
    PYTHONPATH=src python3 examples/visualize_usd.py
    PYTHONPATH=src python3 examples/visualize_usd.py --usd-path /path/to/robot.usda

Dependencies
------------
pip install icub-models viser trimesh usd-core urdf-usd-converter
"""

from __future__ import annotations

import argparse
import pathlib
import tempfile
import time

import numpy as np

from _example_utils import convert_urdf_to_usd
from adam.numpy import KinDynComputations
from adam.visualization import Visualizer


DEFAULT_ICUB_MODEL = "iCubGazeboV2_5"
DEFAULT_BASE_HEIGHT = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a USD robot model with ADAM")
    parser.add_argument(
        "--usd-path",
        default=None,
        help=(
            "Path to an existing USD file. If omitted, the example converts "
            "an iCub URDF to USD and visualizes that output."
        ),
    )
    parser.add_argument(
        "--robot-prim-path",
        default=None,
        help="Optional USD articulation root prim path if the stage has multiple robots",
    )
    parser.add_argument(
        "--icub-model",
        default=DEFAULT_ICUB_MODEL,
        help=(
            "iCub model name used for the default URDF -> USD conversion "
            f"(default: {DEFAULT_ICUB_MODEL})"
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
        "--base-height",
        type=float,
        default=DEFAULT_BASE_HEIGHT,
        help=(
            "World-frame z translation applied to the robot base "
            f"(default: {DEFAULT_BASE_HEIGHT})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the model and print its metadata without starting the server loop",
    )
    return parser.parse_args()


def _load_kindyn_from_usd(
    usd_path: str | pathlib.Path,
    robot_prim_path: str | None,
) -> KinDynComputations:
    if robot_prim_path is None:
        return KinDynComputations.from_usd(usd_path)
    return KinDynComputations.from_usd(
        {"usd_path": usd_path, "robot_prim_path": robot_prim_path}
    )


def main() -> None:
    args = parse_args()

    with tempfile.TemporaryDirectory(prefix="adam_visualize_usd_") as temp_dir:
        generated_from_urdf = None
        if args.usd_path is None:
            try:
                import icub_models
            except Exception as exc:  # pragma: no cover - optional dependency
                raise SystemExit(
                    "This example needs 'icub-models' when --usd-path is omitted. "
                    "Install with: pip install icub-models"
                ) from exc

            icub_path = pathlib.Path(icub_models.get_model_file(args.icub_model)).resolve()
            usd_path = convert_urdf_to_usd(icub_path, temp_dir)
            generated_from_urdf = icub_path
        else:
            usd_path = pathlib.Path(args.usd_path).expanduser().resolve()
            if not usd_path.exists():
                raise SystemExit(f"USD file does not exist: {usd_path}")

        kindyn = _load_kindyn_from_usd(usd_path, args.robot_prim_path)

        if args.dry_run:
            print(f"Loaded USD model: {kindyn.model.name}")
            print(f"USD: {usd_path}")
            if generated_from_urdf is not None:
                print(f"Converted from URDF: {generated_from_urdf}")
            print(f"Links: {len(kindyn.model.links)}")
            print(f"Frames: {len(kindyn.model.frames)}")
            print(f"DoFs: {kindyn.NDoF}")
            return

        visualizer = Visualizer(
            host=args.host,
            port=args.port,
            world_axes=True,
            ground=True,
            ground_width=2.0,
            ground_height=2.0,
            ground_plane="xy",
            ground_cell_size=0.1,
            ground_section_size=0.5,
            camera_position=(2.5, -2.0, 1.5),
            camera_look_at=(0.0, 0.0, 0.6),
        )
        model_handle = visualizer.add_model(
            kindyn,
            root_name=f"/{kindyn.model.name}",
            show_frames=args.show_frames,
        )

        base_transform = np.eye(4)
        base_transform[2, 3] = args.base_height
        joint_positions = np.zeros(kindyn.NDoF)
        model_handle.update(base_transform, joint_positions)
        if kindyn.NDoF > 0:
            model_handle.add_joint_sliders(expand_by_default=False)

        actual_port = (
            visualizer.server.get_port()
            if hasattr(visualizer.server, "get_port")
            else args.port
        )
        print(f"Loaded USD model: {kindyn.model.name}")
        print(f"USD: {usd_path}")
        if generated_from_urdf is not None:
            print(f"Converted from URDF: {generated_from_urdf}")
        print(f"Viser server running at http://{args.host}:{actual_port}")
        if kindyn.NDoF > 0:
            print("Joint sliders are available in the 'Joints' panel.")
        print("Press Ctrl+C to exit.")

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            visualizer.close()


if __name__ == "__main__":  # pragma: no cover - example entry point
    main()
