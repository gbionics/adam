"""
Load an iCub model with ADAM and visualize it with viser.

Usage
-----
    PYTHONPATH=src python3 examples/visualize_icub.py

Dependencies
------------
pip install icub-models viser trimesh
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import icub_models
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "This example needs 'icub-models'. Install with: pip install icub-models"
    ) from exc

from adam.numpy import KinDynComputations
from adam.visualization import Visualizer


DEFAULT_MODEL = "iCubGazeboV2_5"
DEFAULT_BASE_HEIGHT = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize an iCub model with ADAM")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"iCub model name exposed by icub_models (default: {DEFAULT_MODEL})",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = icub_models.get_model_file(args.model)
    kindyn = KinDynComputations.from_urdf(model_path)

    visualizer = Visualizer(
        host=args.host,
        port=args.port,
        world_axes=True,
        ground=True,
        ground_width=2.0,
        ground_height=2.0,
        ground_plane="xy",
        camera_position=(2.5, -2.0, 1.5),
        camera_look_at=(0.0, 0.0, 0.6),
    )
    model_handle = visualizer.add_model(
        kindyn,
        root_name=f"/{args.model}",
        show_frames=args.show_frames,
    )

    base_transform = np.eye(4)
    base_transform[2, 3] = args.base_height
    joint_positions = np.zeros(kindyn.NDoF)
    model_handle.update(base_transform, joint_positions)
    model_handle.add_joint_sliders(
        folder_name=args.model,
        expand_by_default=False,
    )

    actual_port = (
        visualizer.server.get_port()
        if hasattr(visualizer.server, "get_port")
        else args.port
    )
    print(f"Loaded iCub model: {args.model}")
    print(f"URDF: {model_path}")
    print(f"Viser server running at http://{args.host}:{actual_port}")
    print(f"Joint sliders are available in the '{args.model}' panel.")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        visualizer.close()


if __name__ == "__main__":  # pragma: no cover - example entry point
    main()
