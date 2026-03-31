"""
Load the Unitree G1 MuJoCo model with ADAM and visualize it with viser.

Usage
-----
    PYTHONPATH=src python3 examples/visualize_mujoco.py

Dependencies
------------
pip install mujoco robot_descriptions viser
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "This example needs 'mujoco'. Install with: pip install mujoco"
    ) from exc

try:
    from robot_descriptions import g1_mj_description
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "This example needs 'robot_descriptions'. Install with: pip install robot_descriptions"
    ) from exc

from adam.numpy import KinDynComputations
from adam.visualization import Visualizer



# MUJOCO_XML = """
# <mujoco model="adam_visualizer_demo">
#   <worldbody>
#     <body name="base">
#       <geom name="gbox" type="box" size="0.1 0.2 0.3" rgba="0.85 0.25 0.25 1"/>
#       <body name="link1" pos="0 0 0.5">
#         <joint name="joint1" type="hinge" axis="0 0 1"/>
#         <geom name="gsphere" type="sphere" size="0.15" rgba="0.2 0.7 0.35 1"/>
#       </body>
#     </body>
#   </worldbody>
# </mujoco>
# """

DEFAULT_BASE_HEIGHT = 0.793


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the Unitree G1 MuJoCo model with ADAM"
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
            "World-frame z translation applied to the floating base "
            f"(default: {DEFAULT_BASE_HEIGHT})"
        ),
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate the hinge joint with a sinusoid",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.8,
        help="Joint amplitude in radians when --animate is enabled",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=3.0,
        help="Animation period in seconds when --animate is enabled",
    )
    return parser.parse_args()


def build_kindyn() -> KinDynComputations:
    mjcf_path = pathlib.Path(g1_mj_description.MJCF_PATH).expanduser().resolve()
    mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    return KinDynComputations.from_mujoco_model(mj_model)


def main() -> None:
    args = parse_args()

    kindyn = build_kindyn()
    visualizer = Visualizer(
        host=args.host,
        port=args.port,
        world_axes=True,
        ground=True,
        ground_width=2.0,
        ground_height=2.0,
        ground_plane="xy",
        camera_position=(1.8, -1.6, 1.2),
        camera_look_at=(0.0, 0.0, 0.45),
    )
    model_handle = visualizer.add_model(
        kindyn,
        root_name="/mujoco_demo",
        show_frames=args.show_frames,
    )

    base_transform = np.eye(4)
    base_transform[2, 3] = args.base_height
    joint_positions = np.zeros(kindyn.NDoF)
    model_handle.update(base_transform, joint_positions)
    if not args.animate:
        model_handle.add_joint_sliders(
            folder_name="G1",
            expand_by_default=False,
        )

    animated_joint_name = kindyn.model.actuated_joints[0] if kindyn.NDoF > 0 else None
    actual_port = (
        visualizer.server.get_port()
        if hasattr(visualizer.server, "get_port")
        else args.port
    )
    print("Loaded Unitree G1 MuJoCo model.")
    print(f"MJCF: {g1_mj_description.MJCF_PATH}")
    print(f"Viser server running at http://{args.host}:{actual_port}")
    if args.animate and animated_joint_name is not None:
        print(
            f"Animating {animated_joint_name} "
            f"with amplitude={args.amplitude} rad and period={args.period} s."
        )
    elif kindyn.NDoF > 0:
        print("Joint sliders are available in the 'G1' panel.")
    print("Press Ctrl+C to exit.")

    start_time = time.time()
    try:
        while True:
            if args.animate:
                phase = (time.time() - start_time) * (2.0 * np.pi / args.period)
                joint_positions[0] = args.amplitude * np.sin(phase)
                model_handle.update(base_transform, joint_positions)
                time.sleep(1.0 / 60.0)
            else:
                time.sleep(1.0)
    except KeyboardInterrupt:
        visualizer.close()


if __name__ == "__main__":  # pragma: no cover - example entry point
    main()
