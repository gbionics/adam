import argparse
import pathlib
import time
from dataclasses import dataclass

import numpy as np

from adam.numpy import KinDynComputations
from adam.visualization import ModelHandle, Visualizer


@dataclass
class ViewSession:
    visualizer: Visualizer
    model_handle: ModelHandle
    kindyn: KinDynComputations
    label: str
    source_description: str


def _existing_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {value}")
    return path


def _import_mujoco():
    try:
        import mujoco
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "This command needs 'mujoco' to load MuJoCo XML files. "
            "Install with: pip install adam-robotics[mujoco,visualization]"
        ) from exc
    return mujoco


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a URDF, USD, or MuJoCo model with ADAM and viser"
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--urdf",
        "--urdf-path",
        "--urdf_path",
        type=_existing_path,
        dest="urdf",
        help="Path to a URDF model",
    )
    source_group.add_argument(
        "--usd",
        "--usd-path",
        "--usd_path",
        type=_existing_path,
        dest="usd",
        help="Path to a USD model",
    )
    source_group.add_argument(
        "--mujoco",
        "--mujoco-path",
        "--mujoco_path",
        type=_existing_path,
        dest="mujoco",
        help="Path to a MuJoCo XML/MJCF model",
    )

    parser.add_argument(
        "--robot-prim-path",
        help="USD articulation-root prim path to load when using --usd",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the viser server (default: 127.0.0.1; use 0.0.0.0 to bind all interfaces)",
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
        "--root-name",
        help="Scene root name for the robot. Defaults to the input file stem.",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=0.0,
        help="World-frame z translation applied to the floating base",
    )
    parser.add_argument(
        "--no-ground",
        action="store_true",
        help="Disable the default ground plane",
    )
    parser.add_argument(
        "--no-world-axes",
        action="store_true",
        help="Disable world axes",
    )
    parser.add_argument(
        "--no-sliders",
        action="store_true",
        help="Do not add joint sliders",
    )
    parser.add_argument(
        "--ground-width",
        type=float,
        default=6.0,
        help="Ground width in meters",
    )
    parser.add_argument(
        "--ground-height",
        type=float,
        default=6.0,
        help="Ground height in meters",
    )
    parser.add_argument(
        "--camera-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(2.5, -2.0, 1.5),
        help="Initial camera position",
    )
    parser.add_argument(
        "--camera-look-at",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.6),
        help="Initial camera look-at target",
    )
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    if args.robot_prim_path is not None and args.usd is None:
        parser.error("--robot-prim-path can only be used together with --usd")
    return args


def _source_path(args: argparse.Namespace) -> pathlib.Path:
    if args.urdf is not None:
        return args.urdf
    if args.usd is not None:
        return args.usd
    if args.mujoco is not None:
        return args.mujoco
    raise RuntimeError("No model source was provided.")


def _default_label(args: argparse.Namespace) -> str:
    return _source_path(args).stem


def _load_kindyn(args: argparse.Namespace) -> tuple[KinDynComputations, str]:
    if args.urdf is not None:
        return KinDynComputations.from_urdf(args.urdf), f"URDF: {args.urdf}"

    if args.usd is not None:
        return (
            KinDynComputations.from_usd(
                args.usd,
                robot_prim_path=args.robot_prim_path,
            ),
            f"USD: {args.usd}",
        )

    mujoco = _import_mujoco()
    mj_model = mujoco.MjModel.from_xml_path(str(args.mujoco))
    return (
        KinDynComputations.from_mujoco_model(mj_model),
        f"MuJoCo: {args.mujoco}",
    )


def create_view(args: argparse.Namespace) -> ViewSession:
    kindyn, source_description = _load_kindyn(args)
    label = _default_label(args)
    root_name = args.root_name or f"/{label}"

    visualizer = Visualizer(
        host=args.host,
        port=args.port,
        world_axes=not args.no_world_axes,
        ground=not args.no_ground,
        ground_width=args.ground_width,
        ground_height=args.ground_height,
        ground_plane="xy",
        camera_position=tuple(args.camera_position),
        camera_look_at=tuple(args.camera_look_at),
    )
    model_handle = visualizer.add_model(
        kindyn,
        root_name=root_name,
        show_frames=args.show_frames,
        axes_length=0.08,
        axes_radius=0.004,
        origin_radius=0.01,
    )

    base_transform = np.eye(4)
    base_transform[2, 3] = args.base_height
    joint_positions = np.zeros(kindyn.NDoF)
    model_handle.update(base_transform, joint_positions)
    if kindyn.NDoF > 0 and not args.no_sliders:
        model_handle.add_joint_sliders(
            folder_name=label,
            expand_by_default=False,
        )

    return ViewSession(
        visualizer=visualizer,
        model_handle=model_handle,
        kindyn=kindyn,
        label=label,
        source_description=source_description,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = _validate_args(parser, parser.parse_args(argv))
    session = create_view(args)

    actual_port = (
        session.visualizer.server.get_port()
        if hasattr(session.visualizer.server, "get_port")
        else args.port
    )
    public_host = "localhost" if args.host == "0.0.0.0" else args.host

    print(f"Loaded model: {session.label}")
    print(session.source_description)
    print(f"Viser server running at http://{public_host}:{actual_port}")
    if session.kindyn.NDoF > 0 and not args.no_sliders:
        print(f"Joint sliders are available in the '{session.label}' panel.")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        session.visualizer.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
