from __future__ import annotations

import pathlib

import requests


DEFAULT_STICKBOT_URDF = (
    pathlib.Path(__file__).resolve().parent.parent / "stickbot.urdf"
)
STICKBOT_URL = (
    "https://raw.githubusercontent.com/"
    "icub-tech-iit/ergocub-gazebo-simulations/"
    "2a9f2d3849610c5731f925ced54b56a3333d8b0b/models/stickBot/model.urdf"
)


def resolve_stickbot_path(path: str | pathlib.Path) -> pathlib.Path:
    resolved_path = pathlib.Path(path).expanduser().resolve()
    if resolved_path.exists():
        return resolved_path

    response = requests.get(STICKBOT_URL, timeout=30)
    response.raise_for_status()
    resolved_path.write_bytes(response.content)
    return resolved_path


def convert_urdf_to_usd(
    urdf_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> pathlib.Path:
    try:
        import urdf_usd_converter
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "This example needs 'urdf-usd-converter'. "
            "Install with: pip install urdf-usd-converter"
        ) from exc

    resolved_urdf_path = pathlib.Path(urdf_path).expanduser().resolve()
    resolved_output_dir = pathlib.Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    converter = urdf_usd_converter.Converter()
    asset = converter.convert(
        str(resolved_urdf_path),
        str(resolved_output_dir / resolved_urdf_path.stem),
    )
    return pathlib.Path(asset.path).resolve()
