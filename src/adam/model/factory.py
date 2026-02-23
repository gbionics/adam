from __future__ import annotations

import pathlib
from typing import Any

from adam.model.abc_factories import ModelFactory
from adam.model.std_factories.std_model import URDFModelFactory


def _is_mujoco_model(obj: Any) -> bool:
    """Check if obj is a MuJoCo MjModel without importing mujoco."""
    cls = obj.__class__
    cls_name = getattr(cls, "__name__", "")
    cls_module = getattr(cls, "__module__", "")
    if cls_name != "MjModel" or "mujoco" not in cls_module:
        return False
    return all(hasattr(obj, attr) for attr in ("nq", "nv", "nu", "nbody"))


def _is_usd_stage(obj: Any) -> bool:
    """Check if obj is a pxr.Usd.Stage without importing pxr."""
    cls = obj.__class__
    cls_name = getattr(cls, "__name__", "")
    cls_module = getattr(cls, "__module__", "")
    return cls_name == "Stage" and cls_module == "pxr.Usd"


def _is_urdf(obj: Any) -> bool:
    """Check if obj is a URDF."""
    if isinstance(obj, pathlib.Path):
        return obj.suffix.lower() == ".urdf"

    if isinstance(obj, str):
        s = obj.lstrip()
        if s.startswith("<") and "<robot" in s[:2048].lower():
            return True
        try:
            return pathlib.Path(obj).suffix.lower() == ".urdf"
        except Exception:
            return False

    return False


def _is_usd_path(obj: Any) -> bool:
    """Check if obj is a path to an OpenUSD file."""
    usd_exts = {".usd", ".usda", ".usdc", ".usdz"}

    if isinstance(obj, pathlib.Path):
        return obj.suffix.lower() in usd_exts

    if isinstance(obj, str):
        # avoid clashing with inline XML strings
        if obj.lstrip().startswith("<"):
            return False
        try:
            return pathlib.Path(obj).suffix.lower() in usd_exts
        except Exception:
            return False

    return False


def _parse_usd_description(description: Any) -> tuple[Any, str | None] | None:
    """Return (usd_source, robot_prim_path) if description encodes a USD model."""
    if _is_usd_stage(description) or _is_usd_path(description):
        return description, None

    if isinstance(description, dict):
        usd_source = description.get("usd_path", description.get("usd_stage"))
        if _is_usd_stage(usd_source) or _is_usd_path(usd_source):
            robot_prim_path = description.get("robot_prim_path")
            return usd_source, robot_prim_path

    if isinstance(description, (tuple, list)) and len(description) == 2:
        usd_source, robot_prim_path = description
        if (_is_usd_stage(usd_source) or _is_usd_path(usd_source)) and isinstance(
            robot_prim_path, str
        ):
            return usd_source, robot_prim_path

    return None


def build_model_factory(description: Any, math) -> ModelFactory:
    """Return a ModelFactory from a URDF, MuJoCo model, or OpenUSD model."""

    if _is_mujoco_model(description):

        from adam.model.mj_factory.mujoco_model import MujocoModelFactory

        return MujocoModelFactory(mj_model=description, math=math)

    usd_parsed = _parse_usd_description(description)
    if usd_parsed is not None:
        usd_source, robot_prim_path = usd_parsed

        from adam.model.usd_factory.usd_model import USDModelFactory

        return USDModelFactory(
            usd_source=usd_source,
            math=math,
            robot_prim_path=robot_prim_path,
        )

    if _is_urdf(description):
        return URDFModelFactory(path=description, math=math)

    raise ValueError(
        "Unsupported model description. Expected a URDF path/string, a mujoco.MjModel, "
        "or a USD path/stage (optionally with robot_prim_path). "
        f"Got: {type(description)!r}"
    )
