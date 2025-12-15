"""
Dual-hand neural inverse-kinematics with ADAM (PyTorch backend) on a MuJoCo model.

What this does
--------------
- Loads a humanoid robot from `robot_descriptions` using ADAM's `from_mujoco_model`.
- Automatically detects both left and right hand bodies.
- Trains a small MLP that maps (current joints, desired hand positions) -> joint configuration.
- Uses ADAM's differentiable forward kinematics for the loss; MuJoCo is only for
  visualization (no MuJoCo gradients).

Training uses synthetic data:
- Sample random goal configurations within joint limits.
- Compute corresponding hand positions via ADAM FK.
- Train the network to predict joint configurations that reach both target hand positions.
- Regularization keeps uninvolved joints near zero.

Usage
-----
    python3 examples/neural_ik.py \\
        --description g1_mj_description \\
        --epochs 200 --batch-size 512 \\
        --visualize

Dependencies
------------
pip install mujoco robot_descriptions torch
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

try:
    import mujoco
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "This example needs 'mujoco'. Install with: pip install mujoco"
    ) from exc

try:
    from robot_descriptions.loaders.mujoco import load_robot_description
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "This example needs 'robot_descriptions[mujoco]'. "
        "Install with: pip install 'robot_descriptions[mujoco]'"
    ) from exc

import adam
from adam.pytorch import KinDynComputationsBatch


DEFAULT_DESCRIPTION = "g1_mj_description"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class JointLimits:
    lower: torch.Tensor  # shape (n_dof,)
    upper: torch.Tensor  # shape (n_dof,)
    names: list[str]


def _load_model(description: str) -> mujoco.MjModel:
    model = load_robot_description(description)
    if model is None:
        raise RuntimeError(
            f"Could not load '{description}'. "
            "Ensure the description is installed and available."
        )
    return model


def _detect_hand_bodies(model: mujoco.MjModel) -> list[str]:
    """Detect left and right hand/wrist bodies for dual-hand IK.

    Returns list of hand body names [left_hand, right_hand].
    """
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    print("All body names:", body_names)

    left_hand = next(
        (
            n
            for n in body_names
            if "left" in n.lower() and ("hand" in n.lower() or "wrist" in n.lower())
        ),
        None,
    )
    right_hand = next(
        (
            n
            for n in body_names
            if "right" in n.lower() and ("hand" in n.lower() or "wrist" in n.lower())
        ),
        None,
    )

    hands = []
    if left_hand:
        hands.append(left_hand)
    if right_hand:
        hands.append(right_hand)

    if not hands:
        raise RuntimeError("Could not detect hand bodies in the model")

    return hands


def _joint_permutation(
    model: mujoco.MjModel, kd: KinDynComputationsBatch
) -> tuple[np.ndarray, list[str], list[str]]:
    mj_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        for j_id in range(model.njnt)
        if model.jnt_type[j_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    adam_joint_names = kd.rbdalgos.model.actuated_joints
    perm = [mj_joint_names.index(name) for name in adam_joint_names]
    return np.eye(len(mj_joint_names))[:, perm], mj_joint_names, adam_joint_names


def _joint_limits(
    model: mujoco.MjModel,
    permutation: np.ndarray,
    joint_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> JointLimits:
    # MuJoCo jnt_range has shape (njnt, 2). Unbounded joints often have zeros.
    rng = model.jnt_range[
        [
            j_id
            for j_id in range(model.njnt)
            if model.jnt_type[j_id] != mujoco.mjtJoint.mjJNT_FREE
        ]
    ]
    # permutation maps MuJoCo joint order -> ADAM order; pull per-joint limits accordingly.
    perm_idx = permutation.argmax(axis=0)
    rng = rng[perm_idx]
    lower = []
    upper = []
    for i, name in enumerate(joint_names):
        lo, hi = rng[i]
        if math.isclose(lo, 0.0) and math.isclose(hi, 0.0):
            lo, hi = -math.pi, math.pi
        lower.append(lo)
        upper.append(hi)
    return JointLimits(
        lower=torch.as_tensor(lower, device=device, dtype=dtype),
        upper=torch.as_tensor(upper, device=device, dtype=dtype),
        names=list(joint_names),
    )


def _sample_joints(
    limits: JointLimits, batch_size: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    u = torch.rand(
        (batch_size, len(limits.names)),
        device=limits.lower.device,
        dtype=limits.lower.dtype,
        generator=generator,
    )
    return limits.lower + (limits.upper - limits.lower) * u


class NeuralIK(torch.nn.Module):
    """Tiny MLP: (q, target_xyz...) -> q_out. Supports multiple targets."""

    def __init__(self, n_dof: int, hidden: int = 256, n_targets: int = 1):
        super().__init__()
        self.n_targets = n_targets
        input_dim = n_dof + 3 * n_targets  # q + all target positions (xyz each)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_dof),
        )

    def forward(self, q: torch.Tensor, *target_positions: torch.Tensor) -> torch.Tensor:
        """Predict target joint configuration given current q and one or more target XYZ positions."""
        # Concatenate all target positions
        if len(target_positions) == 1:
            targets = target_positions[0]
        else:
            targets = torch.cat(target_positions, dim=-1)
        x = torch.cat([q, targets], dim=-1)
        return self.net(x)


def train_neural_ik(
    kd: KinDynComputationsBatch,
    target_bodies: list[str],
    limits: JointLimits,
    permutation: np.ndarray,
    mj_model: mujoco.MjModel,
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 3e-3,
    seed: int = 0,
    visualize: bool = False,
    viz_fps: float = 60.0,
    joint_regularization: float = 0.01,
) -> None:
    torch.manual_seed(seed)
    device = limits.lower.device
    dtype = limits.lower.dtype
    n_dof = len(limits.names)
    n_targets = len(target_bodies)

    model = NeuralIK(n_dof=n_dof, n_targets=n_targets).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5
    )
    base = torch.eye(4, device=device, dtype=dtype).expand(batch_size, 4, 4)

    for epoch in range(1, epochs + 1):
        q_curr = _sample_joints(limits, batch_size)
        q_goal = _sample_joints(limits, batch_size)

        with torch.no_grad():
            # Compute all target positions
            goal_positions = []
            for body in target_bodies:
                goal_H = kd.forward_kinematics(body, base, q_goal)
                goal_positions.append(goal_H[:, :3, 3])

        # Single-step prediction for stable training
        q_pred = model(q_curr, *goal_positions)
        q_pred = torch.clamp(q_pred, limits.lower, limits.upper)

        # Compute all predicted positions
        pred_positions = []
        for body in target_bodies:
            pred_H = kd.forward_kinematics(body, base, q_pred)
            pred_positions.append(pred_H[:, :3, 3])

        # Primary task loss: reach all target positions
        task_loss = sum(
            torch.nn.functional.mse_loss(pred_pos, goal_pos)
            for pred_pos, goal_pos in zip(pred_positions, goal_positions)
        ) / len(target_bodies)

        # Regularization: penalize joint positions to keep uninvolved joints near zero
        reg_loss = joint_regularization * torch.mean(q_pred**2)

        loss = task_loss + reg_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % max(1, epochs // 10) == 0:
            # Compute average error across all targets
            total_err = sum(
                torch.linalg.norm(pred_pos - goal_pos, dim=1).mean().item()
                for pred_pos, goal_pos in zip(pred_positions, goal_positions)
            ) / len(target_bodies)
            print(
                f"[{epoch:04d}/{epochs}] loss={loss.item():.6f} "
                f"task={task_loss.item():.6f} reg={reg_loss.item():.6f} "
                f"pos_err={total_err:.5f}"
            )

    if visualize:
        _visualize_solution(
            kd=kd,
            model=model,
            mujoco_model=mj_model,
            permutation=permutation,
            target_bodies=target_bodies,
            limits=limits,
            base=base[:1],
            fps=viz_fps,
        )


def _visualize_solution(
    kd: KinDynComputationsBatch,
    model: torch.nn.Module,
    mujoco_model: mujoco.MjModel,
    permutation: np.ndarray,
    target_bodies: list[str],
    limits: JointLimits,
    base: torch.Tensor,
    fps: float,
) -> None:
    """Visualize dual-hand IK convergence in a MuJoCo viewer."""
    try:
        import mujoco.viewer
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Viewer unavailable: {exc}")
        return

    with torch.no_grad():
        # Start from zero joint configuration for debugging
        q_curr = torch.zeros_like(limits.lower).unsqueeze(0)

        # Generate target positions based on zero configuration
        # This places targets near where end effectors are at zero config
        target_positions = []
        for body_name in target_bodies:
            body_fk = kd.forward_kinematics(body_name, base, q_curr)
            target_positions.append(body_fk[:, :3, 3])

        # Print initial state
        print("Starting IK convergence visualization from zero configuration.")
        print(f"Tracking {len(target_bodies)} targets at end-effector positions:")
        for i, (name, pos) in enumerate(zip(target_bodies, target_positions)):
            print(f"  [{i}] {name}: {pos.squeeze().cpu().numpy()}")

    data = mujoco.MjData(mujoco_model)
    frame_time = 1.0 / max(fps, 1e-3)

    try:
        with mujoco.viewer.launch_passive(mujoco_model, data) as viewer:
            # Convert targets to numpy and store centers for motion
            target_centers = [
                pos.squeeze().cpu().numpy().copy() for pos in target_positions
            ]
            target_np = [center.copy() for center in target_centers]

            # Run continuously until user closes the window
            ik_step = 0
            print("Visualization running. Close the MuJoCo window to exit.")

            while viewer.is_running():
                # Update targets with small circular motion for debugging
                t = (ik_step / 100.0) * 2 * np.pi  # One full circle per 100 frames
                radius = 0.25

                # Generate small circular motions around end-effector positions
                # Target order: [right_hand, left_foot, right_foot, left_hand]
                for i, body_name in enumerate(target_bodies):
                    # Phase shift to desynchronize motions
                    phase = t + (i * np.pi / 2)

                    # Small vertical circle motion
                    target_np[i] = np.array(
                        [
                            0.3,
                            0.0 + radius * np.cos(phase),
                            0.2 + radius * np.sin(phase),
                        ]
                    )

                # Convert to tensors for model
                target_tensors = [
                    torch.tensor(
                        tnp, dtype=limits.lower.dtype, device=limits.lower.device
                    ).unsqueeze(0)
                    for tnp in target_np
                ]

                # Perform IK iteration to track moving targets
                with torch.no_grad():

                    # Apply model multiple times per frame for smooth convergence
                    for _ in range(3):
                        q_pred = model(q_curr, *target_tensors)
                        q_curr = torch.clamp(q_pred, limits.lower, limits.upper)

                    # Compute primary target error for logging
                    curr_fk = kd.forward_kinematics(target_bodies[0], base, q_curr)
                    curr_pos = curr_fk[:, :3, 3]
                    pos_err = torch.linalg.norm(
                        curr_pos - target_tensors[0], dim=1
                    ).item()

                    # Get current positions for all tracked bodies
                    current_positions_np = []
                    for body_name in target_bodies:
                        body_fk = kd.forward_kinematics(body_name, base, q_curr)
                        current_positions_np.append(
                            body_fk[:, :3, 3].squeeze().cpu().numpy()
                        )

                # Update MuJoCo visualization
                q_mj = permutation @ q_curr.detach().cpu().numpy().squeeze()
                data.qpos[:] = 0.0
                if mujoco_model.nq >= 7:  # free base assumed
                    data.qpos[3] = 1.0  # identity quaternion
                    data.qpos[7 : 7 + len(q_mj)] = q_mj
                else:
                    data.qpos[: len(q_mj)] = q_mj
                mujoco.mj_forward(mujoco_model, data)

                if ik_step % 10 == 0:
                    print(
                        f"[IK step {ik_step:4d}] "
                        f"pos_error: {pos_err:.6f} m | "
                        f"targets: {len(target_bodies)}"
                    )

                # Add visual markers by modifying the scene directly
                # Define colors for targets and current positions
                target_colors = [
                    np.array([1.0, 0.0, 0.0, 0.6]),  # Red
                    np.array([0.0, 1.0, 1.0, 0.5]),  # Cyan
                    np.array([1.0, 0.0, 1.0, 0.5]),  # Magenta
                    np.array([1.0, 0.5, 0.0, 0.6]),  # Orange
                ]
                current_colors = [
                    np.array([0.0, 1.0, 0.0, 0.8]),  # Green
                    np.array([0.0, 0.0, 1.0, 0.7]),  # Blue
                    np.array([0.5, 0.0, 0.5, 0.7]),  # Purple
                    np.array([1.0, 1.0, 0.0, 0.8]),  # Yellow
                ]

                with viewer.lock():
                    viewer.user_scn.ngeom = len(target_bodies) * 2

                    geom_idx = 0
                    for i in range(len(target_bodies)):
                        # Target sphere (larger, semi-transparent)
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[geom_idx],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.05, 0, 0]),
                            target_np[i],
                            np.eye(3).flatten(),
                            target_colors[i % len(target_colors)],
                        )
                        geom_idx += 1

                        # Current position sphere (smaller, more opaque)
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[geom_idx],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.03, 0, 0]),
                            current_positions_np[i],
                            np.eye(3).flatten(),
                            current_colors[i % len(current_colors)],
                        )
                        geom_idx += 1

                viewer.sync()
                time.sleep(frame_time)

                # Increment and loop
                ik_step = (ik_step + 1) % 200  # Loop motion every 200 steps

    except Exception as exc:  # pragma: no cover - viewer/GL issues
        print(f"Viewer error: {exc}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural IK with ADAM (PyTorch)")
    parser.add_argument(
        "--description",
        type=str,
        default=DEFAULT_DESCRIPTION,
        help=f"robot_descriptions MuJoCo name (default: {DEFAULT_DESCRIPTION})",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
        help="Torch dtype for ADAM computations and the network.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch a MuJoCo viewer after training.",
    )
    parser.add_argument(
        "--viz-fps", type=float, default=60.0, help="Viewer refresh rate."
    )
    parser.add_argument(
        "--joint-reg",
        type=float,
        default=0.001,
        help="Regularization weight to keep non-task joints near zero.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    mj_model = _load_model(args.description)
    print(f"Loaded '{args.description}'.")

    # Detect hand bodies for dual-hand IK
    target_bodies = _detect_hand_bodies(mj_model)
    print(f"Target bodies: {', '.join(target_bodies)}")

    kd = KinDynComputationsBatch.from_mujoco_model(
        mj_model,
        device=DEFAULT_DEVICE,
        dtype=dtype,
    )
    kd.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

    perm, mj_joint_names, adam_joint_names = _joint_permutation(mj_model, kd)
    limits = _joint_limits(mj_model, perm, adam_joint_names, DEFAULT_DEVICE, dtype)

    train_neural_ik(
        kd=kd,
        target_bodies=target_bodies,
        limits=limits,
        permutation=perm,
        mj_model=mj_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        visualize=args.visualize,
        viz_fps=args.viz_fps,
        joint_regularization=args.joint_reg,
    )


if __name__ == "__main__":  # pragma: no cover - example entry point
    main()
