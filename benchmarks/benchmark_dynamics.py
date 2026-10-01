#!/usr/bin/env python3
"""Steady-state benchmark of the batched adam dynamics algorithms.

Measures mass matrix, CMM, bias force (RNEA), ABA, forward kinematics and,
optionally, value-plus-gradient of the mass matrix and ABA, for the NumPy, JAX
and PyTorch backends. Model construction, JAX compilation, input generation
and host/device transfers are kept out of the timed region.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

ICUB_JOINTS = [
    "torso_pitch",
    "torso_roll",
    "torso_yaw",
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "l_hip_pitch",
    "l_hip_roll",
    "l_hip_yaw",
    "l_knee",
    "l_ankle_pitch",
    "l_ankle_roll",
    "r_hip_pitch",
    "r_hip_roll",
    "r_hip_yaw",
    "r_knee",
    "r_ankle_pitch",
    "r_ankle_roll",
]

# op -> (kd, frame) -> f(H, q, v, qd, tau)
OPS = {
    "mass": lambda kd, frame: lambda H, q, v, qd, tau: kd.mass_matrix(H, q),
    "cmm": lambda kd, frame: lambda H, q, v, qd, tau: kd.centroidal_momentum_matrix(
        H, q
    ),
    "rnea": lambda kd, frame: lambda H, q, v, qd, tau: kd.bias_force(H, q, v, qd),
    "aba": lambda kd, frame: lambda H, q, v, qd, tau: kd.aba(H, q, v, qd, tau),
    "fk": lambda kd, frame: lambda H, q, v, qd, tau: kd.forward_kinematics(frame, H, q),
}
# op -> indices of the (H, q, v, qd, tau) inputs to differentiate
GRAD_WRT = {"mass": (1,), "aba": (1, 3, 4)}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", choices=["numpy", "jax", "torch"], required=True)
    p.add_argument(
        "--model",
        nargs="+",
        default=["icub", "stickbot"],
        help="URDF paths, or the aliases 'icub' / 'stickbot' (resolved locally, never downloaded).",
    )
    p.add_argument(
        "--joints",
        default="auto",
        help="'auto' (the 23 iCub joints for the 'icub' alias, else all), 'all', "
        "or a comma-separated list.",
    )
    p.add_argument("--frame", default="l_sole", help="frame used by the 'fk' op")
    p.add_argument("--device", default="cpu", help="cpu / cuda / gpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument(
        "--representation", choices=["mixed", "body", "inertial"], default="mixed"
    )
    p.add_argument(
        "--ops", nargs="+", default=["mass", "cmm", "rnea", "aba"], choices=list(OPS)
    )
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 256, 4096])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--grad",
        action="store_true",
        help="also benchmark value-and-gradient of mass and aba at --grad-batch",
    )
    p.add_argument("--grad-batch", type=int, default=32)
    p.add_argument(
        "--memory",
        action="store_true",
        help="record allocation volume (torch: profiler/CUDA peak, jax: XLA temp bytes)",
    )
    p.add_argument("--output", type=Path, required=True, help="JSON output path")
    return p.parse_args(argv)


def resolve_model(name: str) -> Path | None:
    if os.path.exists(name):
        return Path(name)
    if name == "stickbot":
        path = REPO_ROOT / "stickbot.urdf"
        return path if path.exists() else None
    if name == "icub":
        try:
            import icub_models

            return Path(icub_models.get_model_file("iCubGenova04"))
        except Exception:
            return None
    return None


def resolve_joints(spec: str, model_name: str) -> list[str] | None:
    if spec == "auto":
        return list(ICUB_JOINTS) if model_name == "icub" else None
    if spec == "all":
        return None
    return [j.strip() for j in spec.split(",") if j.strip()]


def make_inputs(rng: np.random.Generator, batch: int, ndof: int):
    """Deterministic float64 NumPy inputs (H, q, v, qd, tau)."""
    quat = rng.normal(size=(batch, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = quat.T
    R = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(batch, 3, 3)
    H = np.tile(np.eye(4), (batch, 1, 1))
    H[:, :3, :3] = R
    H[:, :3, 3] = rng.uniform(-1, 1, size=(batch, 3))
    q, qd, tau = (rng.uniform(-1, 1, size=(batch, ndof)) for _ in range(3))
    v = rng.uniform(-1, 1, size=(batch, 6))
    return H, q, v, qd, tau


def summarize(samples_ms: list[float], batch: int) -> dict:
    p25, median, p75 = statistics.quantiles(samples_ms, n=4, method="inclusive")
    return {
        "median_ms": median,
        "p25_ms": p25,
        "p75_ms": p75,
        "min_ms": min(samples_ms),
        "states_per_s": batch / (median * 1e-3),
        "samples_ms": samples_ms,
    }


def time_calls(run, warmup: int, repeat: int, sync=lambda: None) -> list[float]:
    """Wall-clock milliseconds of ``repeat`` calls of ``run`` after ``warmup`` calls."""
    for _ in range(warmup):
        run()
    sync()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        run()
        sync()
        samples.append((time.perf_counter_ns() - t0) * 1e-6)
    return samples


class Backend:
    """Per-backend construction, input transfer and timing."""

    def __init__(self, args):
        self.args = args

    def build(self, model_path: str, joints):
        from adam import Representations

        self.kd = self._kin_dyn(model_path, joints)
        rep = {
            "mixed": Representations.MIXED_REPRESENTATION,
            "body": Representations.BODY_FIXED_REPRESENTATION,
            "inertial": Representations.INERTIAL_FIXED_REPRESENTATION,
        }[self.args.representation]
        self.kd.set_frame_velocity_representation(rep)
        return self.kd

    def _kin_dyn(self, model_path: str, joints):
        raise NotImplementedError

    def benchmark(self, op: str, arrays, grad: bool) -> dict:
        raise NotImplementedError

    def meta(self) -> dict:
        raise NotImplementedError


class NumpyBackend(Backend):
    def _kin_dyn(self, model_path, joints):
        from adam.numpy import KinDynComputations

        return KinDynComputations(model_path, joints)

    def benchmark(self, op, arrays, grad):
        fn = OPS[op](self.kd, self.args.frame)
        inputs = [np.asarray(a, dtype=self.args.dtype) for a in arrays]
        return {
            "samples": time_calls(
                lambda: fn(*inputs), self.args.warmup, self.args.repeat
            )
        }

    def meta(self):
        return {"numpy": np.__version__}


class JaxBackend(Backend):
    def _kin_dyn(self, model_path, joints):
        import jax
        import jax.numpy as jnp

        if self.args.dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        from adam.jax import KinDynComputations

        self.device = jax.devices(self.args.device)[0]
        return KinDynComputations(
            model_path, joints, dtype=getattr(jnp, self.args.dtype)
        )

    def benchmark(self, op, arrays, grad):
        import jax

        fn = OPS[op](self.kd, self.args.frame)
        if grad:
            forward = fn
            fn = jax.value_and_grad(lambda *a: forward(*a).sum(), argnums=GRAD_WRT[op])
        fn = jax.jit(fn)
        inputs = jax.device_put(
            [a.astype(self.args.dtype) for a in arrays], self.device
        )
        run = lambda: jax.block_until_ready(fn(*inputs))

        t0 = time.perf_counter_ns()
        run()
        result = {"compile_plus_first_exec_ms": (time.perf_counter_ns() - t0) * 1e-6}
        result["samples"] = time_calls(run, self.args.warmup, self.args.repeat)
        if self.args.memory:
            ma = fn.lower(*inputs).compile().memory_analysis()
            result["xla_temp_bytes"] = int(ma.temp_size_in_bytes)
        return result

    def meta(self):
        import jax

        return {"jax": jax.__version__, "jax_device": str(self.device)}


class TorchBackend(Backend):
    def _kin_dyn(self, model_path, joints):
        import torch
        from adam.pytorch import KinDynComputations

        self.dtype = getattr(torch, self.args.dtype)
        self.device = torch.device(self.args.device)
        return KinDynComputations(
            model_path, joints, device=self.device, dtype=self.dtype
        )

    def benchmark(self, op, arrays, grad):
        import torch

        fn = OPS[op](self.kd, self.args.frame)
        inputs = [
            torch.as_tensor(a, dtype=self.dtype, device=self.device).requires_grad_(
                grad
            )
            for a in arrays
        ]
        if grad:
            wrt = [inputs[i] for i in GRAD_WRT[op]]

            def run():
                value = fn(*inputs).sum()
                return value, torch.autograd.grad(value, wrt)

        else:

            def run():
                with torch.inference_mode():
                    return fn(*inputs)

        cuda = self.device.type == "cuda"
        sync = torch.cuda.synchronize if cuda else lambda: None
        result = {"samples": time_calls(run, self.args.warmup, self.args.repeat, sync)}
        if self.args.memory:
            result.update(self._memory(run, cuda))
        return result

    @staticmethod
    def _memory(run, cuda: bool) -> dict:
        import torch

        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            run()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base
            return {"cuda_peak_bytes_above_baseline": int(peak)}
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            run()
        allocated = sum(max(e.self_cpu_memory_usage, 0) for e in prof.key_averages())
        return {"cpu_bytes_allocated_by_ops": int(allocated)}

    def meta(self):
        import torch

        info = {
            "torch": torch.__version__,
            "torch_device": str(self.device),
            "intra_op_threads": torch.get_num_threads(),
        }
        if self.device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(self.device)
        return info


BACKENDS = {"numpy": NumpyBackend, "jax": JaxBackend, "torch": TorchBackend}


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def environment_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "commit": commit,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "os": platform.platform(),
        "cpu": cpu_name(),
        "env": {
            k: os.environ[k]
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "JAX_PLATFORMS")
            if k in os.environ
        },
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    backend = BACKENDS[args.backend](args)
    report = {
        "environment": environment_info(),
        "config": vars(args),
        "results": [],
        "not_measured": [],
    }

    for model_name in args.model:
        model_path = resolve_model(model_name)
        if model_path is None:
            report["not_measured"].append(
                {"model": model_name, "reason": "model file not available locally"}
            )
            print(f"[skip] {model_name}: model not available locally", flush=True)
            continue
        t0 = time.perf_counter()
        kd = backend.build(str(model_path), resolve_joints(args.joints, model_name))
        build_s = time.perf_counter() - t0
        report["environment"].update(backend.meta())
        print(f"== {model_name} ndof={kd.NDoF} build={build_s:.2f}s", flush=True)

        plan = [(op, False, b) for op in args.ops for b in args.batch_sizes]
        if args.grad and args.backend != "numpy":  # NumPy has no autodiff
            plan += [(op, True, args.grad_batch) for op in GRAD_WRT]

        for op, grad, batch in plan:
            mode = "value_and_grad" if grad else "forward"
            entry = {
                "model": model_name,
                "ndof": kd.NDoF,
                "model_construction_s": build_s,
                "op": op,
                "mode": mode,
                "batch": batch,
            }
            arrays = make_inputs(np.random.default_rng(args.seed), batch, kd.NDoF)
            try:
                measured = backend.benchmark(op, arrays, grad)
            except Exception as exc:  # report, never hide
                entry["error"] = f"{type(exc).__name__}: {exc}"
                print(f"{op} {mode} B={batch}: ERROR {entry['error']}", flush=True)
            else:
                entry.update(summarize(measured.pop("samples"), batch), **measured)
                print(
                    f"{op:>6s} {mode:>15s} B={batch:<5d} "
                    f"median={entry['median_ms']:.4f} ms "
                    f"[{entry['p25_ms']:.4f}, {entry['p75_ms']:.4f}]",
                    flush=True,
                )
            report["results"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
