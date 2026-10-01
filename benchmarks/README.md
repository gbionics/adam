# Dynamics benchmarks

`benchmark_dynamics.py` measures steady-state latency of the batched algorithms
(mass matrix, CMM, bias force/RNEA, ABA, forward kinematics) for the NumPy, JAX and
PyTorch backends. It adds no runtime dependency to `adam`.

## Usage

```bash
# Default grid: iCub (23 DoF) and StickBot (34 DoF), float32, B = 1/16/256/4096
python benchmarks/benchmark_dynamics.py --backend torch --device cpu \
    --ops mass cmm rnea aba --grad --grad-batch 32 --memory \
    --output results/torch_cpu_f32.json

JAX_PLATFORMS=cpu python benchmarks/benchmark_dynamics.py --backend jax \
    --dtype float64 --batch-sizes 16 --output results/jax_cpu_f64.json
```

Options: `--backend {numpy,jax,torch}`, `--model` (paths, or the aliases `icub` /
`stickbot`, resolved locally and never downloaded), `--joints` (`auto`, `all` or a
comma-separated list), `--frame`, `--device`, `--dtype`, `--representation`, `--ops`,
`--batch-sizes`, `--warmup`, `--repeat`, `--seed`, `--output`, `--grad`, `--grad-batch`,
`--memory`. A model that cannot be found locally is listed under `not_measured` in the
JSON instead of being downloaded.

## Environments (uv)

`pyproject.toml` defines two uv dependency groups (not part of the published package):
`bench-cpu` (CPU PyTorch wheels) and `bench-gpu` (CUDA 13 PyTorch and `jax[cuda13]`).

```bash
uv run --group bench-gpu python benchmarks/benchmark_dynamics.py --backend torch --device cuda --output out.json
uv run --group bench-gpu pytest tests/test_batched_api.py   # includes CUDA device tests
```

PyTorch >= 2.14 builds Triton launchers for some eager CUDA kernels and needs Python
headers; if the system Python lacks them, create the venv with
`uv venv --managed-python --python 3.12`.

## Measurement rules

- Model construction, input generation and device transfers happen outside the timed region
  (model construction is stored as `model_construction_s`).
- **JAX**: the operation is wrapped in `jax.jit`, compile + first execution is recorded
  separately as `compile_plus_first_exec_ms`, and every timed call is passed through
  `jax.block_until_ready`.
- **PyTorch**: forward-only runs use `torch.inference_mode()`; gradient runs
  (`torch.autograd.grad`) are timed with autograd enabled. On CUDA every sample is
  synchronised before the clock is read.
- JAX warmed-JIT and PyTorch eager are different execution modes; do not compare them as
  equivalent.
- Each result stores the raw per-call samples (`samples_ms`), median, p25/p75 and
  states/second. `--memory` adds allocation volume (PyTorch CPU: sum of positive
  `self_cpu_memory_usage` from the profiler, an upper-bound proxy; CUDA: peak allocation
  above baseline; JAX: XLA `temp_size_in_bytes`).
- No timing thresholds belong in CI; compare runs only on identical hardware and thread
  settings (e.g. `OMP_NUM_THREADS=1`).
