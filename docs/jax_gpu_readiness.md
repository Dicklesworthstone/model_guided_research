# JAX Pathway Audit for GPU Readiness

Bead: `model_guided_research-blm`

An audit of whether the JAX/Flax paths (the 11 demos + `nanochat/train_jax.py`)
are ready to run on GPU, what is hardcoded to CPU, and the minimal changes to
enable a GPU path **without removing the CPU fallbacks**. For the runtime env
flags themselves see `docs/gpu_env_notes.md` (this doc is the code-path audit,
not the env cheat-sheet).

## Current state (this box)

- `jax 0.8.2` / `jaxlib 0.8.2`; `jax.devices()` → `[CpuDevice(id=0)]` (CPU-only
  machine, no `nvidia-smi`).
- `pyproject.toml` already declares both `jax>=0.4.0` and `jax[cuda12]>=0.4.0`,
  so the CUDA wheels are an intended target — the resolver just landed the CPU
  build on this hardware.
- `requires-python >=3.13` — fine for modern jaxlib CUDA wheels.

## CPU-pinning inventory

| location | form | overridable? | verdict |
|---|---|---|---|
| `config.py:93-94` (`setup_jax`) | pins `jax_platform_name=cpu` **only when `use_gpu=False`** | yes — set `ProjectConfig.use_gpu=True` | **good**: GPU path is reachable by config |
| `nanochat/train_jax.py:9` | `os.environ.setdefault("JAX_PLATFORM_NAME","cpu")` | yes — soft default, env wins | **soft**: on a GPU box it still defaults to CPU unless the env is preset |
| `tests/test_practical_utility.py:5`, `test_demos.py:10` | hard `JAX_PLATFORM(S)=cpu` | test-only | **keep**: tests should be deterministic on CPU |
| `debug_gauge.py:10` | hard `JAX_PLATFORM_NAME=cpu` | debug script | minor; leave or soften |
| `.github/workflows/ci.yml:94` | `JAX_PLATFORMS: cpu` | CI-only | **keep** |

Takeaway: the design is already CPU-default-but-GPU-reachable. The only thing
standing between the *library code* and a GPU is `train_jax.py`'s soft default
and the absence of a graceful GPU→CPU fallback check.

## Readiness by concern

| concern | status | note |
|---|---|---|
| Device selection | **ready (config), partial (train_jax)** | `use_gpu` toggles platform pin; train_jax soft-defaults to CPU |
| Precision | **ready** | `jax_enable_x64` opt-in via `jax_precision="float64"`; default float32 is GPU-appropriate |
| Memory preallocation | **needs env** | set `XLA_PYTHON_CLIENT_PREALLOCATE=false` / `..._MEM_FRACTION` (see gpu_env_notes) |
| Graceful fallback | **missing** | `use_gpu=True` does not verify a GPU exists; a no-GPU box would error in JAX rather than fall back |
| Multi-device (pmap/shard) | **absent** | `train_jax.py` is single-device; out of scope for "readiness", note for later |
| Demo portability | **likely ready, unverified** | demos use plain `jax.numpy`/`flax`; no obvious CPU-only ops, but unverified on GPU |

## Minimal changes to enable GPU (recommended, non-breaking)

These are **recommendations** (not applied here — no GPU to verify against); each
preserves the CPU default:

1. **Graceful device selection in `config.setup_jax`.** When `use_gpu=True`,
   check `jax.devices("gpu")`; if empty, emit a rich warning and fall back to the
   CPU pin instead of letting JAX raise. Keeps CPU-only boxes working when a
   config requests GPU.
2. **Gate the `train_jax.py` soft default.** Only `setdefault` CPU when no GPU is
   visible, so a GPU box auto-uses the GPU while a CPU box stays CPU.
3. **Document the install.** GPU requires the CUDA jaxlib plugin:
   `uv pip install "jax[cuda12]"` (jax ≥0.4.x install model); pin alongside
   `jaxlib`. Already declared in `pyproject.toml`.

Each is small and should land as its own bead with a GPU host to verify; do not
remove any CPU fallback.

## Verification checklist (run on a GPU host)

```bash
# 1. devices visible?
python -c "import jax; print(jax.default_backend(), jax.devices())"   # expect gpu / CudaDevice
# 2. a demo on GPU (CPU default is overridden by the env)
JAX_PLATFORM_NAME=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false mgr run matrix-gauge
# 3. confirm no demo silently forces CPU internally (grep already clean except tests/debug)
# 4. spot-check numerics parity vs CPU on one demo (x64 off)
```

## Findings summary

- **No blocker in library code**: `config.use_gpu` already exposes the GPU path
  and precision is configurable.
- **Two soft gaps**: `train_jax.py` defaults to CPU even on GPU hardware, and
  there is no graceful fallback when GPU is requested but absent.
- **No package change needed** beyond installing the already-declared
  `jax[cuda12]` wheel on a CUDA host.
- **Cannot be functionally verified here** (CPU-only box); the checklist above is
  the hand-off for a GPU session.
