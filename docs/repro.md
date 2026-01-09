# Reproducibility & Seed Discipline

Bead: `model_guided_research-8c7`

This repo targets **Python 3.13** and uses **`uv`** for all installs/runs.

Reproducibility goals in this project:

- **Deterministic RNG** where feasible (same seeds → same random draws).
- **Explicit provenance**: record seed(s), config, commit, and environment for runs.
- Avoid “mystery randomness” from implicit globals.

## What “reproducible” means here

1) **Same seed + same code + same deps + same hardware** should produce very similar outputs.
2) Bitwise-identical results are **not guaranteed** (GPU kernels, threading, XLA/torch.compile can introduce
   nondeterminism), but we should at least control the RNG sources.

## Standard seeding pattern

### Python + NumPy

- `random.seed(seed)`
- `np.random.seed(seed)`

### PyTorch

- `torch.manual_seed(seed)`
- If CUDA is available: `torch.cuda.manual_seed_all(seed)`

In `nanochat/`, use `nanochat.common.seed_everything(seed)` or pass `seed=` to `nanochat.common.compute_init(...)`.

### JAX

JAX has no global RNG. Always create and thread keys explicitly:

- `key = jax.random.PRNGKey(seed)`
- Use `key, subkey = jax.random.split(key)` (or `jax.random.fold_in(key, step)`).

The CLI will set `config.random_seed`, but each demo still needs to use that value for its own `PRNGKey`.

## Dataloaders / data order

For reproducible training comparisons, treat “what data did we see?” as part of the seed story.

- Prefer deterministic file ordering (e.g., `sorted(...)`) and deterministic row-group stepping for DDP.
- If you add shuffling, it must be controlled by an explicit seed (and recorded).
- For DDP: keep **model init seed the same on every rank** so weights match; use rank offsets only for
  data shuffling if/when shuffling is introduced.

## CLI entrypoints (seed propagation)

### JAX demos (mgr)

- Single demo: `mgr run <demo> --seed 123`
- All demos: `mgr run-all --seed 123`

### Practical utility suite

- `mgr eval --seed 123`

### Nanochat training

- `python -m nanochat.train --seed 123 ...`

## Numerical stability (NaN/Inf watchpoints)

NaNs/Infs can silently poison comparisons. Use these debug switches when investigating instability.

### JAX demos (mgr)

- `mgr run <demo> --debug` enables:
  - `ProjectConfig.check_numerics = True`
  - `jax_debug_nans/jax_debug_infs = True`
  - `utils.check_nan_inf(...)` watchpoints (where used)

### Nanochat training (PyTorch)

- `python -m nanochat.train --check-numerics ...` adds loss/gradient finite checks and prints rich diagnostics on failure.
- `python -m nanochat.train --detect-anomaly ...` enables `torch.autograd` anomaly detection (very slow; use only for debugging).

Notes:

- Keep these **off** for benchmarking runs; they add overhead.
- Runs record these flags in `summary.json` under `numerics`.

#### Example failure output (nanochat)

When a NaN/Inf is detected, nanochat prints a rich diagnostic table and raises a `FloatingPointError`, e.g.:

```text
Non-finite gradients detected
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━┳━━━━━┓
┃ param                         ┃ shape    ┃ dtype    ┃ device   ┃ nonfinite┃ nan ┃ inf ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━╇━━━━━┩
┃ blocks.0.attn.q_proj.weight   ┃ (64, 64) ┃ torch... ┃ cuda:0   ┃ 3        ┃ 3   ┃ 0   ┃
└──────────────────────────────┴──────────┴──────────┴──────────┴──────────┴─────┴─────┘
FloatingPointError: Non-finite gradients detected (NaN/Inf).
```

## Checklist for new scripts

Before adding a new runnable script or benchmark:

1) Accept a `--seed` (or document the fixed seed).
2) Seed **Python + NumPy + Torch** at process start.
3) Create `jax.random.PRNGKey(seed)` explicitly (JAX only) and thread it through.
4) For DDP: seed after selecting the CUDA device (or use `manual_seed_all`).
5) Print/log the seed and include it in any JSON artifacts.

## Fail-fast config validation (nanochat)

Bead: `model_guided_research-lbp`

The goal is to avoid “mystery failures” by failing early with a clear message when a config is invalid or inconsistent.

Current checks live in:

- `nanochat/train.py` (see `_validate_train_args(...)` and the argparse `choices=` constraints)

**Invariants / invalid combos (current list):**

- Model dims:
  - `--n-embd` must be divisible by `--n-head`
  - `--n-kv-head` must divide `--n-head` and be `<= --n-head` (GQA/MQA constraint)
- Optional features:
  - `--use-flex-attention` requires torch FlexAttention to exist (`torch.nn.attention.flex_attention`, torch>=2.5)
  - `--use-flex-attention` only applies to `--attention-type standard` and requires CUDA (otherwise it is ignored/disabled with a log)
- Synaptic mode:
  - `--optimizer-type hoss` is rejected for `--model-type synaptic` (no HVP closure)
  - `--attention-type` is currently ignored for `--model-type synaptic` (warns)
  - `--synaptic-config` must point to an existing JSON file when `--model-type synaptic` is selected

**Plan (next extensions):**

- Add richer cross-flag checks (e.g., compile flags, dtype/device constraints).
- Centralize validation patterns so `nanochat/train.py`, CMA-ES scripts, and any servers share the same rules.
