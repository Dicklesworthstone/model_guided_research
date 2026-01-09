# Fixed-FLOPs Benchmark Harness

Bead: `model_guided_research-gjm`

This repo standardizes *compute-budgeted* runs so we can do fair comparisons (baseline vs feature, optimizer, etc.)
without accidentally changing total work.

All runs should emit:

- `summary.json` (machine-readable telemetry; see `artifacts/README.md`)
- `run.md` (human-readable summary)

under the unified artifacts layout.

---

## Nanochat FLOPs accounting (reference implementation)

Nanochat uses a simple, explicit accounting model (good enough for consistent comparisons):

1) Estimate per-token FLOPs from the model:

- `f_tok = model.estimate_flops()`  (an *estimate*, but stable across runs for a fixed config)

2) Compute tokens per optimizer step:

- `tokens_per_step_global = batch_size * sequence_len * world_size`

3) Compute per-step FLOPs:

- `flops_per_step = f_tok * tokens_per_step_global`

4) Convert a global compute budget to steps:

- `max_steps = ceil(target_flops / flops_per_step)`

During the run we measure throughput (tokens/s) and report estimated TFLOP/s:

- `tflops_per_second_est = (f_tok * tokens_per_second) / 1e12`

Notes:

- This ignores optimizer/data overhead and any compile-time costs; it’s meant for *comparability*, not exact hardware FLOPs.
- Warmup steps are excluded from throughput measurement.

### Running a fixed-FLOPs nanochat baseline

```bash
uv run python -m nanochat.train \
  --device cpu \
  --auto-download-data \
  --min-parquet-files 2 \
  --attention-type standard \
  --optimizer-type adamw \
  --batch-size 8 \
  --sequence-len 256 \
  --target-flops 1e11 \
  --warmup-steps 2 \
  --artifacts-dir artifacts \
  --artifacts-kind bench \
  --artifacts-topic fixed_flops/nanochat \
  --run-id 20251218_flops_smoke
```

This writes into:

`artifacts/bench/fixed_flops/nanochat/20251218_flops_smoke/`

---

## JAX demos (current status)

Exact FLOPs for JAX demos is trickier because XLA fusion and compilation obscure a clean “FLOPs per step” number.

Current policy:

- Demos should still be run under *explicit* compute knobs (usually iterations/epochs) and record them in artifacts.
- The fixed-FLOPs harness is *fully implemented for nanochat*; demos will gain FLOPs estimators incrementally.

### Running demos with a fixed iteration cap

Many demos respect `ProjectConfig.max_iterations`. You can override it from the CLI:

```bash
mgr run matrix-gauge --max-iterations 50 --artifacts-dir artifacts --run-id 20251218_demo_smoke
```

If a specific demo does not yet honor `max_iterations`, it should be updated *in-place* to do so when practical.

