# Artifacts Layout (Unified)

This repository treats `artifacts/` as the canonical place to store **reproducible run outputs**
for training, benchmarks, correctness checks, profiling, certificates, and CMA-ES sweeps.

## Goals

- Make it easy to find “the latest run” for a given task.
- Keep every run **self-describing** (command line, commit, config, seed, environment).
- Standardize **JSON + Markdown** outputs so downstream tools (dashboards, regressions, CMA-ES) can consume them.

## Directory Structure

Top-level categories (keep these stable):

- `artifacts/baseline/` — baseline training runs (fixed FLOPs or fixed steps).
- `artifacts/bench/` — benchmark runs (A/B comparisons, fixed-FLOPs comparisons, suites).
- `artifacts/perf/` — performance microbenchmarks (throughput, latency, memory).
- `artifacts/profiles/` — profiler traces (e.g., `torch.profiler`, NVTX exports).
- `artifacts/certs/` — correctness checks and “certificates” (math invariants, FlexAttention correctness).
- `artifacts/cmaes/` — CMA-ES searches and evaluation ledgers (see `docs/cmaes_plan_mgr.md`).
- `artifacts/vis/` — saved visualizations (PNG/HTML) of internal states.
- `artifacts/reports/` — post-run summary reports generated from other artifacts.
- `artifacts/regressions/` — historical diffs / regression dashboards.

## Run Directory Naming

Each run should write into a unique run directory:

`artifacts/<category>/<topic>/<run_id>/`

Recommended `run_id` format:

- `YYYYMMDD_HHMMSS` (default)
- Optional suffixes are fine when useful: `YYYYMMDD_HHMMSS_<shorttag>`

Examples:

- `artifacts/baseline/nanochat/20251217_221844/`
- `artifacts/perf/flex_attention/20251218_032000/`
- `artifacts/certs/demos/tropical/20251218_032500/`

## Standard Files Inside a Run

Each run directory should contain:

- `summary.json` — machine-readable metadata + metrics (preferred schema; see `model_guided_research-kt8`).
- `run.md` — human-readable summary (what ran, why, key numbers).

Optional but common:

- `stdout.log`, `stderr.log` — captured logs when running via a harness.
- `metrics.jsonl` — streaming per-step/per-iter metrics (for dashboards).
- `plots/` — images/HTML exports.
- `traces/` — profiler traces.

## Telemetry Schema (model_guided_research-kt8)

Goal: make every run produce a **minimal, stable** `summary.json` that downstream tools (dashboards, regressions,
CMA-ES ledgers) can consume without per-script special-casing.

### summary.json (recommended shape)

**Required top-level keys**

- `schema_version` — string, e.g. `"mgr.telemetry.v1"`.
- `meta` — provenance (who/what/where).
- `budget` — what was intended to run (steps/tokens/FLOPs).
- `results` — what happened (metrics + any time series pointers).

**`meta` (required fields)**

- `run_id` — `YYYYMMDD_HHMMSS[_tag]`.
- `generated_at` — ISO-ish timestamp string.
- `kind` — one of: `baseline`, `bench`, `perf`, `certs`, `cmaes`, `profile`, `report`.
- `topic` — e.g. `nanochat`, `demo:<name>`, `flex_attention`, `fixed_flops`.
- `git` — `{ commit_full, branch, dirty }`.
- `python` — `{ executable, version }`.
- `command` — canonical re-run command (prefer `uv run ...`).
- `argv` — `sys.argv` list.
- `seed` — int (or `seeds: { ... }` when multiple seeds are relevant).
- `device` — `{ type, name }` or a string like `"cuda:0"`.

**`budget` (required fields for training/bench runs)**

- `max_steps` and `warmup_steps`
- `tokens_per_step_global`
- `flops_per_token_est` and `flops_per_step_est`
- Optional: `target_flops` and `planned_total_flops_est`

**`results` (required fields for training/bench runs)**

- `losses` — list[float] (or `final_loss` if you only keep the last value)
- `tokens_per_second` — float
- `tflops_per_second_est` — float
- `peak_memory_allocated_gb` — float or null
- Optional: `metrics_jsonl` — relative path to a streaming metrics file

### Example summary.json (minimal)

```json
{
  "schema_version": "mgr.telemetry.v1",
  "meta": {
    "run_id": "20251218_004200_smoke",
    "generated_at": "2025-12-18T05:42:00Z",
    "kind": "baseline",
    "topic": "nanochat",
    "git": { "commit_full": "abc123...", "branch": "main", "dirty": false },
    "python": { "executable": "/.../python", "version": "3.13.0" },
    "command": "uv run python -m nanochat.train --target-flops ...",
    "argv": ["python", "-m", "nanochat.train", "--target-flops", "..."],
    "seed": 42,
    "device": "cuda:0"
  },
  "budget": {
    "max_steps": 200,
    "warmup_steps": 10,
    "tokens_per_step_global": 65536,
    "flops_per_token_est": 123456,
    "flops_per_step_est": 8090419200,
    "planned_total_flops_est": 1618083840000
  },
  "results": {
    "losses": [10.8, 10.2, 9.9],
    "tokens_per_second": 120000.0,
    "tflops_per_second_est": 14.82,
    "peak_memory_allocated_gb": 6.12,
    "metrics_jsonl": "metrics.jsonl"
  }
}
```

### Rich console summary (recommended)

Every run should print a compact Rich table with at least:

- commit (and dirty/clean), device
- budget (steps, tokens/step, FLOPs/token)
- results (final loss, tokens/s, TFLOP/s, peak mem)
- key feature flags (e.g., `attention_type`, `optimizer_type`, `use_flex_attention`, `compile`)

This keeps terminal output “human-first” while `summary.json` stays “tool-first”.

## Notes / Migration

- Older baseline runs may exist under `artifacts/baseline/<run_id>/`.
  New baselines should prefer `artifacts/baseline/nanochat/<run_id>/`.

## Where Things Should Go

- **Fixed-FLOPs training baselines**: `artifacts/baseline/nanochat/<run_id>/`
- **FlexAttention vs SDPA perf**: `artifacts/perf/flex_attention/<run_id>/`
- **FlexAttention correctness**: `artifacts/certs/flex_attention/<run_id>/`
- **Demo certificates** (e.g., tropical margin, reversible orthogonality): `artifacts/certs/demos/<demo>/<run_id>/`
- **FLOPs-budget harness outputs** (planned): `artifacts/bench/fixed_flops/<run_id>/`

## Regression Guardrails

Use `mgr regressions` to compare a new run against a known-good baseline and optionally fail the process when
regressions exceed thresholds:

```bash
# Compare two run directories (or directly point at summary.json)
uv run mgr regressions \
  -b artifacts/baseline/nanochat/<baseline_run_id> \
  -c artifacts/bench/fixed_flops/nanochat/<candidate_run_id> \
  --run-id <report_run_id> \
  --fail-on-regression
```

Notes:

- **Updating baselines**: choose a new baseline run directory and update the `-b ...` reference wherever you run guardrails.
- **Exceptions / tuning**: widen or tighten thresholds with flags like `--throughput-rel`, `--tflops-rel`, `--loss-abs`,
  `--loss-rel`, `--memory-rel`. To treat missing metrics as failures, add `--fail-on-missing`.

## Authoritative Rules

- Prefer **Rich-first** console output, but always write the final summary to `summary.json` + `run.md`.
- Record: seed, commit (and dirty/clean), resolved config, and key environment info.
- Avoid ad-hoc one-off output paths; keep results discoverable under this structure.
