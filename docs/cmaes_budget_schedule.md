# CMA-ES Budget & Scheduling Plan (MGR)

Bead: `model_guided_research-0nf`

This doc translates the CMA-ES infra plan (`docs/cmaes_plan_mgr.md`) into a **concrete budget + schedule**
for this repo, with explicit **min / target / max** spend and clear **pause / go** triggers.

Scope:
- Applies to CMA-ES runs driven by `scripts/cmaes_phase1.py` and the fixed-FLOPs objective in `nanochat/train.py`.
- Phase 1 is “pilot-quality”: verify reliability + get a first signal.
- Phase 2 is “real search”: add robustness (multi-seed) and/or expand the search space (see `model_guided_research-2co`).

---

## Core Assumptions (explicit + editable)

We don’t assume a specific GPU model in code; instead, we budget based on a measured **eval wall time**.

### Measure first (single eval calibration)

Before committing to a search budget, run **one** representative evaluation and record:

- `eval_seconds`: wall-clock time from `summary.json` (or `run.md`)
- `target_flops`: the eval budget used (e.g., `1e10`)
- `device`: CPU vs GPU, and GPU name if applicable

Example (Phase 1 baseline, CPU or GPU):

```bash
uv run python -m nanochat.train \
  --model-type synaptic \
  --n-layer 4 --n-head 4 --n-kv-head 4 --n-embd 128 \
  --sequence-len 256 --batch-size 8 \
  --target-flops 1e10 \
  --optimizer-type adamw \
  --seed 123 \
  --device auto \
  --artifacts-kind cmaes \
  --artifacts-topic phase1/calibration \
  --run-id eval_calibration_0
```

Then compute:

- `evals_per_hour = 3600 / eval_seconds`
- `gpu_hours = eval_count / evals_per_hour` (use “device-hours” if CPU)

---

## Phase 1 (10D pilot) — Budget Envelope

Reference: `docs/cmaes_phase1.md` (bead `model_guided_research-68v`, already implemented).

Phase 1 is meant to be cheap and decisive: if the pipeline is unstable or the objective is too noisy,
we stop early and invest in guardrails/proxies (beads `model_guided_research-2mj`, `model_guided_research-2xy`).

### Recommended baseline (Phase 1)

- `d = 10` parameters (SynapticConfig knobs)
- `population_size`: 8–16 (choose based on GPU count)
- `eval_seeds_per_candidate`: 1 (robust averaging comes later)
- `target_flops`: start at `1e10` and adjust after calibration

### Min / Target / Max (Phase 1)

Let:
- `P = population_size`
- `G = generations`
- `S = eval_seeds_per_candidate` (Phase 1 default: `S=1`)
- `E = P * G * S` total evals

| Tier | P | G | S | Total evals E | Intent |
|------|---:|---:|---:|-------------:|--------|
| **Min** | 8 | 2 | 1 | 16 | Smoke-test stability; confirm artifacts/telemetry; quick best-so-far. |
| **Target** | 12 | 10 | 1 | 120 | Enough iterations to see real improvement trends. |
| **Max** | 16 | 30 | 1 | 480 | Only if stable + clearly improving; otherwise stop earlier. |

Convert to time with the measured calibration `eval_seconds`.

### Wall-clock estimate (simple formula)

If you can run `W` evaluations concurrently (roughly the number of usable GPUs, assuming one eval per GPU):

- `wall_clock_hours ≈ (E * eval_seconds) / (3600 * W)`

Example (for intuition only): if `eval_seconds = 60s`

- Phase 1 **Min** (E=16):
  - `W=1` → ~0.27 hours (~16 min)
  - `W=4` → ~0.07 hours (~4 min)
- Phase 1 **Target** (E=120):
  - `W=1` → ~2.0 hours
  - `W=4` → ~0.5 hours
- Phase 1 **Max** (E=480):
  - `W=1` → ~8.0 hours
  - `W=4` → ~2.0 hours

---

## Phase 2 (robust / expanded) — Budget Envelope

Phase 2 is blocked by `model_guided_research-2co` (expanded param groups) and should only be started after a Phase 1 “Go”.

Phase 2 variants (pick **one** first):

1) **Robust averaging** (same `d`, more eval seeds)
   - Hold `d=10`, set `S=3` seeds/candidate.
2) **Expanded space** (bigger `d`, same eval seeds)
   - Increase `d` via additional param groups (see `model_guided_research-2co`), keep `S=1` initially.

### Min / Target / Max (Phase 2, robust averaging)

| Tier | P | G | S | Total evals E | Intent |
|------|---:|---:|---:|-------------:|--------|
| **Min** | 12 | 5 | 3 | 180 | Confirm the Phase 1 winner survives multi-seed evaluation. |
| **Target** | 16 | 15 | 3 | 720 | Real optimization with variance control. |
| **Max** | 24 | 30 | 3 | 2160 | Only if correlation is good and budget allows. |

---

## Scheduling (GPU utilization)

Preferred execution model:
- **One evaluation = one process = one GPU** (no DDP inside an eval).
- Use **population-level parallelism**: run multiple candidates concurrently.

### Single GPU

- Run `population_size=P` sequentially.
- Recommended: keep `P` modest (8–12) and adjust `target_flops` to keep each eval short enough for iteration speed.

### Multiple GPUs (2–8)

- Use `P >= #GPUs` to keep devices busy.
- Recommended: `P = 2×GPUs` so stragglers don’t idle the whole generation.

### When to use DDP inside an eval (rare)

Only consider if per-eval budgets grow so large that single-GPU evals become too slow, and the DDP overhead is amortized.
Default remains single-GPU evals for isolation and debuggability.

---

## Pause / Go Triggers (cost + reliability)

These are intentionally blunt; they protect against runaway cost and ambiguous outcomes.

### Pause immediately (investigate)

- **Crash rate**: >10% of evals return `status != ok` (OOM/NAN/error/timeout) in a generation.
- **NaNs**: any sustained NaN/Inf incidence not explained by an obviously-bad candidate region.
- **Runaway time**: median `eval_seconds` drifts upward by >2× vs calibration without a deliberate config change.

### Go to continue (Phase 1 → Phase 2)

- **Stability**: crash rate <5% across multiple generations.
- **Signal**: best score improves meaningfully over baseline (e.g., ≥1–2% on the fixed budget), and improvements persist.
- **Sanity**: best configs do not look pathological (e.g., exploding loss curves, highly unstable steps).

### Stop (no-go / revise)

- Flat/noisy objective: best score doesn’t improve beyond noise after the **Target** budget.
- High variance: candidates “win” due to seed luck (fix via `model_guided_research-2xy` proxy + multi-seed).

---

## Related Beads / Follow-ups

- `model_guided_research-2mj`: implement budget guardrails (max evals/GPU-hours, patience, early stop).
- `model_guided_research-2xy`: define a cheap proxy objective (correlation check vs full eval).
- `model_guided_research-wiz`: dataset snapshot/split pinning (avoid silent drift).
- `model_guided_research-2co`: Phase 2 expanded parameter groups (blocked by this bead’s completion).
