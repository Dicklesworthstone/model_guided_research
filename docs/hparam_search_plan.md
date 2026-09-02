# Hyperparameter Search Plan (CMA-ES / grid)

Bead: `model_guided_research-u8l`

A lightweight decision guide for HP sweeps in this repo: when to use CMA-ES vs a
grid vs the categorical A/B suite, what to search, the budget discipline, the
metric to optimize, and where results land. It is the entry point that points at
the detailed docs rather than duplicating them.

---

## 1. Pick the method by search-space shape

| Space | Method | Tool |
|---|---|---|
| **Categorical** (attention type, flex on/off, optimizer, scheduler) | exhaustive A/B at fixed FLOPs | `mgr bench-fixed-flops -a ... ` + `mgr regressions --fail-on-regression` |
| **1-D / 2-D continuous, cheap, interpretable** | grid (5–9 points) | `nanochat.train` loop + `scripts/cmaes_analyze.py` for readout |
| **≥2-D continuous, expensive objective** | CMA-ES | `scripts/cmaes_phase1.py` (+ `cmaes_analyze.py`) |

Rule of thumb: **don't use CMA-ES below 2 effective dimensions** (`cmaes`
requires `mu>=2`); a grid is cheaper and clearer. **Don't use a grid above ~3
dimensions**; the curse of dimensionality wins and CMA-ES dominates.

## 2. What to search

- **Synaptic biology knobs** (10-D Phase 1, expanded groups Phase 2) →
  `docs/cmaes_phase1.md`, `docs/cmaes_phase2_plan.md`.
- **Math-mechanism knobs** (tropical β, reversible λ, braid τ, ultrametric
  sharpness) → `docs/cmaes_math_targets.md`.
- **Flex vs standard attention** → categorical A/B. *On this CPU-only box
  FlexAttention is unavailable* (needs CUDA, torch≥2.5; see
  `docs/flex_compat_matrix.md`); plan the sweep for a GPU host.
- **Training hparams** (learning rate, batch size, warmup): learning rate is
  the highest-leverage single knob → start with a **log10 grid** (e.g.
  `1e-4 … 3e-3`, 6 points) before adding it to a CMA-ES vector.

## 3. Budget discipline (fixed FLOPs)

All comparisons are at **matched FLOPs**, never matched steps — `--target-flops`
sets the per-eval budget so a cheaper mechanism gets proportionally more steps.
The budget envelope (Min/Target/Max eval tiers, calibration, pause/go triggers)
is in `docs/cmaes_budget_schedule.md`.

The harness enforces budgets directly: `--max-evals`, `--max-wall-seconds`,
`--patience`, `--max-crash-rate`, and `--resume` for preemption.

### The signal gate (non-negotiable)
Before trusting any ranking, run `scripts/cmaes_analyze.py --run-id <id>` and
check `has_signal`. The CPU-scale pilot is **flat** (score std ≈ 2e-6) — at that
budget candidates are indistinguishable and rankings fit noise. A search only
optimizes an objective it can resolve; if flat, **increase budget or task
difficulty**, do not enlarge the search space.

## 4. Metric to optimize

- Default objective: **validation CE** (`--objective-metric val_ce`, what
  `cmaes_phase1.py` minimizes). The training-loss tail
  (`--objective-metric train_tail`, window `--score-tail`) is the cheap
  smoke-only proxy and is rejected on `--resume` of a val_ce run.
- For cross-arm comparison prefer **validation CE** *with care*: raw val CE is
  **not** commensurable across normed vs no-norm arms (logit scale → confidence
  → CE differs at equal accuracy). Use **exact-match accuracy** when comparing
  across arms that change the output scale. Reuse `mgr eval` task metrics where a
  task-level score exists.
- Secondary/diagnostic: route coverage (tropical), symplectic-energy drift
  (reversible), throughput / TFLOP·s (perf), length-generalization (braid/Dyck,
  tested off-floor).

## 5. Artifacts & analysis

Standard layout (`artifacts/README.md`): each run writes `run.json` (param
space + budget + **dataset fingerprint**), `progress.csv`, `best.json`,
`state/` (resume checkpoints), `summary.md`. Analyze every run with:

```bash
uv run python scripts/cmaes_analyze.py --run-id <id>
# -> artifacts/cmaes/analysis/<id>/{sensitivity.json,report.md,sensitivity.png,param_corr.png}
```

Feed `sensitivity.json` (ranked |Spearman| per param) back into the next round:
prune flat params, tighten bounds around the best region, and re-search.

## 6. Worked starting point

```bash
# 1. calibrate one eval (record eval_seconds)
uv run python -m nanochat.train --model-type synaptic --target-flops 1e10 \
  --device auto --artifacts-kind cmaes --artifacts-topic phase1/calibration --run-id cal0

# 2. Phase-1 pilot, multi-seed, guarded
uv run python scripts/cmaes_phase1.py --run-id phase1 --device auto \
  --generations 10 --population-size 12 --eval-seeds 0 1 \
  --patience 3 --max-crash-rate 0.10 --max-wall-seconds 7200

# 3. analyze; only proceed to Phase 2 if has_signal
uv run python scripts/cmaes_analyze.py --run-id phase1
```
