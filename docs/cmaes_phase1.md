# CMA-ES Phase 1 Pilot (nanochat synaptic knobs)

Bead: `model_guided_research-68v`

This doc specifies a **Phase 1 CMA-ES pilot** that tunes a small set of **bio-inspired synaptic parameters**
(implemented in `nanochat/synaptic.py`) using a **fixed-FLOPs** objective backed by `nanochat/train.py`.

Goal: validate the full plumbing end-to-end (parameter decoding → reproducible evaluations → artifacts/telemetry),
and get an early signal on whether the search space is “alive” (nontrivial improvements without frequent failures).

---

## Objective

Minimize a scalar score computed from a short, fixed-budget training run:

- Run `python -m nanochat.train` with `--model-type synaptic`.
- Budget via `--target-flops` (uses `model.estimate_flops()`).
- Score = mean of the last `N` training losses (default `N=3`) from `summary.json`.

Phase 1 uses **train loss** only (no validation pass) to keep the pilot lightweight.

---

## Parameter Vector (10D)

We optimize 10 continuous synaptic knobs from `SynapticConfig`:

| i | name | kind | bounds in x-space | decoded value |
|---:|------|------|-------------------|---------------|
| 0 | `tau_c` | linear | `[0.70, 0.99]` | `tau_c = x` |
| 1 | `alpha_c` | linear | `[0.10, 1.00]` | `alpha_c = x` |
| 2 | `init_rrp` | linear | `[1.0, 18.0]` | `init_rrp = x` |
| 3 | `prime_rate` | linear | `[0.01, 0.20]` | `prime_rate = x` |
| 4 | `rec_rate` | linear | `[0.01, 0.20]` | `rec_rate = x` |
| 5 | `lambda_loge` | linear | `[0.0, 4.0]` | `lambda_loge = x` |
| 6 | `barrier_strength` | linear | `[0.0, 0.50]` | `barrier_strength = x` |
| 7 | `stochastic_train_frac` | linear | `[0.0, 0.40]` | `stochastic_train_frac = x` |
| 8 | `post_fast_lr` | log10 | `[-4.5, -2.0]` | `post_fast_lr = 10**x` |
| 9 | `post_slow_lr` | log10 | `[-5.5, -3.0]` | `post_slow_lr = 10**x` |

All parameters are written into a `SynapticConfig` JSON override file that is passed to
`nanochat/train.py --synaptic-config ...`.

---

## Baseline Training Config (Phase 1)

Fixed baseline for cheap evaluations:

- `--model-type synaptic`
- `--n-layer 4 --n-head 4 --n-kv-head 4 --n-embd 128`
- `--sequence-len 256 --batch-size 8`
- `--target-flops 1e10` (adjust up/down for speed)
- `--optimizer-type adamw` (note: synaptic uses AdamW+Muon internally)
- `--device cpu` for dry-run acceptance when GPU unavailable

Seeds:

- `search_seed`: controls CMA-ES sampling
- `eval_seed`: controls the training run seed (`--seed`)

Phase 1 default is **1 eval seed per candidate** (robust averaging is Phase 2).

---

## Artifacts Layout

All run artifacts live under:

`artifacts/cmaes/phase1/<run_id>/`

Run-level:

- `run.json` – run spec (CLI args, baseline config, param space)
- `progress.csv` – per-candidate results (gen, cand, score, status, summary path)
- `best.json` – best-so-far decoded params + score
- `summary.md` – short human-readable summary + Go/No-Go note

Per-candidate:

- `eval/gen_0000/cand_0000/synaptic_config.json`
- `eval/gen_0000/cand_0000/<eval_id>/summary.json` (written by `nanochat/train.py`)
- `eval/gen_0000/cand_0000/<eval_id>/run.md`

---

## Go / No-Go (Phase 2)

**Go** to Phase 2 if:

- The pilot runs complete reliably (few/no crashes across the population), and
- Best score improves over baseline by a noticeable margin (e.g., ≥1–2% for the tiny budget),
  without obviously pathological configs (e.g., exploding loss).

**No-Go / Revise** if:

- Frequent OOM / NaN / invalid-config errors dominate, or
- Scores are flat/noisy and not meaningfully better than baseline within the pilot budget.

---

## How to Run

Example CPU pilot:

```bash
uv run python scripts/cmaes_phase1.py \
  --run-id pilot_cpu_0 \
  --device cpu \
  --generations 2 \
  --population-size 4 \
  --target-flops 1e10 \
  --search-seed 0 \
  --eval-seed 123
```

