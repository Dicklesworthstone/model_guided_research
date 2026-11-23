# Mechanism scorecard — vdc4-needle-tasktok-shuf-b30-s256-rung

- Original command: `cli.py scorecard --budget 3e13 --cell standard:needle --seeds 3 --eval-seeds 0,1,2 --examples 48 --dataset-size 100000 --dataset-seed 42 --device cpu --batch-size 4 --sequence-len 256 --n-layer 4 --n-head 8 --n-kv-head 4 --n-embd 128 --learning-rate 6e-4 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --val-interval 0 --checkpoint-interval 1000 --min-evidence-steps 10 --timeout-s 36000 --artifacts-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing --run-id vdc4-needle-tasktok-shuf-b30-s256-rung --fresh`
- Resume invocations: `0`
- Budgets: `[30000000000000.0]`
- Mechanisms: `['standard']`
- Tasks: `['needle', 'placebo']`
- Training seeds: `[0, 1, 2]`
- Runtime: `3004.3s`
- Cells: `6/6` done
- Adjudication policy: `ci-v7`

## Cell matrix

| cell | budget | mechanism | task | seed | status | evidence | elapsed s | final loss | logs |
|---|---:|---|---|---:|---|---|---:|---:|---|
| b0-needle-standard-s0 | 3.000e+13 | standard | needle | 0 | done | STEP-ONLY | 289.95 | 2.972810 | logs/b0-needle-standard-s0.eval.stderr.txt |
| b0-needle-standard-s1 | 3.000e+13 | standard | needle | 1 | done | STEP-ONLY | 394.7 | 2.962475 | logs/b0-needle-standard-s1.eval.stderr.txt |
| b0-needle-standard-s2 | 3.000e+13 | standard | needle | 2 | done | STEP-ONLY | 718.04 | 2.961298 | logs/b0-needle-standard-s2.eval.stderr.txt |
| b0-placebo-standard-s0 | 3.000e+13 | standard | placebo | 0 | done | OFF-FLOOR | 579.06 | 1.420200 | logs/b0-placebo-standard-s0.eval.stderr.txt |
| b0-placebo-standard-s1 | 3.000e+13 | standard | placebo | 1 | done | OFF-FLOOR | 379.23 | 1.457837 | logs/b0-placebo-standard-s1.eval.stderr.txt |
| b0-placebo-standard-s2 | 3.000e+13 | standard | placebo | 2 | done | OFF-FLOOR | 643.32 | 1.439224 | logs/b0-placebo-standard-s2.eval.stderr.txt |

> **Evidence quarantine:** 0 cell(s) planned fewer than 10 optimizer steps; 3 additional cell(s) did not belong to a completed standard cohort whose every seed and lower 95% CI cleared the artifact-recorded answer prior. All are excluded from every ci-v6 verdict pool.

## Standard-baseline off-floor gate

- Budget `30000000000000.0`:
  - `needle`: **BLOCKED** — at least one standard training seed did not clear its artifact-recorded prior

## Placebo publication gate

**BLOCKED**
- hyp-placebo-no-winner: universal placebo guard has not been supported
- No operationalized placebo row was found; publication remains blocked.

## Preregistered verdicts

| hypothesis | verdict | q | reason / effect |
|---|---|---:|---|

**FDR:** 0 supported; 0 survive BH at q=0.1 within a family of 0 testable rows.

## Verdict stability across scale

- No decided verdict flips in the available budget cohorts.
- Budget `30000000000000.0` FDR: 0 supported; 0 survive BH at q=0.1.

Raw contracts: `summary.json` and `manifest.json`.
