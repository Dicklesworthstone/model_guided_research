# Mechanism scorecard — vdc4-bag-rel-hier-tasktok-shuf-b10-s256-rung

- Original command: `cli.py scorecard --budget 1e13 --cell standard:bag --cell standard:rel --cell standard:hier --seeds 3 --eval-seeds 0,1,2 --examples 48 --dataset-size 100000 --dataset-seed 42 --device cpu --batch-size 4 --sequence-len 256 --n-layer 4 --n-head 8 --n-kv-head 4 --n-embd 128 --learning-rate 6e-4 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --val-interval 0 --checkpoint-interval 1000 --min-evidence-steps 10 --timeout-s 36000 --artifacts-dir /Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing --run-id vdc4-bag-rel-hier-tasktok-shuf-b10-s256-rung --fresh`
- Resume invocations: `0`
- Budgets: `[10000000000000.0]`
- Mechanisms: `['standard']`
- Tasks: `['bag', 'rel', 'hier', 'placebo']`
- Training seeds: `[0, 1, 2]`
- Runtime: `1629.8s`
- Cells: `12/12` done
- Adjudication policy: `ci-v7`

## Cell matrix

| cell | budget | mechanism | task | seed | status | evidence | elapsed s | final loss | logs |
|---|---:|---|---|---:|---|---|---:|---:|---|
| b0-bag-standard-s0 | 1.000e+13 | standard | bag | 0 | done | STEP-ONLY | 126.12 | 1.244419 | logs/b0-bag-standard-s0.eval.stderr.txt |
| b0-bag-standard-s1 | 1.000e+13 | standard | bag | 1 | done | STEP-ONLY | 211.22 | 1.278842 | logs/b0-bag-standard-s1.eval.stderr.txt |
| b0-bag-standard-s2 | 1.000e+13 | standard | bag | 2 | done | STEP-ONLY | 131.89 | 1.273916 | logs/b0-bag-standard-s2.eval.stderr.txt |
| b0-rel-standard-s0 | 1.000e+13 | standard | rel | 0 | done | STEP-ONLY | 121.83 | 0.873355 | logs/b0-rel-standard-s0.eval.stderr.txt |
| b0-rel-standard-s1 | 1.000e+13 | standard | rel | 1 | done | STEP-ONLY | 156.77 | 0.873509 | logs/b0-rel-standard-s1.eval.stderr.txt |
| b0-rel-standard-s2 | 1.000e+13 | standard | rel | 2 | done | STEP-ONLY | 172.55 | 0.871588 | logs/b0-rel-standard-s2.eval.stderr.txt |
| b0-hier-standard-s0 | 1.000e+13 | standard | hier | 0 | done | STEP-ONLY | 94.35 | 0.475375 | logs/b0-hier-standard-s0.eval.stderr.txt |
| b0-hier-standard-s1 | 1.000e+13 | standard | hier | 1 | done | STEP-ONLY | 115.41 | 0.471410 | logs/b0-hier-standard-s1.eval.stderr.txt |
| b0-hier-standard-s2 | 1.000e+13 | standard | hier | 2 | done | STEP-ONLY | 110.56 | 0.479334 | logs/b0-hier-standard-s2.eval.stderr.txt |
| b0-placebo-standard-s0 | 1.000e+13 | standard | placebo | 0 | done | OFF-FLOOR | 95.81 | 1.474728 | logs/b0-placebo-standard-s0.eval.stderr.txt |
| b0-placebo-standard-s1 | 1.000e+13 | standard | placebo | 1 | done | OFF-FLOOR | 97.0 | 1.442978 | logs/b0-placebo-standard-s1.eval.stderr.txt |
| b0-placebo-standard-s2 | 1.000e+13 | standard | placebo | 2 | done | OFF-FLOOR | 196.23 | 1.467738 | logs/b0-placebo-standard-s2.eval.stderr.txt |

> **Evidence quarantine:** 0 cell(s) planned fewer than 10 optimizer steps; 9 additional cell(s) did not belong to a completed standard cohort whose every seed and lower 95% CI cleared the artifact-recorded answer prior. All are excluded from every ci-v6 verdict pool.

## Standard-baseline off-floor gate

- Budget `10000000000000.0`:
  - `bag`: **BLOCKED** — at least one standard training seed did not clear its artifact-recorded prior
  - `rel`: **BLOCKED** — at least one standard training seed did not clear its artifact-recorded prior
  - `hier`: **BLOCKED** — at least one standard training seed did not clear its artifact-recorded prior

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
- Budget `10000000000000.0` FDR: 0 supported; 0 survive BH at q=0.1.

Raw contracts: `summary.json` and `manifest.json`.
