# Mechanism scorecard — vdc4-group-tasktok-shuf-b2-s256-rung

- Original command: `/data/projects/model_guided_research-worktrees/regime-v2-e7f5845/cli.py scorecard --budget 2e12 --mechanism standard --task group --seeds 3 --eval-seeds 0,1,2 --examples 48 --dataset-size 100000 --dataset-seed 42 --device cpu --batch-size 4 --sequence-len 256 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 6e-4 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --val-interval 0 --checkpoint-interval 1000 --min-evidence-steps 10 --timeout-s 36000 --artifacts-dir /data/projects/model_guided_research/artifacts/probes/sizing --run-id vdc4-group-tasktok-shuf-b2-s256-rung --fresh`
- Resume invocations: `0`
- Budgets: `[2000000000000.0]`
- Mechanisms: `['standard']`
- Tasks: `['group', 'placebo']`
- Training seeds: `[0, 1, 2]`
- Runtime: `1164.2s`
- Cells: `6/6` done
- Adjudication policy: `ci-v6`

## Cell matrix

| cell | budget | mechanism | task | seed | status | evidence | elapsed s | final loss | logs |
|---|---:|---|---|---:|---|---|---:|---:|---|
| b0-group-standard-s0 | 2.000e+12 | standard | group | 0 | done | STEP-ONLY | 74.26 | 0.428036 | logs/b0-group-standard-s0.eval.stderr.txt |
| b0-group-standard-s1 | 2.000e+12 | standard | group | 1 | done | STEP-ONLY | 114.16 | 0.427511 | logs/b0-group-standard-s1.eval.stderr.txt |
| b0-group-standard-s2 | 2.000e+12 | standard | group | 2 | done | STEP-ONLY | 59.04 | 0.419248 | logs/b0-group-standard-s2.eval.stderr.txt |
| b0-placebo-standard-s0 | 2.000e+12 | standard | placebo | 0 | done | OFF-FLOOR | 74.73 | 1.494446 | logs/b0-placebo-standard-s0.eval.stderr.txt |
| b0-placebo-standard-s1 | 2.000e+12 | standard | placebo | 1 | done | OFF-FLOOR | 81.66 | 1.475128 | logs/b0-placebo-standard-s1.eval.stderr.txt |
| b0-placebo-standard-s2 | 2.000e+12 | standard | placebo | 2 | done | OFF-FLOOR | 760.34 | 1.477779 | logs/b0-placebo-standard-s2.eval.stderr.txt |

> **Evidence quarantine:** 0 cell(s) planned fewer than 10 optimizer steps; 3 additional cell(s) did not belong to a completed standard cohort whose every seed and lower 95% CI cleared the artifact-recorded answer prior. All are excluded from every ci-v6 verdict pool.

## Standard-baseline off-floor gate

- Budget `2000000000000.0`:
  - `group`: **BLOCKED** — at least one standard training seed did not clear its artifact-recorded prior

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
- Budget `2000000000000.0` FDR: 0 supported; 0 survive BH at q=0.1.

Raw contracts: `summary.json` and `manifest.json`.
