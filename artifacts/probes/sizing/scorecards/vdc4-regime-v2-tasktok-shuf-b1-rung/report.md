# Mechanism scorecard — vdc4-regime-v2-tasktok-shuf-b1-rung

- Original command: `/data/projects/model_guided_research-worktrees/regime-v2-e7f5845/cli.py scorecard --budget 1e12 --mechanism standard --task regime --seeds 3 --eval-seeds 0,1,2 --examples 48 --dataset-size 50000 --dataset-seed 42 --device cpu --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 6e-4 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --val-interval 0 --checkpoint-interval 1000 --min-evidence-steps 10 --timeout-s 36000 --artifacts-dir /data/projects/model_guided_research/artifacts/probes/sizing --run-id vdc4-regime-v2-tasktok-shuf-b1-rung --fresh`
- Resume invocations: `0`
- Budgets: `[1000000000000.0]`
- Mechanisms: `['standard']`
- Tasks: `['regime', 'placebo']`
- Training seeds: `[0, 1, 2]`
- Runtime: `1592.5s`
- Cells: `6/6` done
- Adjudication policy: `ci-v6`

## Cell matrix

| cell | budget | mechanism | task | seed | status | evidence | elapsed s | final loss | logs |
|---|---:|---|---|---:|---|---|---:|---:|---|
| b0-regime-standard-s0 | 1.000e+12 | standard | regime | 0 | done | STEP-ONLY | 1099.95 | 0.376907 | logs/b0-regime-standard-s0.eval.stderr.txt |
| b0-regime-standard-s1 | 1.000e+12 | standard | regime | 1 | done | STEP-ONLY | 79.77 | 0.392718 | logs/b0-regime-standard-s1.eval.stderr.txt |
| b0-regime-standard-s2 | 1.000e+12 | standard | regime | 2 | done | STEP-ONLY | 79.44 | 0.351491 | logs/b0-regime-standard-s2.eval.stderr.txt |
| b0-placebo-standard-s0 | 1.000e+12 | standard | placebo | 0 | done | OFF-FLOOR | 91.92 | 1.512557 | logs/b0-placebo-standard-s0.eval.stderr.txt |
| b0-placebo-standard-s1 | 1.000e+12 | standard | placebo | 1 | done | OFF-FLOOR | 74.82 | 1.519608 | logs/b0-placebo-standard-s1.eval.stderr.txt |
| b0-placebo-standard-s2 | 1.000e+12 | standard | placebo | 2 | done | OFF-FLOOR | 166.58 | 1.464111 | logs/b0-placebo-standard-s2.eval.stderr.txt |

> **Evidence quarantine:** 0 cell(s) planned fewer than 10 optimizer steps; 3 additional cell(s) did not belong to a completed standard cohort whose every seed and lower 95% CI cleared the artifact-recorded answer prior. All are excluded from every ci-v6 verdict pool.

## Standard-baseline off-floor gate

- Budget `1000000000000.0`:
  - `regime`: **BLOCKED** — at least one standard training seed did not clear its artifact-recorded prior

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
- Budget `1000000000000.0` FDR: 0 supported; 0 survive BH at q=0.1.

Raw contracts: `summary.json` and `manifest.json`.
