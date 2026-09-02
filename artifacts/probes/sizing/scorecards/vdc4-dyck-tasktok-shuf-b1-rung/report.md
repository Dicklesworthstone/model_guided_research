# Mechanism scorecard — vdc4-dyck-tasktok-shuf-b1-rung

- Original command: `/data/projects/model_guided_research-worktrees/vdc4-copyops-tasktok-shuf-5dce958/cli.py scorecard --budget 1e12 --mechanism standard --task dyck --seeds 3 --eval-seeds 0,1,2 --examples 48 --dataset-size 50000 --dataset-seed 42 --device cpu --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 6e-4 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --val-interval 0 --checkpoint-interval 1000 --min-evidence-steps 10 --timeout-s 7200 --artifacts-dir /data/projects/model_guided_research/artifacts/probes/sizing --run-id vdc4-dyck-tasktok-shuf-b1-rung --fresh`
- Resume invocations: `0`
- Budgets: `[1000000000000.0]`
- Mechanisms: `['standard']`
- Tasks: `['dyck', 'placebo']`
- Training seeds: `[0, 1, 2]`
- Runtime: `315.3s`
- Cells: `6/6` done
- Adjudication policy: `ci-v6`

## Cell matrix

| cell | budget | mechanism | task | seed | status | evidence | elapsed s | final loss | logs |
|---|---:|---|---|---:|---|---|---:|---:|---|
| b0-dyck-standard-s0 | 1.000e+12 | standard | dyck | 0 | done | OFF-FLOOR | 52.26 | 1.002629 | logs/b0-dyck-standard-s0.eval.stderr.txt |
| b0-dyck-standard-s1 | 1.000e+12 | standard | dyck | 1 | done | OFF-FLOOR | 54.79 | 1.016330 | logs/b0-dyck-standard-s1.eval.stderr.txt |
| b0-dyck-standard-s2 | 1.000e+12 | standard | dyck | 2 | done | OFF-FLOOR | 52.13 | 0.954449 | logs/b0-dyck-standard-s2.eval.stderr.txt |
| b0-placebo-standard-s0 | 1.000e+12 | standard | placebo | 0 | done | OFF-FLOOR | 50.71 | 1.512557 | logs/b0-placebo-standard-s0.eval.stderr.txt |
| b0-placebo-standard-s1 | 1.000e+12 | standard | placebo | 1 | done | OFF-FLOOR | 51.56 | 1.519608 | logs/b0-placebo-standard-s1.eval.stderr.txt |
| b0-placebo-standard-s2 | 1.000e+12 | standard | placebo | 2 | done | OFF-FLOOR | 53.81 | 1.464111 | logs/b0-placebo-standard-s2.eval.stderr.txt |

## Standard-baseline off-floor gate

- Budget `1000000000000.0`:
  - `dyck`: **CLEAR** — lower CI=0.8365601851851852, prior=0.5625

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
