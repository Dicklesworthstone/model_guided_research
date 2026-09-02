# Mechanism scorecard — dyck-comparison-tasktok-1e12

- Original command: `/data/projects/model_guided_research-worktrees/comparisons-49f59e7/cli.py scorecard --budget 1e12 --mechanism braid --task dyck --seeds 5 --seed-offset 3 --eval-seeds 0,1,2 --examples 48 --dataset-size 50000 --dataset-seed 42 --device cpu --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 6e-4 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --val-interval 0 --checkpoint-interval 1000 --min-evidence-steps 10 --timeout-s 7200 --artifacts-dir /data/projects/model_guided_research/artifacts --run-id dyck-comparison-tasktok-1e12 --fresh`
- Resume invocations: `1`
- Budgets: `[1000000000000.0]`
- Mechanisms: `['standard', 'braid']`
- Tasks: `['dyck', 'placebo']`
- Training seeds: `[3, 4, 5, 6, 7]`
- Runtime: `6692.0s`
- Cells: `20/20` done
- Adjudication policy: `ci-v6`

## Cell matrix

| cell | budget | mechanism | task | seed | status | evidence | elapsed s | final loss | logs |
|---|---:|---|---|---:|---|---|---:|---:|---|
| b0-dyck-standard-s3 | 1.000e+12 | standard | dyck | 3 | done | OFF-FLOOR | 1225.72 | 0.997455 | logs/b0-dyck-standard-s3.eval.stderr.txt |
| b0-dyck-standard-s4 | 1.000e+12 | standard | dyck | 4 | done | OFF-FLOOR | 1128.82 | 1.094494 | logs/b0-dyck-standard-s4.eval.stderr.txt |
| b0-dyck-standard-s5 | 1.000e+12 | standard | dyck | 5 | done | OFF-FLOOR | 878.12 | 0.992038 | logs/b0-dyck-standard-s5.eval.stderr.txt |
| b0-dyck-standard-s6 | 1.000e+12 | standard | dyck | 6 | done | OFF-FLOOR | 1072.88 | 1.042669 | logs/b0-dyck-standard-s6.eval.stderr.txt |
| b0-dyck-standard-s7 | 1.000e+12 | standard | dyck | 7 | done | OFF-FLOOR | 590.46 | 1.076974 | logs/b0-dyck-standard-s7.eval.stderr.txt |
| b0-dyck-braid-s3 | 1.000e+12 | braid | dyck | 3 | done | OFF-FLOOR | 81.39 | 1.105009 | logs/b0-dyck-braid-s3.eval.stderr.txt |
| b0-dyck-braid-s4 | 1.000e+12 | braid | dyck | 4 | done | OFF-FLOOR | 84.57 | 1.061604 | logs/b0-dyck-braid-s4.eval.stderr.txt |
| b0-dyck-braid-s5 | 1.000e+12 | braid | dyck | 5 | done | OFF-FLOOR | 91.04 | 1.077557 | logs/b0-dyck-braid-s5.eval.stderr.txt |
| b0-dyck-braid-s6 | 1.000e+12 | braid | dyck | 6 | done | OFF-FLOOR | 132.23 | 1.021316 | logs/b0-dyck-braid-s6.eval.stderr.txt |
| b0-dyck-braid-s7 | 1.000e+12 | braid | dyck | 7 | done | OFF-FLOOR | 119.5 | 0.967954 | logs/b0-dyck-braid-s7.eval.stderr.txt |
| b0-placebo-standard-s3 | 1.000e+12 | standard | placebo | 3 | done | OFF-FLOOR | 79.58 | 1.443065 | logs/b0-placebo-standard-s3.eval.stderr.txt |
| b0-placebo-standard-s4 | 1.000e+12 | standard | placebo | 4 | done | OFF-FLOOR | 80.04 | 1.498600 | logs/b0-placebo-standard-s4.eval.stderr.txt |
| b0-placebo-standard-s5 | 1.000e+12 | standard | placebo | 5 | done | OFF-FLOOR | 74.63 | 1.435242 | logs/b0-placebo-standard-s5.eval.stderr.txt |
| b0-placebo-standard-s6 | 1.000e+12 | standard | placebo | 6 | done | OFF-FLOOR | 70.69 | 1.475453 | logs/b0-placebo-standard-s6.eval.stderr.txt |
| b0-placebo-standard-s7 | 1.000e+12 | standard | placebo | 7 | done | OFF-FLOOR | 69.59 | 1.489363 | logs/b0-placebo-standard-s7.eval.stderr.txt |
| b0-placebo-braid-s3 | 1.000e+12 | braid | placebo | 3 | done | OFF-FLOOR | 63.33 | 1.396513 | logs/b0-placebo-braid-s3.eval.stderr.txt |
| b0-placebo-braid-s4 | 1.000e+12 | braid | placebo | 4 | done | OFF-FLOOR | 75.99 | 1.443590 | logs/b0-placebo-braid-s4.eval.stderr.txt |
| b0-placebo-braid-s5 | 1.000e+12 | braid | placebo | 5 | done | OFF-FLOOR | 153.06 | 1.451364 | logs/b0-placebo-braid-s5.eval.stderr.txt |
| b0-placebo-braid-s6 | 1.000e+12 | braid | placebo | 6 | done | OFF-FLOOR | 278.1 | 1.525018 | logs/b0-placebo-braid-s6.eval.stderr.txt |
| b0-placebo-braid-s7 | 1.000e+12 | braid | placebo | 7 | done | OFF-FLOOR | 342.16 | 1.571007 | logs/b0-placebo-braid-s7.eval.stderr.txt |

## Standard-baseline off-floor gate

- Budget `1000000000000.0`:
  - `dyck`: **CLEAR** — lower CI=0.8411574643987417, prior=0.5625

## Placebo publication gate

**BLOCKED**
- braid: no selected placebo hypothesis covers this mechanism
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
