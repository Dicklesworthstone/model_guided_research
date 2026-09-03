# Mechanism scorecard — placebo-guard-tasktok-1e12

- Original command: `/data/projects/model_guided_research-worktrees/vdc4-copyops-tasktok-shuf-5dce958/cli.py scorecard --budget 1e12 --mechanism tropical --mechanism ultrametric --mechanism simplicial --mechanism quaternion --mechanism braid --mechanism fractal --mechanism surreal --mechanism reversible --mechanism gauge --mechanism octonion --task placebo --seeds 3 --eval-seeds 0,1,2 --examples 48 --dataset-size 50000 --dataset-seed 42 --device cpu --batch-size 4 --sequence-len 64 --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 64 --learning-rate 6e-4 --optimizer-type adamw --tokenizer task --warmup-steps 0 --log-interval 1 --val-interval 0 --checkpoint-interval 1000 --min-evidence-steps 10 --timeout-s 7200 --artifacts-dir /data/projects/model_guided_research/artifacts --run-id placebo-guard-tasktok-1e12 --fresh`
- Resume invocations: `4`
- Budgets: `[1000000000000.0]`
- Mechanisms: `['standard', 'tropical', 'ultrametric', 'simplicial', 'quaternion', 'braid', 'fractal', 'surreal', 'reversible', 'gauge', 'octonion']`
- Tasks: `['placebo']`
- Training seeds: `[0, 1, 2]`
- Runtime: `13995.7s`
- Cells: `33/33` done
- Adjudication policy: `ci-v6`

## Cell matrix

| cell | budget | mechanism | task | seed | status | evidence | elapsed s | final loss | logs |
|---|---:|---|---|---:|---|---|---:|---:|---|
| b0-placebo-standard-s0 | 1.000e+12 | standard | placebo | 0 | done | OFF-FLOOR | 1111.44 | 1.512557 | logs/b0-placebo-standard-s0.eval.stderr.txt |
| b0-placebo-standard-s1 | 1.000e+12 | standard | placebo | 1 | done | OFF-FLOOR | 962.09 | 1.519608 | logs/b0-placebo-standard-s1.eval.stderr.txt |
| b0-placebo-standard-s2 | 1.000e+12 | standard | placebo | 2 | done | OFF-FLOOR | 906.01 | 1.464111 | logs/b0-placebo-standard-s2.eval.stderr.txt |
| b0-placebo-tropical-s0 | 1.000e+12 | tropical | placebo | 0 | done | OFF-FLOOR | 1224.87 | 1.514039 | logs/b0-placebo-tropical-s0.eval.stderr.txt |
| b0-placebo-tropical-s1 | 1.000e+12 | tropical | placebo | 1 | done | OFF-FLOOR | 712.66 | 1.522137 | logs/b0-placebo-tropical-s1.eval.stderr.txt |
| b0-placebo-tropical-s2 | 1.000e+12 | tropical | placebo | 2 | done | OFF-FLOOR | 106.48 | 1.459287 | logs/b0-placebo-tropical-s2.eval.stderr.txt |
| b0-placebo-ultrametric-s0 | 1.000e+12 | ultrametric | placebo | 0 | done | OFF-FLOOR | 137.01 | 1.487334 | logs/b0-placebo-ultrametric-s0.eval.stderr.txt |
| b0-placebo-ultrametric-s1 | 1.000e+12 | ultrametric | placebo | 1 | done | OFF-FLOOR | 210.26 | 1.452260 | logs/b0-placebo-ultrametric-s1.eval.stderr.txt |
| b0-placebo-ultrametric-s2 | 1.000e+12 | ultrametric | placebo | 2 | done | OFF-FLOOR | 133.56 | 1.518479 | logs/b0-placebo-ultrametric-s2.eval.stderr.txt |
| b0-placebo-simplicial-s0 | 1.000e+12 | simplicial | placebo | 0 | done | OFF-FLOOR | 88.11 | 1.362544 | logs/b0-placebo-simplicial-s0.eval.stderr.txt |
| b0-placebo-simplicial-s1 | 1.000e+12 | simplicial | placebo | 1 | done | OFF-FLOOR | 79.93 | 1.528515 | logs/b0-placebo-simplicial-s1.eval.stderr.txt |
| b0-placebo-simplicial-s2 | 1.000e+12 | simplicial | placebo | 2 | done | OFF-FLOOR | 80.26 | 1.363766 | logs/b0-placebo-simplicial-s2.eval.stderr.txt |
| b0-placebo-quaternion-s0 | 1.000e+12 | quaternion | placebo | 0 | done | OFF-FLOOR | 83.74 | 1.512638 | logs/b0-placebo-quaternion-s0.eval.stderr.txt |
| b0-placebo-quaternion-s1 | 1.000e+12 | quaternion | placebo | 1 | done | OFF-FLOOR | 102.19 | 1.521469 | logs/b0-placebo-quaternion-s1.eval.stderr.txt |
| b0-placebo-quaternion-s2 | 1.000e+12 | quaternion | placebo | 2 | done | OFF-FLOOR | 281.63 | 1.460949 | logs/b0-placebo-quaternion-s2.eval.stderr.txt |
| b0-placebo-braid-s0 | 1.000e+12 | braid | placebo | 0 | done | OFF-FLOOR | 302.73 | 1.499685 | logs/b0-placebo-braid-s0.eval.stderr.txt |
| b0-placebo-braid-s1 | 1.000e+12 | braid | placebo | 1 | done | OFF-FLOOR | 225.06 | 1.383682 | logs/b0-placebo-braid-s1.eval.stderr.txt |
| b0-placebo-braid-s2 | 1.000e+12 | braid | placebo | 2 | done | OFF-FLOOR | 96.17 | 1.545245 | logs/b0-placebo-braid-s2.eval.stderr.txt |
| b0-placebo-fractal-s0 | 1.000e+12 | fractal | placebo | 0 | done | OFF-FLOOR | 97.75 | 1.489247 | logs/b0-placebo-fractal-s0.eval.stderr.txt |
| b0-placebo-fractal-s1 | 1.000e+12 | fractal | placebo | 1 | done | OFF-FLOOR | 101.17 | 1.452998 | logs/b0-placebo-fractal-s1.eval.stderr.txt |
| b0-placebo-fractal-s2 | 1.000e+12 | fractal | placebo | 2 | done | OFF-FLOOR | 98.32 | 1.521386 | logs/b0-placebo-fractal-s2.eval.stderr.txt |
| b0-placebo-surreal-s0 | 1.000e+12 | surreal | placebo | 0 | done | OFF-FLOOR | 97.17 | 1.435496 | logs/b0-placebo-surreal-s0.eval.stderr.txt |
| b0-placebo-surreal-s1 | 1.000e+12 | surreal | placebo | 1 | done | OFF-FLOOR | 100.8 | 1.455496 | logs/b0-placebo-surreal-s1.eval.stderr.txt |
| b0-placebo-surreal-s2 | 1.000e+12 | surreal | placebo | 2 | done | OFF-FLOOR | 78.02 | 1.421780 | logs/b0-placebo-surreal-s2.eval.stderr.txt |
| b0-placebo-reversible-s0 | 1.000e+12 | reversible | placebo | 0 | done | OFF-FLOOR | 123.86 | 1.508910 | logs/b0-placebo-reversible-s0.eval.stderr.txt |
| b0-placebo-reversible-s1 | 1.000e+12 | reversible | placebo | 1 | done | OFF-FLOOR | 134.5 | 1.480382 | logs/b0-placebo-reversible-s1.eval.stderr.txt |
| b0-placebo-reversible-s2 | 1.000e+12 | reversible | placebo | 2 | done | OFF-FLOOR | 98.46 | 1.477812 | logs/b0-placebo-reversible-s2.eval.stderr.txt |
| b0-placebo-gauge-s0 | 1.000e+12 | gauge | placebo | 0 | done | OFF-FLOOR | 103.44 | 1.451283 | logs/b0-placebo-gauge-s0.eval.stderr.txt |
| b0-placebo-gauge-s1 | 1.000e+12 | gauge | placebo | 1 | done | OFF-FLOOR | 145.57 | 1.418849 | logs/b0-placebo-gauge-s1.eval.stderr.txt |
| b0-placebo-gauge-s2 | 1.000e+12 | gauge | placebo | 2 | done | OFF-FLOOR | 193.7 | 1.479185 | logs/b0-placebo-gauge-s2.eval.stderr.txt |
| b0-placebo-octonion-s0 | 1.000e+12 | octonion | placebo | 0 | done | OFF-FLOOR | 572.0 | 1.511272 | logs/b0-placebo-octonion-s0.eval.stderr.txt |
| b0-placebo-octonion-s1 | 1.000e+12 | octonion | placebo | 1 | done | OFF-FLOOR | 475.89 | 1.521320 | logs/b0-placebo-octonion-s1.eval.stderr.txt |
| b0-placebo-octonion-s2 | 1.000e+12 | octonion | placebo | 2 | done | OFF-FLOOR | 4730.68 | 1.458906 | logs/b0-placebo-octonion-s2.eval.stderr.txt |

## Standard-baseline off-floor gate

- Budget `1000000000000.0`:

## Placebo publication gate

**BLOCKED**
- braid: no selected placebo hypothesis covers this mechanism
- fractal: no selected placebo hypothesis covers this mechanism
- gauge: no selected placebo hypothesis covers this mechanism
- octonion: no selected placebo hypothesis covers this mechanism
- quaternion: no selected placebo hypothesis covers this mechanism
- reversible: no selected placebo hypothesis covers this mechanism
- simplicial: no selected placebo hypothesis covers this mechanism
- surreal: no selected placebo hypothesis covers this mechanism
- tropical: no selected placebo hypothesis covers this mechanism
- ultrametric: no selected placebo hypothesis covers this mechanism
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
