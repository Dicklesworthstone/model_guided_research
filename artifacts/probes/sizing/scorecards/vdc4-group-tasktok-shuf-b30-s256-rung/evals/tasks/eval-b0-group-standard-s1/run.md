# eval-tasks — eval-b0-group-standard-s1

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-group-tasktok-shuf-b30-s256-rung/runs/budget_0/group/standard/seed_1/checkpoints` @ step 4730
- attention_type: standard
- n_params: 819,200
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| group | 0.924 | 0.035 | 1.4/2.5 | +0.0022 [+0.0000,+0.0044] | [curve](curve_group.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
