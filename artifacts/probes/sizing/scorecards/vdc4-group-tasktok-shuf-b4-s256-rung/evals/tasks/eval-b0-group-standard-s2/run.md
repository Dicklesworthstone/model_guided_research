# eval-tasks — eval-b0-group-standard-s2

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-group-tasktok-shuf-b4-s256-rung/runs/budget_0/group/standard/seed_2/checkpoints` @ step 3612
- attention_type: standard
- n_params: 139,264
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| group | 0.674 | 0.028 | 1.5/2.2 | -0.0004 [-0.0023,+0.0016] | [curve](curve_group.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
