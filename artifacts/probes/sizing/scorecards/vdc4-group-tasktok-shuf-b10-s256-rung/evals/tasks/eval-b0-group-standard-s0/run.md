# eval-tasks — eval-b0-group-standard-s0

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-group-tasktok-shuf-b10-s256-rung/runs/budget_0/group/standard/seed_0/checkpoints` @ step 1576
- attention_type: standard
- n_params: 819,200
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| group | 0.736 | 0.042 | 1.5/2.3 | +0.0001 [-0.0022,+0.0025] | [curve](curve_group.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
