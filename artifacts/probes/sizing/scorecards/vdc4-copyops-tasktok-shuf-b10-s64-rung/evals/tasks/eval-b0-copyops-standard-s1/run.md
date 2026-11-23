# eval-tasks — eval-b0-copyops-standard-s1

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-copyops-tasktok-shuf-b10-s64-rung/runs/budget_0/copyops/standard/seed_1/checkpoints` @ step 7868
- attention_type: standard
- n_params: 802,816
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| copyops | 0.000 | 0.000 | 9.0/74.1 | +0.0000 [+0.0000,+0.0000] | [curve](curve_copyops.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
