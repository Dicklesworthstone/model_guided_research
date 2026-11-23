# eval-tasks — eval-b0-rot-standard-s0

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-rot-tasktok-shuf-b4-rung/runs/budget_0/rot/standard/seed_0/checkpoints` @ step 20509
- attention_type: standard
- n_params: 131,072
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| rot | 0.306 | 0.042 | 2.1/6.6 | +0.0006 [-0.0187,+0.0198] | [curve](curve_rot.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
