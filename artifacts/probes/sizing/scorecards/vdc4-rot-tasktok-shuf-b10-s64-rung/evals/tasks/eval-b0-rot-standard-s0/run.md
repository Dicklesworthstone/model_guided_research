# eval-tasks — eval-b0-rot-standard-s0

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-rot-tasktok-shuf-b10-s64-rung/runs/budget_0/rot/standard/seed_0/checkpoints` @ step 7868
- attention_type: standard
- n_params: 802,816
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| rot | 0.083 | 0.014 | 2.2/4.9 | +0.0042 [-0.0071,+0.0154] | [curve](curve_rot.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
