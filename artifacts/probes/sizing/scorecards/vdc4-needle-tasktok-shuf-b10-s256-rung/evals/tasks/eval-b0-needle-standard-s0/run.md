# eval-tasks — eval-b0-needle-standard-s0

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-needle-tasktok-shuf-b10-s256-rung/runs/budget_0/needle/standard/seed_0/checkpoints` @ step 1564
- attention_type: standard
- n_params: 835,584
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| needle | 0.014 | 0.000 | 19.2/34.6 | - | [curve](curve_needle.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
