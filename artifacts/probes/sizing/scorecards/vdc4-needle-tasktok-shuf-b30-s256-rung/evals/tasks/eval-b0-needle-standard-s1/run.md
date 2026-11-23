# eval-tasks — eval-b0-needle-standard-s1

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-needle-tasktok-shuf-b30-s256-rung/runs/budget_0/needle/standard/seed_1/checkpoints` @ step 4693
- attention_type: standard
- n_params: 835,584
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| needle | 0.007 | 0.000 | 19.1/86.4 | - | [curve](curve_needle.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
