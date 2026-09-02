# eval-tasks — eval-b0-arith-standard-s2

- checkpoint: `/data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-arith-tasktok-shuf-b1-rung/runs/budget_0/arith/standard/seed_2/checkpoints` @ step 4967
- attention_type: standard
- n_params: 139,264
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| arith | 1.000 | 0.847 | 2.7/16.4 | -0.0329 [-0.0531,-0.0127] | [curve](curve_arith.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
