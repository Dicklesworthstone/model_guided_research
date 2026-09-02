# eval-tasks — standard-s3

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/control-no-context/standard-s3/checkpoints` @ step 4967
- attention_type: standard
- n_params: 139,264
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| arith | 1.000 | 0.812 | 2.7/15.9 | -0.0349 [-0.0568,-0.0130] | [curve](curve_arith.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
