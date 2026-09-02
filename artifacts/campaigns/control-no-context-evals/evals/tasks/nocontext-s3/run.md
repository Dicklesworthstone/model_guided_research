# eval-tasks — nocontext-s3

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/control-no-context/nocontext-s3/checkpoints` @ step 4967
- attention_type: standard
- n_params: 139,264
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| arith | 0.521 | 0.507 | 3.1/12.6 | -0.0013 [-0.0303,+0.0277] | [curve](curve_arith.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
