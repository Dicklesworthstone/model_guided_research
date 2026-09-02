# eval-tasks — standard-s5

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/control-no-context/standard-s5/checkpoints` @ step 4967
- attention_type: standard
- n_params: 139,264
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| arith | 1.000 | 0.875 | 2.7/12.9 | -0.0235 [-0.0423,-0.0047] | [curve](curve_arith.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
