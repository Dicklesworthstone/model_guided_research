# eval-tasks — rmatrix-s3

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/group-comparison/rmatrix-s3/checkpoints` @ step 4780
- attention_type: braid
- n_params: 144,392
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| group | 0.590 | 0.056 | 1.5/27.4 | +0.0009 [-0.0018,+0.0037] | [curve](curve_group.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
