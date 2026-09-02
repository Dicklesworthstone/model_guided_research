# eval-tasks — rmatrix-s5

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/group-comparison/rmatrix-s5/checkpoints` @ step 4780
- attention_type: braid
- n_params: 144,392
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| group | 0.521 | 0.028 | 1.6/34.9 | +0.0000 [-0.0020,+0.0020] | [curve](curve_group.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
