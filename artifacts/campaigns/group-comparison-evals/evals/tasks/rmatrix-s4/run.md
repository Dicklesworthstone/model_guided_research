# eval-tasks — rmatrix-s4

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/group-comparison/rmatrix-s4/checkpoints` @ step 4780
- attention_type: braid
- n_params: 144,392
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| group | 0.611 | 0.021 | 1.5/78.5 | +0.0002 [-0.0015,+0.0019] | [curve](curve_group.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
