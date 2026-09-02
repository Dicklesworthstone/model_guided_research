# eval-tasks — standard-s4

- checkpoint: `/data/projects/model_guided_research/artifacts/campaigns/group-comparison/standard-s4/checkpoints` @ step 4967
- attention_type: standard
- n_params: 139,264
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| group | 0.653 | 0.021 | 1.5/5.3 | +0.0015 [-0.0002,+0.0032] | [curve](curve_group.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
