# eval-tasks — eval-b0-dyck-braid-s3

- checkpoint: `/data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12/runs/budget_0/dyck/braid/seed_3/checkpoints` @ step 5125
- attention_type: braid
- n_params: 131,104
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| dyck | 0.993 | 0.868 | 3.0/3.9 | -0.0663 [-0.0928,-0.0399] | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
