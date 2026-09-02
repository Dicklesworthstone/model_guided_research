# eval-tasks — eval-b0-dyck-standard-s6

- checkpoint: `/data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12/runs/budget_0/dyck/standard/seed_6/checkpoints` @ step 5127
- attention_type: standard
- n_params: 131,072
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| dyck | 0.979 | 0.854 | 2.9/3.7 | -0.0589 [-0.0871,-0.0307] | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
