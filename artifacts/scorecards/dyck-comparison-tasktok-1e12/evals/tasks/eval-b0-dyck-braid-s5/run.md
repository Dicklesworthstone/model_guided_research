# eval-tasks — eval-b0-dyck-braid-s5

- checkpoint: `/data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12/runs/budget_0/dyck/braid/seed_5/checkpoints` @ step 5125
- attention_type: braid
- n_params: 131,104
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| dyck | 1.000 | 0.910 | 2.9/3.9 | -0.0668 [-0.0884,-0.0452] | [curve](curve_dyck.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
