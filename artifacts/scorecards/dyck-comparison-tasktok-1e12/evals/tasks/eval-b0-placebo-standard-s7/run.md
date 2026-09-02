# eval-tasks — eval-b0-placebo-standard-s7

- checkpoint: `/data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12/runs/budget_0/placebo/standard/seed_7/checkpoints` @ step 4967
- attention_type: standard
- n_params: 139,264
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| placebo | - | - | 4.2/48.9 | - | - |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
