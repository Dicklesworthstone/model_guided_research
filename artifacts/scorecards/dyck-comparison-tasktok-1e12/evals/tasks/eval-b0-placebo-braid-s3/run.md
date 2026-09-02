# eval-tasks — eval-b0-placebo-braid-s3

- checkpoint: `/data/projects/model_guided_research/artifacts/scorecards/dyck-comparison-tasktok-1e12/runs/budget_0/placebo/braid/seed_3/checkpoints` @ step 4965
- attention_type: braid
- n_params: 139,296
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| placebo | - | - | 4.2/16.2 | - | - |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
