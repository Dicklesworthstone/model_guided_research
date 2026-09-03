# eval-tasks — eval-b0-placebo-surreal-s1

- checkpoint: `/data/projects/model_guided_research/artifacts/scorecards/placebo-guard-tasktok-1e12/runs/budget_0/placebo/surreal/seed_1/checkpoints` @ step 4952
- attention_type: surreal
- n_params: 139,648
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| placebo | - | - | 4.2/50.1 | - | - |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
