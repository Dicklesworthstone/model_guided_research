# eval-tasks — eval-b0-placebo-ultrametric-s0

- checkpoint: `/data/projects/model_guided_research/artifacts/scorecards/placebo-guard-tasktok-1e12/runs/budget_0/placebo/ultrametric/seed_0/checkpoints` @ step 4947
- attention_type: ultrametric
- n_params: 139,776
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| placebo | - | - | 4.2/54.0 | - | - |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
