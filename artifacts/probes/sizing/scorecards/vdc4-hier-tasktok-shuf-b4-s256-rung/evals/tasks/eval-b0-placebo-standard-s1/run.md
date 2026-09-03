# eval-tasks — eval-b0-placebo-standard-s1

- checkpoint: `/data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-hier-tasktok-shuf-b4-s256-rung/runs/budget_0/placebo/standard/seed_1/checkpoints` @ step 3612
- attention_type: standard
- n_params: 139,264
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| placebo | - | - | 4.3/82.1 | - | - |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
