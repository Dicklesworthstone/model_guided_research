# eval-tasks — eval-b0-needle-standard-s0

- checkpoint: `/data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-needle-tasktok-shuf-b1-rung/runs/budget_0/needle/standard/seed_0/checkpoints` @ step 4816
- attention_type: standard
- n_params: 147,456
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| needle | 0.007 | 0.000 | 19.3/29.5 | - | [curve](curve_needle.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
