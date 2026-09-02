# eval-tasks — eval-b0-bag-standard-s0

- checkpoint: `/data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-bag-tasktok-shuf-b2-s256-rung/runs/budget_0/bag/standard/seed_0/checkpoints` @ step 1848
- attention_type: standard
- n_params: 131,072
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| bag | 0.451 | 0.292 | 3.6/13.8 | - | [curve](curve_bag.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
