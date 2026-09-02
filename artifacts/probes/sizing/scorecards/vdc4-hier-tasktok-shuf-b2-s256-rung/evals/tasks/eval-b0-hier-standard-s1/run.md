# eval-tasks — eval-b0-hier-standard-s1

- checkpoint: `/data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-hier-tasktok-shuf-b2-s256-rung/runs/budget_0/hier/standard/seed_1/checkpoints` @ step 1806
- attention_type: standard
- n_params: 139,264
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| hier | 0.021 | 0.021 | 1.6/33.3 | +0.0099 [-0.0216,+0.0414] | [curve](curve_hier.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
