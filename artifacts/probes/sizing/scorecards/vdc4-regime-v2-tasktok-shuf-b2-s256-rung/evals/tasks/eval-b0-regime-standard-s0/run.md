# eval-tasks — eval-b0-regime-standard-s0

- checkpoint: `/data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-regime-v2-tasktok-shuf-b2-s256-rung/runs/budget_0/regime/standard/seed_0/checkpoints` @ step 1806
- attention_type: standard
- n_params: 139,264
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| regime | 0.104 | 0.083 | 2.1/2.4 | - | [curve](curve_regime.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
