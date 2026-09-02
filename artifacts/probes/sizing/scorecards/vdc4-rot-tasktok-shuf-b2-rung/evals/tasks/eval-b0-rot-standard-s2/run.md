# eval-tasks — eval-b0-rot-standard-s2

- checkpoint: `/data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-rot-tasktok-shuf-b2-rung/runs/budget_0/rot/standard/seed_2/checkpoints` @ step 10254
- attention_type: standard
- n_params: 131,072
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| rot | 0.090 | 0.035 | 2.2/6.6 | -0.0122 [-0.0297,+0.0053] | [curve](curve_rot.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
