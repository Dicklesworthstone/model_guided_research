# eval-tasks — eval-b0-rot-standard-s1

- checkpoint: `/data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-rot-tasktok-shuf-b2-rung/runs/budget_0/rot/standard/seed_1/checkpoints` @ step 10254
- attention_type: standard
- n_params: 131,072
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| rot | 0.125 | 0.042 | 2.1/4.1 | +0.0077 [-0.0115,+0.0270] | [curve](curve_rot.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
