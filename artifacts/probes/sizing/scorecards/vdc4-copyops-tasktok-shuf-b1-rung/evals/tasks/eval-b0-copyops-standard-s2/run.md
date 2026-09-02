# eval-tasks — eval-b0-copyops-standard-s2

- checkpoint: `/data/projects/model_guided_research/artifacts/probes/sizing/scorecards/vdc4-copyops-tasktok-shuf-b1-rung/runs/budget_0/copyops/standard/seed_2/checkpoints` @ step 5127
- attention_type: standard
- n_params: 131,072
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| copyops | 0.000 | 0.000 | 9.4/56.7 | +0.0000 [+0.0000,+0.0000] | [curve](curve_copyops.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
