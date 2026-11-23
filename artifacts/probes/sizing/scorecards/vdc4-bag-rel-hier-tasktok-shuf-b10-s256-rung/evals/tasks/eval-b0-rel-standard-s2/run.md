# eval-tasks — eval-b0-rel-standard-s2

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-bag-rel-hier-tasktok-shuf-b10-s256-rung/runs/budget_0/rel/standard/seed_2/checkpoints` @ step 1589
- attention_type: standard
- n_params: 802,816
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| rel | 0.083 | 0.083 | 2.4/12.5 | - | [curve](curve_rel.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
