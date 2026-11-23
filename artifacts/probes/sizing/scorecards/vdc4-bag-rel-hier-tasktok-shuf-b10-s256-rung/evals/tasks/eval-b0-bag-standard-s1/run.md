# eval-tasks — eval-b0-bag-standard-s1

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/scorecards/vdc4-bag-rel-hier-tasktok-shuf-b10-s256-rung/runs/budget_0/bag/standard/seed_1/checkpoints` @ step 1589
- attention_type: standard
- n_params: 802,816
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| bag | 0.486 | 0.236 | 3.6/13.9 | - | [curve](curve_bag.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
