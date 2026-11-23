# eval-tasks — batch16-s0

- checkpoint: `/Volumes/SANDISK_SSD/scratch/mgr-fixes-20261122.E905R8/research/artifacts/probes/sizing/copyops-lrprobe/nanochat/batch16-s0/checkpoints` @ step 1281
- attention_type: standard
- n_params: 131,072
- seeds: [0, 1, 2] · examples/seed: 48 · decode: ['greedy']

| task | EM in-range | EM held-out | ppl in/held | slope held-out [CI95] | curve |
|---|---|---|---|---|---|
| copyops | 0.000 | 0.000 | 9.4/31.0 | +0.0000 [+0.0000,+0.0000] | [curve](curve_copyops.png) |

See `summary.json` (schema mgr.evaltasks.v4) for the full contract output and `generations.jsonl` for per-example receipts.
