# vdc.4 Phase-0 copyops sizing result

This is a quarantined sizing artifact. It is not eligible as evidence for any
mechanism hypothesis.

- Bead: `model_guided_research-vdc.4`
- Producer: clean detached commit `f6f7d7b61c8afa379de82c5976552bd3a521c431`
- Runtime: 4,654.6 seconds
- Completion: 18/18 cells done, 0 failed
- Coordinate: standard attention, L2/H2/KV2/d64, T64, B4
- Training seeds: 0, 1, 2
- Eval seeds: 0, 1, 2; 48 examples per split
- Dataset: 1,000 documents, seed 42
- Full provenance: [manifest.json](manifest.json)

## Preregistered selection rule

Select the lowest rung where every standard training seed exceeds its own
artifact-recorded held-out answer prior and the per-seed held-out distribution
is unimodal rather than floor-bimodal. If no rung clears, preregister a larger
probe rather than treating these runs as evidence.

## Result

| Target FLOPs | Steps | Final loss by training seed | Held-out exact match | Recorded prior | Mean held-out PPL | Qualified |
| ---: | ---: | --- | --- | --- | ---: | :---: |
| 5e10 | 10 | 10.1956, 10.2258, 10.2481 | 0, 0, 0 | 0.020833 each | 30,648.28 | No |
| 2e11 | 40 | 7.4717, 7.4914, 7.5363 | 0, 0, 0 | 0.020833 each | 3,204.23 | No |
| 5e11 | 98 | 2.4665, 2.3056, 2.3964 | 0, 0, 0 | 0.020833 each | 24.90 | No |

No rung qualified. In-range exact match was also zero for every checkpoint, so
this is not merely a held-out-length failure. Optimization was healthy: losses
and perplexities fell sharply without numerical failures, but the model had not
yet learned complete output sequences. The correct result is therefore
“insufficient sizing exposure,” not a verdict about braid, reversible, or any
other candidate mechanism.

The successor probe must use the exact fair production coordinate and change
only the target FLOPs. Checkpoints, raw logs, datasets, and generations remain
local and untouched under this quarantine directory.
