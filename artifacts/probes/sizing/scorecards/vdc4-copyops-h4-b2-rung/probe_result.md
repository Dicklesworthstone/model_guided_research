# vdc.4 Phase-0c exact-H4 copyops sizing result

This is a quarantined sizing artifact. It is not eligible as evidence for any
mechanism hypothesis.

- Bead: `model_guided_research-vdc.4`
- Producer: clean detached commit `c71f7f67fabfe83db66bae2bda0cf071135bb40a`
- Runtime: 2,990.5 seconds
- Completion: 6/6 cells done, 0 failed
- Coordinate: standard attention, L2/H4/KV2/d64, T64, B4
- Target: 2e12 FLOPs, 392 steps, 2.002595610624e12 planned FLOPs
- Training seeds: 0, 1, 2
- Eval seeds: 0, 1, 2; 48 examples per split
- Dataset: 1,000 documents, seed 42
- Full provenance: [manifest.json](manifest.json)

## Preregistered selection rule

The rung qualifies only if all three standard copyops checkpoints exceed their
own artifact-recorded held-out answer prior and the per-seed held-out
distribution is non-pathological and unimodal rather than floor-bimodal. If
the rung fails, preregister one fresh budget-only successor and change no
model, optimizer, corpus, task, seed, or evaluation knob.

## Result

| Train seed | Final loss | Held-out exact match | In-range exact match | Recorded prior | Held-out PPL | In-range PPL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.8556 | 0 | 0 | 0.020833 | 13.8310 | 7.1388 |
| 1 | 1.8627 | 0 | 0 | 0.020833 | 13.9105 | 7.1265 |
| 2 | 1.8675 | 0 | 0 | 0.020833 | 13.8259 | 7.1284 |

No checkpoint qualified. All 864 scored copyops examples completed with zero
skips, but every checkpoint produced zero exact matches in both regions. The
distribution is uniformly floored, not bimodal. All three checkpoints emitted
the same prompt-blind `e OUT` local grammar observed at 1e12. Doubling exposure
therefore did not change the qualitative failure mode, and low cross-entropy
is still not evidence that the model learned the stateful COPY/REV/ROT
operation.

The quarantined placebo cells also completed cleanly. Their held-out
perplexities were 33.9827, 47.2194, and 53.8676; their in-range perplexities
were 4.3504, 4.2245, and 4.2832, with zero skipped examples.

## Decision

The preregistered 2e12 rung fails. The fresh successor is 4e12 FLOPs at the
same exact coordinate, giving 783 steps and 4.000082558976e12 planned FLOPs.
Target FLOPs is the only changed knob. That successor remains quarantined and
cannot contribute claim evidence. Checkpoints, raw logs, datasets, and
generation receipts remain local and untouched under this directory.
